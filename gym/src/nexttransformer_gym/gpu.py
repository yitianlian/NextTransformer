"""Operator GPU allocations, read-only discovery and cooperative local leases."""

import csv
import fcntl
import io
import os
import re
import stat
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

from .common import GymError, require
from .isolation import BACKENDS


UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def configuration(value=None):
    require(
        value is None or isinstance(value, dict),
        "invalid_request",
        "Execution configuration must be an object.",
    )
    value = {} if value is None else dict(value)
    require(
        not (
            set(value)
            - {
                "schema_version",
                "gpu_ids",
                "gpu_memory_limit_bytes",
                "max_concurrent_jobs",
                "allow_idle_graphics",
                "isolation",
            }
        ),
        "invalid_request",
        "Unknown execution configuration field.",
    )
    require(
        value.get("schema_version", "gym.execution.v1") == "gym.execution.v1",
        "invalid_request",
        "Unsupported execution configuration schema.",
    )
    ids = value.get("gpu_ids", [])
    require(
        isinstance(ids, list)
        and all(
            (type(x) is int and x >= 0)
            or (isinstance(x, str) and (x.isdecimal() or UUID.fullmatch(x)))
            for x in ids
        ),
        "invalid_request",
        "gpu_ids must contain physical indexes or complete NVIDIA GPU UUIDs.",
    )
    ids = [str(x) for x in ids]
    require(len(ids) == len(set(ids)), "invalid_request", "Duplicate GPU allocations.")
    memory = value.get("gpu_memory_limit_bytes", 4 * 1024**3)
    require(
        type(memory) is int and memory > 0,
        "invalid_request",
        "GPU memory budget must be a positive integer.",
    )
    require(
        type(value.get("max_concurrent_jobs", 1)) is int
        and value.get("max_concurrent_jobs", 1) == 1,
        "invalid_request",
        "This backend supports exactly one concurrent job.",
    )
    allow_graphics = value.get("allow_idle_graphics", False)
    require(
        type(allow_graphics) is bool,
        "invalid_request",
        "allow_idle_graphics must be an operator-supplied boolean.",
    )
    isolation = value.get("isolation", "bwrap")
    require(
        isolation in BACKENDS,
        "invalid_request",
        "isolation must be one of: " + ", ".join(BACKENDS) + ".",
    )
    return {
        "schema_version": "gym.execution.v1",
        "gpu_ids": ids,
        "gpu_memory_limit_bytes": memory,
        "max_concurrent_jobs": 1,
        "allow_idle_graphics": allow_graphics,
        "isolation": isolation,
    }


def _query(fields, kind="gpu"):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def _process_counts(xml):
    """Count every live compute/graphics process, without exposing host PIDs."""
    root = ET.fromstring(xml)
    require(
        root.tag == "nvidia_smi_log",
        "resource_unavailable",
        "Unknown GPU process observation format.",
    )
    result = {}
    for card in root.findall("gpu"):
        uuid = card.findtext("uuid", "").strip()
        require(
            bool(UUID.fullmatch(uuid)) and uuid not in result,
            "resource_unavailable",
            "Missing or duplicate GPU process identity.",
        )
        processes = card.find("processes")
        require(
            processes is not None and (processes.text or "").strip() in {"", "None"},
            "resource_unavailable",
            "GPU process observation unavailable.",
        )
        compute, graphics, all_processes = set(), set(), set()
        for process in processes:
            require(
                process.tag == "process_info",
                "resource_unavailable",
                "Unknown GPU process record.",
            )
            pid = int(process.findtext("pid", ""))
            kind = process.findtext("type", "").strip()
            require(
                pid > 0 and kind in {"C", "G", "C+G"},
                "resource_unavailable",
                "Unknown GPU process identity or type.",
            )
            all_processes.add(pid)
            if "C" in kind:
                compute.add(pid)
            if "G" in kind:
                graphics.add(pid)
        result[uuid] = {
            "compute_process_count": len(compute),
            "graphics_process_count": len(graphics),
            "process_count": len(all_processes),
        }
    return result


def discover():
    """Unknown utilization or process data makes the entire observation unavailable."""
    try:
        rows = _query(
            "index,uuid,name,memory.total,memory.free,utilization.gpu,pci.bus_id"
        )
        process_query = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        )
        counts = _process_counts(process_query.stdout)
        require(
            set(counts) == {row[1] for row in rows},
            "resource_unavailable",
            "GPU inventory and process observations disagree.",
        )
        cards = []
        for index, uuid, name, total, free, utilization, pci in rows:
            require(
                bool(UUID.fullmatch(uuid)),
                "resource_unavailable",
                "Unknown GPU identity.",
            )
            # Device minors need not equal nvidia-smi indexes (e.g. GPU ordering).
            domain, bus, device = pci.lower().split(":")
            info = Path(
                f"/proc/driver/nvidia/gpus/{domain[-4:]}:{bus}:{device}/information"
            ).read_text()
            minor = int(re.search(r"Device Minor:\s*(\d+)", info).group(1))
            node = Path(f"/dev/nvidia{minor}")
            node_stat = node.stat()
            require(
                stat.S_ISCHR(node_stat.st_mode)
                and os.minor(node_stat.st_rdev) == minor,
                "resource_unavailable",
                "GPU device mapping unavailable.",
            )
            cards.append(
                {
                    "id": uuid,
                    "uuid": uuid,
                    "index": int(index),
                    "device_minor": minor,
                    "name": name,
                    "memory_total_bytes": int(total) * 1024**2,
                    "memory_free_bytes": int(free) * 1024**2,
                    "utilization_percent": int(utilization),
                    **counts[uuid],
                }
            )
        return {"status": "ok", "gpus": cards}
    except (
        OSError,
        ValueError,
        AttributeError,
        ET.ParseError,
        subprocess.SubprocessError,
        GymError,
    ):
        return {
            "status": "unavailable",
            "gpus": [],
            "reason": "GPU discovery or process/device observation unavailable",
        }


def allocated(config, observation):
    selectors = set(config["gpu_ids"])
    return [
        card
        for card in observation["gpus"]
        if card["id"] in selectors or str(card["index"]) in selectors
    ]


def idle(card, config):
    graphics = card.get("graphics_process_count")
    return (
        type(graphics) is int
        and graphics >= 0
        and card.get("compute_process_count") == 0
        and card.get("process_count") == graphics
        and (graphics == 0 or config.get("allow_idle_graphics", False))
        and card["utilization_percent"] == 0
        and card["memory_free_bytes"]
        >= config["gpu_memory_limit_bytes"] + 256 * 1024**2
    )


def _lock_path(uuid):
    require(bool(UUID.fullmatch(uuid)), "invalid_request", "Invalid GPU UUID.")
    directory = Path(tempfile.gettempdir()) / f"nexttransformer-gym-gpus-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    require(
        not directory.is_symlink() and directory.stat().st_uid == os.getuid(),
        "resource_unavailable",
        "Unsafe GPU lease directory.",
    )
    return directory / (uuid + ".lock")


def leased(uuid):
    """Check another cooperating Gym's reservation without retaining it."""
    try:
        fd = os.open(_lock_path(uuid), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        return False
    except (OSError, GymError):
        return True


@contextmanager
def reserve(config, requested=None):
    """A lease coordinates Gym instances; it cannot exclude unrelated host jobs."""
    observed = discover()
    candidates = [
        card
        for card in allocated(config, observed)
        if requested is None or card["id"] == requested
    ]
    for card in candidates:
        if not idle(card, config):
            continue
        try:
            fd = os.open(
                _lock_path(card["id"]), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
        except (OSError, GymError):
            continue
        with os.fdopen(fd, "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            fresh = next(
                (x for x in allocated(config, discover()) if x["id"] == card["id"]),
                None,
            )
            if fresh is None or not idle(fresh, config):
                continue
            yield {**fresh, "memory_limit_bytes": config["gpu_memory_limit_bytes"]}
            return
    raise GymError(
        "resource_unavailable",
        "Configured GPU is busy, leased, has insufficient free memory, or cannot be observed.",
    )
