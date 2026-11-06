"""Filesystem/process/network isolation for native research plugin workers.

The operator selects ``bwrap`` (default), ``landlock`` or ``none`` in the execution
configuration. This isolates OS resources. It does not make arbitrary in-process
Python a trusted accounting authority; adversarial accounting certification is a
separate gate.
"""

import glob
import shutil
import os
import sys
from pathlib import Path
from .common import canonical, require

BACKENDS = ("bwrap", "landlock", "none")
UNISOLATED_WARNING = (
    "WARNING: Gym research workers run without isolation. Candidate code can read "
    "controller state, reach the network and signal other processes. Use this only "
    "with trusted candidates."
)


def preflight(backend):
    """Check landlock prerequisites and warn about ``none`` at controller startup."""
    if backend == "landlock":
        from .landlock import require_available

        require_available()
        require(
            shutil.which("unshare") is not None,
            "resource_unavailable",
            "The landlock isolation backend requires util-linux unshare.",
        )
        # Workers then run as a separate user, which Landlock relies on to keep them
        # from signalling the controller or connecting to its Unix sockets.
        require(
            os.geteuid() == 0,
            "resource_unavailable",
            "The landlock isolation backend requires a root controller.",
        )
    elif backend == "none":
        print(UNISOLATED_WARNING, file=sys.stderr)


def command(python, platform, request, backend):
    worker = [python, "-m", "nexttransformer_gym.worker"]
    if backend == "none":
        return worker
    if backend == "landlock":
        return landlock_command(python, platform, request, worker)
    return bwrap_command(python, platform, request, worker)


def linker_directories(config=Path("/etc/ld.so.conf")):
    """Library directories configured for the dynamic linker."""
    try:
        lines = config.read_text().splitlines()
    except OSError:
        return []
    directories = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line.startswith("include "):
            pattern = str(config.parent / line.split(None, 1)[1])
            for included in sorted(glob.glob(pattern)):
                directories += linker_directories(Path(included))
        elif line and not line.startswith("hwcap "):
            directories.append(line)
    return directories


def runtime_roots(python, platform):
    """Read-only interpreter, library and platform source trees."""
    roots = {Path(p) for p in ("/usr", "/bin", "/lib", "/lib64") if Path(p).exists()}
    roots |= {Path(p) for p in linker_directories() if Path(p).is_dir()}
    roots |= {
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(python).resolve().parent.parent,
        Path(python).absolute().parent.parent,
        Path(__file__).resolve().parent,
        Path(platform) / "nexttransformer",
        Path(platform) / "nanochat",
    }
    if Path(python).is_symlink():
        target = Path(os.readlink(python))
        if target.is_absolute():
            roots.add(target.parent.parent)
    return [
        str(path)
        for path in sorted(roots, key=lambda p: (len(p.parts), str(p)))
        if not any(path != parent and path.is_relative_to(parent) for parent in roots)
    ]


def system_files():
    return [p for p in ("/etc/ld.so.cache", "/etc/localtime") if Path(p).exists()]


def job_inputs(request):
    """Read-only checkpoint and approved data for the current job."""
    paths = [request["checkpoint"]] if request.get("checkpoint") else []
    data = request["profile"].get("data", {"kind": "synthetic_fixture"})
    if request["mode"] != "inspect" and data["kind"] != "synthetic_fixture":
        for path in [
            *data["train_paths"],
            data["dev_path"],
            data["tokenizer"],
            data["token_bytes"],
        ]:
            paths.append(str(Path(path).resolve()))
    return paths


def cuda_nodes(request):
    """The allocated card, control and UVM device nodes, or None for CPU work."""
    device = str(request["profile"].get("device", "cpu"))
    if request["mode"] == "inspect" or not device.startswith("cuda"):
        return None
    gpu = request.get("gpu")
    require(
        gpu is not None
        and type(gpu.get("device_minor")) is int
        and gpu["device_minor"] >= 0,
        "resource_unavailable",
        "A scheduler GPU allocation is required.",
    )
    nodes = [f"/dev/nvidia{gpu['device_minor']}", "/dev/nvidiactl", "/dev/nvidia-uvm"]
    require(
        all(Path(node).exists() for node in nodes),
        "resource_unavailable",
        "Allocated CUDA device nodes are unavailable.",
    )
    return nodes


def bwrap_command(python, platform, request, worker):
    executable = shutil.which("bwrap")
    require(
        executable is not None,
        "resource_unavailable",
        "The bwrap isolation backend requires bubblewrap.",
    )
    args = [
        executable,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    for path in [*runtime_roots(python, platform), *system_files()]:
        args += ["--ro-bind", path, path]
    allocated = request["mode"] != "inspect" and request.get("gpu")
    args += [
        "--proc" if allocated else "--dir",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/tmp",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "XDG_CACHE_HOME",
        "/tmp/cache",
    ]
    if request.get("output"):
        directory = Path(request["output"]).parent
        args += ["--bind", str(directory), str(directory)]
    for path in job_inputs(request):
        args += ["--ro-bind", path, path]
    nodes = cuda_nodes(request)
    if nodes is not None:
        for node in nodes:
            args += ["--dev-bind", node, node]
        # Private PID namespace procfs exposes only the worker; driver metadata is
        # read-only. No host process tree, host network, or other GPU nodes.
        if Path("/proc/driver/nvidia").exists():
            args += ["--ro-bind", "/proc/driver/nvidia", "/proc/driver/nvidia"]
    return [*args, "--", *worker]


def landlock_command(python, platform, request, worker):
    workspace = str(Path(request["output"]).parent) if request.get("output") else None
    policy = {
        "read": [
            *runtime_roots(python, platform),
            *system_files(),
            *job_inputs(request),
        ],
        # Directories under the import path parents are listable; files outside
        # the read paths stay unreadable.
        "list": [str(Path(__file__).parent.parent), str(platform)],
        "write": [
            *([workspace] if workspace else []),
            "/dev/null",
            "/dev/zero",
            "/dev/full",
            "/dev/random",
            "/dev/urandom",
        ],
        # The job workspace is removed by the controller even if the worker is killed.
        "tmp": workspace,
        "unix_sockets": False,
    }
    namespaces = ["--user", "--net", "--ipc", "--uts"]
    nodes = cuda_nodes(request)
    if nodes is None:
        namespaces += ["--pid", "--kill-child"]
    else:
        # CUDA needs procfs entries that match its PID and writes to its own entry,
        # so GPU workers keep the host PID namespace. CUDA also creates Unix sockets
        # and opens every card node during startup, so CUDA_VISIBLE_DEVICES selects
        # the allocated card but does not confine the worker to it.
        policy["unix_sockets"] = True
        policy["read"].append("/proc/driver/nvidia")
        policy["write"] += [
            "/proc/self",
            *nodes[1:],
            *sorted(glob.glob("/dev/nvidia[0-9]*")),
        ]
    return [
        shutil.which("unshare"),
        *namespaces,
        "--",
        python,
        "-m",
        "nexttransformer_gym.landlock",
        canonical(policy),
        "--",
        *worker,
    ]
