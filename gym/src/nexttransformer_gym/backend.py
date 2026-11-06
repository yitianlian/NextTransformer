"""Pinned worker subprocesses and a fenced, crash-conservative scheduler."""

import fcntl
import hashlib
import json
import os
import queue
import threading
import subprocess
import sys
import time
import tempfile
import shutil
import copy
from contextlib import ExitStack
from pathlib import Path
from .common import GymError, canonical, identity, new_id, require, file_hash


class NativeBackend:
    def __init__(
        self,
        profiles,
        state_dir,
        platform_root,
        python=None,
        server_id=None,
        execution_config=None,
    ):
        from .resources import local_server_id

        self.server_id = server_id or local_server_id()
        from .gpu import configuration
        from .isolation import preflight

        self.execution_config = configuration(execution_config)
        preflight(self.execution_config["isolation"])
        self.profiles = copy.deepcopy(profiles)
        for profile in self.profiles.values():
            if self.execution_config["gpu_ids"] and not profile.get("official_plan"):
                profile["device"] = "cuda:0"
                profile.pop("cuda_visible_devices", None)
                profile["accelerator_class"] = "cuda"
                profile["memory_limit_bytes"] = self.execution_config[
                    "gpu_memory_limit_bytes"
                ]
                profile["resource_profile_id"] += "/cuda"
            elif not profile.get("official_plan"):
                require(
                    not str(profile.get("device", "cpu")).startswith("cuda"),
                    "invalid_request",
                    "Research CUDA profiles require an operator execution config.",
                )
        self.state_dir, self.platform_root = (
            Path(state_dir).resolve(),
            Path(platform_root).resolve(),
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.python = str(Path(python or sys.executable).absolute())
        for pid, profile in profiles.items():
            require(
                profile["id"] == pid and profile["kind"] in {"research", "final"},
                "invalid_request",
                "Invalid profile manifest.",
            )
            require(
                all(
                    type(profile[k]) is int and profile[k] > 0
                    for k in (
                        "width",
                        "heads",
                        "sequence_len",
                        "vocab_size",
                        "batch_size",
                        "max_updates",
                        "dev_batches",
                        "timeout_sec",
                    )
                ),
                "invalid_request",
                "Profile bounds must be explicit positive integers.",
            )
            require(
                profile["max_updates"] <= 46875,
                "invalid_request",
                "Schedule horizon cannot be extended.",
            )
        source_files = [
            p
            for root in (
                self.platform_root / "nexttransformer",
                self.platform_root / "nanochat",
                Path(__file__).parent,
            )
            for p in root.rglob("*.py")
        ]
        self.revision = identity(
            {
                str(p.relative_to(self.platform_root))
                if p.is_relative_to(self.platform_root)
                else "gym/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source_files)
            }
        )
        # Query the actual numerical interpreter, not the controller environment.
        metadata_script = """
import ast, importlib.metadata as m, json, sys
names = ('torch', 'numpy', 'pyarrow', 'tiktoken', 'rustbpe', 'filelock', 'fsspec', 'sympy')
version = m.distribution('torch').locate_file('torch/version.py').read_text()
cuda = None
for node in ast.walk(ast.parse(version)):
    if isinstance(node, ast.AnnAssign) and getattr(node.target, 'id', '') == 'cuda':
        cuda = ast.literal_eval(node.value)
print(json.dumps({'dependencies': {n:m.version(n) for n in names}, 'python':list(sys.version_info[:3]), 'cuda_version':cuda}))
"""
        try:
            self.worker_runtime = json.loads(
                subprocess.check_output(
                    [self.python, "-c", metadata_script], text=True, timeout=30
                )
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            raise GymError(
                "resource_unavailable",
                "Cannot inspect the configured numerical Python runtime.",
            ) from None
        self.revision = identity(
            {
                "source": self.revision,
                "worker_runtime": self.worker_runtime,
                "execution_config": self.execution_config,
                "compute_dtype": "float32",
            }
        )
        self.cache = {}

    def bounded_profile(self, profile):
        """Single-device diagnostic view; preserve the operator's final plan."""
        if not profile.get("official_plan"):
            return profile
        return {
            **profile,
            "device": "cuda:0",
            "accelerator_class": "cuda",
            "memory_limit_bytes": min(
                profile["memory_limit_bytes"],
                self.execution_config["gpu_memory_limit_bytes"],
            ),
        }

    def validate_placement(self, profile, gpu_id=None, server_id=None):
        from .gpu import allocated, discover

        profile = self.bounded_profile(profile)
        require(
            server_id is None or server_id == self.server_id,
            "invalid_request",
            "Only this configured local execution node is supported.",
        )
        if not str(profile.get("device", "cpu")).startswith("cuda"):
            require(
                gpu_id is None,
                "invalid_request",
                "CPU profiles do not accept GPU placement.",
            )
            return None
        require(
            bool(self.execution_config["gpu_ids"]),
            "resource_unavailable",
            "Bounded CUDA work requires an operator execution allocation.",
        )
        if gpu_id is None:
            return None
        require(
            isinstance(gpu_id, (str, int)) and not isinstance(gpu_id, bool),
            "invalid_request",
            "Invalid GPU selection.",
        )
        card = next(
            (
                card
                for card in allocated(self.execution_config, discover())
                if str(gpu_id) in (card["id"], str(card["index"]))
            ),
            None,
        )
        require(
            card is not None,
            "invalid_request",
            "GPU is not present in this node's configured allocation.",
        )
        return card["id"]

    def resources(self, global_busy=False):
        from .resources import observe
        from .gpu import allocated, discover, idle, leased

        node = observe(self.server_id, self.profiles.values(), global_busy)
        inventory = discover()
        allowed = {card["id"] for card in allocated(self.execution_config, inventory)}
        runtime_ready = bool(self.worker_runtime["cuda_version"])
        cards = []
        for card in inventory["gpus"]:
            assigned = card["id"] in allowed
            held = assigned and leased(card["id"])
            available = (
                assigned
                and idle(card, self.execution_config)
                and not held
                and not global_busy
                and runtime_ready
            )
            cards.append(
                {
                    **{k: v for k, v in card.items() if k != "device_minor"},
                    "allocated": assigned,
                    "available": available,
                    "busy_reason": None
                    if available
                    else "not_allocated"
                    if not assigned
                    else "cuda_runtime_unavailable"
                    if not runtime_ready
                    else "worker_busy"
                    if global_busy
                    else "gpu_leased"
                    if held
                    else "gpu_busy_or_insufficient_memory",
                    "memory_limit_bytes": self.execution_config[
                        "gpu_memory_limit_bytes"
                    ]
                    if assigned
                    else 0,
                }
            )
        node.update(
            gpu_discovery_status=inventory["status"],
            discovered_gpu_count=len(cards),
            allocated_gpu_count=len(allowed),
            available_gpu_count=sum(card["available"] for card in cards),
            gpus=cards,
            configured_cuda_devices=self.execution_config["gpu_ids"],
            execution_config=self.execution_config,
            worker_runtime=self.worker_runtime,
            research_devices=["cuda:0"]
            if self.execution_config["gpu_ids"]
            else ["cpu"],
        )
        if self.execution_config["gpu_ids"]:
            node["allocatable_slots"] = int(any(card["available"] for card in cards))
        return node

    def public_profiles(self):
        from .candidate_contract import profile_constraints

        for profile in self.profiles.values():
            devices = profile.get("cuda_visible_devices", [])
            bounded = self.bounded_profile(profile)
            public = {
                k: profile[k]
                for k in (
                    "id",
                    "kind",
                    "width",
                    "heads",
                    "sequence_len",
                    "vocab_size",
                    "max_updates",
                    "resource_profile_id",
                )
            }
            public.update(
                model_config_constraints=profile_constraints(profile),
                accelerator_class=bounded.get("accelerator_class", "cpu"),
                device=bounded.get("device", "cpu"),
                usable_memory_bytes_per_device=bounded.get("memory_limit_bytes"),
                supported_device_counts=[1],
                supported_precisions=["float32"],
                runtime_manifest_id=self.revision,
                checkpoint_quota_bytes=profile.get("checkpoint_quota_bytes"),
            )
            if profile.get("official_plan"):
                public["operator_final_training"] = {
                    "device_count": len(devices),
                    "precision": "bfloat16",
                    "agent_dispatch_allowed": False,
                }
            yield public

    def environment(self, profile=None, official=False, gpu=None):
        # Explicit environment prevents ambient plugin paths, credentials and proxies.
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": os.pathsep.join(
                [str(Path(__file__).parent.parent), str(self.platform_root)]
            ),
            "NANOCHAT_DTYPE": "float32",
            "TORCHDYNAMO_DISABLE": "1",
            "OMP_NUM_THREADS": "1",
            "USER": "gym-worker",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(self.state_dir),
            "XDG_CACHE_HOME": str(self.state_dir / "cache"),
            "CUDA_VISIBLE_DEVICES": "",
        }
        if (
            not official
            and profile
            and str(profile.get("device", "cpu")).startswith("cuda")
        ):
            require(
                gpu is not None,
                "resource_unavailable",
                "Research GPU execution requires a scheduler allocation.",
            )
            env["CUDA_VISIBLE_DEVICES"] = gpu["id"]
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if official:
            require(
                profile is not None and profile.get("cuda_visible_devices"),
                "resource_unavailable",
                "Official GPU allocation must be explicit.",
            )
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(x) for x in profile["cuda_visible_devices"]
            )
            env["NANOCHAT_DTYPE"] = "bfloat16"
            env["TORCHDYNAMO_DISABLE"] = "0"
        return env

    def inspect(self, candidate, profile):
        key = identity([candidate["hash"], profile, self.revision])
        if key not in self.cache:
            request = {"mode": "inspect", "candidate": candidate, "profile": profile}
            try:
                result = subprocess.run(
                    self.worker_command(request),
                    input=canonical(request) + "\n",
                    text=True,
                    capture_output=True,
                    env=self.environment(),
                    cwd=self.state_dir,
                    timeout=profile["timeout_sec"],
                    check=True,
                    **self.worker_identity(request),
                )
                events = [
                    json.loads(line[len("GYM_EVENT ") :])
                    for line in result.stdout.splitlines()
                    if line.startswith("GYM_EVENT ")
                ]
                self.cache[key] = next(
                    e["value"] for e in events if e["kind"] == "inspection"
                )
            except (subprocess.SubprocessError, ValueError, StopIteration):
                raise GymError(
                    "incomplete_coverage",
                    "Trusted shape-only cost audit failed; inspect controller diagnostics.",
                ) from None
        return self.cache[key]

    def worker_command(self, request):
        if request.get("quote", {}).get("operation") == "official":
            # Official source is separately reviewed/admitted in an operator-owned snapshot.
            return [self.python, "-m", "nexttransformer_gym.worker"]
        from .isolation import command

        return command(
            self.python,
            self.platform_root,
            request,
            self.execution_config["isolation"],
        )

    @staticmethod
    def worker_identity(request):
        # Docker controllers may be root for volume bootstrap. Namespace creation
        # runs as an unprivileged UID; the worker receives no controller capabilities.
        if (
            os.geteuid() == 0
            and request.get("quote", {}).get("operation") != "official"
        ):
            return {"user": 65534, "group": 65534, "extra_groups": []}
        return {}

    @staticmethod
    def bound(report, operation, updates, profile, check=None):
        init, fwd, bwd, opt = (
            int(report[k])
            for k in ("initialization", "forward", "backward", "optimizer")
        )
        if operation == "validate":
            return 2 * init + fwd + bwd + opt + 3 * int(report["prediction"])
        if operation == "debug":
            costs = {
                "construction": init,
                "cost_coverage": init,
                "numerical": init + fwd + bwd + opt,
                "causal": init + 2 * int(report["prediction"]),
                "checkpoint": 2 * init
                + fwd
                + bwd
                + opt
                + 2 * int(report["prediction"]),
            }
            require(check in costs, "invalid_request", "Unknown named diagnostic.")
            return costs[check]
        return (
            init
            + updates * (fwd + bwd + opt)
            + profile["dev_batches"] * int(report["evaluation"])
        )

    def checkpoint_path(self, checkpoint_id):
        require(
            isinstance(checkpoint_id, str)
            and len(checkpoint_id) == 32
            and all(c in "0123456789abcdef" for c in checkpoint_id),
            "invalid_request",
            "Invalid checkpoint identity.",
        )
        return self.state_dir / "checkpoints" / (checkpoint_id + ".pt")


class Scheduler:
    def __init__(self, service):
        self.service, self.backend, self.store = service, service.backend, service.store

    def work_once(self):
        with (
            ExitStack() as leases,
            (self.backend.state_dir / "worker.lock").open("a") as lock,
        ):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            with self.store.transaction() as db:
                rows = db.execute(
                    "SELECT owner,value FROM objects WHERE kind='job' ORDER BY rowid"
                ).fetchall()
                selected = next(
                    (
                        (sid, json.loads(raw))
                        for sid, raw in rows
                        if json.loads(raw)["state"] == "queued"
                    ),
                    None,
                )
                if selected is None:
                    return False
                sid, job = selected
                # Uncertain work is a resource lease, not a free slot.
                if any(
                    json.loads(raw)["state"]
                    in {"running", "cancelling", "reconciliation"}
                    for _, raw in rows
                ):
                    return False
                quote = self.store.get(db, sid, "quote", job["quote_id"])
                profile = self.backend.profiles[quote["profile_id"]]
                if quote["operation"] != "official":
                    profile = self.backend.bounded_profile(profile)
                if (
                    str(profile.get("device", "cpu")).startswith("cuda")
                    and quote["operation"] != "official"
                ):
                    from .gpu import reserve

                    try:
                        require(
                            bool(self.backend.worker_runtime["cuda_version"]),
                            "resource_unavailable",
                            "Configured worker Python has no CUDA runtime.",
                        )
                        policy = {
                            **self.backend.execution_config,
                            "gpu_memory_limit_bytes": min(
                                profile["memory_limit_bytes"],
                                self.backend.execution_config["gpu_memory_limit_bytes"],
                            ),
                        }
                        allocation = leases.enter_context(
                            reserve(policy, quote.get("gpu_id"))
                        )
                    except GymError as error:
                        if job.get("waiting_reason") != str(error):
                            job["waiting_reason"] = str(error)
                            self.store.update(db, sid, "job", job)
                        return False
                    job["assigned_gpu"] = allocation
                job.pop("waiting_reason", None)
                job.update(
                    state="running", lease=new_id(), server_id=self.backend.server_id
                )
                self.store.update(db, sid, "job", job)
                quote = self.store.get(db, sid, "quote", job["quote_id"])
                if quote["checkpoint_id"]:
                    checkpoint = self.store.get(
                        db, sid, "checkpoint", quote["checkpoint_id"]
                    )
                    try:
                        valid_checkpoint = (
                            file_hash(self.backend.checkpoint_path(checkpoint["id"]))
                            == checkpoint["sha256"]
                        )
                    except OSError:
                        valid_checkpoint = False
                    if not valid_checkpoint:
                        job.update(
                            state="failed",
                            lease=None,
                            error_code="incompatible_checkpoint",
                        )
                        self.store.release(db, sid, job["id"])
                        self.store.update(db, sid, "job", job)
                        return True
                candidate = self.store.get(db, sid, "candidate", quote["candidate_id"])
                candidate["files"] = self.store.get(
                    db, sid, "package", candidate["id"]
                )["files"]
            try:
                self.execute(sid, job, quote, candidate)
            except Exception:
                with self.store.transaction() as db:
                    current = self.store.get(db, sid, "job", job["id"])
                    current.update(
                        state="reconciliation", error_code="infrastructure_failure"
                    )
                    self.store.update(db, sid, "job", current)
            return True

    def execute(self, sid, job, quote, candidate):
        profile = self.backend.profiles[quote["profile_id"]]
        if quote["operation"] != "official":
            profile = self.backend.bounded_profile(profile)
        output_id = new_id()
        workspace = tempfile.TemporaryDirectory(
            prefix="worker-", dir=self.backend.state_dir
        )
        output = Path(workspace.name) / "checkpoint.pt"
        if os.geteuid() == 0:
            self.backend.state_dir.chmod(self.backend.state_dir.stat().st_mode | 0o111)
            Path(workspace.name).chmod(0o777)
        request = {
            "mode": "execute",
            "gpu": job.get("assigned_gpu"),
            "candidate": candidate,
            "profile": profile,
            "quote": quote,
            "output": str(output),
            "platform_root": str(self.backend.platform_root),
            "final_output_root": str(self.backend.state_dir / "final-runs"),
            "checkpoint": str(self.backend.checkpoint_path(quote["checkpoint_id"]))
            if quote["checkpoint_id"]
            else None,
        }
        process = subprocess.Popen(
            self.backend.worker_command(request),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self.backend.environment(
                profile, quote["operation"] == "official", request["gpu"]
            ),
            cwd=self.backend.state_dir,
            start_new_session=True,
            **self.backend.worker_identity(request),
        )
        pending, result, cancelled, failure = None, None, False, None
        # Drain stderr to a bounded private log without blocking stdout.
        messages = queue.Queue()

        def drain(stream, channel):
            for line in stream:
                messages.put((channel, line))
            messages.put((channel, None))

        readers = [
            threading.Thread(target=drain, args=(stream, channel), daemon=True)
            for stream, channel in (
                (process.stdout, "stdout"),
                (process.stderr, "stderr"),
            )
        ]
        for reader in readers:
            reader.start()
        diagnostics = []

        def send(value):
            process.stdin.write(canonical(value) + "\n")
            process.stdin.flush()

        send(request)
        deadline = time.monotonic() + profile["timeout_sec"]
        try:
            closed = set()
            while len(closed) < 2:
                require(
                    time.monotonic() < deadline,
                    "infrastructure_failure",
                    "Worker deadline exceeded.",
                )
                try:
                    channel, line = messages.get(timeout=0.25)
                except queue.Empty:
                    continue
                if line is None:
                    closed.add(channel)
                    continue
                if channel == "stderr":
                    if sum(map(len, diagnostics)) < 65536:
                        diagnostics.append(line[:4096])
                    continue
                if not line.startswith("GYM_EVENT "):
                    continue
                event = json.loads(line[len("GYM_EVENT ") :])
                with self.store.transaction() as db:
                    current = self.store.get(db, sid, "job", job["id"])
                    require(
                        current["lease"] == job["lease"]
                        and current["state"] in {"running", "cancelling"},
                        "stale_lease",
                        "Worker no longer owns this job.",
                    )
                    kind = event["kind"]
                    if kind == "intent":
                        require(
                            pending is None,
                            "integrity_violation",
                            "Overlapping work intents.",
                        )
                        remaining = self.store.get(db, sid, "reservation", job["id"])[
                            "remaining"
                        ]
                        require(
                            0 <= int(event["amount"]) <= int(remaining),
                            "insufficient_budget",
                            "Operation exceeds reservation.",
                        )
                        if current["cancel_requested"]:
                            send({"cancel": True})
                        else:
                            pending = event
                            current["inflight"] = {
                                "label": event["label"],
                                "amount": event["amount"],
                            }
                            self.store.update(db, sid, "job", current)
                            # Acknowledgement follows durable intent commit below.
                    elif kind == "charge":
                        require(
                            pending is not None
                            and event["label"] == pending["label"]
                            and event["amount"] == pending["amount"],
                            "integrity_violation",
                            "Charge does not match in-flight operation.",
                        )
                        self.store.charge(
                            db,
                            sid,
                            job["id"],
                            identity([job["id"], event["label"]]),
                            int(event["amount"]),
                        )
                        pending = None
                        current.pop("inflight", None)
                        self.store.update(db, sid, "job", current)
                    elif kind == "runtime":
                        current["execution"] = event["value"]
                        self.store.update(db, sid, "job", current)
                    elif kind == "progress":
                        current["progress"] = event["step"]
                        self.store.update(db, sid, "job", current)
                    elif kind == "result":
                        result = event["value"]
                    elif kind == "cancelled":
                        cancelled = True
                    elif kind == "failed":
                        require(
                            pending is None,
                            "integrity_violation",
                            "Failure has unsettled work.",
                        )
                        failure = event["error_code"]
                    else:
                        raise GymError("integrity_violation", "Unknown worker event.")
                if kind == "charge" or kind == "intent" and pending is not None:
                    send({"ack": event["label"]})
            returncode = process.wait(timeout=5)
            require(
                returncode == 0
                and pending is None
                and (result is not None or cancelled or failure is not None),
                "infrastructure_failure",
                "Worker ended without reconciled completion.",
            )
            with self.store.transaction() as db:
                current = self.store.get(db, sid, "job", job["id"])
                require(
                    current["lease"] == job["lease"], "stale_lease", "Lost job lease."
                )
                if not cancelled and failure is None:
                    result.update(
                        id=job["id"],
                        cost=self.store.budget(db, sid),
                        server_id=self.backend.server_id,
                        assigned_gpu=job.get("assigned_gpu"),
                    )
                    if quote["operation"] == "validate":
                        self.store.put(db, sid, "report", {**result, "id": new_id()})
                    elif quote["operation"] == "train":
                        require(
                            output.is_file()
                            and not output.is_symlink()
                            and output.stat().st_nlink == 1
                            and output.stat().st_size
                            <= job.get(
                                "checkpoint_reservation_bytes",
                                profile["checkpoint_quota_bytes"],
                            ),
                            "infrastructure_failure",
                            "Missing complete checkpoint.",
                        )
                        retained = self.backend.checkpoint_path(output_id)
                        retained.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(output, retained)
                        self.store.put(
                            db,
                            sid,
                            "checkpoint",
                            {
                                "id": output_id,
                                "complete": True,
                                "step": result["step"],
                                "candidate_hash": candidate["hash"],
                                "profile_hash": quote["profile_hash"],
                                "runtime_revision": self.backend.revision,
                                "size_bytes": output.stat().st_size,
                                "sha256": file_hash(output),
                            },
                        )
                        result["checkpoint_id"] = output_id
                    if "execution" in result:
                        current["execution"] = result["execution"]
                    self.store.put(db, sid, "result", result)
                self.store.release(db, sid, job["id"])
                current.update(
                    state="cancelled"
                    if cancelled
                    else "budget_stopped"
                    if failure == "insufficient_budget"
                    else "failed"
                    if failure
                    else "succeeded",
                    lease=None,
                )
                if failure:
                    current.update(
                        error_code=failure, settlement="conservative_operation_bound"
                    )
                self.store.update(db, sid, "job", current)
        except Exception:
            if process.poll() is None:
                import signal

                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            with self.store.transaction() as db:
                current = self.store.get(db, sid, "job", job["id"])
                current.update(
                    state="reconciliation", error_code="infrastructure_failure"
                )
                self.store.update(db, sid, "job", current)
        finally:
            for reader in readers:
                reader.join(timeout=2)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
            (self.backend.state_dir / (job["id"] + ".diagnostic.log")).write_text(
                "".join(diagnostics)
            )
            workspace.cleanup()
