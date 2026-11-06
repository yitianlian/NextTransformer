import base64
import concurrent.futures
import io
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from nexttransformer_gym.backend import NativeBackend, Scheduler
from nexttransformer_gym.common import GymError
from nexttransformer_gym.evidence import snapshot, verify_receipt
from nexttransformer_gym.final import launch_final_run, reconcile
from nexttransformer_gym.landlock import abi_version
from nexttransformer_gym.package import ingest
from nexttransformer_gym.server import bind
from nexttransformer_gym.service import Service

ROOT = Path(__file__).resolve().parents[2]


def bwrap_usable():
    executable = shutil.which("bwrap")
    if executable is None:
        return False
    probe = [executable, "--unshare-all", "--ro-bind", "/", "/", "true"]
    return subprocess.run(probe, capture_output=True).returncode == 0


BWRAP = bwrap_usable()
LANDLOCK = abi_version() >= 1 and os.geteuid() == 0
# Tests use bwrap where it works, then landlock; with neither, bwrap tests fail.
ISOLATION = "bwrap" if BWRAP or not LANDLOCK else "landlock"


def candidate_bytes(activation="square"):
    blob = io.BytesIO()
    runpy.run_path(str(ROOT / "gym/examples/make_candidate.py"))["package"](
        blob, activation
    )
    return blob.getvalue()


def session(tmp_path, isolation=ISOLATION):
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    backend = NativeBackend(
        profiles, tmp_path, ROOT, execution_config={"isolation": isolation}
    )
    service = Service(tmp_path / "gym.sqlite3", backend)
    credentials = service.create_session(10**12, "test-trial")

    def call(operation, **params):
        return service.call(credentials["agent_token"], operation, params)

    candidate = call(
        "submit_candidate",
        package_base64=base64.b64encode(candidate_bytes()).decode(),
        idempotency_key="first",
    )
    return service, credentials, call, candidate


@pytest.fixture
def context(tmp_path):
    return session(tmp_path)


def run_job(context, **params):
    service, _, call, candidate = context
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        **params,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key=quote["id"])
    Scheduler(service).work_once()
    state = call("get_job", job_id=job["id"])
    if state["state"] != "succeeded":
        pytest.fail(
            (service.backend.state_dir / (job["id"] + ".diagnostic.log")).read_text()
        )
    return call("get_result", job_id=job["id"])


def test_continuation_matches_uninterrupted_training(context):
    import torch

    first = run_job(context, target_update=1)
    resumed = run_job(context, target_update=2, checkpoint_id=first["checkpoint_id"])
    uninterrupted = run_job(context, target_update=2)
    backend = context[0].backend
    a = torch.load(
        backend.checkpoint_path(resumed["checkpoint_id"]), weights_only=False
    )
    b = torch.load(
        backend.checkpoint_path(uninterrupted["checkpoint_id"]), weights_only=False
    )
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["data"] == b["data"]
    assert resumed["dev_bpb"] == uninterrupted["dev_bpb"]
    assert context[2]("get_budget")["reserved"] == "0"


def test_full_submission_verifier_and_separate_final(
    context, monkeypatch, capsys, tmp_path_factory
):
    service, credentials, call, candidate = context
    job = call(
        "validate_candidate",
        candidate_id=candidate["id"],
        profile_id="final-fixture",
        idempotency_key="checks",
    )
    Scheduler(service).work_once()
    assert call("get_job", job_id=job["id"])["state"] == "succeeded"
    checks = call("get_result", job_id=job["id"])
    assert all(checks["checks"].values()) and checks["admitted"]
    submission = call(
        "finalize",
        candidate_id=candidate["id"],
        profile_id="final-fixture",
        idempotency_key="final",
    )
    assert submission["final_run_state"] == "not_scheduled"
    with pytest.raises(GymError, match="Verify the frozen deliverable first"):
        launch_final_run(
            service, credentials["session_id"], submission["id"], 10**12, "dispatch"
        )
    verifier = {
        "session_id": credentials["session_id"],
        "token": credentials["verifier_token"],
    }
    signed = snapshot(service, verifier)
    receipt = {
        "session_id": credentials["session_id"],
        "submission_id": submission["id"],
        "candidate_hash": candidate["hash"],
    }
    assert verify_receipt(receipt, signed, verifier)["valid"]
    assert not verify_receipt({**receipt, "session_id": "forged"}, signed, verifier)[
        "valid"
    ]
    profile = service.backend.profiles["final-fixture"]
    original_updates = profile["max_updates"]
    profile["max_updates"] += 1
    with pytest.raises(GymError, match="Final profile changed after submission"):
        launch_final_run(
            service, credentials["session_id"], submission["id"], 10**12, "dispatch"
        )
    with service.store.transaction() as db:
        assert service.store.all(db, credentials["session_id"], "final_run") == []
    profile["max_updates"] = original_updates
    for op in ("submit_candidate", "run_experiment", "launch_final_run"):
        with pytest.raises(GymError):
            service.call(credentials["verifier_token"], op, {})
    # The final dispatcher needs only the receipt and durable Gym state, not
    # an Agent workspace or the original in-memory candidate/service objects.
    backend = NativeBackend(
        service.backend.profiles,
        service.backend.state_dir,
        ROOT,
        execution_config=service.backend.execution_config,
    )
    service = Service(backend.state_dir / "gym.sqlite3", backend)
    policy = tmp_path_factory.mktemp("operator") / "execution.json"
    policy.write_text(json.dumps(backend.execution_config))
    import sys
    from nexttransformer_gym.cli import main

    def controller(*arguments):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "nt-gym",
                "--state",
                str(backend.state_dir),
                "--profiles",
                str(ROOT / "gym/examples/profiles.fixture.json"),
                "--platform",
                str(ROOT),
                "--execution-config",
                str(policy),
                *arguments,
            ],
        )
        main()
        return json.loads(capsys.readouterr().out)

    run = controller(
        "launch-final",
        "--session-id",
        receipt["session_id"],
        "--submission-id",
        receipt["submission_id"],
        "--allocation",
        str(10**12),
        "--idempotency-key",
        "dispatch",
    )
    assert run == launch_final_run(
        service, credentials["session_id"], submission["id"], 10**12, "dispatch"
    )
    assert controller("work-once")["dispatched"]
    final = controller(
        "get-final", "--session-id", credentials["session_id"], "--run-id", run["id"]
    )
    assert final["state"] == "succeeded"
    assert final["result"]["rank"] is None and final["result"]["core"] is None
    assert final["result"]["official_compatible"] is False
    assert final["result"]["candidate_hash"] == receipt["candidate_hash"]
    assert (
        call("get_submission", submission_id=submission["id"])["final_run_state"]
        == "succeeded"
    )
    with pytest.raises(GymError):
        call(
            "quote_experiment",
            candidate_id=candidate["id"],
            profile_id="research-fixture",
            target_update=1,
        )


def test_idempotency_and_cross_session(context):
    service, credentials, call, candidate = context
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(
                lambda _: call(
                    "submit_candidate",
                    package_base64=base64.b64encode(candidate_bytes()).decode(),
                    idempotency_key="first",
                ),
                range(6),
            )
        )
    assert all(r == candidate for r in results)
    with pytest.raises(GymError, match="different request"):
        call(
            "submit_candidate",
            package_base64=base64.b64encode(candidate_bytes("tanh")).decode(),
            idempotency_key="first",
        )
    other = service.create_session(100, "other")
    with pytest.raises(GymError):
        service.call(
            other["agent_token"], "get_candidate", {"candidate_id": candidate["id"]}
        )
    restored = Service(service.store.path, service.backend)
    assert restored.call(credentials["agent_token"], "get_budget", {}) == call(
        "get_budget"
    )


def test_atomic_budget_and_charge_replay(context):
    service, credentials, call, _ = context
    sid = credentials["session_id"]
    with service.store.transaction() as db:
        service.store.reserve(db, sid, "one", 10**12)

    def reserve():
        try:
            with service.store.transaction() as db:
                service.store.reserve(db, sid, "two", 1)
        except GymError as error:
            return error.code

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert (
            list(pool.map(lambda _: reserve(), range(4))) == ["insufficient_budget"] * 4
        )
    with service.store.transaction() as db:
        service.store.charge(db, sid, "one", "charge", 5)
        service.store.charge(db, sid, "one", "charge", 5)
    assert call("get_budget")["spent"] == "5"
    assert call("get_budget")["reserved"] == str(10**12 - 5)


def test_queued_cancellation_releases_only_unexecuted_work(context):
    service, _, call, candidate = context
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=1,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="run")
    call("cancel_job", job_id=job["id"], idempotency_key="cancel")
    assert call("get_budget")["reserved"] == "0"
    assert call("get_budget")["spent"] == "0"
    assert not Scheduler(service).work_once()


def test_crash_recovery_keeps_reservation(context):
    service, credentials, call, candidate = context
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=1,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="run")
    with service.store.transaction() as db:
        job.update(state="running", lease="old-controller")
        service.store.update(db, credentials["session_id"], "job", job)
    assert call("get_budget")["reserved"] == quote["bound"]
    reconcile(
        service, credentials["session_id"], job["id"], worker_confirmed_stopped=True
    )
    assert call("get_budget")["spent"] == quote["bound"]
    assert call("get_budget")["reserved"] == "0"


def rewrite_package(changes):
    import zipfile

    with zipfile.ZipFile(io.BytesIO(candidate_bytes())) as original:
        files = {name: original.read(name) for name in original.namelist()}
    files.update({name: value.encode() for name, value in changes.items()})
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as out:
        for name, source in files.items():
            out.writestr(name, source)
    return blob.getvalue()


@pytest.mark.parametrize(
    "architecture, field",
    [
        ({"layers": True}, "layers"),
        ({"layers": "4"}, "layers"),
        ({"layers": 4.5}, "layers"),
        ({"window_pattern": None}, "window_pattern"),
        ({"act_epsilon": False}, "act_epsilon"),
        ({"act_epsilon": float("inf")}, "act_epsilon"),
        ({"options": {"activation": {"expression": "x"}}}, "options"),
        ({"options": {"output_scale": float("nan")}}, "options"),
    ],
)
def test_model_config_types_rejected_before_plugin_import(
    tmp_path, architecture, field
):
    marker = tmp_path / "must-not-exist"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()"
    with pytest.raises(GymError, match=f"ModelConfig.{field}") as error:
        ingest(
            rewrite_package(
                {
                    "architecture.json": json.dumps(architecture),
                    "plugin/research_model/__init__.py": source,
                }
            )
        )
    assert error.value.code == "invalid_artifact"
    assert not marker.exists()


def test_invalid_packages_never_execute(tmp_path):
    for name in (
        "../outside",
        "torch.py",
        "plugin/another/model.py",
        "plugin/research_model/weights.pt",
    ):
        with pytest.raises(GymError):
            ingest(rewrite_package({name: "payload"}))
    marker = tmp_path / "must-not-exist"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    accepted = ingest(rewrite_package({"plugin/research_model/__init__.py": source}))
    assert accepted["files"]["plugin/research_model/__init__.py"] == source
    assert not marker.exists()  # Ingestion does not import or interpret model source.
    with pytest.raises(GymError):
        ingest(rewrite_package({"architecture.json": '{"optimizer":"changed"}'}))


def test_evidence_tampering_rejected(context):
    service, credentials, _, _ = context
    verifier = {
        "session_id": credentials["session_id"],
        "token": credentials["verifier_token"],
    }
    signed = snapshot(service, verifier)
    signed["payload"]["budget"]["spent"] = "-1"
    with pytest.raises(GymError):
        verify_receipt({}, signed, verifier)


def test_host_port_persisted_and_conflict_explicit(context, tmp_path):
    service = context[0]
    first = bind(service, tmp_path / "listener")
    port = first.server_port
    conflict = tmp_path / "conflict"
    conflict.mkdir()
    (conflict / "listener.json").write_text(json.dumps({"port": port}))
    with pytest.raises(OSError):
        bind(service, conflict)
    first.server_close()
    first.service_lock.close()
    second = bind(service, tmp_path / "listener")
    assert second.server_port == port
    second.server_close()
    second.service_lock.close()


def test_harbor_native_schema():
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_task_verifier_mode

    task = Task(ROOT / "gym/tasks/architecture-research")
    assert str(resolve_task_verifier_mode(task.config).value) == "separate"
    assert {a.destination for a in task.config.artifacts} == {
        "submission.json",
        "agent-runtime.json",
        "gym-evidence.json",
    }
    assert all(a.service in (None, "main") for a in task.config.artifacts)
    assert [h.service for h in task.config.verifier.collect] == ["main"]


def test_native_model_loss_gradients_and_cost_parity():
    import subprocess
    import sys

    script = """import json,sys,torch
from nexttransformer_gym.native_plugin import activate
candidate,profile=json.load(sys.stdin)
activate(candidate)
from nexttransformer_gym.numerics import build,inspect,model_config
from nexttransformer.models import build_model
from nexttransformer.protocol.compute import audit_graph
adapted=build(candidate,profile)
reference=build_model(model_config(candidate,profile),'cpu',seed=42)
assert type(adapted).__name__ == 'ResearchModel'
x=torch.arange(8).reshape(1,8)
a,b=adapted(x,x),reference(x,x)
assert torch.equal(a,b)
a.backward();b.backward()
for (n,p),(m,q) in zip(adapted.named_parameters(),reference.named_parameters()):
    assert n==m
    assert (p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad),n
report=inspect(candidate,profile)
original=audit_graph(model_config(candidate,profile),batch=1)
assert sum(int(report[k]) for k in ('forward','backward','optimizer'))==original['total']
"""
    profile = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())[
        "research-fixture"
    ]
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps([ingest(candidate_bytes()), profile]),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("split", ["train", "val", "dev"])
def test_native_train_validation_stream_parity(tmp_path, split):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from nexttransformer_gym.data import parquet_stream
    from nexttransformer.runtime.data import BestFitStream

    class Tokenizer:
        def encode(self, texts, **kwargs):
            return [[1, *[ord(c) % 31 for c in text]] for text in texts]

        def get_bos_token_id(self):
            return 1

    pq.write_table(
        pa.table({"text": ["training text"] * 3000}), tmp_path / "shard_00000.parquet"
    )
    pq.write_table(
        pa.table({"text": ["validation text"] * 3000}),
        tmp_path / "shard_06542.parquet",
    )
    native_split = "train" if split == "train" else "val"
    shard = "shard_00000.parquet" if split == "train" else "shard_06542.parquet"
    adapted = parquet_stream([tmp_path / shard], Tokenizer(), 8, split)
    reference = BestFitStream(tmp_path, Tokenizer(), native_split, sequence_len=8)
    assert adapted.paths == reference.paths and adapted.split == reference.split
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            adapted.batch(3, torch.device("cpu")),
            reference.batch(3, torch.device("cpu")),
        )
    )
    state = adapted.state_dict()
    expected = adapted.batch(3, torch.device("cpu"))
    resumed = parquet_stream([tmp_path / shard], Tokenizer(), 8, split)
    resumed.load_state_dict(state)
    assert all(
        torch.equal(a, b)
        for a, b in zip(expected, resumed.batch(3, torch.device("cpu")))
    )


def test_last_shard_is_validation_and_penultimate_is_training(monkeypatch, tmp_path):
    from nexttransformer_gym import data
    from nexttransformer.runtime.files import VALIDATION_SHARD

    # Check the full inventory boundary without creating thousands of parquet files.
    monkeypatch.setattr(data, "file_hash", lambda path: path.name)
    paths = [tmp_path / f"shard_{i:05d}.parquet" for i in range(6542)]
    stream = data.parquet_stream(paths, None, 8, "train")
    assert stream.paths[-1].name == "shard_06541.parquet"
    validation = tmp_path / VALIDATION_SHARD
    for invalid_paths, split in (
        (paths + [validation], "train"),
        ([validation], "train"),
        ([paths[-1]], "val"),
        ([paths[-1]], "dev"),
        ([paths[-1], validation], "val"),
        (list(reversed(paths)), "train"),
    ):
        with pytest.raises(GymError):
            data.parquet_stream(invalid_paths, None, 8, split)


def test_unknown_parameters_are_not_silently_accepted(context):
    with pytest.raises(GymError):
        context[2]("get_budget", command="anything")


def test_trial_receipt_never_persists_provider_credentials(
    monkeypatch, tmp_path, context
):
    import subprocess
    import sys
    from types import SimpleNamespace
    from nexttransformer_gym import host_trial

    async def no_trial_result(*_args):
        return None

    monkeypatch.setattr(host_trial, "load", lambda _args: context[0])
    monkeypatch.setattr(host_trial, "docker_gateway", lambda: "127.0.0.1")
    monkeypatch.setattr(host_trial, "run_job", no_trial_result)

    runner = runpy.run_path(str(ROOT / "gym/run_trial.py"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-provider-secret-never-persist")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="sha256:fixture-image\n"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_trial.py",
            "--agent",
            "oracle",
            "--research-flops-limit",
            "1000",
            "--jobs-dir",
            str(tmp_path),
            "--client-image",
            "client",
            "--verifier-image",
            "verifier",
        ],
    )
    with pytest.raises(SystemExit) as exit_status:
        runner["main"]()
    assert exit_status.value.code == 1
    receipt = next(tmp_path.glob("*.binding.json")).read_text()
    assert "ANTHROPIC_API_KEY" not in receipt and "test-provider-secret" not in receipt


def test_rank_adapter_rendezvous_and_meter_on_cpu(tmp_path):
    from fractions import Fraction
    import torch.multiprocessing
    from nexttransformer_gym.common import file_hash
    from nexttransformer_gym.official import train_rank

    # A tiny CPU protocol probe tests process/rendezvous/accounting mechanics;
    # it makes no claim to replace GPU scientific validation.
    root = tmp_path / "snapshot"
    files = {
        "nexttransformer/__init__.py": "",
        "nexttransformer/models/__init__.py": "",
        "nexttransformer/models/lm.py": "import torch\ndef build_model():\n    return torch.nn.Linear(2, 1)\n",
        "nexttransformer/train.py": """import torch
import torch.distributed as dist
from nexttransformer.models.lm import build_model
dist.init_process_group('gloo')
model = build_model()
value = model(torch.ones(1, 2)).sum()
value.backward()
count = torch.ones(1)
dist.all_reduce(count)
assert count.item() == 2
""",
        "config.json": json.dumps({"calibration": {"cost_record": "cost.json"}}),
        "cost.json": "{}",
        "dense.json": "{}",
    }
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    plan = {
        "candidate_hash": "probe",
        "snapshot": str(root),
        "manifest": {name: file_hash(root / name) for name in files},
        "config": "config.json",
        "resources": str(tmp_path),
        "method": "probe",
        "initialization_flops": 1,
    }
    request = {
        "candidate": {"hash": "probe"},
        "profile": {"official_plan": plan, "batch_size": 1},
        "quote": {"bound": "100"},
    }
    output = tmp_path / "output"
    output.mkdir()
    torch.multiprocessing.spawn(
        train_rank,
        args=(2, request, str(output), Fraction(1), Fraction(1)),
        nprocs=2,
        join=True,
    )
    assert [
        json.loads((output / f"rank-{rank}.json").read_text())["spent"]
        for rank in range(2)
    ] == ["3", "3"]


def test_final_meter_refuses_before_dispatch():
    from nexttransformer_gym.official import FinalMeter

    meter = FinalMeter(100, 3, 10)
    meter.charge(90)
    with pytest.raises(GymError):
        meter.charge(11)
    assert meter.spent == 90


def test_final_meter_keeps_integer_precision_and_requires_registered_world():
    from fractions import Fraction
    from types import SimpleNamespace
    from nexttransformer_gym.official import FinalMeter, official_world

    amount = 2**60 + 1
    meter = FinalMeter(amount, Fraction(amount, 3), Fraction(amount, 3))
    meter.before_forward(
        SimpleNamespace(training=False), [SimpleNamespace(numel=lambda: 3)]
    )
    assert meter.spent == amount
    with pytest.raises(GymError):
        meter.before_forward(
            SimpleNamespace(training=False), [SimpleNamespace(numel=lambda: 1)]
        )
    for devices in ([], [0], [0] * 8):
        with pytest.raises(GymError):
            official_world({"cuda_visible_devices": devices})
    assert official_world({"cuda_visible_devices": list(range(8))}) == 8


def test_known_numerical_failure_is_charged_before_reporting(monkeypatch):
    from nexttransformer_gym import worker

    events = []
    monkeypatch.setattr(worker, "emit", events.append)
    monkeypatch.setattr(
        worker.sys, "stdin", io.StringIO('{"ack":"update"}\n{"ack":"update"}\n')
    )

    def failing_work():
        raise GymError("numerical_failure", "Non-finite gradient")

    with pytest.raises(GymError, match="Non-finite"):
        worker.operation("update", 123, failing_work)
    assert [event["kind"] for event in events] == ["intent", "charge"]
    assert events[-1]["amount"] == "123"


def test_running_cancel_stops_at_next_accounted_boundary(context):
    import time

    service, _, call, candidate = context
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=8,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="cancel-running")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(Scheduler(service).work_once)
        deadline = time.monotonic() + 15
        while call("get_job", job_id=job["id"])["state"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        call("cancel_job", job_id=job["id"], idempotency_key="stop")
        assert running.result(timeout=20)
    assert call("get_job", job_id=job["id"])["state"] == "cancelled"
    budget = call("get_budget")
    assert budget["reserved"] == "0" and int(budget["available"]) >= 0


@pytest.mark.parametrize("damage", ["missing", "modified"])
def test_checkpoint_damage_refused_before_dispatch(context, damage):
    first = run_job(context, target_update=1)
    service, _, call, candidate = context
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=2,
        checkpoint_id=first["checkpoint_id"],
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="resume-damaged")
    path = service.backend.checkpoint_path(first["checkpoint_id"])
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"changed")
    spent = call("get_budget")["spent"]
    assert Scheduler(service).work_once()
    state = call("get_job", job_id=job["id"])
    assert (
        state["state"] == "failed" and state["error_code"] == "incompatible_checkpoint"
    )
    assert call("get_budget")["spent"] == spent
    assert call("get_budget")["reserved"] == "0"


def test_expired_quote_cannot_reserve_budget(context):
    service, credentials, call, candidate = context
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=1,
    )
    with service.store.transaction() as db:
        quote["expires_at"] = 0
        service.store.update(db, credentials["session_id"], "quote", quote)
    with pytest.raises(GymError):
        call("run_experiment", quote_id=quote["id"], idempotency_key="expired")
    assert call("get_budget")["reserved"] == "0"


def test_native_plugin_staging_preserves_every_source_file(tmp_path):
    from nexttransformer_gym.native_plugin import write_plugin

    candidate = ingest(candidate_bytes("tanh"))
    candidate["files"]["plugin/research_model/model.py"] = candidate["files"][
        "plugin/research_model/model.py"
    ].replace("\n", "\r\n")
    plugin = write_plugin(candidate, tmp_path)
    assert len(list(plugin.glob("*.py"))) == 3
    prefix = "plugin/" + candidate["method"] + "/"
    for name, source in candidate["files"].items():
        if name.startswith(prefix):
            assert (plugin / name[len(prefix) :]).read_bytes() == source.encode("utf-8")
    assert candidate["architecture"]["options"]["activation"] == "tanh"


def test_same_package_loads_directly_in_native_nexttransformer(tmp_path):
    import os
    import subprocess
    import sys
    from nexttransformer_gym.native_plugin import write_plugin

    candidate = ingest(candidate_bytes("silu"))
    write_plugin(candidate, tmp_path)
    script = """import hashlib,json,sys,torch
candidate,mode=json.load(sys.stdin)
if mode == "gym":
    from nexttransformer_gym.native_plugin import activate
    activate(candidate)
    from nexttransformer_gym.numerics import build
    model=build(candidate,dict(width=32,heads=2,sequence_len=8,vocab_size=64,device="cpu"))
else:
    # Independent native entry: no Gym imports, parser, config adapter or source transformation.
    from nexttransformer.models.config import ModelConfig
    from nexttransformer.models import build_model
    declaration=candidate["declaration"]
    choices=dict(candidate["architecture"])
    choices["options"]={**declaration["options"],**choices.get("options",{})}
    config=ModelConfig(method=candidate["method"],width=32,heads=2,sequence_len=8,vocab_size=64,scale="debug",**choices)
    model=build_model(config,"cpu",seed=42)
x=torch.arange(8).reshape(1,8)
loss=model(x,x);loss.backward()
h=hashlib.sha256()
for name,p in model.named_parameters():
    h.update(name.encode());h.update(p.detach().numpy().tobytes())
    if p.grad is not None:h.update(p.grad.numpy().tobytes())
print(json.dumps({"loss":loss.item(),"parameters_and_gradients":h.hexdigest(),"class":type(model).__name__,"config":model.config.serialized()}))
"""
    results = []
    for mode in ("native", "gym"):
        env = dict(
            os.environ,
            NEXTTRANSFORMER_METHOD_PATHS=str(tmp_path) if mode == "native" else "",
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps([candidate, mode]),
            text=True,
            capture_output=True,
            env=env,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        results.append(json.loads(result.stdout.splitlines()[-1]))
    assert results[0] == results[1]
    assert results[0]["class"] == "ResearchModel"
    accepted = ingest(
        rewrite_package({"architecture.json": json.dumps(results[0]["config"])})
    )
    assert accepted["architecture"] == results[0]["config"]


@pytest.mark.parametrize(
    "isolation",
    [
        pytest.param(
            "bwrap",
            marks=pytest.mark.skipif(not BWRAP, reason="bwrap cannot run here"),
        ),
        pytest.param(
            "landlock",
            marks=pytest.mark.skipif(
                not LANDLOCK, reason="landlock needs Landlock and a root controller"
            ),
        ),
    ],
)
def test_native_worker_cannot_read_controller_files_or_reach_host(tmp_path, isolation):
    import socket

    service, _, call, _ = session(tmp_path, isolation)
    private = tmp_path / "controller-secret"
    private.write_text("not-for-workers")
    unix_path = tmp_path / "controller.sock"
    with socket.socket() as listener, socket.socket(socket.AF_UNIX) as unix:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        unix.bind(str(unix_path))
        unix.listen()
        checks = f"""import socket
for path in ({str(private)!r}, {str(service.store.path)!r}):
    try:
        open(path, "rb").close()
    except OSError:
        continue
    raise AssertionError("worker read " + path)
s=socket.socket();s.settimeout(0.2)
assert s.connect_ex(("127.0.0.1",{port})) != 0
s.close()
try:
    socket.socket(socket.AF_UNIX).connect({str(unix_path)!r})
except OSError:
    pass
else:
    raise AssertionError("worker reached a controller Unix socket")
"""
        source = runpy.run_path(str(ROOT / "gym/examples/make_candidate.py"))["ENTRY"]
        candidate = call(
            "submit_candidate",
            package_base64=base64.b64encode(
                rewrite_package({"plugin/research_model/__init__.py": checks + source})
            ).decode(),
            idempotency_key="isolation-probe",
        )
        report = call(
            "inspect_cost", candidate_id=candidate["id"], profile_id="research-fixture"
        )
        assert report["coverage"] == "complete"
