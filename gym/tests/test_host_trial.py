"""Host Gym remains outside the native Harbor Agent and verifier containers."""

import json
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml
from harbor.models.task.config import TaskConfig

from nexttransformer_gym.client import GymClient
from nexttransformer_gym.evidence import verify_receipt
from nexttransformer_gym import host_trial
from nexttransformer_gym.host_trial import HostGym, materialize_task

from test_gym import ROOT, candidate_bytes, context as make_context


def make_host(tmp_path):
    service, _, _, _ = make_context.__wrapped__(tmp_path / "service")
    return HostGym(
        service,
        tmp_path / "host",
        10**12,
        "host-fixture",
        bind_host="127.0.0.1",
        advertise_host="127.0.0.1",
    )


def client_for(host):
    return GymClient(host.agent_credential["endpoint"], host.agent_credential["token"])


def test_host_listener_holds_random_port_and_reuses_it_on_restart(tmp_path):
    host = make_host(tmp_path)
    host.start()
    try:
        port = host.server.server_port
        assert port > 0
        client = client_for(host)
        assert client.get_budget()["limit"] == str(10**12)
        assert int(host.agent_credential["endpoint"].rsplit(":", 1)[1]) == port
        service = host.service
    finally:
        host.close()
    restarted = HostGym(
        service,
        tmp_path / "host",
        10**12,
        "host-fixture",
        bind_host="127.0.0.1",
        advertise_host="127.0.0.1",
    )
    restarted.start()
    try:
        assert restarted.server.server_port == port
        assert client_for(restarted).get_budget()["limit"] == str(10**12)
    finally:
        restarted.close()


def test_materialized_task_uses_native_mcp_and_only_readonly_scoped_mounts(tmp_path):
    host = make_host(tmp_path)
    try:
        task = tmp_path / "task"
        materialize_task(
            ROOT / "gym/tasks/architecture-research",
            task,
            host.directory,
            "client@sha256:fixture",
            "verifier@sha256:fixture",
            120,
        )
        native = TaskConfig.model_validate_toml((task / "task.toml").read_text())
        assert native.agent.timeout_sec == 120
        assert native.agent.user == "agent"
        assert native.verifier.environment_mode == "separate"
        assert len(native.environment.mcp_servers) == 1
        mcp = native.environment.mcp_servers[0]
        assert mcp.transport == "stdio" and mcp.command == "python"
        assert mcp.args == ["-m", "nexttransformer_gym.mcp_server"]
        assert all(artifact.service != "gym" for artifact in native.artifacts)
        assert all(collect.service != "gym" for collect in native.verifier.collect)
        compose = yaml.safe_load((task / "environment/docker-compose.yaml").read_text())
        assert set(compose["services"]) == {"main"}
        main = compose["services"]["main"]
        assert (
            "depends_on" not in main and "devices" not in main and "deploy" not in main
        )
        assert "privileged" not in main and main.get("network_mode") != "host"
        mounts = {mount["target"]: mount for mount in main["volumes"]}
        assert set(mounts) == {"/run/gym-auth", "/export"}
        assert all(mount["read_only"] for mount in mounts.values())
        assert all(
            Path(mount["source"]).is_relative_to(host.directory)
            for mount in mounts.values()
        )
        assert all(Path(mount["source"]) != host.directory for mount in mounts.values())
        verifier_compose = yaml.safe_load(
            (task / "tests/docker-compose.yaml").read_text()
        )
        verifier = verifier_compose["services"]["main"]
        assert verifier["network_mode"] == "none"
        assert len(verifier["volumes"]) == 1
        mount = verifier["volumes"][0]
        assert mount["target"] == "/run/verifier-auth" and mount["read_only"]
        assert Path(mount["source"]) != Path(mounts["/run/gym-auth"]["source"])
    finally:
        host.close()


def test_host_lifecycle_hook_signs_current_service_submission_and_budget(
    tmp_path, monkeypatch
):
    from harbor.job import Job

    host = make_host(tmp_path)
    host.start()
    try:
        client = client_for(host)
        candidate = client.submit_candidate(candidate_bytes(), "host-candidate")
        job = client.validate_candidate(
            candidate_id=candidate["id"],
            profile_id="final-fixture",
            idempotency_key="host-checks",
        )
        assert client.wait_result(job["id"], timeout=30)["admitted"]
        submission = client.finalize(
            candidate_id=candidate["id"],
            profile_id="final-fixture",
            idempotency_key="host-final",
        )
        receipt = {
            "session_id": host.verifier_credential["session_id"],
            "submission_id": submission["id"],
            "candidate_hash": candidate["hash"],
        }
        signed = host.export_evidence()
        assert signed == json.loads(
            (host.directory / "export/evidence.json").read_text()
        )
        assert verify_receipt(receipt, signed, host.verifier_credential)["valid"]
        assert signed["payload"]["budget"] == client.get_budget()
        assert int(signed["payload"]["budget"]["spent"]) > 0
        with host.service.store.transaction() as db:
            session = host.service.store.get(
                db,
                host.verifier_credential["session_id"],
                "session",
                host.verifier_credential["session_id"],
            )
            session["integrity_violation"] = True
            host.service.store.update(db, session["id"], "session", session)
        captured = {}

        class JobStub:
            def on_agent_ended(self, callback):
                captured["callback"] = callback

            async def run(self):
                await captured["callback"](None)

        async def create(config):
            captured["config"] = config
            return JobStub()

        monkeypatch.setattr(Job, "create", create)
        args = SimpleNamespace(
            agent="claude-code",
            model="fixture-model",
            agent_version="2.1.312",
            max_turns=8,
            max_budget_usd="0.10",
            jobs_dir=tmp_path / "jobs",
        )
        asyncio.run(
            host_trial.run_job(
                args, "native-fixture", ROOT / "gym/tasks/architecture-research", host
            )
        )
        agent = captured["config"].agents[0]
        assert agent.name == "claude-code"
        assert agent.model_name == "fixture-model"
        assert agent.kwargs["max_turns"] == 8
        latest = json.loads((host.directory / "export/evidence.json").read_text())
        assert latest != signed
        assert not verify_receipt(receipt, latest, host.verifier_credential)["valid"]
    finally:
        host.close()


def test_host_run_passes_operator_config_and_worker_python_to_real_backend(
    tmp_path, monkeypatch
):
    actual_load = host_trial.load
    observed = {}

    class Loaded(Exception):
        pass

    def capture(settings):
        observed["settings"] = settings
        observed["service"] = actual_load(settings)
        raise Loaded

    monkeypatch.setattr(host_trial, "load", capture)
    policy = {
        "schema_version": "gym.execution.v1",
        "gpu_ids": ["GPU-00000000-0000-0000-0000-000000000001"],
        "gpu_memory_limit_bytes": 1024**3,
        "max_concurrent_jobs": 1,
    }
    args = SimpleNamespace(
        jobs_dir=tmp_path / "jobs",
        profiles=ROOT / "gym/examples/profiles.fixture.json",
        execution_policy=policy,
        server_id="fixture-node",
        worker_python=sys.executable,
    )
    with pytest.raises(Loaded):
        host_trial.run(
            args,
            lambda name: pytest.fail("No image operation should precede backend setup"),
        )
    settings = observed["settings"]
    backend = observed["service"].backend
    assert settings.platform == ROOT
    assert json.loads(settings.execution_config.read_text()) == policy
    assert backend.execution_config == {
        **policy,
        "allow_idle_graphics": False,
        "isolation": "bwrap",
    }
    assert backend.python == str(Path(sys.executable).absolute())
    assert backend.worker_runtime["python"] == list(sys.version_info[:3])
    assert backend.server_id == "fixture-node"
    assert backend.profiles["research-fixture"]["device"] == "cuda:0"
