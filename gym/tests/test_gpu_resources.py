"""Resource visibility, allowed placement and conservative GPU queue behavior."""

import json

import pytest

from nexttransformer_gym import gpu
from nexttransformer_gym.backend import NativeBackend, Scheduler
from nexttransformer_gym.common import GymError

from test_gym import ISOLATION, ROOT, context as make_context

GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"


def card(identity, index, processes=0):
    return {
        "id": identity,
        "uuid": identity,
        "index": index,
        "device_minor": index,
        "name": "Fixture GPU",
        "memory_total_bytes": 16 * 1024**3,
        "memory_free_bytes": 12 * 1024**3,
        "utilization_percent": 0,
        "compute_process_count": processes,
        "graphics_process_count": 0,
        "process_count": processes,
    }


@pytest.fixture
def observed(monkeypatch, tmp_path):
    cards = [card(GPU_A, 0), card(GPU_B, 1, processes=1)]
    monkeypatch.setattr(gpu, "discover", lambda: {"status": "ok", "gpus": cards})
    monkeypatch.setattr(gpu, "_lock_path", lambda uuid: tmp_path / (uuid + ".lock"))
    return cards


def config(ids=None):
    return {
        "schema_version": "gym.execution.v1",
        "gpu_ids": [GPU_A] if ids is None else ids,
        "gpu_memory_limit_bytes": 1024**3,
        "max_concurrent_jobs": 1,
        "isolation": ISOLATION,
    }


@pytest.mark.parametrize(
    "override",
    [
        {"gpu_ids": [True]},
        {"gpu_ids": ["all"]},
        {"gpu_ids": [GPU_A, GPU_A]},
        {"gpu_memory_limit_bytes": 0},
        {"gpu_memory_limit_bytes": True},
        {"max_concurrent_jobs": 2},
        {"allow_idle_graphics": "true"},
        {"isolation": "chroot"},
        {"command": "nvidia-smi"},
    ],
)
def test_execution_config_rejects_unbounded_or_invalid_allocations(override):
    with pytest.raises(GymError):
        gpu.configuration({**config(), **override})


def test_operator_can_allow_idle_graphics_but_not_active_or_compute_work(observed):
    device = observed[0]
    device.update(graphics_process_count=2, process_count=2)
    assert not gpu.idle(device, gpu.configuration(config()))
    policy = gpu.configuration({**config(), "allow_idle_graphics": True})
    assert gpu.idle(device, policy)
    with gpu.reserve(policy, GPU_A) as lease:
        assert lease["id"] == GPU_A
    for change in (
        {"utilization_percent": 1},
        {"compute_process_count": 1, "process_count": 3},
        {"process_count": 3},
        {"graphics_process_count": None},
        {"memory_free_bytes": 1},
    ):
        assert not gpu.idle({**device, **change}, policy)


def test_gpu_lease_rechecks_occupancy_and_releases_after_exception(observed):
    settings = config()
    with pytest.raises(RuntimeError, match="worker failed"):
        with gpu.reserve(settings, GPU_A) as lease:
            assert lease["id"] == GPU_A
            with pytest.raises(GymError):
                with gpu.reserve(settings, GPU_A):
                    pytest.fail("An already leased GPU cannot be assigned twice")
            raise RuntimeError("worker failed")
    with gpu.reserve(settings, GPU_A) as lease:
        assert lease["id"] == GPU_A
    observed[0].update(compute_process_count=1, process_count=1)
    with pytest.raises(GymError):
        with gpu.reserve(settings, GPU_A):
            pytest.fail("A host process on the GPU must block dispatch")


def test_gpu_lease_rechecks_after_lock_acquisition(observed, monkeypatch):
    observations = iter(
        [
            {"status": "ok", "gpus": [card(GPU_A, 0)]},
            {"status": "ok", "gpus": [card(GPU_A, 0, processes=1)]},
        ]
    )
    monkeypatch.setattr(gpu, "discover", lambda: next(observations))
    with pytest.raises(GymError):
        with gpu.reserve(config(), GPU_A):
            pytest.fail(
                "A GPU becoming occupied during lock acquisition must be refused"
            )


def test_missing_discovery_cannot_turn_into_cpu_fallback(
    observed, monkeypatch, tmp_path
):
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    backend = NativeBackend(profiles, tmp_path, ROOT, execution_config=config())
    backend.worker_runtime["cuda_version"] = "fixture-cuda"
    monkeypatch.setattr(gpu, "discover", lambda: {"status": "unavailable", "gpus": []})
    resources = backend.resources(False)
    assert resources["gpu_discovery_status"] == "unavailable"
    assert resources["available_gpu_count"] == resources["allocatable_slots"] == 0
    assert resources["research_devices"] == ["cuda:0"]
    with pytest.raises(GymError):
        with gpu.reserve(config(), GPU_A):
            pytest.fail("Missing device observation must not dispatch a worker")


def test_discovery_counts_are_distinct_from_allocated_and_available(observed, tmp_path):
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    backend = NativeBackend(profiles, tmp_path, ROOT, execution_config=config([GPU_B]))
    backend.worker_runtime["cuda_version"] = "fixture-cuda"
    resources = backend.resources(False)
    assert resources["discovered_gpu_count"] == 2
    assert resources["allocated_gpu_count"] == 1
    assert resources["available_gpu_count"] == 0
    observed[1].update(compute_process_count=0, process_count=0)
    assert backend.resources(False)["available_gpu_count"] == 1
    with gpu.reserve(config([GPU_B]), GPU_B):
        leased = backend.resources(False)
        assert leased["available_gpu_count"] == 0
        assert (
            next(card for card in leased["gpus"] if card["id"] == GPU_B)["busy_reason"]
            == "gpu_leased"
        )
    assert backend.resources(False)["available_gpu_count"] == 1
    assert backend.resources(True)["available_gpu_count"] == 0
    backend.worker_runtime["cuda_version"] = None
    assert backend.resources(False)["available_gpu_count"] == 0


def test_gpu_placement_is_operator_owned(observed, tmp_path):
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    profile = profiles["research-fixture"]
    profile.update(device="cuda:0", cuda_visible_devices=[GPU_A])
    backend = NativeBackend(profiles, tmp_path, ROOT, execution_config=config())
    assert backend.validate_placement(profile, "0", backend.server_id) == GPU_A
    assert backend.validate_placement(profile, GPU_A) == GPU_A
    for options in (
        {"gpu_id": GPU_B},
        {"server_id": "unconfigured-remote"},
        {"gpu_id": "all"},
    ):
        with pytest.raises(GymError):
            backend.validate_placement(profile, **options)
    with pytest.raises(GymError):
        backend.validate_placement(profiles["final-fixture"], GPU_A)


def test_busy_gpu_keeps_job_queued_and_budget_reserved_until_cancel(observed, tmp_path):
    service, _, call, candidate = make_context.__wrapped__(tmp_path)
    # Shape-only cost inspection does not execute CUDA; dispatch must stop at
    # the occupancy check before starting any numerical worker.
    service.backend.execution_config = gpu.configuration(config())
    service.backend.profiles["research-fixture"].update(
        device="cuda:0", cuda_visible_devices=[GPU_A]
    )
    observed[0].update(compute_process_count=1, process_count=1)
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=1,
        gpu_id=GPU_A,
        server_id=service.backend.server_id,
    )
    assert quote["gpu_id"] == GPU_A
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="queued-gpu")
    before = call("get_budget")
    assert int(before["reserved"]) > 0 and before["spent"] == "0"
    assert Scheduler(service).work_once() is False
    assert call("get_job", job_id=job["id"])["state"] == "queued"
    assert call("get_budget") == before
    call("cancel_job", job_id=job["id"], idempotency_key="cancel-gpu")
    after = call("get_budget")
    assert after["spent"] == "0" and after["reserved"] == "0"
    assert after["available"] == after["limit"]


def test_graphics_process_blocks_idle_and_reservation(observed):
    observed[0].update(
        compute_process_count=0,
        graphics_process_count=1,
        process_count=1,
        utilization_percent=0,
    )
    assert not gpu.idle(observed[0], config())
    with pytest.raises(GymError):
        with gpu.reserve(config(), GPU_A):
            pytest.fail("Graphics jobs must block allocation even at zero utilization")


def test_xml_process_counts_include_compute_graphics_and_mixed():
    xml = f"""<nvidia_smi_log><gpu><uuid>{GPU_A}</uuid><processes>
    <process_info><pid>11</pid><type>C</type></process_info>
    <process_info><pid>12</pid><type>G</type></process_info>
    <process_info><pid>13</pid><type>C+G</type></process_info>
    </processes></gpu><gpu><uuid>{GPU_B}</uuid><processes /></gpu></nvidia_smi_log>"""
    assert gpu._process_counts(xml) == {
        GPU_A: {
            "compute_process_count": 2,
            "graphics_process_count": 2,
            "process_count": 3,
        },
        GPU_B: {
            "compute_process_count": 0,
            "graphics_process_count": 0,
            "process_count": 0,
        },
    }


@pytest.mark.parametrize(
    "processes",
    [
        "",
        "<processes>N/A</processes>",
        "<processes><process_info><pid>11</pid><type>N/A</type></process_info></processes>",
        "<processes><process_info><pid>0</pid><type>G</type></process_info></processes>",
    ],
)
def test_xml_missing_or_unknown_process_information_fails_closed(processes):
    with pytest.raises(GymError):
        gpu._process_counts(
            f"<nvidia_smi_log><gpu><uuid>{GPU_A}</uuid>{processes}</gpu></nvidia_smi_log>"
        )


def test_missing_total_process_observation_is_not_idle():
    observed = card(GPU_A, 0)
    observed.pop("process_count")
    assert not gpu.idle(observed, config())
