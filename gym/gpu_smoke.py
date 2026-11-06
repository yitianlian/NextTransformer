"""Bounded real-CUDA HTTP/MCP integration check; never calls a model provider.

Run with an explicitly allocated idle GPU and a CUDA-enabled worker interpreter.
Artifacts, credentials and checkpoints remain in the supplied ignored state path.
Only tiny synthetic data are used: four optimizer updates plus admission checks.
"""

import argparse
import asyncio
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from nexttransformer_gym.backend import NativeBackend, Scheduler
from nexttransformer_gym.client import GymClient
from nexttransformer_gym.isolation import BACKENDS
from nexttransformer_gym.server import atomic_json, bind
from nexttransformer_gym.service import Service

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def exercise(client, credential, service, candidate, gpu_id):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nexttransformer_gym.mcp_server", "--credential", str(credential)],
    )
    results, jobs = [], []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()
            tools = {tool.name: tool for tool in (await mcp.list_tools()).tools}
            schemas = client.describe_space()["operation_schemas"]
            assert {name: tool.inputSchema for name, tool in tools.items()} == {
                name: schema["inputSchema"] for name, schema in schemas.items()
            }
            for target, checkpoint_index in ((1, None), (2, 0), (2, None)):
                quote = client.quote_experiment(
                    candidate_id=candidate["id"],
                    profile_id="research-fixture",
                    target_update=target,
                    gpu_id=gpu_id,
                    server_id=service.backend.server_id,
                    checkpoint_id=results[checkpoint_index]["checkpoint_id"]
                    if checkpoint_index is not None
                    else None,
                )
                response = await mcp.call_tool(
                    "run_experiment",
                    {
                        "quote_id": quote["id"],
                        "idempotency_key": quote["id"],
                    },
                )
                assert not response.isError, response
                job = json.loads(response.content[0].text)
                assert await asyncio.to_thread(Scheduler(service).work_once)
                result = client.wait_result(job["id"], timeout=5)
                execution = result["execution"]
                assert execution["cuda_available"]
                assert execution["device"].startswith("cuda")
                assert all(
                    device.startswith("cuda")
                    for device in execution["model_parameter_devices"]
                )
                assert execution["peak_memory_allocated_bytes"] > 0
                results.append(result)
                jobs.append(client.get_job(job_id=job["id"]))
            budget = await mcp.call_tool("get_budget", {})
            assert not budget.isError
            assert json.loads(budget.content[0].text) == client.get_budget()
    return results, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu", required=True, help="Explicit allocated GPU UUID or index"
    )
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--isolation", choices=BACKENDS, default="bwrap")
    args = parser.parse_args()
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    for profile in profiles.values():
        profile.update(
            device="cuda:0",
            cuda_visible_devices=[args.gpu],
            resource_profile_id="cuda-smoke-v1",
            accelerator_class="cuda",
            memory_limit_bytes=8 * 1024**3,
            timeout_sec=120,
        )
    backend = NativeBackend(
        profiles,
        state / "worker",
        ROOT,
        python=args.worker_python,
        execution_config={
            "schema_version": "gym.execution.v1",
            "gpu_ids": [args.gpu],
            "gpu_memory_limit_bytes": 2 * 1024**3,
            "max_concurrent_jobs": 1,
            "isolation": args.isolation,
        },
    )
    resources = backend.resources(False)
    # The backend's occupancy check must refuse busy devices before any work.
    if resources.get("available_gpu_count", 0) < 1:
        raise RuntimeError("The explicitly allocated GPU is not currently idle.")
    service = Service(state / "gym.sqlite3", backend)
    credentials = service.create_session(10**12, "real-gpu-smoke")
    server = bind(service, state / "listener")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    credential = state / "credential.json"
    atomic_json(credential, {"endpoint": endpoint, "token": credentials["agent_token"]})
    client = GymClient(endpoint, credentials["agent_token"])
    try:
        package = io.BytesIO()
        runpy.run_path(str(ROOT / "gym/examples/make_candidate.py"))["package"](
            package, "square"
        )
        candidate = client.submit_candidate(package.getvalue(), "gpu-smoke-candidate")
        results, jobs = asyncio.run(
            exercise(client, credential, service, candidate, args.gpu)
        )
        checkpoint_paths = [
            str(backend.checkpoint_path(result["checkpoint_id"]))
            for result in results[1:]
        ]
        subprocess.run(
            [
                args.worker_python,
                "-c",
                "import sys,torch; a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in sys.argv[1:]]; "
                "assert a['step']==b['step']==2; assert a['data']==b['data']; "
                "assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])",
                *checkpoint_paths,
            ],
            check=True,
            timeout=30,
        )
        assert results[1]["dev_bpb"] == results[2]["dev_bpb"]
        check = client.validate_candidate(
            candidate_id=candidate["id"],
            profile_id="research-fixture",
            idempotency_key="gpu-admission",
        )
        assert Scheduler(service).work_once()
        admission = client.wait_result(check["id"], timeout=5)
        assert all(admission["checks"].values())
        budget = client.get_budget()
        assert int(budget["spent"]) > 0 and budget["reserved"] == "0"
        report = {
            "ok": True,
            "transport": ["http", "mcp-stdio"],
            "resources": client.get_resources(),
            "jobs": jobs,
            "results": results,
            "admission": admission,
            "budget": budget,
            "resumed_equals_uninterrupted": True,
            "optimizer_updates": 4,
        }
        atomic_json(state / "report.json", report)
        print(
            json.dumps(
                {
                    "ok": True,
                    "optimizer_updates": 4,
                    "spent_flops": budget["spent"],
                    "reserved_flops": budget["reserved"],
                    "report": str(state / "report.json"),
                }
            )
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        server.service_lock.close()


if __name__ == "__main__":
    main()
