# NextTransformer Gym

[中文](README.zh-CN.md) · [Design](DESIGN.md)

Gym runs model architecture research experiments: the host runs training, and Harbor runs Claude Code and the submission verifier in separate containers.
The examples use synthetic data. They do not represent production ClimbMix or CORE-22 performance.

- [Install](#install) · [Start a research session](#start-a-research-session) · [Run an experiment and submit](#run-an-experiment-and-submit)
- [Gym API](#gym-api) · [Submit a native candidate](#submit-a-native-candidate) · [Worker isolation](#worker-isolation)
- [Errors and recovery](#errors-and-recovery) · [Check the installation](#check-the-installation)

The Agent's requirements are in the [task instructions](tasks/architecture-research/instruction.md). Run the host commands below from the repository root.

## Install

Requirements:

- Linux, Docker and unprivileged user namespaces.
- Python 3.12 or 3.13, and `uv`.
- Bubblewrap, or a kernel with Landlock; see [Worker isolation](#worker-isolation).

```sh
uv sync --project gym --extra worker
docker build -f gym/Dockerfile --target client -t nt-gym-client:local .
docker build -f gym/Dockerfile --target verifier -t nt-gym-verifier:local .
```

The `worker` extra installs the CPU dependencies. GPU workers need a CUDA Python environment with the versions in `gym/pyproject.toml`.
Gym does not need a CUDA container image. For deployment, pin the Docker base images to reviewed digests.
The launcher requires local images and records their immutable IDs.

## Start a research session

Set the provider credentials in the controller environment. Harbor passes them to Claude Code.
The example sets the model, CLI version, time limit, turn limit and Agent USD limit.

The Agent USD limit is optional. Omit `agent.max_budget_usd` to use no launcher-defined API cost limit.
Set `--max-budget-usd` to override the configured limit. A specified limit must be finite and greater than zero.

`budget.research_flops_limit` sets one research allocation for the whole session. Final training uses a separate allocation.

```sh
uv run --project gym python gym/run_trial.py --config gym/examples/claude-code.toml --dry-run
uv run --project gym python gym/run_trial.py --config gym/examples/claude-code.toml
```

For DeepSeek Flash, select `gym/examples/deepseek-v4.1-flash.toml`.
Set `GYM_RESEARCH_MODEL` to the provider's model ID.
Set `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` for an endpoint that supports Anthropic Messages.
For a strict billing cap, set a provider limit. The CLI USD option depends on the provider's usage and price data.

For GPU work, create a local configuration:

```sh
mkdir -p .agent-work
cp gym/examples/resources.gpu.json .agent-work/resources.local.json
```

Set `gpu_ids` to allocated indices or UUIDs. Set `gpu_memory_limit_bytes` to the allowed memory per GPU.
Keep `max_concurrent_jobs` at `1`. Start the session with that configuration:

```sh
uv run --project gym python gym/run_trial.py \
  --config gym/examples/deepseek-v4.1-flash.toml \
  --execution-config .agent-work/resources.local.json \
  --worker-python /path/to/cuda-environment/bin/python
```

Omit the GPU configuration for CPU examples. Run one research session at a time.
Each Gym state has one worker slot. Separate states do not share a host-wide scheduler.
A busy GPU keeps a job queued. Gym does not switch a GPU job to CPU.

<a id="gym-api"></a>

## Gym API

Inside the Agent container, read `/run/gym-auth/credential.json`. It contains `endpoint`, `token` and `session_id`.
Harbor registers the `gym` MCP tools. The Python `GymClient` uses the same HTTP API.
An external MCP client can run `python -m nexttransformer_gym.mcp_server --credential <credential.json>`.

For direct HTTP calls, send JSON to `POST <endpoint>/v1/call`.
Set `Content-Type: application/json` and `Authorization: Bearer <token>`.

```json
{"schema_version": "gym.api.v1", "operation": "describe_space", "params": {}}
```

A successful HTTP response contains `schema_version` and `result`. SDK and MCP return the result value.

Call `describe_space` first. Its `operation_schemas` gives the parameter types, required fields and limits.
Its `candidate_contract` gives the model schema and merge rules. Each profile gives its fixed configuration values.

| Operation | Parameters | Result |
| --- | --- | --- |
| `describe_space` | None | API schemas, candidate rules and profiles |
| `get_resources` | Optional `resource_profile_id` | Host resources, GPU allocation and availability |
| `list_components` | Optional `cursor`, `limit` | Approved reference IDs |
| `read_reference` | `resource_id` | Reference text |
| `list_experiments` | Optional `scope`, `cursor`, `limit` | Experiment summaries |
| `read_experiment` | `experiment_id`, optional `scope` | Results, configuration and source references |
| `submit_candidate` | `package_base64` | Candidate ID, hashes and metadata |
| `list_candidates` | Optional `cursor`, `limit` | Session candidates |
| `get_candidate` | `candidate_id` | Candidate metadata and submitted configuration |
| `inspect_cost` | `candidate_id`, `profile_id` | Traced FLOPs, parameters and coverage |
| `calibrate_candidate` | `candidate_id`, `profile_id`, `grid` | Static-analysis job with derived candidates and costs |
| `validate_candidate` | `candidate_id`, `profile_id` | Full check job, including final admission |
| `debug_candidate` | `candidate_id`, `profile_id`, `check`, optional `shape` | Selected diagnostic job |
| `quote_experiment` | `candidate_id`, `profile_id`, optional placement and continuation fields below | Cost bound and affordability |
| `run_experiment` | `quote_id` | Queued job |
| `get_job` | `job_id` | Job state, progress and execution details |
| `get_result` | `job_id` | Completed result |
| `cancel_job` | `job_id` | Updated cancellation state |
| `read_events` | Optional integer `cursor`, `limit` | Session events |
| `get_budget` | None | Limit, spent, reserved and available FLOPs |
| `finalize` | `candidate_id`, `profile_id` | Frozen submission |
| `get_submission` | `submission_id` | Submission and final-run state |

Use `scope="published"` for published experiments or `scope="session"` for your own experiments. The default is `published`.
Pagination uses a string cursor, except for `read_events`. The limit is 1–100 and defaults to 50.

Add `idempotency_key` to `submit_candidate`, `calibrate_candidate`, `validate_candidate`, `debug_candidate`, `run_experiment`, `cancel_job` and `finalize`.
Use a 1–128 character key. Retry identical input with the same key. Use a new key for changed input.
Quotation creates state but reserves no FLOPs. `run_experiment` reserves the quoted amount.

Optional quote fields are `target_update`, `checkpoint_id`, `resource_profile_id`, `server_id` and `gpu_id`.
`target_update` is an absolute step, not an additional step count. It defaults to 1 and must fit the profile's limit.
For continuation, the target must exceed the checkpoint step.
Use the local server ID and an allocated GPU UUID or index string from `get_resources`.
Omit placement fields to let Gym select an allocated idle card.

Diagnostic names are `construction`, `numerical`, `causal`, `checkpoint` and `cost_coverage`.
If you supply `shape`, use the profile's batch size and sequence length.
A diagnostic charges its required work. It does not grant final admission.

## Submit a native candidate

A candidate contains native plugin source and a partial `ModelConfig`.
Create a ZIP with this structure:

```text
proposal.json
architecture.json
plugin/<method>/method.json
plugin/<method>/__init__.py
plugin/<method>/<optional local Python modules>
```

Limit the ZIP and total uncompressed content to 1 MiB each. Use at most 32 UTF-8 text files, without directory entries.
Do not include weights, installers or training scripts. Gym stores the source without importing it during upload.
Use [make_candidate.py](examples/make_candidate.py) as the package example.

Example `proposal.json`:

```json
{
  "schema_version": "gym.candidate.v1",
  "sdk_version": "reference_plugin_v1",
  "method": "research_model",
  "parent_ids": [],
  "hypothesis": "Compare a deeper model under the same training protocol."
}
```

Supply the first three fields. The method must match the plugin directory and `method.json.name`.
Do not replace a registered method. Optional `parent_ids` can contain at most 16 candidate IDs from this session.
Gym stores the hypothesis as metadata.

Example `architecture.json` for the example plugin:

```json
{
  "layers": 4,
  "mlp_width": 128,
  "window_pattern": "L",
  "options": {"activation": "silu", "output_scale": 1.0}
}
```

Use native `ModelConfig` fields and options declared in `method.json.options`.
Option values can be strings, finite numbers, booleans or nested lists of these values.
Gym does not parse model expressions. Native checks still control valid field combinations.
A native `model` hook must subclass `ReferenceLM`.
Read the `plugin-contract` and `reference-model` references for hook interfaces.

| Field | Input rule |
| --- | --- |
| `layers`, `mlp_width`, `window_pattern` and other architecture fields | Follow the published types and native checks. Some fields apply only to specific methods. |
| `method` | Omit it or match the proposal method. |
| `width`, `heads`, `sequence_len`, `vocab_size`, `scale` | Omit them or match the selected profile. |
| `routing_seed` | Omit it or use 1729. |
| Training settings, paths, commands and environment variables | Keep these outside the model configuration. Gym rejects them as unknown fields. |

Integer fields reject strings and booleans. Number fields require finite numbers. Do not use `null` for an omitted field.
For missing fields, Gym applies these sources in order. Later values take precedence:

1. Native `ModelConfig` defaults.
2. Gym defaults: `layers=24`, `mlp_width=6144`, `window_pattern="L"`.
3. `method.json.shell`.
4. `architecture.json`.

For options, the order is `method.json.options`, then `debug_options` for debug profiles, then `architecture.json.options`.
Gym fills omitted profile fields. It rejects explicit values that differ from the profile.

To change only the configuration, upload the same source with a new `architecture.json`.
Alternatively, pass 1–32 complete architecture objects in `calibrate_candidate.grid`.
Each object replaces the whole architecture. It does not patch the parent.
Call `get_result` for the returned job ID to read `derived[].candidate` and `derived[].cost`.
Calibration does not train or measure throughput. Source or configuration changes create a new candidate.
A changed candidate cannot use the parent's checkpoint.

## Run an experiment and submit

Use the following example inside the Agent container. First create the candidate package:

```sh
python /opt/gym/make_candidate.py /workspace/candidate.zip
```

The profile names below refer to the synthetic examples.

```python
import json
from pathlib import Path
from nexttransformer_gym import GymClient

binding = json.loads(Path("/run/gym-auth/credential.json").read_text())
gym = GymClient(binding["endpoint"], binding["token"])
candidate = gym.submit_candidate(Path("/workspace/candidate.zip").read_bytes(), "idea-1")
quote = gym.quote_experiment(
    candidate_id=candidate["id"], profile_id="research-fixture", target_update=2
)
job = gym.run_experiment(quote_id=quote["id"], idempotency_key="run-1")
result = gym.wait_result(job["id"])
checks = gym.validate_candidate(
    candidate_id=candidate["id"], profile_id="final-fixture", idempotency_key="checks-1"
)
assert gym.wait_result(checks["id"])["admitted"]
submission = gym.finalize(
    candidate_id=candidate["id"], profile_id="final-fixture", idempotency_key="final-1"
)
Path("/workspace/submission.json").write_text(json.dumps({
    "session_id": binding["session_id"],
    "submission_id": submission["id"],
    "candidate_hash": candidate["hash"],
}))
```

`submit_candidate` takes ZIP bytes in the SDK. Direct HTTP uses their base64 text as `package_base64`.
The response contains `id`, `hash`, `configuration_hash`, `architecture`, `declaration`, `method`, `parent_ids` and `contract`.
`architecture` contains the submitted partial configuration. Gym resolves defaults for the chosen profile.

Check the job state before you read a result. `get_result` returns `not_found` until a result exists.
A successful check job can contain failed checks. Read `checks` and `admitted` before you submit.
Settle or cancel all jobs before `finalize`. The account must have no reserved budget.
Gym retains the source. The verifier checks the receipt but does not start final training.
See [research_with_placement.py](examples/research_with_placement.py) for resource selection.

<a id="worker-isolation"></a>

## Worker isolation

Candidate code runs in worker processes on the host. Set `isolation` in the execution configuration to select the backend:

| Value | Isolation | Use |
| --- | --- | --- |
| `bwrap` (default) | Fresh bubblewrap user, mount, PID and network namespaces | Hosts that allow unprivileged mount namespaces |
| `landlock` | Landlock path rules with user, network, IPC and UTS namespaces. CPU workers also get a PID namespace and a seccomp filter against new Unix sockets | Hosts that deny mounts, such as unprivileged Kubernetes pods |
| `none` | No isolation | Trusted candidates only. Gym prints a warning at startup |

`landlock` requires a root controller, util-linux `unshare` and an x86_64 or aarch64 kernel with Landlock.
A CPU session can use a configuration with only this field:

```json
{"schema_version": "gym.execution.v1", "isolation": "landlock"}
```

`landlock` is weaker than `bwrap`:

- Workers can detect whether a path exists and can list every directory under the platform root and the Gym source root. They cannot read files outside the allowed paths.
- Landlock does not restrict metadata changes such as chmod and utime.
- CUDA needs the host PID namespace and Unix sockets, so CUDA workers keep both. Processes they detach can outlive the job.
  As `nobody`, they can connect only to Unix sockets that every user may write to.
- CUDA workers can open every NVIDIA device node. `CUDA_VISIBLE_DEVICES` selects the allocated GPU but does not confine the worker to it.

If the controller runs as root, workers run as `nobody` under every backend. That user must be able to read the platform source and the worker Python.
`get_resources` reports the configured backend in `execution_config`.

## Errors and recovery

The HTTP `error` object contains `code`, `hint`, `phase`, `retryable` and `details`.
The SDK raises `GymError`. MCP preserves the error and sets `isError=true`.

| Code | Meaning |
| --- | --- |
| `invalid_request` | Operation parameters fail the API schema or request rules. |
| `invalid_artifact` | The package, model field names or field types are invalid. |
| `incompatible_contract` | The proposal method or routing seed conflicts with the contract. |
| `conflict` | An idempotency key has different input, or another state conflict occurs. |
| `incomplete_coverage` | The static audit fails. This also covers native configuration or profile failures in that path. |

Keep the trial's host state, source revision, profiles and worker Python environment.
For controller commands, use the frozen `<trial>.host/execution.json`, which contains normalized GPU UUIDs.
Keep the original server ID. For `serve`, also use the original bind address through `--host`.
The listener file preserves the port only.

After a crash, establish that the worker exited. Then run the trusted controller command.
Replace the placeholders with your paths and IDs:

```sh
gym/.venv/bin/python -m nexttransformer_gym.cli \
  --state <retained-state> --profiles <profiles.json> --platform . \
  --execution-config <trial>.host/execution.json --worker-python <worker-python> \
  reconcile --session-id <owner-id> --job-id <job-id> --worker-confirmed-stopped
```

The controller refuses a worker slot that another scheduler owns. It charges the remaining conservative reservation.
Repeating the command does not charge twice. The flag asserts worker termination. It does not stop a process.
Use `gym/.venv/bin/python -m nexttransformer_gym.cli --help` to read the controller commands.
They include `prepare-final`, `launch-final`, `work-once`, `get-final`, `snapshot` and `serve`.
Only trusted controller commands can start separately allocated final training.

## Check the installation

Run these checks from the repository root:

```sh
PYTHONPATH=gym/src:. NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 \
  uv run --project gym --extra worker python -m pytest -q gym/tests
PYTEST_XDIST_AUTO_NUM_WORKERS=2 uv run --extra cpu python -m pytest -q -n auto
gym/.venv/bin/python -m nexttransformer.checks.changes --base origin/main
uvx --from ruff==0.14.10 ruff check --select F gym
uvx --from ruff==0.14.10 ruff format --check gym
uv run --project gym python gym/smoke.py
```

The smoke uses native Claude Code and a local model stub. It makes no external model calls.
Harbor installs the pinned CLI unless the client image contains it.
If the installer is unreachable, use `--client-image <image>` with that CLI version and the Gym client files.

For a short GPU check, select an allocated card:

```sh
PYTHONPATH=gym/src:. gym/.venv/bin/python gym/gpu_smoke.py \
  --state-dir .agent-work/gpu-smoke \
  --gpu <allocated-index-or-uuid> --worker-python <cuda-python>
```

These checks establish integration behavior. Production data and final benchmark acceptance need separate checks.
See the [design limits](DESIGN.md).
