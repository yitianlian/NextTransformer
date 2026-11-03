# Architecture design interface for the research gym

Status: exploration and proposed contract, not an implemented service.
Source baseline: public fork `yitianlian/NextTransformer`, commit `41b6016`.
This proposal does not change Arena v1 or the frozen training protocol.

## Outcome

An agent should be able to inspect the design space, write an architecture plugin,
submit an immutable candidate, understand validation failures, run a budgeted dev
experiment, and use the returned evidence to design the next candidate. The gym
owns execution and measurements; the agent owns the architecture and hypothesis.
Python, CLI and MCP should expose the same application contract. Start with Python
and JSON schemas; add transport wrappers after the local loop works.

## What the public code provides

| Capability | Evidence | Implication |
| --- | --- | --- |
| Declarative discovery without importing candidate code | `nexttransformer/models/plugins.py:discover` | Keep `method.json` as the source of plugin metadata. |
| External plugin roots | `plugins.py:roots`, `NEXTTRANSFORMER_METHOD_PATHS` | Store attempts outside the source checkout. |
| Mixer, FFN, block and model hooks | `plugins.py:load`, `tests/plugins/` | Generate templates from these existing contracts. |
| Model hook restricted to `ReferenceLM` subclasses | `plugins.py:load` and `check_shell` | Whole-model hooks do not currently mean arbitrary `nn.Module` architectures. |
| Fixed formal shell and calibration grid | `models/config.py:ModelConfig.__post_init__`, `protocol/calibrate.py` | Expose supported dimensions and bounds instead of promising unlimited architecture changes. |
| Staged computation | `plugins.py:stages`, `tests/plugins/toy_two_pass/` | Declare stage schedules before auditing; charge each stage. |
| Cost audit and provenance | `protocol/calibrate.py`, `protocol/fingerprint.py` | Reuse the ledger; retain a separate hash of external candidate source. |
| Research mode and smoke runs | `nexttransformer/train.py` | These labels prevent formal export, but do not isolate held-out evaluation. |
| Train and validation data streams | `runtime/data.py:BestFitStream` | There is no `dev` split argument today. |

No committed gym implementation exists at this baseline. In particular, a README
describing a future service must not be interpreted as working admission, compute
metering, job execution or MCP support.

### Two boundaries to expose explicitly

**Architecture freedom.** Formal configurations fix width 1536, 12 heads, vocabulary
32768 and sequence length 2048. The model hook inherits `ReferenceLM`; shared
initialization expects named attention and FFN linear projections. The initial
interface should advertise a `reference_plugin_v1` capability with these limits.
An arbitrary-model contract would require a separately reviewed design for parameter
roles, initialization, checkpoint state, calibration and formal comparability.
Do not silently treat that larger design space as already supported.

**Selection data.** `train.py:scheduled_evaluations` calls validation in smoke mode
at step zero and the stopping step. `validation` defaults to `split="val"`.
`BestFitStream` accepts only `train` and `val`. Consequently, invoking the existing
training CLI with `--research` or `--smoke-steps` is not a dev-only evaluation backend.
Filtering its output afterward is insufficient: candidate execution must not have
access to held-out data in the first place.

## Candidate artifact

```text
candidate/
  proposal.json             hypothesis, parent IDs, entrypoint, contract version
  plugin/<method>/
    method.json             existing plugin declaration
    __init__.py             existing hooks; optional sibling Python modules
  tests/                    candidate-specific checks
```

Submission freezes file contents and records their content hash, base commit,
environment identity and researcher identity. A change creates a new candidate,
not a mutation of an evaluated candidate. The service resolves artifact IDs rather
than accepting arbitrary host paths or shell commands from remote clients.
The calibrated configuration is a derived immutable artifact. Agent-supplied
scores, proof hashes and cost estimates are never authoritative.

## Proposed agent operations

All operations return versioned JSON. Long work returns a job ID immediately.

| Operation | Input | Output |
| --- | --- | --- |
| `describe_space` | contract version | fixed controls, available hooks, option types, bounds, templates, evaluation policy |
| `observe` | filters, cursor | remaining compute, comparable dev results, candidate lineage, failure summaries |
| `get_candidate` | candidate ID | source artifact, hypothesis, parents, reports and provenance |
| `submit_candidate` | artifact ID, hypothesis, parent IDs, idempotency key | immutable candidate ID and source hash |
| `validate_candidate` | candidate ID | job ID; eventual structured static, correctness and audit reports |
| `calibrate_candidate` | candidate ID, supported search grid | job ID; derived config ID, cost proof, deviation from anchor |
| `quote_experiment` | candidate ID, config ID, research profile | expiring quote, maximum cost, required resources, comparable cohort |
| `run_experiment` | quote ID, idempotency key | job ID and reservation ID |
| `get_job` / `read_events` | job ID / cursor | state, progress, bounded diagnostics and next cursor |
| `cancel_job` | job ID, idempotency key | cancellation status; compute charged through actual termination |
| `get_result` | job ID | dev BPB, training curve, cost, completion status, artifact IDs |

Formal admission remains the separate Arena workflow. The initial autonomous loop
selects on dev BPB or training loss. Formal validation and CORE are not feedback
signals for this loop. Published baseline results can be read as public context,
but cannot stand in for measurements at a different research horizon.

Example failure payload:

```json
{
  "schema_version": "gym.v1",
  "error": {
    "code": "unsupported_operator",
    "phase": "cost_audit",
    "retryable": false,
    "details": {"operator": "example::custom_op"},
    "hint": "Provide an auditable reference path and an equivalence check."
  }
}
```

Return machine-readable error codes, offending paths/operators, observed and
expected values, and a useful repair hint. Distinguish invalid candidates, failed
infrastructure, exhausted budget and cancelled jobs. Never rank incomplete jobs as
successful evaluations.

## Execution and scientific boundaries

- Perform static metadata validation before importing a plugin. Run dynamic checks
  in a fresh isolated worker per candidate: discovery and module loading are cached
  in the existing plugin code. A subprocess alone is not a sandbox.
- Workers receive only the required data, artifacts and resource limits. Agent code
  cannot read credentials, held-out data, the service database or other candidates'
  writable state. Static AST checks are diagnostics, not isolation.
- Verify causal outputs, finite losses and gradients, parameter coverage, checkpoint
  round trips, stage boundaries, and accelerated/reference equivalence where relevant.
- Reuse audited operator accounting. Unknown operators fail admission. Snapshot the
  external plugin itself because the public fingerprint machinery only includes
  plugin roots inside the repository.
- A service-side ledger atomically reserves cost before dispatch, charges executed
  work including failures and cancellation, and settles after the worker terminates.
  Keep theory FLOPs and measured GPU-hours distinct; record unknown values as null.
- Replays with the same idempotency key must not launch or reserve twice. Recovery
  must reconcile running workers and reservations before releasing resources.
- Compare only results sharing source/config identity where required, data split,
  token horizon, optimizer schedule, evaluation policy and accounting version.
  Small research profiles are explicit configurations, not new formal results.
- A dev evaluator must read `shard_06541.parquet` explicitly, retain tokenizer byte
  accounting and declared packing, and never rename held-out files or patch shared
  evaluation functions. A separate gym research driver needs parity tests for the
  training primitives it reuses. Formal execution stays on the frozen trainer.
- Start with stdio/CLI; no host listener is needed. If HTTP is added, bind port 0 on
  first use, persist the actual service port, reuse it on restart, and fail clearly
  if occupied. Keep the socket bound through startup.

## Implementation order and acceptance gates

1. **Design space and submissions.** Typed request/response schemas, hook templates,
   immutable artifact ingestion, fresh-process plugin loading and structured errors.
   Gate: one valid sample for each hook and useful failures for malformed submissions.
2. **Validation and calibration.** Adapt existing model checks and the ledger without
   editing shared scientific code. Gate: built-in/test-plugin controls reproduce
   expected behavior; source mutation invalidates candidate evidence.
3. **Dev-only bounded evaluation.** Implement the explicit dev data adapter and a
   research driver. Gate: tests prove held-out files are inaccessible, BPB uses the
   correct token-byte denominator, and checkpoint resume reproduces the data stream.
4. **Budgeted jobs and feedback.** Persistent state, atomic reservation, events,
   cancellation, recovery and result cohorts. Gate: retries do not duplicate jobs;
   failed/cancelled work remains charged; agent completes two candidate iterations.
5. **MCP wrapper and remote backends.** Thin adapters over the tested service contract.
   Gate: identical semantics through Python/CLI/MCP, with no transport-specific policy.

Do not choose screening horizons solely from final-validation curves and present
that as independent dev evidence. Establish horizon usefulness with dev measurements;
short horizons remain configurable research policy until supported by such evidence.

## Scope of this exploration

Only documentation and fork workflow guidance are changed. No runtime service,
training run, benchmark result or architectural admission is claimed. Private code
and the earlier staged gym draft are not migrated into the public fork. The next
implementation target is the design-space/submission contract, followed by the
dev-only evaluation path before any autonomous training loop.
