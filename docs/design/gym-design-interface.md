# Architecture design interface for the research gym

Status: agreed v1 design boundaries and proposed API; not an implemented service.
Source baseline: public fork `yitianlian/NextTransformer`, commit `41b6016`.
This proposal does not change Arena v1 or the frozen training protocol.

## Outcome

An agent should be able to inspect the design space, write an architecture plugin,
submit an immutable candidate, understand validation failures, run a budgeted dev
experiment, and use the returned evidence to design the next candidate. The gym
owns execution and measurements; the agent owns the architecture and hypothesis.
Python, CLI and MCP should expose the same application contract. Start with Python
and JSON schemas; add transport wrappers after the local loop works.

## Agreed v1 boundaries

- The product is a **gym**, separate from the RSI system. It evaluates architecture
  research under a common execution contract.
- Research is offline. Every session receives the same source baseline, components,
  documentation, dependencies and baseline records. New session artifacts are private
  to that session. Agent inference uses a platform-controlled broker; the research
  workspace and candidate workers have no arbitrary network access or credentials.
- The final output is **one architecture artifact**: plugin source, declaration and
  architecture configuration. It contains no pretrained weights, executable training
  entrypoint, environment installer, evaluator or claimed score.
- The gym automatically builds, checks, initializes, trains, evaluates, audits and
  exports the artifact with its own versioned runner. Neither research nor final
  execution requires a participant-written launch script. Admission and operational
  failures are structured outcomes, not occasions to silently patch the candidate.
- Stage one permits composition and local invention through the existing plugin
  contract. The shared shell, optimizer recipe, data pipeline and evaluation remain
  controlled. Arbitrary whole-model architectures are a stage-two TODO.
- Research candidates may have different FLOPs/token. This is separate from the
  fixed cost requirement for final official admission. Return cost and training
  horizon alongside every score; unequal-cost observations are not matched comparisons.
- RSI controls candidate proposals, bounded experiment lengths, early stopping,
  continuation and selection. Concurrent jobs are allowed under a configured limit
  and one atomic session budget. Scheduling must not alter the declared LR schedule.
- Continuing training requires the same immutable architecture and configuration,
  compatible environment and complete checkpoint state. Changed architecture or
  configuration starts from scratch. Consumed budget never resets on continuation.
- All candidate execution, including debug forward/backward passes, tests and
  profiling, goes through the gym. The workspace has no direct experiment device
  access or alternative execution route to experiment resources.
- No unaccountable candidate computation is admitted. Unknown operations yield
  static diagnostics and block dynamic execution until a supported, auditable path
  exists. A reference implementation alone does not prove the actual kernel cost.
- Research feedback includes dev BPB, training curves, FLOPs/token, cumulative cost,
  throughput, memory, check reports and bounded logs. Formal validation and CORE are
  not returned as feedback for an active research session.
- The researcher freezes one final candidate. The platform performs a fresh final
  run under a common, separately allocated scoring budget. Final results cannot be
  used to reopen that research session. Presentation follows the official leaderboard,
  including BPB, CORE and provenance; no new composite score is required.

## Research budget contract

The v1 hard research limit is `research_flops_limit`, an explicitly configured finite
positive value per session. No amount has been selected yet. Missing configuration
blocks experiment execution; it never means unlimited compute. Agent inference
cost accounting is deferred. GPU-hours and elapsed time are operational diagnostics,
not substitutes for the FLOPs comparison. Resource/time limits still bound workers.

Every model computation in debugging, correctness checks, training, evaluation and
profiling consumes this account, including failed and cancelled attempts. Static
metadata/source checks and symbolic cost estimation are tracked separately from
executed model FLOPs. The accounting convention and coverage are versioned and fixed
for a comparison; no work is silently assigned zero cost because it is unsupported.

Before execution, return a cost report with:

- training FLOPs per input token, its forward/backward/optimizer/auxiliary components,
  the configured accounting version and coverage status;
- stage-specific costs and the averaging horizon for staged or dynamic architectures;
- ratio to the official SSSL anchor, final-admission eligibility and reasons;
- quoted total experiment cost, including checks and evaluation, and the fraction
  of the remaining research budget it would reserve.

Keep `spent`, `reserved` and `available = limit - spent - reserved` distinct. Reserve
atomically before dispatch, including a conservative allowance for bounded work in
flight and termination. Reject a launch or next work unit that cannot fit. Release
unused reservations only after termination and reconciliation. Unknown consumption
after a crash remains reserved pending reconciliation; it is never automatically
refunded. Retries are idempotent, and infrastructure failures are marked separately
without erasing consumed compute. Resume and finalization survive controller restarts.

Formal scoring has a separately configured common allocation and accounting record.
The official profile uses the admitted cost, 49.152B tokens and full evaluation
schedule. Smaller profiles must be explicitly named and cannot enter the official
leaderboard as equivalent runs. Final submission does not bypass resource readiness,
admission or budget checks.

## Isolation and trusted measurements

The control service and ledger run outside candidate workers. Each worker receives
an immutable candidate and read-only runner, only the data needed for its operation,
bounded resources and a private output area. It has no host socket, host credentials,
other session storage or writable score/budget database. Artifact access is authorized
by session ownership; IDs and paths alone confer no access.

Candidates remain untrusted Python even after static checks. Do not import them into
the privileged control service. Model execution and trusted scoring need separate
privilege boundaries: the evaluation prediction interface takes inputs, not target
labels; the trusted scorer derives BPB from predictions and labels. Candidate-reported
losses or score files are not authoritative. Bound and validate prediction outputs.
The official protocol remains the compatibility target; a safe execution adapter
requires equivalence tests rather than claiming the existing in-process trainer is
already an isolation boundary.

Cost checks must cover the frozen code and the actual executed paths, including
training/evaluation differences and any accelerated path. Hash equality alone does
not establish identical behavior. Before accepting general plugin execution, tests
must demonstrate refusal of unsupported work, score forgery, filesystem/network
escape, cross-session access and budget oversubscription. A container alone is not
proof of these properties. The specific sandbox/runtime is an implementation choice
still to validate, not a completed capability.

Existing validation and CORE data are public. Offline execution limits access in
this session but cannot remove knowledge already present in a researcher model.
Describe this as controlled access to a public benchmark, not a secret test set.
An additional private evaluation set remains a separate, undecided future protocol.

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
| `validate_candidate` | candidate ID | static coverage report; budgeted dynamic-check job only after coverage passes |
| `calibrate_candidate` | candidate ID, supported search grid | job ID; derived config ID, cost proof, deviation from anchor |
| `inspect_cost` | candidate ID, config ID, profile | component/stage FLOPs/token, anchor ratio, accounting coverage and admission status |
| `debug_candidate` | candidate ID, config ID, named check and bounded inputs | budgeted isolated job; diagnostic output and charged compute |
| `quote_experiment` | candidate ID, config ID, research profile | expiring quote, maximum cost, required resources, comparable cohort |
| `run_experiment` | quote ID, target update, optional checkpoint ID, idempotency key | job ID and reservation ID; no arbitrary command or training script |
| `get_job` / `read_events` | job ID / cursor | state, progress, bounded diagnostics and next cursor |
| `cancel_job` | job ID, idempotency key | cancellation status; compute charged through actual termination |
| `get_result` | job ID | dev BPB, training curve, cost, completion status, artifact IDs |
| `get_budget` | session ID | limit, spent, reserved, available, accounting version |
| `finalize` | candidate ID, frozen config ID, idempotency key | immutable final submission ID and platform scoring status |

Candidate, experiment, checkpoint and research session have distinct IDs. A quote
binds the immutable candidate/config/profile, target update and continuation state;
`run_experiment` cannot override these after quotation. Checkpoints belong to one
candidate/config and capture optimizer, RNG, schedule and data state, not just weights.
Finalization locks candidate selection and prevents new research launches; it must
terminate or reconcile outstanding research jobs before final scoring. The scorer
never loads research checkpoints. Session state is durable across RSI process restarts.

Provide a blocking `evaluate(candidate, profile)` convenience adapter over the same
job API for frameworks expecting a scalar/result callback. It must preserve detailed
feedback, provenance and accounting. RSIHub can adapt through its evaluator script
and `EvaluationRecord`; FlashRSI can adapt at its snapshot evaluation boundary. Neither
framework's source snapshot should be confused with a training checkpoint.

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

### Deferred work

- Stage two: an arbitrary-model contract beyond `ReferenceLM`, with explicit parameter
  roles, initialization, state/checkpoint schema, calibration and compatibility rules.
- Agent inference cost metering and limits, including broker usage provenance.
- Cross-architecture weight transfer, distillation and parameter inheritance.
- Internet-enabled research as a separately specified setting; no v1 network toggle
  that silently changes a comparison cohort.
- Optional private held-out evaluation and additional tasks, under a separate protocol.

### Deployment settings still required

Choose concrete research FLOPs limits, final scoring allocation, available hardware,
concurrency and per-worker limits, supported dependencies/operations and research
profiles before enabling execution. These are not implicitly authorized by agreeing
to this design. No GPU runs or external services are launched by this document.

Only documentation and fork workflow guidance are changed. No runtime service,
training run, benchmark result or architectural admission is claimed. Private code
and the earlier staged gym draft are not migrated into the public fork. The next
implementation target is the design-space/submission contract, followed by the
dev-only evaluation path before any autonomous training loop.
