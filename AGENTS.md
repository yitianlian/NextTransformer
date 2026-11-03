# Instructions for coding agents

## Fork development

Development in this checkout targets the public `yitianlian/NextTransformer` fork.
Use `yitianlian/<topic>` branches; keep `main` as the upstream baseline.
`origin` is the public fork and the default push destination; `upstream` is
`Interactive-Active-AI/NextTransformer`. Do not use private repositories or their
branches as development targets. Preserve existing results and cost proofs.

NextTransformer is built to be worked on by AI agents as well as people. If you are an agent working in this
repository, follow these rules in addition to the user's request.

## The contract

All methods train under one fixed protocol: the same data, data order, tokenizer, 2,048-token context,
49.152B input tokens, 1,048,576 tokens per update, 46,875 updates, seed 42, Muon + AdamW schedule, 189 validation
events and six full CORE-22 events. Training FLOPs per input token must stay within 1% of the SSSL anchor at
admission and within 2% cumulatively. Never change any of these to help one method. The full contract is in the
[protocol documentation](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/).

## Adding a method

- Add it as a plugin under `nexttransformer/methods/<method>/` ([plugin guide](https://interactive-active-ai.github.io/NextTransformer/docs/guides/plugins/)).
  Do not edit shared model files, the trainer, evaluation, data pipeline, cost ledger, `nexttransformer/models/backbone.py`
  or `nanochat/`.
- Calibrate it with `python -m nexttransformer.protocol.calibrate --methods <method> ...` and keep
  `python -m nexttransformer.protocol.fingerprint` passing.
- Select candidates with training loss or the dev split (`shard_06541.parquet`). The validation shard and CORE-22
  are reserved for the formal run.

## Running things

- `python -m nexttransformer.train` runs in research mode by default; leaderboard runs need `--formal` and an
  admitted configuration. Research runs are never published.
- Bounded checks only, unless the user explicitly asks for a full training run. Check GPU occupancy first and do not
  disturb other jobs, data, checkpoints or caches.
- Before a pull request: `python -m nexttransformer.checks.changes --base origin/main` and `python -m pytest -q -n auto`.

## Arena entries

If you were given the [arena task](https://interactive-active-ai.github.io/NextTransformer/docs/arena/task/), the
[arena rules](https://interactive-active-ai.github.io/NextTransformer/docs/arena/rules/) also bind you. Record every
candidate, formal launch and human intervention truthfully in the disclosure card.

## Hygiene

- Code, comments and documentation in English.
- Disposable scripts and logs go in `.agent-work/` (ignored). Never commit credentials, datasets, checkpoints, caches,
  machine names or absolute local paths.
- Do not rewrite published results, audits or cost proofs.
