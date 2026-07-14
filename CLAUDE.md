# Relay — Working Agreement

- SPEC.md drives the build. datasets/TIERING_RUBRIC.md is the labeling/split/eval contract and
  OUTRANKS SPEC.md on tier semantics, splitting, and the length-control gate.
- datasets/* is FROZEN (relay-datasets-v1.1.0). Never edit those files. Changes require the human
  to run a version bump.
- NEVER tune cache thresholds against the test split of cache_traps.jsonl. Dev split (40 pairs) for
  tuning; test split (80 pairs) for reporting only. Same rule for the complexity dataset: the
  load-test corpus draws from the TRAIN split only.
- Splitting is group-aware by split_group. A pytest must assert zero group leakage across splits.
  The embedded `split`/`split_fold` fields in complexity_labels.jsonl are the normative
  train/dev/test assignment (frozen with the dataset). The length baseline recomputes its own
  7-fold protocol per the rubric — those are two different things; do not conflate them.
- The length-only baseline (~45.3% logreg / ~46.0% tree) is a release gate: CI fails if it hits 50%.
- Build the mock provider FIRST. Everything is tested against it; real APIs are for demos only.
- One milestone at a time: read the SPEC section, propose a short plan, wait for approval.
- Test-first for: the Lua token bucket, cache key derivation + namespacing, singleflight, breaker FSM,
  fallback selection, feature extraction, savings attribution math.
- Integration tests use testcontainers (real Redis + Postgres). Do not mock Redis for bucket/cache
  tests. Integration tests are marked `@pytest.mark.integration`; plain `pytest -m "not integration"`
  must pass with no Docker available.
- Commands: make up, seed, train, test, loadtest, drill, harvest.
- Style: Python 3.11, full type hints, Pydantic v2, ruff + mypy pass required.
- All provider calls go through gateway/adapters (cost tracking + MAX_DAILY_SPEND cap).
- Nontrivial choices → docs/adr/NNNN-*.md. No LangChain, Celery, or Kafka. Do not rewrite in Go.
- Small commits per subtask. Secrets via .env only (.env.example provided).
