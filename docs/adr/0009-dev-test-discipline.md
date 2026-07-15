# ADR-0009: Dev/test discipline on the trap corpus

**Status:** accepted

## Context
Tuning a threshold on the same pairs you report is overfitting to your own
benchmark — the reported false-hit rate becomes an advertisement, not a
measurement.

## Decision
The 120 trap pairs are frozen with an embedded split: 40 dev, 80 test.
- Thresholds AND guard rules are tuned exclusively on dev
  (`scripts/eval_cache_traps.py`, `tests/test_cache_unit.py` regression-gate
  dev behavior only).
- The README/resume numbers come exclusively from the held-out 80,
  evaluated once per matcher version.
- Same rule for the complexity dataset: the load-test corpus draws from the
  TRAIN split only; classifier accuracy is reported on its test fold.
- CI enforces dataset immutability (any change to datasets/* requires a
  version bump in TIERING_RUBRIC.md).

## Consequences
- The reported 2.5% false-hit rate is a genuine out-of-sample measurement —
  and non-zero, as an honest measurement of this problem should be.
- Improving the matcher means re-tuning on dev and accepting whatever the
  test split then says.
