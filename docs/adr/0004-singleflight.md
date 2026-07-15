# ADR-0004: Singleflight stampede protection

**Status:** accepted

## Context
A burst of identical misses (cache cold, popular prompt) would fan out into
N identical upstream calls — paying N× for one answer.

## Decision
On exact-key miss, `SET NX lock:{ns}:{key} <token> PX 30000`. The winner goes
upstream and publishes via its cache write; losers poll the exact entry until
it appears, the lock vanishes, or a timeout elapses — then fall through to
their own upstream call (availability beats deduplication). Release is a
compare-and-delete on the token so only the owner can unlock.

Per-exact-key, not per-semantic-neighborhood: the exact key is a cheap,
unambiguous identity. Locking a similarity region would require a vector
search inside the lock path and can false-share across genuinely different
prompts.

## Consequences
- 10 concurrent identical requests → exactly 1 upstream call (integration-tested).
- A crashed owner delays waiters by at most the lock TTL; waiters then serve
  themselves.
- Non-identical paraphrases still stampede — accepted (rare, and the
  semantic hit closes the window after the first write).
