# ADR-0002: Cache key namespacing

**Status:** accepted

## Context
A semantic cache that matches across system prompts, models, sampling
parameters, or tenants will eventually serve an answer generated under
different instructions to the wrong caller. That is a correctness failure,
not a tuning problem.

## Decision
`namespace = sha256(system_prompt) | model | temp_bucket(0.2) |
top_p_bucket(0.2) | team_scope`, hashed to 24 hex chars. Vector KNN filters
on the namespace TAG — similarity search physically cannot cross the
boundary. `team_scope` defaults to per-team; global sharing is opt-in per
team (`cache_scope: global`).

## Consequences
- Same prompt under a different system prompt / model / temperature bucket /
  team never shares an entry (integration-tested).
- Sharding by namespace lowers hit rates vs a global cache — accepted; it is
  the price of correctness.
- Parameter bucketing (0.2) trades a small correctness risk (0.61 vs 0.79
  share a bucket) for not fragmenting the cache per float value.
