# ADR-0006: Only cache complete, successful, cacheable responses

**Status:** accepted

## Context
A cache that stores truncated, errored, or intrinsically-variable responses
replays those failures to every future matching request.

## Decision
Write to the cache only when ALL hold:
- `finish_reason == "stop"` (complete — not `length`, not an error),
- the request succeeded end-to-end,
- the prompt's cache class allows it: `stable` → 24 h TTL, `temporal` → 1 h,
  `no_cache` (creative/subjective/high-temperature) → never written, never
  served. The runtime classifier for cache class was validated against the
  600 labeled prompts (96% agreement) rather than invented blind.
- Streaming: chunks tee into a buffer; client disconnect or mid-stream
  provider failure discards the buffer entirely.

## Consequences
- Partial answers can never be replayed.
- Temporal prompts go stale in bounded time; creative prompts always hit the
  model (cache hit rate is deliberately sacrificed there).
- The cache-class heuristic mislabels ~4% of prompts (measured); the failure
  mode is a lost caching opportunity or a 24h-cached judgment call — bounded
  by TTL.
