# ADR-0001: Local embeddings for the semantic cache

**Status:** accepted

## Context
Every request needs an embedding for cache lookup. A paid embedding API would
add per-request cost and 20–60 ms of latency to the exact component meant to
remove both, and would couple cache availability to a provider.

## Decision
fastembed with `BAAI/bge-small-en-v1.5` (384-d, ONNX, CPU): ~5–15 ms per
prompt, $0, fully deterministic for identical inputs. Cache lookups need
*consistency*, not state-of-the-art retrieval quality — and the trap corpus
showed that no embedding threshold alone separates paraphrases from traps
anyway (see ADR-0010); guards carry the correctness burden.

Implementation note: vectors live in Redis 8's query engine (FT.* / HNSW /
cosine) accessed directly through redis-py rather than the RedisVL wrapper —
same engine and index structure, one fewer dependency to track.

## Consequences
- Zero marginal cost/latency coupling; embeddings survive provider outages.
- bge-small's similarity ceiling is real: paraphrase/trap distributions
  overlap heavily (medians 0.911 vs 0.913). Mitigated by guards (ADR-0010).
- Model file (~130 MB) downloads on first use; compose persists it in a volume.
