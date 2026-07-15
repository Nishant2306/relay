# ADR-0003: Exact-hash fast path before vector search

**Status:** accepted

## Context
True repeats are common under production traffic (retries, polling UIs,
copy-pasted prompts). Embedding + KNN costs ~5–15 ms; a GET costs ~0.2 ms.

## Decision
`exact_key = sha256(namespace | normalized_prompt)` checked first; embed +
KNN run only on exact miss. Normalization is deliberately conservative —
trim, collapse whitespace, casefold — never touching punctuation, numbers,
or word order (those can be answer-determining; the trap corpus proves it).

## Consequences
- O(1) hits for true repeats (measured ~4 ms total gateway overhead).
- Exact hits bypass threshold/guard logic entirely — safe because the
  normalized text is identical.
- Singleflight (ADR-0004) keys off the exact key.
