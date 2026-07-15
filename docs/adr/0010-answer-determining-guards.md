# ADR-0010: Answer-determining guards on semantic candidates

**Status:** accepted (added during C5 — the corpus forced it)

## Context
The plan was "cosine similarity ≥ per-tier threshold ⇒ hit." The frozen trap
corpus falsified it: hit and miss pairs have nearly identical similarity
distributions (held-out medians 0.911 vs 0.913; "Convert 20 Celsius to
Fahrenheit" vs its reverse embeds at 0.995). No threshold exists that passes
both the ≥80% hit-rate and the low false-hit acceptance — bge-small (and
realistically any bi-encoder) encodes topic, not answer identity.

## Decision
A KNN candidate above the threshold must also survive deterministic guards
comparing query vs stored prompt on answer-determining content:
1. **numbers** — normalized numeric multisets ("1,000"="1000", "5 PM"="17",
   "ten"="10") must match;
2. **negation** — negation presence must match;
3. **inverse pairs** — min/max, encrypt/decrypt, sum/average, before/after…;
4. **direction** — "X to Y" vs "Y to X" argument swaps;
5. **content substitution** — structurally identical prompts differing by a
   small same-length word swap (Paris→London, GET→POST) are rejected unless
   the swap is morphological or numeric-equivalent.

Ambiguity resolves toward rejection: a false reject costs one upstream call;
a false accept serves a wrong answer. Guards were tuned on the 40 dev pairs
only (ADR-0009); with them, the similarity threshold dropped to 0.755 and
held-out results reached 95% hit / 2.5% false-hit.

## Consequences
- The similarity score is a candidate generator, not the decision-maker.
- Residual risk is multi-token entity swaps ("AWS"→"Google Cloud") — the one
  held-out false hit; documented in Known Gaps.
- Guards are English-centric regex/lexicon logic; other languages would need
  their own tables (v2).
