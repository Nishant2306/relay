# ADR-0007: Fallback never downgrades quality

**Status:** accepted

## Context
When a provider fails, the tempting fallback is "any model that answers."
Silently serving a weaker model turns an availability incident into a silent
quality incident.

## Decision
Fallback chains (routing.yaml) list same-tier alternate providers first, then
tier+1 — never a lower tier. Non-retryable errors (400/401/403) surface
immediately without fallback: a request that is wrong by construction is
wrong everywhere.

Related tier semantics: fail-safe promotion (classifier confidence < 0.6)
moves *routing* up one tier, but the semantic-cache threshold keys off the
classifier's *base* tier — promotion expresses generation caution, not
increased cache-match risk. A promoted tier-1 paraphrase still matches at
the tier-1 threshold.

## Consequences
- Outages can only make responses more expensive, never worse.
- `x-relay-fallback: true` + request_log + `relay_fallbacks_total{from,to}`
  make every fallback observable and attributable.
- Chain exhaustion returns an honest 502 rather than a degraded answer.
