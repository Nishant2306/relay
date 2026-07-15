# ADR-0005: TPM pre-charge + reconcile

**Status:** accepted

## Context
Token-per-minute limits must be enforced *before* the call, but output length
is unknown until after. Charging only prompt tokens lets bursts overshoot the
limit by the entire output volume.

## Decision
Pre-charge `prompt_tokens + max_tokens` against the TPM bucket, then refund
the unused remainder once actual usage is known. If TPM rejects, the RPM
token already taken is refunded (a rejected request costs nothing).

## Consequences
- Bursts cannot overshoot; the worst case is temporary under-utilization
  between charge and refund.
- Clients that set no `max_tokens` are charged a 512-token default estimate;
  usage above the pre-charge is absorbed (bounded by estimate accuracy) and
  documented rather than double-charged.
- Both operations are single atomic Lua round-trips (no read-modify-write races).
