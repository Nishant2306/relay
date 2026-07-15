# ADR-0008: Honest savings accounting

**Status:** accepted

## Context
"We saved X%" is meaningless without a defined counterfactual, and inflated
when cache savings and routing savings are blended into one number.

## Decision
Per request, store both `actual_cost_usd` (what was paid, $0 for cache hits)
and `counterfactual_cost_usd` (the same tokens priced at the flagship model,
gpt-4o rates). Savings = 1 − Σactual/Σcounterfactual, attributed separately:
- **cache**: counterfactual value of requests served from cache,
- **routing**: counterfactual − actual on requests served by a model.

Exposed as `relay_saved_usd_total{source=cache|routing}`, split in the
Business dashboard, `/admin/stats`, and `harvest_metrics.py`. Never blended.

## Consequences
- Every reported number is reproducible from the request_log.
- The counterfactual ("everything at flagship price") is stated with the
  number; simulated-workload caveats live in the README's Known Gaps.
- Mock models carry realistic simulated prices so $0 load tests still
  exercise the full accounting path.
