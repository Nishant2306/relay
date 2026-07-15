"""Prometheus metrics (SPEC C9 — exact metric names are contractual).

`RelayMetrics` is the single funnel the pipeline/resilience/verifier call
into; a fresh CollectorRegistry per instance keeps tests isolated.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

BREAKER_STATE_VALUE = {"closed": 0, "half_open": 1, "open": 2}

OVERHEAD_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
SIMILARITY_BUCKETS = (0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0)


class RelayMetrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests_total = Counter(
            "relay_requests_total", "Requests through the gateway",
            ["team", "provider", "model", "tier", "cache", "status"], registry=r,
        )
        self.latency_seconds = Histogram(
            "relay_latency_seconds", "End-to-end request latency",
            ["path", "cache"], buckets=LATENCY_BUCKETS, registry=r,
        )
        self.overhead_seconds = Histogram(
            "relay_overhead_seconds", "Gateway-added overhead (total minus provider)",
            buckets=OVERHEAD_BUCKETS, registry=r,
        )
        self.cost_usd_total = Counter(
            "relay_cost_usd_total", "Cost accounting (attribution=actual|counterfactual)",
            ["team", "attribution"], registry=r,
        )
        self.saved_usd_total = Counter(
            "relay_saved_usd_total",
            "Dollars saved vs flagship, attributed (source=cache|routing, ADR-0008)",
            ["team", "source"], registry=r,
        )
        self.cache_hits_total = Counter(
            "relay_cache_hits_total", "Cache hits by kind", ["kind"], registry=r,
        )
        self.cache_similarity = Histogram(
            "relay_cache_similarity", "Top-1 KNN similarity observed on lookups",
            buckets=SIMILARITY_BUCKETS, registry=r,
        )
        self.ratelimit_rejections_total = Counter(
            "relay_ratelimit_rejections_total", "429s by team and bucket kind",
            ["team", "kind"], registry=r,
        )
        self.fallbacks_total = Counter(
            "relay_fallbacks_total", "Fallbacks walked", ["from", "to"], registry=r,
        )
        self.breaker_state = Gauge(
            "relay_breaker_state", "0=closed 1=half_open 2=open",
            ["provider", "model"], registry=r,
        )
        self.verification_disagreements_total = Counter(
            "relay_verification_disagreements_total", "Judge disagreements", ["tier"],
            registry=r,
        )

    # -- pipeline-facing helpers ------------------------------------------------
    def request(self, team: str, model_key: str | None, tier: int | None,
                cache: str, status: int) -> None:
        provider, model = (model_key.split("/", 1) if model_key and "/" in model_key
                           else ("none", model_key or "none"))
        self.requests_total.labels(
            team=team, provider=provider, model=model,
            tier=str(tier) if tier else "none", cache=cache, status=str(status),
        ).inc()

    def latency(self, path: str, cache: str, seconds: float) -> None:
        self.latency_seconds.labels(path=path, cache=cache).observe(seconds)

    def overhead(self, seconds: float) -> None:
        self.overhead_seconds.observe(seconds)

    def cost(self, team: str, actual: float, counterfactual: float) -> None:
        if actual > 0:
            self.cost_usd_total.labels(team=team, attribution="actual").inc(actual)
        if counterfactual > 0:
            self.cost_usd_total.labels(team=team, attribution="counterfactual").inc(counterfactual)

    def saved(self, team: str, source: str, amount: float) -> None:
        if amount > 0:
            self.saved_usd_total.labels(team=team, source=source).inc(amount)

    def cache_hit(self, kind: str, similarity: float | None) -> None:
        self.cache_hits_total.labels(kind=kind).inc()
        if similarity is not None:
            self.cache_similarity.observe(similarity)

    def ratelimit_rejection(self, team: str, kind: str) -> None:
        self.ratelimit_rejections_total.labels(team=team, kind=kind).inc()

    def fallback(self, from_key: str, to_key: str) -> None:
        self.fallbacks_total.labels(**{"from": from_key, "to": to_key}).inc()

    def set_breaker(self, model_key: str, state: str) -> None:
        provider, model = model_key.split("/", 1)
        self.breaker_state.labels(provider=provider, model=model).set(
            BREAKER_STATE_VALUE.get(state, 0)
        )

    def verification_disagreement(self, tier: int) -> None:
        self.verification_disagreements_total.labels(tier=str(tier)).inc()

    def export(self) -> bytes:
        return generate_latest(self.registry)
