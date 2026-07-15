"""Generate the three committed Grafana dashboards (SPEC C9).

Regenerate after metric changes: python scripts/gen_dashboards.py
Outputs land in infra/grafana/dashboards/*.json (committed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).resolve().parent.parent / "infra" / "grafana" / "dashboards"
DS = {"type": "prometheus", "uid": "prometheus"}


def target(expr: str, legend: str = "") -> dict[str, Any]:
    return {"datasource": DS, "expr": expr, "legendFormat": legend or "__auto", "refId": "A"}


def panel(title: str, targets: list[dict], x: int, y: int, w: int = 12, h: int = 8,
          kind: str = "timeseries", unit: str = "short",
          overrides: dict | None = None) -> dict[str, Any]:
    for i, t in enumerate(targets):
        t["refId"] = chr(ord("A") + i)
    p: dict[str, Any] = {
        "title": title, "type": kind, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": targets,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {},
    }
    if overrides:
        p.update(overrides)
    return p


def dashboard(uid: str, title: str, panels: list[dict]) -> dict[str, Any]:
    return {
        "uid": uid, "title": title, "tags": ["relay"], "timezone": "browser",
        "schemaVersion": 39, "version": 1, "refresh": "5s",
        "time": {"from": "now-30m", "to": "now"},
        "panels": panels,
    }


def latency_quantiles(metric: str, by_cache: bool = False) -> list[dict]:
    group = "(le, cache)" if by_cache else "(le)"
    legend_suffix = " {{cache}}" if by_cache else ""
    return [
        target(
            f"histogram_quantile({q}, sum(rate({metric}_bucket[1m])) by {group})",
            f"p{int(q * 100)}{legend_suffix}",
        )
        for q in (0.5, 0.95, 0.99)
    ]


operations = dashboard("relay-ops", "Relay / Operations", [
    panel("Requests per second", [
        target('sum(rate(relay_requests_total[1m]))', "RPS"),
        target('sum(rate(relay_requests_total{cache="hit"}[1m]))', "served from cache"),
    ], 0, 0, unit="reqps"),
    panel("Error rate (non-2xx)", [
        target('sum(rate(relay_requests_total{status!~"2.."}[1m])) '
               '/ sum(rate(relay_requests_total[1m]))', "error rate"),
    ], 12, 0, unit="percentunit"),
    panel("Latency p50/p95/p99 — hit vs miss",
          latency_quantiles("relay_latency_seconds", by_cache=True), 0, 8, unit="s"),
    panel("Circuit breaker state (0 closed / 1 half-open / 2 open)", [
        target("relay_breaker_state", "{{provider}}/{{model}}"),
    ], 12, 8, overrides={"options": {"legend": {"displayMode": "table"}}}),
    panel("Fallbacks per second", [
        target("sum(rate(relay_fallbacks_total[1m])) by (from, to)", "{{from}} -> {{to}}"),
    ], 0, 16),
    panel("Rate-limit rejections (429s)", [
        target("sum(rate(relay_ratelimit_rejections_total[1m])) by (team, kind)",
               "{{team}} {{kind}}"),
    ], 12, 16),
])

business = dashboard("relay-business", "Relay / Business", [
    panel("Cumulative $ saved — cache vs routing (ADR-0008, never blended)", [
        target('sum(relay_saved_usd_total) by (source)', "{{source}}"),
    ], 0, 0, unit="currencyUSD"),
    panel("Actual vs counterfactual spend", [
        target('sum(relay_cost_usd_total) by (attribution)', "{{attribution}}"),
    ], 12, 0, unit="currencyUSD"),
    panel("Spend by team (actual)", [
        target('sum(relay_cost_usd_total{attribution="actual"}) by (team)', "{{team}}"),
    ], 0, 8, unit="currencyUSD"),
    panel("Cache hit rate (5m)", [
        target('sum(rate(relay_requests_total{cache="hit"}[5m])) '
               '/ sum(rate(relay_requests_total[5m]))', "hit rate"),
    ], 12, 8, unit="percentunit"),
    panel("Cache hits by kind", [
        target("sum(rate(relay_cache_hits_total[5m])) by (kind)", "{{kind}}"),
    ], 0, 16),
    panel("Verification disagreements by tier", [
        target("sum(increase(relay_verification_disagreements_total[1h])) by (tier)",
               "tier {{tier}}"),
    ], 12, 16),
])

performance = dashboard("relay-perf", "Relay / Performance", [
    panel("Gateway overhead p50/p95/p99",
          latency_quantiles("relay_overhead_seconds"), 0, 0, unit="s"),
    panel("Overhead distribution (rate by bucket)", [
        target("sum(rate(relay_overhead_seconds_bucket[5m])) by (le)", "{{le}}s"),
    ], 12, 0, kind="heatmap"),
    panel("Cache similarity distribution", [
        target("sum(rate(relay_cache_similarity_bucket[5m])) by (le)", "<= {{le}}"),
    ], 0, 8, kind="heatmap"),
    panel("Near-miss band (0.75 <= sim < tuned threshold)", [
        target('sum(increase(relay_cache_similarity_bucket{le="0.94"}[1h])) '
               '- sum(increase(relay_cache_similarity_bucket{le="0.75"}[1h]))',
               "near misses / h"),
    ], 12, 8),
    panel("Requests by tier", [
        target("sum(rate(relay_requests_total[5m])) by (tier)", "tier {{tier}}"),
    ], 0, 16),
    panel("TPM/RPM rejections by kind", [
        target("sum(rate(relay_ratelimit_rejections_total[5m])) by (kind)", "{{kind}}"),
    ], 12, 16),
])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, dash in (("operations", operations), ("business", business),
                       ("performance", performance)):
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(dash, indent=2))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
