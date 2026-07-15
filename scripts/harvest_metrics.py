"""Harvest every README/resume number (SPEC C11, make harvest).

Pulls from: Prometheus (overhead/latency/hit-rate/fallbacks), Postgres
(request_log savings attribution, verification, breaker events), and the
results/ + models/ artifacts (trap eval, classifier meta, scenario stats).
Writes results/summary.md and prints it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROM = "http://localhost:9090"
RESULTS = Path("results")


def prom(query: str) -> float | None:
    try:
        r = httpx.get(f"{PROM}/api/v1/query", params={"query": query}, timeout=10).json()
        result = r["data"]["result"]
        return float(result[0]["value"][1]) if result else None
    except Exception:
        return None


def pg_stats() -> dict[str, Any]:
    try:
        from sqlalchemy import create_engine, text

        from gateway.config import settings

        engine = create_engine(settings.database_url_sync)
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT count(*) AS n,
                       coalesce(sum(actual_cost_usd), 0) AS actual,
                       coalesce(sum(counterfactual_cost_usd), 0) AS counterfactual,
                       coalesce(sum(CASE WHEN cache = 'hit' THEN 1 ELSE 0 END), 0) AS hits,
                       coalesce(sum(CASE WHEN cache = 'hit' AND cache_kind = 'exact'
                                         THEN 1 ELSE 0 END), 0) AS exact_hits,
                       coalesce(sum(CASE WHEN cache = 'hit' AND cache_kind = 'semantic'
                                         THEN 1 ELSE 0 END), 0) AS semantic_hits,
                       coalesce(sum(CASE WHEN fallback_used THEN 1 ELSE 0 END), 0) AS fallbacks
                FROM request_log WHERE status = 200
            """)).mappings().one()
            saved_cache = conn.execute(text(
                "SELECT coalesce(sum(counterfactual_cost_usd), 0) FROM request_log "
                "WHERE status = 200 AND cache = 'hit'"
            )).scalar()
            saved_routing = conn.execute(text(
                "SELECT coalesce(sum(counterfactual_cost_usd - actual_cost_usd), 0) "
                "FROM request_log WHERE status = 200 AND cache != 'hit'"
            )).scalar()
            verification = conn.execute(text("""
                SELECT tier, verified, count(*) FROM request_log
                WHERE verified IS NOT NULL GROUP BY tier, verified
            """)).all()
            breaker_events = conn.execute(text(
                "SELECT provider, model, from_state, to_state, reason, ts "
                "FROM provider_health_events ORDER BY ts"
            )).all()
        n = row["n"]
        actual, counterfactual = float(row["actual"]), float(row["counterfactual"])
        return {
            "requests_ok": n,
            "hit_rate": round(row["hits"] / n, 4) if n else None,
            "exact_hits": row["exact_hits"], "semantic_hits": row["semantic_hits"],
            "savings_pct": round(1 - actual / counterfactual, 4) if counterfactual else None,
            "saved_usd_cache": round(float(saved_cache), 4),
            "saved_usd_routing": round(float(saved_routing), 4),
            "fallbacks": row["fallbacks"],
            "verification": [
                {"tier": t, "verdict": v, "count": c} for t, v, c in verification
            ],
            "breaker_events": [
                {"model": f"{p}/{m}", "from": f, "to": to, "reason": r, "ts": str(ts)}
                for p, m, f, to, r, ts in breaker_events
            ],
        }
    except Exception as e:
        return {"error": f"postgres unavailable: {e}"}


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def fmt_pct(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "n/a"


def main() -> int:
    overhead = {
        q: prom(f'histogram_quantile({q}, sum(rate(relay_overhead_seconds_bucket[10m])) by (le))')
        for q in (0.5, 0.95, 0.99)
    }
    pg = pg_stats()
    trap_eval = load_json(Path("results/cache_trap_eval.json"))
    traps_live = load_json(Path("results/traps_under_load.json"))
    classifier = load_json(Path("models/classifier_meta.json"))
    steady = load_json(Path("results/steady_cache_stats.json"))
    storm = load_json(Path("results/storm_stats.json"))
    budget = load_json(Path("results/budget_stats.json"))
    outage = load_json(Path("results/outage_stats.json"))

    lines = ["# Relay — harvested numbers", ""]

    lines += ["## Gateway overhead (Prometheus, last 10m)"]
    for q, v in overhead.items():
        lines.append(f"- p{int(q * 100)}: "
                     f"{f'{v * 1000:.1f} ms' if v is not None else 'n/a (no traffic scraped)'}")

    lines += ["", "## Savings (request_log, ADR-0008 attribution)"]
    if "error" in pg:
        lines.append(f"- {pg['error']}")
    else:
        lines += [
            f"- requests (200): {pg['requests_ok']:,}",
            f"- cache hit rate: {fmt_pct(pg['hit_rate'])} "
            f"(exact {pg['exact_hits']:,} / semantic {pg['semantic_hits']:,})",
            f"- savings vs flagship: {fmt_pct(pg['savings_pct'])}",
            f"- saved by cache: ${pg['saved_usd_cache']:.4f} | "
            f"saved by down-routing: ${pg['saved_usd_routing']:.4f}",
            f"- fallbacks used: {pg['fallbacks']:,}",
        ]

    lines += ["", "## Cache correctness (frozen trap corpus)"]
    if trap_eval:
        t = trap_eval["test"]
        lines += [
            f"- OFFLINE (held-out 80, threshold {trap_eval['tuned_threshold']}): "
            f"hit {fmt_pct(t['hit_rate'])}, false-hit {fmt_pct(t['false_hit_rate'])}",
        ]
    if traps_live:
        lines += [
            f"- UNDER LOAD (live gateway, production per-tier thresholds): "
            f"hit {fmt_pct(traps_live.get('hit_rate'))}, "
            f"false-hit {fmt_pct(traps_live.get('false_hit_rate'))} "
            f"({traps_live.get('miss_false_hits', 0)}/{traps_live.get('miss_pairs', 0)} pairs)",
        ]

    lines += ["", "## Classifier vs the length-only control"]
    if classifier:
        control = classifier["length_only_control"]
        lines += [
            f"- test accuracy: {fmt_pct(classifier['test_accuracy'])} "
            f"vs control logreg {fmt_pct(control['logreg'])} / tree {fmt_pct(control['tree'])}",
            f"- confusion matrix (rows true 1..3): {classifier['test_confusion_matrix']}",
        ]

    if steady:
        lines += ["", "## Steady-state load",
                  f"- client-observed cache stats: {steady['counts']}",
                  f"- hit rate at convergence: {fmt_pct(steady['cache_hit_rate'])} "
                  f"(mix: 50% unique / 30% repeat / 20% paraphrase)"]
    if storm:
        lines += ["", "## Rate-limit storm",
                  f"- stormy 429s: {storm.get('stormy_429', 0):,} "
                  f"(with Retry-After: {storm.get('stormy_429_with_retry_after', 0):,}), "
                  f"ok: {storm.get('stormy_ok', 0):,}",
                  f"- innocent-team isolation: "
                  f"{'CLEAN — zero 429s' if storm.get('isolation_ok') else 'VIOLATED'}"]
    if budget:
        lines += ["", "## Budget exhaustion",
                  f"- outcomes: { {k: v for k, v in budget.items() if k != 'budget_block_seen'} }",
                  f"- budget block observed: {budget.get('budget_block_seen')}"]
    if outage:
        lines += ["", "## Outage drill (3-min primary kill)",
                  f"- client-visible 5xx: {outage.get('client_5xx', 0)} (target 0) -> "
                  f"{'ZERO-DROP PASS' if outage.get('zero_dropped') else 'FAIL'}",
                  f"- served via fallback: {outage.get('served_via_fallback', 0):,} "
                  f"of {outage.get('ok', 0):,} ok"]
    if not pg.get("error") and pg.get("breaker_events"):
        lines += ["", "## Breaker / health transitions"]
        lines += [f"- {e['ts']} {e['model']}: {e['from']} -> {e['to']} ({e['reason']})"
                  for e in pg["breaker_events"][:20]]
    if not pg.get("error") and pg.get("verification"):
        lines += ["", "## Verification loop"]
        lines += [f"- tier {v['tier']}: {v['verdict']} x{v['count']}"
                  for v in pg["verification"]]

    RESULTS.mkdir(exist_ok=True)
    summary = "\n".join(lines) + "\n"
    (RESULTS / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"(written to {RESULTS / 'summary.md'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
