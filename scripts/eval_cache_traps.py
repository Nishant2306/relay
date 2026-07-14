"""The headline evaluation (SPEC C5): semantic-cache correctness on the frozen
120-pair trap corpus.

Discipline (ADR-0009, enforced here, not by convention):
  - the threshold is tuned ONLY on the 40 dev pairs
  - every reported number comes ONLY from the 80 held-out test pairs
  - pairs flagged known_hard_embedding_case are reported as expected failures,
    never tuned around

Outputs:
  results/cache_trap_eval.json   — full report (the README numbers)
  models/trap_dev_curve.json     — dev-only threshold curve for the runtime
                                   near-miss tuning endpoint
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.cache.embedding import embed_sync
from gateway.cache.guards import answer_determining_conflict
from gateway.datasets import load_cache_traps

SPEC_MISS_FAMILIES = [
    "negation_flip", "source_target_swap", "aggregation_swap",
    "operation_inverse", "temporal_boundary_flip", "number_change",
]
THRESHOLD_GRID = [round(t, 3) for t in np.arange(0.70, 0.996, 0.005)]


def matcher(row: dict[str, Any], sim: float, threshold: float) -> bool:
    """The full production matcher: similarity gate + answer-determining
    guards (gateway/cache/guards.py). Guards exist because raw cosine cannot
    separate this corpus — see results/cache_trap_eval.json distributions."""
    if sim < threshold:
        return False
    return answer_determining_conflict(row["prompt_a"], row["prompt_b"]) is None


def pair_similarities(rows: list[dict[str, Any]]) -> dict[str, float]:
    texts: list[str] = []
    for r in rows:
        texts.extend([r["prompt_a"], r["prompt_b"]])
    vecs = embed_sync(texts)
    sims: dict[str, float] = {}
    for i, r in enumerate(rows):
        a, b = vecs[2 * i], vecs[2 * i + 1]
        sims[r["id"]] = float(np.dot(a, b))
    return sims


def rates_at(rows: list[dict[str, Any]], sims: dict[str, float],
             threshold: float) -> tuple[float, float]:
    hits = [r for r in rows if r["expected"] == "hit"]
    misses = [r for r in rows if r["expected"] == "miss"]
    hit_rate = sum(1 for r in hits if matcher(r, sims[r["id"]], threshold)) / len(hits)
    false_hit_rate = sum(
        1 for r in misses if matcher(r, sims[r["id"]], threshold)
    ) / len(misses)
    return hit_rate, false_hit_rate


def tune_on_dev(dev: list[dict[str, Any]], sims: dict[str, float]) -> tuple[float, dict]:
    """Maximize (hit recall - false-hit rate) on dev; tie-break to the safer
    (higher) threshold."""
    curve = {}
    best_t, best_score = THRESHOLD_GRID[0], -1.0
    for t in THRESHOLD_GRID:
        hit_rate, false_hit_rate = rates_at(dev, sims, t)
        curve[f"{t:.3f}"] = {"hit_rate": round(hit_rate, 4),
                             "false_hit_rate": round(false_hit_rate, 4)}
        score = hit_rate - false_hit_rate
        if score > best_score or (score == best_score and t > best_t):
            best_t, best_score = t, score
    return best_t, curve


def phenomenon_breakdown(rows: list[dict[str, Any]], sims: dict[str, float],
                         threshold: float, expected: str) -> list[dict[str, Any]]:
    """For miss pairs: which trap families produce false hits.
    For hit pairs: which paraphrase families fail to collide."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "wrong": 0})
    for r in rows:
        if r["expected"] != expected:
            continue
        collided = matcher(r, sims[r["id"]], threshold)
        wrong = collided if expected == "miss" else not collided
        for p in r["phenomena"]:
            stats[p]["n"] += 1
            stats[p]["wrong"] += int(wrong)
    out = []
    for p, s in sorted(stats.items(), key=lambda kv: (-kv[1]["wrong"], -kv[1]["n"])):
        out.append({
            "phenomenon": p, "pairs": s["n"], "wrong": s["wrong"],
            "wrong_rate": round(s["wrong"] / s["n"], 3),
            "spec_family": p in SPEC_MISS_FAMILIES,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/cache_trap_eval.json"))
    parser.add_argument("--curve-out", type=Path, default=Path("models/trap_dev_curve.json"))
    args = parser.parse_args()

    rows = load_cache_traps()
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    assert len(dev) == 40 and len(test) == 80

    print("Embedding 240 prompts (bge-small-en-v1.5, local CPU)...")
    sims = pair_similarities(rows)

    threshold, dev_curve = tune_on_dev(dev, sims)
    dev_hit, dev_false = rates_at(dev, sims, threshold)
    print(f"\nTuned on 40 DEV pairs only -> threshold {threshold:.3f} "
          f"(dev hit {dev_hit:.1%}, dev false-hit {dev_false:.1%})")

    test_hit, test_false = rates_at(test, sims, threshold)
    miss_breakdown = phenomenon_breakdown(test, sims, threshold, "miss")
    hit_breakdown = phenomenon_breakdown(test, sims, threshold, "hit")

    # which guard families do the work on the held-out traps
    guard_fires: dict[str, int] = defaultdict(int)
    for r in test:
        if r["expected"] == "miss":
            reason = answer_determining_conflict(r["prompt_a"], r["prompt_b"])
            if reason:
                guard_fires[reason.split(":")[0]] += 1

    def dist(rows_subset: list[dict[str, Any]]) -> dict[str, float]:
        vals = np.array([sims[r["id"]] for r in rows_subset])
        return {"min": round(float(vals.min()), 4), "median": round(float(np.median(vals)), 4),
                "max": round(float(vals.max()), 4)}

    known_hard = [
        {
            "id": r["id"], "split": r["split"], "expected": r["expected"],
            "similarity": round(sims[r["id"]], 4),
            "guard": answer_determining_conflict(r["prompt_a"], r["prompt_b"]),
            "outcome": "collides" if matcher(r, sims[r["id"]], threshold) else "separates",
        }
        for r in rows if "known_hard_embedding_case" in r.get("review_flags", [])
    ]

    residual_false_hits = [
        {"id": r["id"], "phenomena": r["phenomena"], "similarity": round(sims[r["id"]], 4),
         "prompt_a": r["prompt_a"], "prompt_b": r["prompt_b"]}
        for r in test
        if r["expected"] == "miss" and matcher(r, sims[r["id"]], threshold)
    ]

    print(f"\n=== HELD-OUT TEST (80 pairs) at threshold {threshold:.3f} ===")
    print(f"  paraphrase hit rate : {test_hit:.1%}   (target >= 80%)")
    print(f"  trap false-hit rate : {test_false:.1%}   (expected 2-8%; 0% would mean "
          f"threshold too high or dev leakage)")
    print("\n  Guard families doing the work on test traps:")
    for family, n in sorted(guard_fires.items(), key=lambda kv: -kv[1]):
        print(f"     {family:28s} {n} pairs rejected")
    print("\n  Trap families that defeat embedding similarity (test miss pairs):")
    for row in miss_breakdown:
        if row["wrong"]:
            marker = "*" if row["spec_family"] else " "
            print(f"   {marker} {row['phenomenon']:28s} {row['wrong']}/{row['pairs']} false hits")
    print("\n  Paraphrase families that fail to collide (test hit pairs):")
    for row in hit_breakdown:
        if row["wrong"]:
            print(f"     {row['phenomenon']:28s} {row['wrong']}/{row['pairs']} missed")
    if known_hard:
        print("\n  known_hard_embedding_case pairs:")
        for k in known_hard:
            print(f"     {k['id']} ({k['split']}, expected {k['expected']}): "
                  f"sim {k['similarity']}, guard={k['guard']} -> {k['outcome']}")
    if residual_false_hits:
        print("\n  Residual test false hits (the honest 2-8%):")
        for f in residual_false_hits:
            print(f"     {f['id']} {f['phenomena']} sim {f['similarity']}")
            print(f"        A: {f['prompt_a'][:90]}")
            print(f"        B: {f['prompt_b'][:90]}")

    report = {
        "dataset_version": "1.1.0",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "matcher": "cosine threshold + answer-determining guards (gateway/cache/guards.py)",
        "why_guards": {
            "note": "raw cosine cannot separate this corpus; traps are designed lookalikes",
            "test_hit_similarity": dist([r for r in test if r["expected"] == "hit"]),
            "test_miss_similarity": dist([r for r in test if r["expected"] == "miss"]),
        },
        "guard_fires_on_test_misses": dict(guard_fires),
        "tuned_threshold": threshold,
        "tuning_protocol": "dev-only (40 pairs); reported numbers test-only (80 pairs); ADR-0009",
        "dev": {"hit_rate": round(dev_hit, 4), "false_hit_rate": round(dev_false, 4)},
        "test": {"hit_rate": round(test_hit, 4), "false_hit_rate": round(test_false, 4)},
        "test_miss_phenomena": miss_breakdown,
        "test_hit_phenomena": hit_breakdown,
        "known_hard_cases": known_hard,
        "residual_test_false_hits": residual_false_hits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    args.curve_out.parent.mkdir(parents=True, exist_ok=True)
    args.curve_out.write_text(json.dumps(dev_curve, indent=2))
    print(f"\nWrote {args.out} and {args.curve_out}")

    ok = test_hit >= 0.80
    if not ok:
        print("ACCEPTANCE FAILED: test hit rate below 80%")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
