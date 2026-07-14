"""Length-only control baseline — the release gate (SPEC C0.4, rubric line 20).

Protocol (frozen in datasets/TIERING_RUBRIC.md):
  - single feature: prompt character count (standard-scaled per fold)
  - StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42),
    groups = split_group, stratification target = tier
  - pooled out-of-fold accuracy
  - LogisticRegression(C=0.3)                       -> committed 45.3%
  - DecisionTree(depth=3, min_samples_leaf=5,
                 class_weight="balanced")           -> committed 46.0%

Both must stay BELOW 50% or the release gate fails (exit 1). The real
classifier is reported against this control: the gap is the result.

Note: this recomputes folds per the rubric protocol. It intentionally does NOT
use the embedded split_fold field (that is the frozen train/dev/test assignment
for classifier work — a separate contract).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.datasets import load_complexity_labels

COMMITTED = {"logreg": 0.453, "tree": 0.460}
GATE = 0.50
TOLERANCE = 0.010  # drift alarm vs the committed numbers, ±1.0pp


def run_baseline() -> dict[str, float]:
    rows = load_complexity_labels()
    X = np.array([len(r["prompt"]) for r in rows], dtype=float).reshape(-1, 1)
    y = np.array([r["tier"] for r in rows])
    groups = np.array([r["split_group"] for r in rows])

    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y, groups))

    def pooled_oof(make_model) -> float:
        pred = np.empty(len(y), dtype=int)
        for tr, te in folds:
            scaler = StandardScaler().fit(X[tr])
            model = make_model().fit(scaler.transform(X[tr]), y[tr])
            pred[te] = model.predict(scaler.transform(X[te]))
        return float(accuracy_score(y, pred))

    return {
        "logreg": pooled_oof(lambda: LogisticRegression(C=0.3, max_iter=5000)),
        "tree": pooled_oof(
            lambda: DecisionTreeClassifier(
                max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="also write results to this JSON file")
    args = parser.parse_args()

    results = run_baseline()

    print("Length-only control (7-fold StratifiedGroupKFold, seed 42, pooled OOF):")
    failed = False
    for name, acc in results.items():
        committed = COMMITTED[name]
        drift = abs(acc - committed)
        status = "OK" if acc < GATE else "GATE FAILED (>= 50%)"
        if acc < GATE and drift > TOLERANCE:
            status = f"WARN: drifted {drift:+.1%} from committed {committed:.1%}"
        print(f"  {name:8s} {acc:.3%}   (committed {committed:.1%})   {status}")
        if acc >= GATE:
            failed = True

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"length_baseline": results, "gate": GATE}, indent=2))

    if failed:
        print("\nRELEASE GATE FAILED: a length-only model reached 50% — the dataset's "
              "length confound control is broken. Do not ship classifier numbers.")
        return 1
    print("\nGate passed: length alone cannot predict tier (<50%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
