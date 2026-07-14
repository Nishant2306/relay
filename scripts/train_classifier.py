"""Train the complexity classifier on the frozen 600 (SPEC C6, make train).

Split discipline: the embedded `split` fields are the normative assignment
(folds 2-6 train, fold 0 dev, fold 1 test — group-aware, leakage-checked in
CI). Hyperparameters were chosen against DEV; the reported accuracy +
confusion matrix come from TEST, evaluated once.

Sanity rails baked in:
  - report next to the 45.3%/46.0% length-only control (the gap is the result)
  - accuracy > 95% aborts with a leakage warning (SPEC: "go check")

Outputs: models/classifier.joblib, models/classifier_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.datasets import load_complexity_labels
from gateway.router.features import FEATURE_NAMES, extract_features
from scripts.length_baseline import COMMITTED

LEAKAGE_CEILING = 0.95


def build_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([extract_features(r["prompt"]) for r in rows])
    y = np.array([r["tier"] for r in rows])
    return X, y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-out", type=Path, default=Path("models/classifier.joblib"))
    parser.add_argument("--meta-out", type=Path, default=Path("models/classifier_meta.json"))
    parser.add_argument("--extra-training-rows", type=Path, default=None,
                        help="JSONL of {prompt, tier} rows harvested from verification "
                             "failures (scripts/retrain_from_failures.py)")
    args = parser.parse_args()

    rows = load_complexity_labels()
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    test_rows = [r for r in rows if r["split"] == "test"]

    extra_n = 0
    if args.extra_training_rows and args.extra_training_rows.exists():
        with args.extra_training_rows.open(encoding="utf-8") as f:
            extra = [json.loads(line) for line in f if line.strip()]
        train_rows = train_rows + extra
        extra_n = len(extra)
        print(f"Including {extra_n} harvested rows from the verification failure loop.")

    X_train, y_train = build_matrix(train_rows)
    X_dev, y_dev = build_matrix(dev_rows)
    X_test, y_test = build_matrix(test_rows)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=5000, C=1.0)),
    ])
    model.fit(X_train, y_train)

    dev_acc = float(accuracy_score(y_dev, model.predict(X_dev)))
    test_pred = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_pred))
    cm = confusion_matrix(y_test, test_pred, labels=[1, 2, 3]).tolist()

    control = COMMITTED["tree"]  # the stronger of the two length-only controls
    print(f"train n={len(train_rows)}  dev n={len(dev_rows)}  test n={len(test_rows)}")
    print(f"dev accuracy : {dev_acc:.1%}")
    print(f"TEST accuracy: {test_acc:.1%}   vs length-only control {control:.1%} "
          f"(gap {test_acc - control:+.1%})")
    print("TEST confusion matrix (rows=true tier 1..3, cols=predicted):")
    for i, row in enumerate(cm, start=1):
        print(f"  tier {i}: {row}")

    if test_acc > LEAKAGE_CEILING:
        print(f"\nSUSPICIOUS: {test_acc:.1%} > 95% — check for split leakage before "
              f"believing this number (SPEC C6).")
        return 1
    if test_acc <= max(COMMITTED.values()):
        print("\nFAILED: classifier does not beat the length-only control.")
        return 1

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    meta = {
        "feature_names": FEATURE_NAMES,
        "dev_accuracy": round(dev_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_confusion_matrix": cm,
        "length_only_control": COMMITTED,
        "train_rows": len(train_rows),
        "extra_training_rows": extra_n,
        "split_source": "embedded normative split (fold 0 dev / 1 test / 2-6 train)",
    }
    args.meta_out.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {args.model_out} and {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
