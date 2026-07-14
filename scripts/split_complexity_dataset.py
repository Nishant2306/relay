"""Verify (and export) the group-aware split of the complexity dataset (SPEC C0.3).

The frozen dataset already embeds the normative assignment in `split_fold` /
`split` (fold 0 = dev, fold 1 = test, folds 2-6 = train). This script:

  1. verifies the embedded assignment honors the rubric's split contract
     (zero split_group / template_group leakage, audit strata preserved in
     every split, fold->split mapping intact);
  2. reports the tier x stratum composition per split;
  3. optionally writes results/splits.csv (id, split_group, tier, fold, split);
  4. informationally re-runs StratifiedGroupKFold(7, shuffle, seed 42) and
     reports agreement — fold assignment depends on row order and sklearn
     version, so disagreement here is expected and NOT an error. The embedded
     fields are normative (see CLAUDE.md).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.datasets import FOLD_TO_SPLIT, load_complexity_labels
from scripts.validate_datasets import check_group_leakage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None,
                        help="write the assignment CSV here (e.g. results/splits.csv)")
    args = parser.parse_args()

    rows = load_complexity_labels()

    errors = check_group_leakage(rows)
    for r in rows:
        if r["split"] != FOLD_TO_SPLIT[r["split_fold"]]:
            errors.append(f"{r['id']}: split {r['split']} != mapping for fold {r['split_fold']}")

    # Every audit stratum must be represented in every split.
    for split in ("dev", "test", "train"):
        strata = {r["length_confound_stratum"] for r in rows if r["split"] == split}
        missing = {"normal", "long_tier1", "short_tier3"} - strata
        if missing:
            errors.append(f"split {split}: missing audit strata {missing}")

    print("Embedded split composition (tier x length_confound_stratum):")
    for split in ("dev", "test", "train"):
        subset = [r for r in rows if r["split"] == split]
        tiers = Counter(r["tier"] for r in subset)
        strata = Counter(r["length_confound_stratum"] for r in subset)
        groups = len({r["split_group"] for r in subset})
        print(f"  {split:5s} n={len(subset):3d}  groups={groups:3d}  "
              f"tiers={dict(sorted(tiers.items()))}  strata={dict(strata)}")

    # Informational: recompute the rubric protocol and report agreement.
    y = np.array([r["tier"] for r in rows])
    groups_arr = np.array([r["split_group"] for r in rows])
    X = np.zeros((len(rows), 1))
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    recomputed = np.empty(len(rows), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(X, y, groups_arr)):
        recomputed[te] = k
    embedded = np.array([r["split_fold"] for r in rows])
    agree = float((recomputed == embedded).mean())
    print(f"\nInformational: recomputed SGKF folds agree with embedded on {agree:.1%} of rows "
          "(embedded fields are normative; disagreement is expected and not an error).")

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        with args.write.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "split_group", "tier", "split_fold", "split"])
            for r in rows:
                writer.writerow([r["id"], r["split_group"], r["tier"], r["split_fold"], r["split"]])
        print(f"Wrote {args.write}")

    if errors:
        print(f"\nFAIL — {len(errors)} split-contract violations:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nOK — split contract holds: zero group leakage, strata preserved, mapping intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
