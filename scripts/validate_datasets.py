"""Dataset contract validator for relay-datasets-v1.1.0 (SPEC C0).

Runs standalone (`python scripts/validate_datasets.py`) and is imported by
tests/test_dataset_contract.py so the same checks gate CI.

Every check returns a list of violation strings; empty list == pass.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.datasets import (
    DATASET_VERSION,
    FOLD_TO_SPLIT,
    load_cache_traps,
    load_complexity_labels,
)

VALID_HIT_PHENOMENA = {
    "word_order", "framing", "synonym", "politeness", "verbosity",
    "number_format", "formatting",
}
VALID_MISS_PHENOMENA = {
    "negation_flip", "source_target_swap", "aggregation_swap", "operation_inverse",
    "temporal_boundary_flip", "number_change",
}


def check_cache_traps(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []

    if len(rows) != 120:
        errors.append(f"traps: expected 120 rows, got {len(rows)}")

    by_expected = Counter(r["expected"] for r in rows)
    if by_expected.get("hit") != 60 or by_expected.get("miss") != 60:
        errors.append(f"traps: expected 60 hit / 60 miss, got {dict(by_expected)}")

    split_expected = Counter((r["split"], r["expected"]) for r in rows)
    want = {("dev", "hit"): 20, ("dev", "miss"): 20, ("test", "hit"): 40, ("test", "miss"): 40}
    if dict(split_expected) != want:
        errors.append(f"traps: split/expected counts {dict(split_expected)} != {want}")

    for r in rows:
        rid = r.get("id", "<no id>")
        if r.get("dataset_version") != DATASET_VERSION:
            errors.append(f"traps {rid}: dataset_version {r.get('dataset_version')}")
        if r.get("adjudication_status") != "human_confirmed":
            errors.append(f"traps {rid}: adjudication_status {r.get('adjudication_status')}")
        if not r.get("human_verified"):
            errors.append(f"traps {rid}: not human_verified")
        if r.get("effective_expected") != r.get("expected"):
            errors.append(f"traps {rid}: effective_expected != expected")
        if not r.get("prompt_a") or not r.get("prompt_b"):
            errors.append(f"traps {rid}: empty prompt")
        if not r.get("phenomena"):
            errors.append(f"traps {rid}: no phenomena")

    # No prompt text may appear in both dev and test (leakage into the reported number).
    dev_prompts = {p for r in rows if r["split"] == "dev" for p in (r["prompt_a"], r["prompt_b"])}
    test_prompts = {p for r in rows if r["split"] == "test" for p in (r["prompt_a"], r["prompt_b"])}
    overlap = dev_prompts & test_prompts
    if overlap:
        errors.append(f"traps: {len(overlap)} prompts appear in BOTH dev and test: "
                      f"{sorted(overlap)[:3]}...")

    ids = Counter(r["id"] for r in rows)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        errors.append(f"traps: duplicate ids {dupes}")

    return errors


def check_complexity_labels(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []

    if len(rows) != 600:
        errors.append(f"labels: expected 600 rows, got {len(rows)}")

    tiers = Counter(r["tier"] for r in rows)
    if dict(tiers) != {1: 205, 2: 196, 3: 199}:
        errors.append(f"labels: tier counts {dict(tiers)} != {{1: 205, 2: 196, 3: 199}}")

    strata = Counter(r["length_confound_stratum"] for r in rows)
    if dict(strata) != {"normal": 457, "long_tier1": 82, "short_tier3": 61}:
        errors.append(f"labels: length_confound_stratum counts {dict(strata)}")

    cache_classes = Counter(r["cache_class"] for r in rows)
    if dict(cache_classes) != {"stable": 554, "no_cache": 39, "temporal": 7}:
        errors.append(f"labels: cache_class counts {dict(cache_classes)}")

    corrections = sum(1 for r in rows if r.get("correction"))
    if corrections != 34:
        errors.append(f"labels: expected 34 logged bootstrap corrections, got {corrections}")

    for r in rows:
        rid = r.get("id", "<no id>")
        if r.get("dataset_version") != DATASET_VERSION:
            errors.append(f"labels {rid}: dataset_version {r.get('dataset_version')}")
        if not r.get("human_verified"):
            errors.append(f"labels {rid}: not human_verified")
        if r.get("tier") not in (1, 2, 3):
            errors.append(f"labels {rid}: tier {r.get('tier')}")
        if r.get("tier") != r.get("human_tier"):
            errors.append(f"labels {rid}: tier != human_tier")
        if not r.get("split_group"):
            errors.append(f"labels {rid}: split_group missing/empty")
        if r.get("split_fold") not in range(7):
            errors.append(f"labels {rid}: split_fold {r.get('split_fold')}")
        if r.get("split") != FOLD_TO_SPLIT.get(r.get("split_fold")):
            errors.append(f"labels {rid}: split {r.get('split')!r} inconsistent with "
                          f"split_fold {r.get('split_fold')}")
        if r.get("cache_class") not in ("stable", "no_cache", "temporal"):
            errors.append(f"labels {rid}: cache_class {r.get('cache_class')}")
        if not r.get("prompt", "").strip():
            errors.append(f"labels {rid}: empty prompt")
        audit = r.get("features_for_audit", {})
        if audit.get("char_count") != len(r["prompt"]):
            errors.append(f"labels {rid}: char_count {audit.get('char_count')} != "
                          f"len(prompt) {len(r['prompt'])}")

    prompts = Counter(r["prompt"] for r in rows)
    dupes = [p for p, n in prompts.items() if n > 1]
    if dupes:
        errors.append(f"labels: {len(dupes)} exact duplicate prompts, e.g. {dupes[:2]}")

    ids = Counter(r["id"] for r in rows)
    dup_ids = [i for i, n in ids.items() if n > 1]
    if dup_ids:
        errors.append(f"labels: duplicate ids {dup_ids}")

    return errors


def check_group_leakage(rows: list[dict[str, Any]]) -> list[str]:
    """Every split_group (and template_group) must live in exactly one split."""
    errors: list[str] = []
    for key in ("split_group", "template_group"):
        group_splits: dict[str, set[str]] = {}
        for r in rows:
            group = r.get(key)
            if group:
                group_splits.setdefault(group, set()).add(r["split"])
        leaked = {g: s for g, s in group_splits.items() if len(s) > 1}
        if leaked:
            errors.append(f"labels: {key} leakage across splits: {leaked}")
    return errors


def main() -> int:
    traps = load_cache_traps()
    labels = load_complexity_labels()

    errors = (
        check_cache_traps(traps)
        + check_complexity_labels(labels)
        + check_group_leakage(labels)
    )

    print(f"cache_traps.jsonl:        {len(traps)} rows")
    print(f"complexity_labels.jsonl:  {len(labels)} rows")
    if errors:
        print(f"\nFAIL — {len(errors)} contract violations:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nOK — all dataset contracts hold (relay-datasets-v1.1.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
