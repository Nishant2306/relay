"""Loaders for the frozen Relay datasets (relay-datasets-v1.1.0).

These files are immutable. Anything that reads them goes through here so the
version expectation lives in exactly one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
DATASET_VERSION = "1.1.0"

CACHE_TRAPS_PATH = DATASETS_DIR / "cache_traps.jsonl"
COMPLEXITY_LABELS_PATH = DATASETS_DIR / "complexity_labels.jsonl"
TIERING_RUBRIC_PATH = DATASETS_DIR / "TIERING_RUBRIC.md"

# Embedded fold -> split mapping frozen with the dataset.
FOLD_TO_SPLIT = {0: "dev", 1: "test", 2: "train", 3: "train", 4: "train", 5: "train", 6: "train"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_cache_traps() -> list[dict[str, Any]]:
    """120 paraphrase/trap pairs: 40 dev (tuning only), 80 test (reporting only)."""
    return load_jsonl(CACHE_TRAPS_PATH)


def load_complexity_labels() -> list[dict[str, Any]]:
    """600 human-verified prompts with tier, split, cache_class, and audit features."""
    return load_jsonl(COMPLEXITY_LABELS_PATH)


def complexity_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Filter by the embedded (frozen, normative) split: 'dev' | 'test' | 'train'."""
    if split not in {"dev", "test", "train"}:
        raise ValueError(f"unknown split {split!r}")
    return [r for r in rows if r["split"] == split]


def trap_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Filter trap pairs by split: 'dev' (tune) | 'test' (report)."""
    if split not in {"dev", "test"}:
        raise ValueError(f"unknown split {split!r}")
    return [r for r in rows if r["split"] == split]
