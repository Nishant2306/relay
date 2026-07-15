"""Shared load-test plumbing: corpus loading, keys, cache-header accounting."""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

CORPUS_PATH = Path(__file__).resolve().parent.parent / "results" / "loadtest_corpus.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

LOADTEST_KEY = os.environ.get("RELAY_LOADTEST_KEY", "relay-loadtest-key")
STORMY_KEY = os.environ.get("RELAY_STORMY_KEY", "relay-stormy-key")
SPENDY_KEY = os.environ.get("RELAY_SPENDY_KEY", "relay-spendy-key")
DEMO_KEY = os.environ.get("RELAY_DEMO_KEY", "relay-demo-key")

RNG = random.Random(1234)


def load_corpus() -> dict[str, list[dict]]:
    rows = [json.loads(line) for line in CORPUS_PATH.open(encoding="utf-8")]
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    return by_kind


def headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def chat_body(prompt: str, model: str = "relay-auto") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": prompt}]}


class CacheStats:
    """Client-side accounting from x-relay-* headers."""

    def __init__(self) -> None:
        self.counter: Counter = Counter()

    def record(self, response) -> None:
        cache = response.headers.get("x-relay-cache", "none")
        kind = response.headers.get("x-relay-cache-kind", "")
        self.counter[f"cache_{cache}{('_' + kind) if kind else ''}"] += 1
        if response.headers.get("x-relay-fallback") == "true":
            self.counter["fallback"] += 1
        tier = response.headers.get("x-relay-tier")
        if tier:
            self.counter[f"tier_{tier}"] += 1

    def dump(self, name: str) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        total_hits = sum(v for k, v in self.counter.items() if k.startswith("cache_hit"))
        total = sum(v for k, v in self.counter.items() if k.startswith("cache_"))
        out = {
            "counts": dict(self.counter),
            "cache_hit_rate": round(total_hits / total, 4) if total else None,
        }
        (RESULTS_DIR / f"{name}_cache_stats.json").write_text(json.dumps(out, indent=2))
