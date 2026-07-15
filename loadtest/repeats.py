"""Repeat-heavy scenario (SPEC C11): hit-rate convergence curve.

A small hot set gets hammered; the hit rate should converge toward 100% as
the cache warms. Convergence samples land in results/repeats_convergence.json.
"""

from __future__ import annotations

import json
import random
import time

from locust import HttpUser, between, events, task

from loadtest.common import (
    LOADTEST_KEY,
    RESULTS_DIR,
    CacheStats,
    chat_body,
    headers,
    load_corpus,
)

corpus = load_corpus()
HOT_SET = [r["prompt"] for r in corpus["unique"][:25]]
stats = CacheStats()
samples: list[dict] = []
window = {"hits": 0, "total": 0, "started": time.time()}


class RepeatUser(HttpUser):
    wait_time = between(0.1, 0.4)

    @task
    def hot_repeat(self) -> None:
        prompt = random.choice(HOT_SET)
        with self.client.post(
            "/v1/chat/completions", json=chat_body(prompt),
            headers=headers(LOADTEST_KEY), name="chat[hot]", catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                stats.record(resp)
                window["total"] += 1
                if resp.headers.get("x-relay-cache") == "hit":
                    window["hits"] += 1
                if window["total"] >= 50:  # one convergence sample per 50 requests
                    samples.append({
                        "t": round(time.time() - window["started"], 1),
                        "hit_rate": round(window["hits"] / window["total"], 4),
                    })
                    window["hits"] = window["total"] = 0
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


@events.test_stop.add_listener
def _dump(environment, **kwargs) -> None:
    stats.dump("repeats")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "repeats_convergence.json").write_text(json.dumps(samples, indent=2))
