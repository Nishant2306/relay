"""Rate-limit storm (SPEC C11): team 'stormy' (60 RPM) hammers past its limit
and must get clean 429s with Retry-After; team 'demo' must be unaffected."""

from __future__ import annotations

import json
import random
from collections import Counter

from locust import HttpUser, between, constant, events, task

from loadtest.common import DEMO_KEY, RESULTS_DIR, STORMY_KEY, chat_body, headers, load_corpus

corpus = load_corpus()
counts: Counter = Counter()


class StormyUser(HttpUser):
    """Deliberately exceeds 60 RPM. 429s here are EXPECTED and correct."""

    weight = 4
    wait_time = constant(0.2)

    @task
    def hammer(self) -> None:
        row = random.choice(corpus["unique"])
        with self.client.post(
            "/v1/chat/completions", json=chat_body(row["prompt"], model="mock/cheap-a"),
            headers=headers(STORMY_KEY), name="chat[stormy]", catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                counts["stormy_429"] += 1
                if resp.headers.get("Retry-After"):
                    counts["stormy_429_with_retry_after"] += 1
                resp.success()  # a clean 429 IS the correct behavior
            elif resp.status_code == 200:
                counts["stormy_ok"] += 1
                resp.success()
            else:
                counts[f"stormy_{resp.status_code}"] += 1
                resp.failure(f"HTTP {resp.status_code}")


class InnocentUser(HttpUser):
    """Well-behaved team on the same gateway — must see zero 429s."""

    weight = 1
    wait_time = between(1.0, 2.0)

    @task
    def normal(self) -> None:
        row = random.choice(corpus["unique"])
        with self.client.post(
            "/v1/chat/completions", json=chat_body(row["prompt"], model="mock/cheap-a"),
            headers=headers(DEMO_KEY), name="chat[innocent]", catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                counts["innocent_ok"] += 1
                resp.success()
            elif resp.status_code == 429:
                counts["innocent_429"] += 1
                resp.failure("innocent team was rate-limited by the storm")
            else:
                resp.failure(f"HTTP {resp.status_code}")


@events.test_stop.add_listener
def _dump(environment, **kwargs) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(counts)
    out["isolation_ok"] = counts["innocent_429"] == 0
    (RESULTS_DIR / "storm_stats.json").write_text(json.dumps(out, indent=2))
