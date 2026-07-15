"""Budget exhaustion (SPEC C11): drive team 'stormy' ($0.50/day) to its daily
cap -> Slack warn at 80%, clean budget-exhausted 429s at 100%."""

from __future__ import annotations

import json
import random
from collections import Counter

from locust import HttpUser, constant, events, task

from loadtest.common import RESULTS_DIR, STORMY_KEY, chat_body, headers, load_corpus

corpus = load_corpus()
counts: Counter = Counter()


class BudgetBurner(HttpUser):
    wait_time = constant(0.5)

    @task
    def burn(self) -> None:
        row = random.choice(corpus["unique"])
        # top-c carries the highest simulated price -> fastest burn
        with self.client.post(
            "/v1/chat/completions",
            json=chat_body(row["prompt"] + " Give a thorough answer.", model="mock/top-c"),
            headers=headers(STORMY_KEY), name="chat[budget]", catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                counts["ok"] += 1
                resp.success()
            elif resp.status_code == 429:
                body = resp.json() if resp.text else {}
                code = (body.get("error") or {}).get("code", "")
                counts[code or "429"] += 1
                resp.success()  # blocked-by-budget is the scenario's success
            else:
                counts[f"http_{resp.status_code}"] += 1
                resp.failure(f"HTTP {resp.status_code}")


@events.test_stop.add_listener
def _dump(environment, **kwargs) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(counts)
    out["budget_block_seen"] = counts.get("budget_exhausted", 0) > 0
    (RESULTS_DIR / "budget_stats.json").write_text(json.dumps(out, indent=2))
