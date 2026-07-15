"""Outage drill (SPEC C11, C8 acceptance): chaos-kill the PRIMARY mock
provider for 3 minutes mid-run. Target: ZERO client-visible 5xx — fallbacks
walk to the alternate provider (mockb), breakers open, then auto-recover.

Timeline (driven by a background greenlet):
  t=30s   primary hard-down
  t=210s  primary restored
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter

import gevent
import httpx
from locust import HttpUser, between, events, task

from loadtest.common import DEMO_KEY, RESULTS_DIR, chat_body, headers, load_corpus

MOCK_ADMIN = os.environ.get("MOCK_PROVIDER_URL", "http://localhost:8100")
OUTAGE_START_S = 30
OUTAGE_END_S = 210

corpus = load_corpus()
counts: Counter = Counter()
timeline: list[dict] = []


def _set_chaos(hard_down: bool) -> None:
    httpx.post(f"{MOCK_ADMIN}/chaos", json={"hard_down": hard_down}, timeout=10)
    timeline.append({"t": time.time(), "hard_down": hard_down})


@events.test_start.add_listener
def _schedule_outage(environment, **kwargs) -> None:
    def orchestrate() -> None:
        gevent.sleep(OUTAGE_START_S)
        _set_chaos(True)
        gevent.sleep(OUTAGE_END_S - OUTAGE_START_S)
        _set_chaos(False)

    gevent.spawn(orchestrate)


class OutageUser(HttpUser):
    wait_time = between(0.2, 0.6)

    @task
    def steady(self) -> None:
        row = random.choice(corpus["unique"])
        with self.client.post(
            "/v1/chat/completions", json=chat_body(row["prompt"]),
            headers=headers(DEMO_KEY), name="chat[outage]", catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                counts["ok"] += 1
                if resp.headers.get("x-relay-fallback") == "true":
                    counts["served_via_fallback"] += 1
                resp.success()
            elif resp.status_code >= 500:
                counts["client_5xx"] += 1  # the number that must be zero
                resp.failure(f"client-visible {resp.status_code}")
            else:
                counts[f"http_{resp.status_code}"] += 1
                resp.failure(f"HTTP {resp.status_code}")


@events.test_stop.add_listener
def _dump(environment, **kwargs) -> None:
    try:
        _set_chaos(False)  # never leave the mock dead
    except Exception:
        pass
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(counts)
    out["zero_dropped"] = counts.get("client_5xx", 0) == 0
    out["timeline"] = timeline
    (RESULTS_DIR / "outage_stats.json").write_text(json.dumps(out, indent=2))
