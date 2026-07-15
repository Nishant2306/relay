"""Steady mix (SPEC C11): ~50% unique / 30% repeat / 20% paraphrase."""

from __future__ import annotations

from locust import HttpUser, between, events, task

from loadtest.common import LOADTEST_KEY, CacheStats, chat_body, headers, load_corpus

corpus = load_corpus()
stats = CacheStats()


class SteadyUser(HttpUser):
    wait_time = between(0.2, 0.8)

    @task(50)
    def unique(self) -> None:
        self._send(corpus["unique"])

    @task(30)
    def repeat(self) -> None:
        self._send(corpus["repeat"])

    @task(20)
    def paraphrase(self) -> None:
        self._send(corpus["paraphrase"])

    def _send(self, pool: list[dict]) -> None:
        import random

        row = random.choice(pool)
        with self.client.post(
            "/v1/chat/completions", json=chat_body(row["prompt"]),
            headers=headers(LOADTEST_KEY), name=f"chat[{row['kind']}]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                stats.record(resp)
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


@events.test_stop.add_listener
def _dump(environment, **kwargs) -> None:
    stats.dump("steady")
