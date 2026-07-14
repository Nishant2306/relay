"""Provider health tracking (SPEC C8).

Passive: every adapter call outcome (via the resilient caller) lands in a
rolling 5-minute window per provider/model. Active: a 30s prober loop
evaluates each window -> healthy / degraded / down, and every status change
emits an event (Prometheus gauge + Slack + provider_health_events row).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger("relay.health")

WINDOW_S = 300.0
PROBE_INTERVAL_S = 30.0
DEGRADED_ERROR_RATE = 0.10
DOWN_ERROR_RATE = 0.50
DEGRADED_P99_S = 10.0


@dataclass
class _Sample:
    ts: float
    ok: bool
    latency_s: float


class HealthTracker:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._samples: dict[str, deque[_Sample]] = {}
        self._status: dict[str, str] = {}

    def record(self, model_key: str, ok: bool, latency_s: float) -> None:
        window = self._samples.setdefault(model_key, deque())
        now = self.clock()
        window.append(_Sample(now, ok, latency_s))
        cutoff = now - WINDOW_S
        while window and window[0].ts < cutoff:
            window.popleft()

    def evaluate(self, model_key: str) -> str:
        window = self._samples.get(model_key)
        if not window:
            return self._status.get(model_key, "healthy")
        errors = sum(1 for s in window if not s.ok)
        error_rate = errors / len(window)
        latencies = sorted(s.latency_s for s in window)
        p99 = latencies[min(len(latencies) - 1, int(0.99 * len(latencies)))]
        if error_rate >= DOWN_ERROR_RATE:
            return "down"
        if error_rate >= DEGRADED_ERROR_RATE or p99 >= DEGRADED_P99_S:
            return "degraded"
        return "healthy"

    def sweep(self) -> list[tuple[str, str, str]]:
        """Evaluate every tracked model; return (model_key, old, new) changes."""
        changes = []
        for model_key in list(self._samples):
            new = self.evaluate(model_key)
            old = self._status.get(model_key, "healthy")
            if new != old:
                self._status[model_key] = new
                changes.append((model_key, old, new))
        return changes


async def prober_loop(
    tracker: HealthTracker,
    on_change: Callable[[str, str, str], Awaitable[None]],
    interval_s: float = PROBE_INTERVAL_S,
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            for model_key, old, new in tracker.sweep():
                await on_change(model_key, old, new)
        except Exception:
            logger.exception("health sweep failed")
