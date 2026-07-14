"""MAX_DAILY_SPEND kill-switch for real-dollar providers.

Every paid adapter call must pass through `SpendGuard`: a pre-call gate that
refuses new paid work once the day's real spend reaches the cap, plus a
post-call charge with the actual cost. Free/simulated providers (mock, ollama)
are never blocked.

Backed by Redis when available (shared across workers), with an in-memory
fallback so unit tests and offline scripts need no infrastructure.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

from redis.asyncio import Redis

from gateway.pricing import is_real_spend


class SpendCapExceeded(Exception):
    def __init__(self, spent: float, cap: float):
        self.spent, self.cap = spent, cap
        super().__init__(
            f"MAX_DAILY_SPEND kill-switch: ${spent:.4f} of ${cap:.2f} daily real-API "
            f"budget already spent — refusing new paid provider calls"
        )


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


class SpendGuard:
    def __init__(self, cap_usd: float, redis: Redis | None = None):
        self.cap_usd = cap_usd
        self._redis = redis
        self._memory: dict[str, float] = defaultdict(float)

    async def spent_today(self) -> float:
        key = f"spend:real:{_today()}"
        if self._redis is not None:
            val = await self._redis.get(key)
            return float(val) if val else 0.0
        return self._memory[key]

    async def check(self, model_key: str) -> None:
        """Pre-call gate. Raises SpendCapExceeded for paid providers at the cap."""
        if not is_real_spend(model_key):
            return
        spent = await self.spent_today()
        if spent >= self.cap_usd:
            raise SpendCapExceeded(spent, self.cap_usd)

    async def charge(self, model_key: str, cost_usd: float) -> None:
        """Post-call: record actual real-dollar spend."""
        if not is_real_spend(model_key) or cost_usd <= 0:
            return
        key = f"spend:real:{_today()}"
        if self._redis is not None:
            pipe = self._redis.pipeline()
            pipe.incrbyfloat(key, cost_usd)
            pipe.expire(key, 60 * 60 * 48)
            await pipe.execute()
        else:
            self._memory[key] += cost_usd
