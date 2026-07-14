"""Singleflight stampede protection (ADR-0004).

On an exact-key miss, SET NX a lock; concurrent identical misses wait for the
winner's cache write instead of stampeding upstream. 10 concurrent identical
requests = 1 upstream call.

Per-exact-key, not per-semantic-neighborhood: the exact key is a cheap,
unambiguous identity — locking a similarity region would need a vector search
inside the lock path and can false-share across genuinely different prompts.

The lock value is a random token so only the owner can release it
(compare-and-delete in Lua); PX expiry bounds the blast radius of a crashed
owner, and waiters fall through to their own upstream call on timeout.
"""

from __future__ import annotations

import asyncio
import secrets

from redis.asyncio import Redis

RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

LOCK_TTL_MS = 30_000
WAIT_INTERVAL_S = 0.05
WAIT_TIMEOUT_S = 10.0


def lock_key(namespace: str, exact: str) -> str:
    return f"lock:{namespace}:{exact}"


class Singleflight:
    def __init__(self, redis: Redis, wait_timeout_s: float = WAIT_TIMEOUT_S):
        self.redis = redis
        self.wait_timeout_s = wait_timeout_s
        self._release = redis.register_script(RELEASE_LUA)
        self._tokens: dict[str, str] = {}

    async def acquire(self, namespace: str, exact: str) -> bool:
        """True -> caller owns the flight and must call release() when done."""
        token = secrets.token_hex(16)
        key = lock_key(namespace, exact)
        acquired = await self.redis.set(key, token, nx=True, px=LOCK_TTL_MS)
        if acquired:
            self._tokens[key] = token
            return True
        return False

    async def release(self, namespace: str, exact: str) -> None:
        key = lock_key(namespace, exact)
        token = self._tokens.pop(key, None)
        if token is not None:
            await self._release(keys=[key], args=[token])

    async def wait_for_result(self, namespace: str, exact: str, fetch) -> dict | None:
        """Poll `fetch()` until the flight owner publishes, the lock vanishes
        without a result (owner failed -> we go upstream), or timeout."""
        deadline = asyncio.get_event_loop().time() + self.wait_timeout_s
        key = lock_key(namespace, exact)
        while asyncio.get_event_loop().time() < deadline:
            entry = await fetch()
            if entry is not None:
                return entry
            if not await self.redis.exists(key):
                return await fetch()  # owner released: one final look
            await asyncio.sleep(WAIT_INTERVAL_S)
        return None
