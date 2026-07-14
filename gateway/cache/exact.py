"""Exact-hash cache entries (ADR-0003): O(1) fast path for true repeats.

Entry lives at ce:{namespace}:{exact_key} as a JSON blob:
  {response, model_key, tokens_in, tokens_out, cache_class}
"""

from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis


def entry_key(namespace: str, exact: str) -> str:
    return f"ce:{namespace}:{exact}"


class ExactStore:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def get(self, namespace: str, exact: str) -> dict[str, Any] | None:
        raw = await self.redis.get(entry_key(namespace, exact))
        return json.loads(raw) if raw else None

    async def set(self, namespace: str, exact: str, entry: dict[str, Any], ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        await self.redis.set(entry_key(namespace, exact), json.dumps(entry), ex=ttl_s)

    async def delete_namespace(self, namespace: str) -> int:
        deleted = 0
        async for key in self.redis.scan_iter(match=f"ce:{namespace}:*", count=500):
            await self.redis.delete(key)
            deleted += 1
        return deleted
