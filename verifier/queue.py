"""Verification queue: gateway samples tier-1/2 responses in, worker pops out.

Redis list in production; the in-memory variant keeps unit tests and offline
runs infrastructure-free.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Protocol

from redis.asyncio import Redis

from gateway.models import AdapterResult, ChatCompletionRequest

QUEUE_KEY = "verify:queue"


class VerifyItem(dict):
    """{log_id, tier, messages, model_served, response_content}"""


class VerifyQueue(Protocol):
    async def push(self, item: dict[str, Any]) -> None: ...
    async def pop(self, timeout_s: float = 5.0) -> dict[str, Any] | None: ...


class RedisVerifyQueue:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def push(self, item: dict[str, Any]) -> None:
        await self.redis.lpush(QUEUE_KEY, json.dumps(item))

    async def pop(self, timeout_s: float = 5.0) -> dict[str, Any] | None:
        raw = await self.redis.brpop([QUEUE_KEY], timeout=timeout_s)
        return json.loads(raw[1]) if raw else None


class InMemoryVerifyQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def push(self, item: dict[str, Any]) -> None:
        await self._queue.put(item)

    async def pop(self, timeout_s: float = 5.0) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        except TimeoutError:
            return None


class VerifierSampler:
    """Pipeline-side (implements the VerifierQueue protocol in pipeline.py):
    sample tier-1/2 responses at verification.sample_rate."""

    def __init__(self, queue: VerifyQueue, sample_rate=lambda: 0.15,
                 rng: random.Random | None = None):
        self.queue = queue
        self.sample_rate = sample_rate
        self.rng = rng or random.Random()

    async def maybe_enqueue(self, request: ChatCompletionRequest, result: AdapterResult,
                            tier: int | None, request_log_id: int | None) -> None:
        if tier not in (1, 2):
            return
        if self.rng.random() >= self.sample_rate():
            return
        content = ""
        if result.response.get("choices"):
            content = result.response["choices"][0].get("message", {}).get("content", "") or ""
        await self.queue.push({
            "log_id": request_log_id,
            "tier": tier,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "prompt_text": request.full_prompt_text(),
            "model_served": f"{result.provider}/{result.model}",
            "response_content": content,
        })
