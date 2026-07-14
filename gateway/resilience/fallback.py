"""ResilientCaller (SPEC C8): retry w/ backoff -> walk the fallback chain ->
circuit breakers — implementing the pipeline's Caller protocol.

Chain order comes from routing.yaml and never downgrades quality (ADR-0007):
same-tier alternates first, then tier+1. Per model in the chain:

  breaker.allow()? -> attempt + up to max_retries retryable retries
                      (each retryable failure feeds the breaker)
  exhausted        -> next model in the chain (fallback_used=true)
  non-retryable    -> surfaces immediately (no fallback: a 400 is wrong
                      everywhere; retrying elsewhere just duplicates it)

Streaming: the first chunk is pulled INSIDE the retry/fallback loop, so a
provider that fails at stream-open falls back cleanly; after first byte we
are committed to that stream (mid-stream failure -> SSE error event upstream).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from typing import Any

from gateway.adapters.base import AdapterError, StreamCollector
from gateway.models import AdapterResult, ChatCompletionRequest
from gateway.pipeline import CallMeta
from gateway.registry import ProviderRegistry
from gateway.resilience.breaker import BreakerRegistry
from gateway.resilience.health import HealthTracker
from gateway.resilience.retry import RetryConfig, backoff_delay

logger = logging.getLogger("relay.resilience")


class AllProvidersFailed(AdapterError):
    def __init__(self, chain: list[str], last: AdapterError | None):
        provider, model = (chain[-1].split("/", 1) if chain else ("?", "?"))
        message = f"all providers in chain {chain} failed or breaker-open"
        if last is not None:
            message += f"; last error: {last}"
        super().__init__(provider, model, message,
                         status_code=last.status_code if last else 503,
                         retryable=False)


class ResilientCaller:
    def __init__(
        self,
        registry: ProviderRegistry,
        breakers: BreakerRegistry,
        retry_config: RetryConfig | None = None,
        health: HealthTracker | None = None,
        sleep=asyncio.sleep,
        rng: random.Random | None = None,
    ):
        self.registry = registry
        self.breakers = breakers
        self.retry_config = retry_config or RetryConfig()
        self.health = health
        self.sleep = sleep
        self.rng = rng or random.Random()

    def _record(self, model_key: str, ok: bool, latency_s: float) -> None:
        if self.health is not None:
            self.health.record(model_key, ok, latency_s)

    async def call(self, chain: list[str],
                   request: ChatCompletionRequest) -> tuple[AdapterResult, CallMeta]:
        last_error: AdapterError | None = None
        total_retries = 0
        breaker_open_seen = False

        for position, model_key in enumerate(chain):
            breaker = self.breakers.get(model_key)
            if not breaker.allow():
                breaker_open_seen = True
                logger.info("breaker open for %s — skipping", model_key)
                continue
            adapter, model = self.registry.resolve(model_key)

            for attempt in range(self.retry_config.max_retries + 1):
                try:
                    result = await adapter.chat(request, model)
                except AdapterError as e:
                    last_error = e
                    self._record(model_key, ok=False, latency_s=0.0)
                    if not e.retryable:
                        raise  # wrong by construction — no retry, no fallback
                    breaker.record_failure()
                    if attempt < self.retry_config.max_retries and breaker.allow():
                        await self.sleep(backoff_delay(self.retry_config, attempt, self.rng))
                        total_retries += 1
                        continue
                    break  # exhausted here -> next model in chain
                breaker.record_success()
                self._record(model_key, ok=True, latency_s=result.latency_ms / 1000)
                return result, CallMeta(
                    model_key=model_key, retries=total_retries,
                    fallback_used=position > 0, breaker_open=breaker_open_seen,
                )

        raise AllProvidersFailed(chain, last_error)

    async def open_stream(self, chain: list[str], request: ChatCompletionRequest) -> tuple[
        AsyncIterator[dict[str, Any]], StreamCollector, CallMeta, Any, str
    ]:
        last_error: AdapterError | None = None
        total_retries = 0
        breaker_open_seen = False

        for position, model_key in enumerate(chain):
            breaker = self.breakers.get(model_key)
            if not breaker.allow():
                breaker_open_seen = True
                continue
            adapter, model = self.registry.resolve(model_key)

            for attempt in range(self.retry_config.max_retries + 1):
                collector = StreamCollector()
                aiter = adapter.stream_chat(request, model, collector)
                try:
                    first = await anext(aiter)
                except StopAsyncIteration:
                    first = None
                except AdapterError as e:
                    last_error = e
                    self._record(model_key, ok=False, latency_s=0.0)
                    if not e.retryable:
                        raise
                    breaker.record_failure()
                    if attempt < self.retry_config.max_retries and breaker.allow():
                        await self.sleep(backoff_delay(self.retry_config, attempt, self.rng))
                        total_retries += 1
                        continue
                    break
                # first chunk arrived -> committed to this stream
                breaker.record_success()
                meta = CallMeta(
                    model_key=model_key, retries=total_retries,
                    fallback_used=position > 0, breaker_open=breaker_open_seen,
                )
                return _prepend(first, aiter), collector, meta, adapter, model

        raise AllProvidersFailed(chain, last_error)


async def _prepend(first: dict[str, Any] | None,
                   rest: AsyncIterator[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    if first is not None:
        yield first
    async for chunk in rest:
        yield chunk
