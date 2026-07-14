"""Adapter base classes.

Contract: every provider call — paid or free — goes through an adapter, which
(1) checks the MAX_DAILY_SPEND kill-switch before any paid call,
(2) computes the call's cost from usage and charges the SpendGuard,
(3) normalizes errors into AdapterError with a `retryable` flag the resilience
    layer keys off.

`OpenAIStyleAdapter` implements the OpenAI wire format and is reused by the
OpenAI, Ollama, and mock adapters (they differ only in base URL, auth, and
pricing). Anthropic overrides the request/response translation.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.models import AdapterResult, ChatCompletionRequest, Usage, approx_tokens
from gateway.pricing import cost_usd
from gateway.spend import SpendGuard

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class AdapterError(Exception):
    def __init__(self, provider: str, model: str, message: str, *,
                 status_code: int | None = None, retryable: bool = False):
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(f"[{provider}/{model}] {message}")


class StreamCollector:
    """Tee target for streaming responses: accumulates deltas while the raw
    chunks pass through to the client, and yields a complete AdapterResult
    once the stream finishes cleanly."""

    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.finish_reason: str | None = None
        self.usage: Usage | None = None
        self.completion_id: str | None = None
        self.created: int | None = None

    def feed(self, chunk: dict[str, Any]) -> None:
        self.completion_id = chunk.get("id", self.completion_id)
        self.created = chunk.get("created", self.created)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.content_parts.append(delta["content"])
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if chunk.get("usage"):
            self.usage = Usage(**chunk["usage"])

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    def complete(self) -> bool:
        return self.finish_reason == "stop"


class ProviderAdapter(ABC):
    provider: str = "base"

    def __init__(self, http: httpx.AsyncClient, spend_guard: SpendGuard, base_url: str,
                 api_key: str = ""):
        self.http = http
        self.spend_guard = spend_guard
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def model_key(self, model: str) -> str:
        return f"{self.provider}/{model}"

    @abstractmethod
    async def chat(self, request: ChatCompletionRequest, model: str) -> AdapterResult: ...

    @abstractmethod
    def stream_chat(
        self, request: ChatCompletionRequest, model: str, collector: StreamCollector
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield chat.completion.chunk dicts, feeding each into `collector`."""
        ...

    async def finalize_stream(
        self, request: ChatCompletionRequest, model: str,
        collector: StreamCollector, latency_ms: int,
    ) -> AdapterResult:
        """Build the AdapterResult for a finished stream and charge spend."""
        usage = collector.usage or Usage(
            prompt_tokens=approx_tokens(request.full_prompt_text() + request.system_prompt()),
            completion_tokens=approx_tokens(collector.content),
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        cost = cost_usd(self.model_key(model), usage.prompt_tokens, usage.completion_tokens)
        await self.spend_guard.charge(self.model_key(model), cost)
        from gateway.models import openai_response

        response = openai_response(
            model, collector.content, usage.prompt_tokens, usage.completion_tokens
        )
        return AdapterResult(
            response=response, usage=usage, provider=self.provider, model=model,
            cost_usd=cost, latency_ms=latency_ms, finish_reason=collector.finish_reason,
        )


class OpenAIStyleAdapter(ProviderAdapter):
    """Any provider speaking the OpenAI chat-completions wire format."""

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: ChatCompletionRequest, model: str, stream: bool) -> dict[str, Any]:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = model
        payload["stream"] = stream
        return payload

    def _raise_for_status(self, resp: httpx.Response, model: str) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            raise AdapterError(
                self.provider, model, f"HTTP {resp.status_code}: {detail}",
                status_code=resp.status_code, retryable=resp.status_code in RETRYABLE_STATUS,
            )

    async def chat(self, request: ChatCompletionRequest, model: str) -> AdapterResult:
        key = self.model_key(model)
        await self.spend_guard.check(key)
        started = time.perf_counter()
        try:
            resp = await self.http.post(
                f"{self.base_url}/v1/chat/completions",
                json=self._payload(request, model, stream=False),
                headers=self._headers(),
            )
        except httpx.TimeoutException as e:
            raise AdapterError(self.provider, model, f"timeout: {e}", retryable=True) from e
        except httpx.HTTPError as e:
            raise AdapterError(self.provider, model, f"transport: {e}", retryable=True) from e
        self._raise_for_status(resp, model)

        latency_ms = int((time.perf_counter() - started) * 1000)
        body = resp.json()
        usage = Usage(**(body.get("usage") or {}))
        cost = cost_usd(key, usage.prompt_tokens, usage.completion_tokens)
        await self.spend_guard.charge(key, cost)
        finish = None
        if body.get("choices"):
            finish = body["choices"][0].get("finish_reason")
        return AdapterResult(
            response=body, usage=usage, provider=self.provider, model=model,
            cost_usd=cost, latency_ms=latency_ms, finish_reason=finish,
        )

    async def stream_chat(
        self, request: ChatCompletionRequest, model: str, collector: StreamCollector
    ) -> AsyncIterator[dict[str, Any]]:
        key = self.model_key(model)
        await self.spend_guard.check(key)
        try:
            async with self.http.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=self._payload(request, model, stream=True),
                headers=self._headers(),
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_for_status(resp, model)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    collector.feed(chunk)
                    yield chunk
        except httpx.TimeoutException as e:
            raise AdapterError(self.provider, model, f"timeout: {e}", retryable=True) from e
        except httpx.HTTPError as e:
            raise AdapterError(self.provider, model, f"transport: {e}", retryable=True) from e
