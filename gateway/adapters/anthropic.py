"""Anthropic adapter — translates OpenAI chat format <-> Anthropic /v1/messages."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.adapters.base import RETRYABLE_STATUS, AdapterError, ProviderAdapter, StreamCollector
from gateway.models import (
    AdapterResult,
    ChatCompletionRequest,
    Usage,
    openai_response,
)
from gateway.pricing import cost_usd

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


class AnthropicAdapter(ProviderAdapter):
    provider = "anthropic"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _payload(self, request: ChatCompletionRequest, model: str, stream: bool) -> dict[str, Any]:
        system = request.system_prompt()
        messages = [
            {"role": m.role, "content": m.text()}
            for m in request.messages
            if m.role in ("user", "assistant")
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
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
                f"{self.base_url}/v1/messages",
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
        content = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        usage = Usage(
            prompt_tokens=body.get("usage", {}).get("input_tokens", 0),
            completion_tokens=body.get("usage", {}).get("output_tokens", 0),
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        finish = "stop" if body.get("stop_reason") in ("end_turn", "stop_sequence") else "length"
        cost = cost_usd(key, usage.prompt_tokens, usage.completion_tokens)
        await self.spend_guard.charge(key, cost)
        return AdapterResult(
            response=openai_response(
                model, content, usage.prompt_tokens, usage.completion_tokens, finish_reason=finish
            ),
            usage=usage, provider=self.provider, model=model,
            cost_usd=cost, latency_ms=latency_ms, finish_reason=finish,
        )

    async def stream_chat(
        self, request: ChatCompletionRequest, model: str, collector: StreamCollector
    ) -> AsyncIterator[dict[str, Any]]:
        """Translate Anthropic SSE events into OpenAI chat.completion.chunk dicts."""
        key = self.model_key(model)
        await self.spend_guard.check(key)
        completion_id = f"chatcmpl-anthropic-{int(time.time() * 1000)}"
        created = int(time.time())
        input_tokens = 0
        output_tokens = 0

        def to_chunk(delta: dict[str, Any], finish: str | None = None,
                     usage: dict[str, int] | None = None) -> dict[str, Any]:
            chunk: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage:
                chunk["usage"] = usage
            return chunk

        try:
            async with self.http.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=self._payload(request, model, stream=True),
                headers=self._headers(),
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_for_status(resp, model)
                first = to_chunk({"role": "assistant", "content": ""})
                collector.feed(first)
                yield first
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[len("data:"):].strip())
                    etype = event.get("type")
                    if etype == "message_start":
                        input_tokens = event["message"].get("usage", {}).get("input_tokens", 0)
                    elif etype == "content_block_delta":
                        text = event.get("delta", {}).get("text", "")
                        if text:
                            chunk = to_chunk({"content": text})
                            collector.feed(chunk)
                            yield chunk
                    elif etype == "message_delta":
                        output_tokens = event.get("usage", {}).get("output_tokens", output_tokens)
                        stop = event.get("delta", {}).get("stop_reason")
                        if stop:
                            finish = "stop" if stop in ("end_turn", "stop_sequence") else "length"
                            usage = {
                                "prompt_tokens": input_tokens,
                                "completion_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens,
                            }
                            chunk = to_chunk({}, finish=finish, usage=usage)
                            collector.feed(chunk)
                            yield chunk
        except httpx.TimeoutException as e:
            raise AdapterError(self.provider, model, f"timeout: {e}", retryable=True) from e
        except httpx.HTTPError as e:
            raise AdapterError(self.provider, model, f"transport: {e}", retryable=True) from e
