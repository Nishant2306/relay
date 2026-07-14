"""OpenAI-compatible chaos mock provider (SPEC C1).

- Responses are deterministic from a hash of (model, conversation text) so
  cache tests are stable: same prompt -> byte-identical content and usage.
- `/chaos` reconfigures failure injection at runtime: error rate, 429 rate,
  latency distribution, streaming pace, hard-down toggle.
- Built as an app factory with per-app chaos state so tests are isolated and
  need no network (drive it with httpx.ASGITransport).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

WORDS = (
    "relay gateway cache vector token bucket breaker fallback tier route budget "
    "latency stream chunk provider model prompt response semantic exact usage "
    "cost saving verify judge health probe retry jitter cooldown quorum shard"
).split()

MODELS = ["cheap-a", "mid-b", "top-c"]


class ChaosConfig(BaseModel):
    error_rate: float = Field(0.0, ge=0.0, le=1.0)  # fraction of calls -> HTTP 500
    rate_429: float = Field(0.0, ge=0.0, le=1.0)  # fraction of calls -> HTTP 429
    base_latency_ms: int = Field(40, ge=0)
    latency_jitter_ms: int = Field(20, ge=0)
    stream_tokens_per_sec: int = Field(200, gt=0)
    hard_down: bool = False  # every call -> HTTP 503


def _seed(model: str, conversation: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{model}::{conversation}".encode()).digest()[:8], "big")


def _conversation_text(body: dict[str, Any]) -> str:
    parts = []
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        parts.append(f"{m.get('role')}:{content}")
    return "\n".join(parts)


def deterministic_completion(model: str, body: dict[str, Any]) -> tuple[str, int, int]:
    """(content, prompt_tokens, completion_tokens) — pure function of model+messages."""
    conversation = _conversation_text(body)
    rng = random.Random(_seed(model, conversation))
    n_tokens = rng.randint(30, 120)
    if body.get("max_tokens"):
        n_tokens = min(n_tokens, int(body["max_tokens"]))
    content = " ".join(rng.choice(WORDS) for _ in range(n_tokens))
    prompt_tokens = max(1, len(conversation) // 4)
    return content, prompt_tokens, n_tokens


def create_app() -> FastAPI:
    app = FastAPI(title="relay-mock-provider")
    chaos = ChaosConfig()
    app.state.chaos = chaos
    app.state.chat_calls = 0  # upstream-call counter (singleflight tests read this)

    def _chaos_gate(rng: random.Random) -> Response | None:
        if chaos.hard_down:
            return JSONResponse(
                {"error": {"message": "mock provider is hard down", "type": "server_error"}},
                status_code=503,
            )
        if rng.random() < chaos.error_rate:
            return JSONResponse(
                {"error": {"message": "injected upstream failure", "type": "server_error"}},
                status_code=500,
            )
        if rng.random() < chaos.rate_429:
            return JSONResponse(
                {"error": {"message": "injected rate limit", "type": "rate_limit_error"}},
                status_code=429,
                headers={"Retry-After": "1"},
            )
        return None

    async def _latency(rng: random.Random) -> None:
        delay = chaos.base_latency_ms + rng.random() * chaos.latency_jitter_ms
        if delay > 0:
            await asyncio.sleep(delay / 1000)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "down" if chaos.hard_down else "ok"}

    @app.get("/chaos")
    async def get_chaos() -> ChaosConfig:
        return chaos

    @app.post("/chaos")
    async def set_chaos(patch: dict[str, Any]) -> ChaosConfig:
        updated = chaos.model_copy(update=patch)
        ChaosConfig.model_validate(updated.model_dump())  # reject bad values
        for field, value in updated.model_dump().items():
            setattr(chaos, field, value)
        return chaos

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": "relay-mock"}
                for m in MODELS
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        app.state.chat_calls += 1
        body = await request.json()
        model = body.get("model", "cheap-a")
        rng = random.Random()  # chaos draws are random; content is not
        failure = _chaos_gate(rng)
        if failure is not None:
            return failure
        await _latency(rng)

        content, prompt_tokens, completion_tokens = deterministic_completion(model, body)
        created = int(time.time())
        completion_id = "chatcmpl-mock" + hashlib.sha256(content.encode()).hexdigest()[:20]

        if body.get("stream"):
            return StreamingResponse(
                _stream(completion_id, created, model, content, prompt_tokens, completion_tokens),
                media_type="text/event-stream",
            )

        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    async def _stream(
        completion_id: str,
        created: int,
        model: str,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> AsyncIterator[bytes]:
        def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n".encode()

        yield chunk({"role": "assistant", "content": ""})
        pace = 1.0 / chaos.stream_tokens_per_sec
        words = content.split(" ")
        for i, word in enumerate(words):
            # last word carries no trailing space so the streamed text
            # reassembles byte-identical to the non-stream response
            yield chunk({"content": word if i == len(words) - 1 else word + " "})
            await asyncio.sleep(pace)
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
