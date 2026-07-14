"""C1 acceptance: deterministic content, scriptable chaos, no network needed."""

from __future__ import annotations

import httpx
import pytest

from mockprovider.app import create_app


@pytest.fixture
async def client():
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as c:
        yield c


def body(prompt: str, stream: bool = False) -> dict:
    return {
        "model": "cheap-a",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }


async def test_deterministic_from_prompt_hash(client):
    r1 = await client.post("/v1/chat/completions", json=body("What is Python?"))
    r2 = await client.post("/v1/chat/completions", json=body("What is Python?"))
    r3 = await client.post("/v1/chat/completions", json=body("What is Rust?"))
    assert r1.status_code == 200
    c1, c2, c3 = (r.json()["choices"][0]["message"]["content"] for r in (r1, r2, r3))
    assert c1 == c2
    assert c1 != c3
    assert r1.json()["usage"] == r2.json()["usage"]


async def test_openai_response_shape(client):
    r = await client.post("/v1/chat/completions", json=body("hello"))
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["finish_reason"] == "stop"
    usage = data["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    models = (await client.get("/v1/models")).json()
    assert {m["id"] for m in models["data"]} == {"cheap-a", "mid-b", "top-c"}


async def test_chaos_full_outage(client):
    await client.post("/chaos", json={"error_rate": 1.0})
    r = await client.post("/v1/chat/completions", json=body("x"))
    assert r.status_code == 500

    await client.post("/chaos", json={"error_rate": 0.0, "hard_down": True})
    r = await client.post("/v1/chat/completions", json=body("x"))
    assert r.status_code == 503
    assert (await client.get("/health")).json()["status"] == "down"

    await client.post("/chaos", json={"hard_down": False, "rate_429": 1.0})
    r = await client.post("/v1/chat/completions", json=body("x"))
    assert r.status_code == 429
    assert "Retry-After" in r.headers


async def test_chaos_recovery(client):
    await client.post("/chaos", json={"error_rate": 1.0})
    assert (await client.post("/v1/chat/completions", json=body("x"))).status_code == 500
    await client.post("/chaos", json={"error_rate": 0.0})
    assert (await client.post("/v1/chat/completions", json=body("x"))).status_code == 200


async def test_stream_reassembles_to_nonstream_content(client):
    await client.post("/chaos", json={"stream_tokens_per_sec": 100000, "base_latency_ms": 0})
    non_stream = await client.post("/v1/chat/completions", json=body("stream me"))
    expected = non_stream.json()["choices"][0]["message"]["content"]

    parts: list[str] = []
    saw_done = False
    final_usage = None
    async with client.stream(
        "POST", "/v1/chat/completions", json=body("stream me", stream=True)
    ) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                continue
            import json

            chunk = json.loads(payload)
            if chunk.get("usage"):
                final_usage = chunk["usage"]
            for choice in chunk["choices"]:
                if (choice.get("delta") or {}).get("content"):
                    parts.append(choice["delta"]["content"])

    assert saw_done
    assert "".join(parts) == expected
    assert final_usage is not None
    assert final_usage["completion_tokens"] == non_stream.json()["usage"]["completion_tokens"]
