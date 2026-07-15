"""C2/C3 acceptance: OpenAI-compatible proxy + auth outcomes + request logging."""

from __future__ import annotations

import json

import httpx
from asgi_lifespan import LifespanManager

from tests.conftest import DEMO_KEY, RESTRICTED_KEY, auth, build_harness, chat_body


class TestAuth:
    async def test_missing_key_is_401(self, harness):
        r = await harness.client.post("/v1/chat/completions", json=chat_body("hi"))
        assert r.status_code == 401

    async def test_unknown_key_is_401(self, harness):
        r = await harness.client.post(
            "/v1/chat/completions", json=chat_body("hi"), headers=auth("not-a-key")
        )
        assert r.status_code == 401

    async def test_disallowed_model_is_403_with_clear_message(self, harness):
        r = await harness.client.post(
            "/v1/chat/completions",
            json=chat_body("hi", model="mock/top-c"),
            headers=auth(RESTRICTED_KEY),
        )
        assert r.status_code == 403
        assert "mock/top-c" in r.json()["detail"]
        assert "restricted" in r.json()["detail"]

    async def test_allowed_model_passes(self, harness):
        r = await harness.client.post(
            "/v1/chat/completions",
            json=chat_body("hi", model="mock/cheap-a"),
            headers=auth(RESTRICTED_KEY),
        )
        assert r.status_code == 200


class TestProxy:
    async def test_openai_shape_and_relay_headers(self, harness):
        r = await harness.client.post(
            "/v1/chat/completions", json=chat_body("What is Python?"), headers=auth()
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert r.headers["x-relay-cache"] == "miss"
        assert r.headers["x-relay-model"] == "mock/cheap-a"
        assert float(r.headers["x-relay-cost"]) > 0
        assert r.headers["x-relay-fallback"] == "false"
        assert int(r.headers["x-relay-overhead-ms"]) >= 0

    async def test_request_logged_with_costs(self, harness):
        await harness.client.post(
            "/v1/chat/completions", json=chat_body("What is Python?"), headers=auth()
        )
        await harness.drain()
        assert len(harness.request_log.entries) == 1
        entry = harness.request_log.entries[0]
        assert entry.team_id == 1
        assert entry.model_served == "mock/cheap-a"
        assert entry.status == 200
        assert entry.tokens_out > 0
        assert entry.actual_cost_usd > 0
        assert entry.counterfactual_cost_usd > entry.actual_cost_usd  # cheap-a < flagship

    async def test_invalid_body_is_400(self, harness):
        r = await harness.client.post(
            "/v1/chat/completions", json={"model": "mock/cheap-a"}, headers=auth()
        )
        assert r.status_code == 400

    async def test_model_without_provider_prefix_is_400(self, harness):
        r = await harness.client.post(
            "/v1/chat/completions", json=chat_body("hi", model="gpt-4o"), headers=auth()
        )
        assert r.status_code == 400
        assert "provider/model" in r.json()["error"]["message"]

    async def test_upstream_500_surfaces_as_502_and_logged(self, harness):
        harness.chaos.error_rate = 1.0
        r = await harness.client.post(
            "/v1/chat/completions", json=chat_body("hi"), headers=auth()
        )
        assert r.status_code == 502
        await harness.drain()
        entry = harness.request_log.entries[-1]
        assert entry.status == 502
        assert entry.error_class == "upstream_500"

    async def test_models_endpoint(self, harness):
        r = await harness.client.get("/v1/models", headers=auth())
        ids = {m["id"] for m in r.json()["data"]}
        assert {"mock/cheap-a", "mock/mid-b", "mock/top-c", "relay-auto"} <= ids


class TestStreaming:
    async def test_stream_matches_nonstream_and_logs(self, harness):
        non_stream = await harness.client.post(
            "/v1/chat/completions", json=chat_body("stream me"), headers=auth()
        )
        expected = non_stream.json()["choices"][0]["message"]["content"]

        parts: list[str] = []
        saw_done = False
        async with harness.client.stream(
            "POST", "/v1/chat/completions",
            json=chat_body("stream me", stream=True), headers=auth(),
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers["x-relay-model"] == "mock/cheap-a"
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    saw_done = True
                    continue
                chunk = json.loads(payload)
                for choice in chunk.get("choices", []):
                    if (choice.get("delta") or {}).get("content"):
                        parts.append(choice["delta"]["content"])

        assert saw_done
        assert "".join(parts) == expected
        await harness.drain()
        stream_entries = [e for e in harness.request_log.entries if e.status == 200]
        assert len(stream_entries) == 2  # non-stream + stream
        assert stream_entries[-1].tokens_out == non_stream.json()["usage"]["completion_tokens"]


class TestOpenAIClientConformance:
    """SPEC C2 acceptance: the official openai client works unmodified."""

    async def test_official_client_nonstream_and_stream(self):
        import openai

        h = await build_harness()
        async with LifespanManager(h.app):
            http_client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=h.app), base_url="http://relay"
            )
            client = openai.AsyncOpenAI(
                api_key=DEMO_KEY, base_url="http://relay/v1", http_client=http_client
            )

            completion = await client.chat.completions.create(
                model="mock/cheap-a",
                messages=[{"role": "user", "content": "What is Python?"}],
            )
            assert completion.choices[0].message.content
            assert completion.usage.total_tokens > 0

            stream = await client.chat.completions.create(
                model="mock/cheap-a",
                messages=[{"role": "user", "content": "What is Python?"}],
                stream=True,
            )
            streamed = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    streamed += chunk.choices[0].delta.content
            assert streamed == completion.choices[0].message.content

            models = await client.models.list()
            assert any(m.id == "relay-auto" for m in models.data)
            await http_client.aclose()
