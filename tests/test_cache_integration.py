"""C5 acceptance (integration — real Redis 8 vector search + local embeddings):

- exact-hash fast path, semantic paraphrase hit, trap correctly missing
- namespace isolation (system prompt / temperature / team)
- TTL by cache_class (temporal ~1h, no_cache bypassed entirely)
- singleflight: 10 concurrent identical misses -> exactly 1 upstream call
- invalidation
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from asgi_lifespan import LifespanManager
from redis.asyncio import Redis

from gateway.cache.service import SemanticCache
from gateway.models import ChatCompletionRequest
from gateway.pipeline import RouteDecision
from tests.conftest import auth, build_harness, chat_body

pytestmark = pytest.mark.integration


class StubRouter:
    """Until C6 lands: everything is tier 1 (threshold 0.755 from routing.yaml)."""

    def decide(self, request: ChatCompletionRequest) -> RouteDecision:
        return RouteDecision(tier=1, chain=["mock/cheap-a"], confidence=1.0)


@pytest.fixture(scope="module")
def redis_url():
    from testcontainers.redis import RedisContainer

    with RedisContainer("redis:8") as rc:
        yield f"redis://{rc.get_container_host_ip()}:{rc.get_exposed_port(6379)}/0"


@pytest.fixture
async def cache_harness(redis_url):
    redis = Redis.from_url(redis_url)
    await redis.flushall()
    cache = SemanticCache(redis)
    await cache.start()
    h = await build_harness(pipeline_kwargs={"cache": cache, "router": StubRouter()})
    h.cache = cache  # type: ignore[attr-defined]
    async with LifespanManager(h.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=h.app), base_url="http://relay"
        ) as client:
            h.client = client
            yield h
    await redis.aclose()


async def ask(h, prompt: str, **kw):
    r = await h.client.post("/v1/chat/completions", json=chat_body(prompt, **kw), headers=auth())
    assert r.status_code == 200, r.text
    await h.drain()  # let the cache write land before the next request
    return r


class TestCacheCorrectness:
    async def test_exact_hit_on_repeat(self, cache_harness):
        h = cache_harness
        first = await ask(h, "What is a unit test?")
        assert first.headers["x-relay-cache"] == "miss"
        second = await ask(h, "  what is a unit   test? ")  # normalization applies
        assert second.headers["x-relay-cache"] == "hit"
        assert second.headers["x-relay-cache-kind"] == "exact"
        assert second.headers["x-relay-cost"] == "0.00000000"
        assert second.json()["choices"] == first.json()["choices"]

    async def test_semantic_hit_on_paraphrase(self, cache_harness):
        h = cache_harness
        await ask(h, "What is Python?")
        r = await ask(h, "Could you explain what Python is?")
        assert r.headers["x-relay-cache"] == "hit"
        assert r.headers["x-relay-cache-kind"] == "semantic"
        assert float(r.headers["x-relay-cache-similarity"]) >= 0.755

    async def test_trap_pair_does_not_hit(self, cache_harness):
        h = cache_harness
        await ask(h, "Convert 20 Celsius to Fahrenheit.")
        r = await ask(h, "Convert 20 Fahrenheit to Celsius.")  # sim 0.99, guard rejects
        assert r.headers["x-relay-cache"] == "miss"
        stats = await h.cache.stats()
        assert stats.get("guard_reject", 0) >= 1

    async def test_namespace_isolation_system_prompt(self, cache_harness):
        h = cache_harness
        body = chat_body("What is DNS?")
        body["messages"].insert(0, {"role": "system", "content": "You are helpful."})
        r1 = await h.client.post("/v1/chat/completions", json=body, headers=auth())
        await h.drain()
        assert r1.headers["x-relay-cache"] == "miss"

        body2 = chat_body("What is DNS?")
        body2["messages"].insert(0, {"role": "system", "content": "Answer in French."})
        r2 = await h.client.post("/v1/chat/completions", json=body2, headers=auth())
        await h.drain()
        assert r2.headers["x-relay-cache"] == "miss"  # never shares across system prompts

    async def test_namespace_isolation_temperature_bucket(self, cache_harness):
        h = cache_harness
        await ask(h, "Define recursion.", temperature=0.0)
        r = await ask(h, "Define recursion.", temperature=0.8)
        assert r.headers["x-relay-cache"] == "miss"

    async def test_ttl_by_cache_class(self, cache_harness):
        h = cache_harness
        await ask(h, "Summarize the weather report for today in two lines.")  # temporal
        keys = [k async for k in h.cache.redis.scan_iter(match="ce:*")]
        assert keys
        ttls = [await h.cache.redis.ttl(k) for k in keys]
        assert any(0 < t <= 3600 for t in ttls)

    async def test_no_cache_class_bypasses(self, cache_harness):
        h = cache_harness
        before = [k async for k in h.cache.redis.scan_iter(match="ce:*")]
        r = await ask(h, "Write a poem about the sea.")
        assert r.headers["x-relay-cache"] == "miss"
        after = [k async for k in h.cache.redis.scan_iter(match="ce:*")]
        assert len(after) == len(before)  # nothing written
        r2 = await ask(h, "Write a poem about the sea.")
        assert r2.headers["x-relay-cache"] == "miss"  # and never served

    async def test_invalidation_flush(self, cache_harness):
        h = cache_harness
        await ask(h, "What is TCP?")
        assert (await ask(h, "What is TCP?")).headers["x-relay-cache"] == "hit"
        await h.cache.flush_all()
        assert (await ask(h, "What is TCP?")).headers["x-relay-cache"] == "miss"


class TestSingleflight:
    async def test_10_concurrent_identical_misses_one_upstream_call(self, cache_harness):
        h = cache_harness
        h.mock_app.state.chaos.base_latency_ms = 150  # force overlap
        h.mock_app.state.chat_calls = 0

        async def one():
            r = await h.client.post(
                "/v1/chat/completions",
                json=chat_body("Explain the CAP theorem in one paragraph."),
                headers=auth(),
            )
            assert r.status_code == 200
            return r

        responses = await asyncio.gather(*(one() for _ in range(10)))
        await h.drain()
        assert h.mock_app.state.chat_calls == 1, (
            f"singleflight leaked: {h.mock_app.state.chat_calls} upstream calls"
        )
        bodies = {r.json()["choices"][0]["message"]["content"] for r in responses}
        assert len(bodies) == 1  # everyone got the same answer
