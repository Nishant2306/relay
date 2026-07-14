"""C4 acceptance (integration — real Redis, no mocks per CLAUDE.md):

- property: 120-request burst at 60 RPM -> ~60 accepted
- atomicity under 50 parallel clients (no over-admission)
- Retry-After computed from deficit/refill rate
- TPM pre-charge + refund reconcile (ADR-0005)
- budget: Slack warn once at 80%, block at 100%
"""

from __future__ import annotations

import asyncio
import time

import pytest
from redis.asyncio import Redis

from gateway.middleware.budget import RedisBudgetGuard
from gateway.middleware.ratelimit import RedisRateLimiter
from gateway.models import TeamContext
from gateway.obs.slack import SlackNotifier

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def redis_url():
    from testcontainers.redis import RedisContainer

    with RedisContainer("redis:8") as rc:
        yield f"redis://{rc.get_container_host_ip()}:{rc.get_exposed_port(6379)}/0"


@pytest.fixture
async def redis(redis_url):
    client = Redis.from_url(redis_url)
    await client.flushall()
    yield client
    await client.aclose()


def team(team_id: int = 1, rpm: int = 60, tpm: int = 1_000_000, daily: float = 100.0,
         monthly: float = 1000.0) -> TeamContext:
    return TeamContext(
        team_id=team_id, name=f"team{team_id}", rpm=rpm, tpm=tpm,
        daily_budget_usd=daily, monthly_budget_usd=monthly,
        allowed_models=["*"], cache_scope="team",
    )


class TestTokenBucket:
    async def test_burst_property_120_requests_at_60_rpm(self, redis):
        limiter = RedisRateLimiter(redis)
        t = team(rpm=60)
        results = [await limiter.check(t, est_tokens=10) for _ in range(120)]
        accepted = sum(1 for r in results if r.allowed)
        # bucket starts full at capacity 60; refill during the burst is negligible
        assert 58 <= accepted <= 63
        # rejections carry a sane Retry-After
        rejected = [r for r in results if not r.allowed]
        assert all(r.kind == "rpm" for r in rejected)
        assert all(0 < (r.retry_after_s or 0) <= 2.0 for r in rejected)

    async def test_no_overadmission_under_50_parallel_clients(self, redis):
        limiter = RedisRateLimiter(redis)
        t = team(team_id=2, rpm=50)

        async def worker(n: int) -> int:
            allowed = 0
            for _ in range(n):
                if (await limiter.check(t, est_tokens=1)).allowed:
                    allowed += 1
            return allowed

        counts = await asyncio.gather(*(worker(4) for _ in range(50)))  # 200 attempts
        assert 50 <= sum(counts) <= 53  # atomic take: no over-admission beyond refill

    async def test_retry_after_matches_deficit_over_refill_rate(self, redis):
        limiter = RedisRateLimiter(redis)
        t = team(team_id=3, rpm=60)  # refill 1 token/s
        while (await limiter.check(t, est_tokens=1)).allowed:
            pass
        decision = await limiter.check(t, est_tokens=1)
        assert not decision.allowed
        assert decision.retry_after_s == pytest.approx(1.0, abs=0.5)

    async def test_bucket_refills_over_time(self, redis):
        limiter = RedisRateLimiter(redis)
        t = team(team_id=4, rpm=60)
        while (await limiter.check(t, est_tokens=1)).allowed:
            pass
        start = time.monotonic()
        while not (await limiter.check(t, est_tokens=1)).allowed:
            if time.monotonic() - start > 3:
                pytest.fail("bucket did not refill")
            await asyncio.sleep(0.1)
        assert 0.5 <= time.monotonic() - start <= 2.5

    async def test_tpm_precharge_and_refund(self, redis):
        limiter = RedisRateLimiter(redis)
        t = team(team_id=5, rpm=1000, tpm=1000)

        first = await limiter.check(t, est_tokens=600)
        assert first.allowed
        second = await limiter.check(t, est_tokens=600)  # only ~400 left
        assert not second.allowed and second.kind == "tpm"

        # response actually used 100 tokens -> refund 500
        await limiter.reconcile(t, est_tokens=600, actual_tokens=100)
        third = await limiter.check(t, est_tokens=600)
        assert third.allowed

    async def test_rpm_token_refunded_when_tpm_rejects(self, redis):
        limiter = RedisRateLimiter(redis)
        t = team(team_id=6, rpm=2, tpm=100)

        denied = await limiter.check(t, est_tokens=10_000)  # TPM rejects instantly
        assert not denied.allowed and denied.kind == "tpm"
        # both RPM tokens must still be available
        assert (await limiter.check(t, est_tokens=10)).allowed
        assert (await limiter.check(t, est_tokens=10)).allowed


class TestBudget:
    async def test_warn_once_at_80_percent_then_block_at_100(self, redis):
        slack = SlackNotifier(webhook_url="")  # capture only
        guard = RedisBudgetGuard(redis, slack=slack)
        t = team(team_id=7, daily=1.00)

        await guard.charge(t, 0.50)
        assert (await guard.check(t)).allowed
        assert slack.sent == []

        await guard.charge(t, 0.35)  # 85%
        assert (await guard.check(t)).allowed
        assert len(slack.sent) == 1 and "85%" in slack.sent[0]

        await guard.charge(t, 0.05)  # 90% — no second warning the same day
        assert len(slack.sent) == 1

        await guard.charge(t, 0.15)  # 105%
        verdict = await guard.check(t)
        assert not verdict.allowed
        assert "Daily budget exhausted" in (verdict.reason or "")

    async def test_monthly_block(self, redis):
        guard = RedisBudgetGuard(redis, slack=None)
        t = team(team_id=8, daily=0, monthly=1.00)  # daily cap disabled
        await guard.charge(t, 1.20)
        verdict = await guard.check(t)
        assert not verdict.allowed
        assert "Monthly" in (verdict.reason or "")
