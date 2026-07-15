"""C8: table-driven breaker FSM, retry taxonomy, fallback selection, health.

The breaker table encodes every transition the SPEC requires:
closed->open->half_open->closed and half_open->open, plus the hysteresis
rules (window expiry, single probe, cooldown gating).
"""

from __future__ import annotations

import httpx
import pytest

from gateway.adapters import MockAdapter
from gateway.adapters.base import AdapterError
from gateway.models import ChatCompletionRequest
from gateway.registry import ProviderRegistry
from gateway.resilience.breaker import (
    BreakerConfig,
    BreakerRegistry,
    BreakerState,
    CircuitBreaker,
)
from gateway.resilience.fallback import AllProvidersFailed, ResilientCaller
from gateway.resilience.health import HealthTracker
from gateway.resilience.retry import RetryConfig, backoff_delay
from gateway.spend import SpendGuard
from mockprovider.app import create_app as create_mock_app


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# Each step: (action, arg, expected_state_after)
#   fail/ok        — record a call outcome
#   tick           — advance the clock by `arg` seconds
#   allow          — assert allow() returns `arg` (drives open->half_open)
BREAKER_TABLE = [
    (
        "closed -> open at threshold, then half-open probe succeeds -> closed",
        [
            ("fail", None, "closed"),
            ("fail", None, "closed"),
            ("allow", True, "closed"),  # still serving while under threshold
            ("fail", None, "open"),  # 3rd failure in window trips it
            ("allow", False, "open"),  # skipped while cooling down
            ("tick", 9.0, "open"),
            ("allow", False, "open"),  # cooldown not elapsed
            ("tick", 1.5, "open"),
            ("allow", True, "half_open"),  # cooldown elapsed -> single probe
            ("allow", False, "half_open"),  # second caller gets no probe
            ("ok", None, "closed"),  # probe success closes
            ("allow", True, "closed"),
        ],
    ),
    (
        "half-open probe fails -> open again with fresh cooldown",
        [
            ("fail", None, "closed"),
            ("fail", None, "closed"),
            ("fail", None, "open"),
            ("tick", 10.0, "open"),
            ("allow", True, "half_open"),
            ("fail", None, "open"),  # probe failure re-opens
            ("allow", False, "open"),  # new cooldown applies
            ("tick", 10.0, "open"),
            ("allow", True, "half_open"),
            ("ok", None, "closed"),
        ],
    ),
    (
        "window expiry: stale failures do not accumulate",
        [
            ("fail", None, "closed"),
            ("fail", None, "closed"),
            ("tick", 31.0, "closed"),  # window is 30s
            ("fail", None, "closed"),  # old failures aged out -> only 1 in window
            ("allow", True, "closed"),
        ],
    ),
    (
        "success in closed clears the failure window",
        [
            ("fail", None, "closed"),
            ("fail", None, "closed"),
            ("ok", None, "closed"),
            ("fail", None, "closed"),
            ("fail", None, "closed"),
            ("allow", True, "closed"),  # 2 < threshold after the reset
            ("fail", None, "open"),
        ],
    ),
]


class TestBreakerFSM:
    @pytest.mark.parametrize("name,steps", BREAKER_TABLE, ids=[t[0] for t in BREAKER_TABLE])
    def test_table(self, name, steps):
        clock = FakeClock()
        transitions: list[tuple[str, str]] = []
        breaker = CircuitBreaker(
            name="mock/cheap-a",
            config=BreakerConfig(failure_threshold=3, window_s=30.0, cooldown_s=10.0),
            clock=clock,
            on_transition=lambda _n, old, new, _r: transitions.append((old.value, new.value)),
        )
        for i, (action, arg, expected) in enumerate(steps):
            if action == "fail":
                breaker.record_failure()
            elif action == "ok":
                breaker.record_success()
            elif action == "tick":
                clock.advance(arg)
            elif action == "allow":
                assert breaker.allow() is arg, f"step {i}: allow() != {arg}"
            assert breaker.state.value == expected, (
                f"step {i} ({action}): state {breaker.state.value} != {expected}"
            )

    def test_every_transition_reported(self):
        clock = FakeClock()
        seen: list[tuple[str, str]] = []
        breaker = CircuitBreaker(
            name="x", config=BreakerConfig(failure_threshold=1, cooldown_s=5.0),
            clock=clock, on_transition=lambda _n, old, new, _r: seen.append((old.value, new.value)),
        )
        breaker.record_failure()
        clock.advance(5.0)
        breaker.allow()
        breaker.record_success()
        assert seen == [("closed", "open"), ("open", "half_open"), ("half_open", "closed")]


class TestRetryPolicy:
    def test_backoff_schedule_with_jitter_bounds(self):
        config = RetryConfig()
        for attempt, base in enumerate([0.5, 1.0, 2.0]):
            for _ in range(50):
                delay = backoff_delay(config, attempt)
                assert base * 0.5 <= delay <= base * 1.5


def request_for(prompt: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="relay-auto",
                                 messages=[{"role": "user", "content": prompt}])


@pytest.fixture
async def resilient_setup():
    mock_app = create_mock_app()
    mock_app.state.chaos.base_latency_ms = 0
    mock_app.state.chaos.latency_jitter_ms = 0
    mock_app.state.chaos.stream_tokens_per_sec = 100_000
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app))
    registry = ProviderRegistry()
    registry.register(MockAdapter(http, SpendGuard(5.0), base_url="http://mockprov"))

    async def no_sleep(_s: float) -> None:
        return None

    clock = FakeClock()
    breakers = BreakerRegistry(
        BreakerConfig(failure_threshold=3, window_s=30.0, cooldown_s=10.0), clock=clock
    )
    health = HealthTracker(clock=clock)
    caller = ResilientCaller(registry, breakers,
                             RetryConfig(max_retries=2), health=health, sleep=no_sleep)
    yield mock_app, caller, breakers, clock, health
    await http.aclose()


class TestResilientCaller:
    async def test_happy_path_no_fallback(self, resilient_setup):
        _, caller, _, _, _ = resilient_setup
        result, meta = await caller.call(["mock/cheap-a", "mock/mid-b"], request_for("hi"))
        assert meta.model_key == "mock/cheap-a"
        assert not meta.fallback_used and meta.retries == 0

    async def test_retries_then_fallback_within_chain(self, resilient_setup):
        mock_app, caller, breakers, _, _ = resilient_setup
        # cheap-a permanently erroring; mid-b healthy (same app, so flip
        # error off after cheap-a's attempts by counting calls)
        calls = {"n": 0}
        original = mock_app.state.chaos

        # error only while the request model is cheap-a
        from fastapi import Request

        @mock_app.middleware("http")
        async def selective_chaos(request: Request, call_next):
            if request.url.path.endswith("/chat/completions"):
                body = await request.json()
                if body.get("model") == "cheap-a":
                    calls["n"] += 1
                    from fastapi.responses import JSONResponse

                    return JSONResponse({"error": {"message": "boom"}}, status_code=500)
            return await call_next(request)

        result, meta = await caller.call(["mock/cheap-a", "mock/mid-b"], request_for("hi"))
        assert meta.model_key == "mock/mid-b"
        assert meta.fallback_used
        assert meta.retries == 2  # exhausted retries on cheap-a first
        assert calls["n"] == 3  # 1 attempt + 2 retries
        assert original.error_rate == 0.0  # chaos config untouched

    async def test_breaker_opens_after_repeated_failures_and_skips(self, resilient_setup):
        mock_app, caller, breakers, clock, _ = resilient_setup
        mock_app.state.chaos.error_rate = 1.0
        for _ in range(2):
            with pytest.raises(AdapterError):
                await caller.call(["mock/cheap-a"], request_for("hi"))
        assert breakers.get("mock/cheap-a").state is BreakerState.OPEN

        # while open, the chain skips straight to the next model
        mock_app.state.chaos.error_rate = 0.0
        result, meta = await caller.call(["mock/cheap-a", "mock/mid-b"], request_for("hi"))
        assert meta.model_key == "mock/mid-b"
        assert meta.breaker_open

        # after cooldown the probe closes it again
        clock.advance(10.0)
        result, meta = await caller.call(["mock/cheap-a"], request_for("hi"))
        assert meta.model_key == "mock/cheap-a"
        assert breakers.get("mock/cheap-a").state is BreakerState.CLOSED

    async def test_non_retryable_surfaces_immediately(self, resilient_setup):
        _, caller, _, _, _ = resilient_setup

        def unauthorized(_request):
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        registry = ProviderRegistry()
        async with httpx.AsyncClient(transport=httpx.MockTransport(unauthorized)) as http:
            registry.register(MockAdapter(http, SpendGuard(5.0), base_url="http://x"))
            caller2 = ResilientCaller(caller.registry.__class__ and registry,
                                      BreakerRegistry(), RetryConfig(), sleep=caller.sleep)
            with pytest.raises(AdapterError) as exc:
                await caller2.call(["mock/cheap-a", "mock/mid-b"], request_for("hi"))
            assert exc.value.status_code == 401
            assert not isinstance(exc.value, AllProvidersFailed)  # no fallback happened

    async def test_all_failed_raises_chain_error(self, resilient_setup):
        mock_app, caller, _, _, _ = resilient_setup
        mock_app.state.chaos.hard_down = True
        with pytest.raises(AllProvidersFailed):
            await caller.call(["mock/cheap-a", "mock/mid-b"], request_for("hi"))

    async def test_stream_open_falls_back(self, resilient_setup):
        mock_app, caller, _, _, _ = resilient_setup
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @mock_app.middleware("http")
        async def cheap_a_down(request: Request, call_next):
            if request.url.path.endswith("/chat/completions"):
                body = await request.json()
                if body.get("model") == "cheap-a":
                    return JSONResponse({"error": {"message": "boom"}}, status_code=503)
            return await call_next(request)

        aiter, collector, meta, adapter, model = await caller.open_stream(
            ["mock/cheap-a", "mock/mid-b"], request_for("stream me")
        )
        chunks = [c async for c in aiter]
        assert chunks
        assert meta.model_key == "mock/mid-b" and meta.fallback_used
        assert collector.complete()
        assert collector.content


class TestHealthTracker:
    def test_status_thresholds(self):
        clock = FakeClock()
        tracker = HealthTracker(clock=clock)
        for _ in range(20):
            tracker.record("mock/cheap-a", ok=True, latency_s=0.05)
        assert tracker.evaluate("mock/cheap-a") == "healthy"
        for _ in range(4):
            tracker.record("mock/cheap-a", ok=False, latency_s=0.0)
        assert tracker.evaluate("mock/cheap-a") == "degraded"  # 4/24 ~ 17%
        for _ in range(30):
            tracker.record("mock/cheap-a", ok=False, latency_s=0.0)
        assert tracker.evaluate("mock/cheap-a") == "down"
        changes = tracker.sweep()
        assert ("mock/cheap-a", "healthy", "down") in changes
