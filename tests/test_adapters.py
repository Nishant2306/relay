"""Adapter layer: cost math, MAX_DAILY_SPEND kill-switch, streaming tee, error taxonomy."""

from __future__ import annotations

import httpx
import pytest

from gateway.adapters import MockAdapter, OpenAIAdapter
from gateway.adapters.base import AdapterError, StreamCollector
from gateway.models import ChatCompletionRequest
from gateway.pricing import cost_usd, counterfactual_cost_usd, is_real_spend
from gateway.spend import SpendCapExceeded, SpendGuard
from mockprovider.app import create_app


def request_for(prompt: str, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="mock/cheap-a",
        messages=[{"role": "user", "content": prompt}],
        stream=stream,
    )


@pytest.fixture
def mock_app():
    return create_app()


@pytest.fixture
async def mock_adapter(mock_app):
    transport = httpx.ASGITransport(app=mock_app)
    async with httpx.AsyncClient(transport=transport) as http:
        yield MockAdapter(http, SpendGuard(cap_usd=5.0), base_url="http://mock")


class TestPricing:
    def test_cost_math(self):
        # 1M in + 1M out at gpt-4o = $2.50 + $10.00
        assert cost_usd("openai/gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.50)
        assert cost_usd("ollama/llama3.1-8b", 1_000_000, 1_000_000) == 0.0

    def test_counterfactual_is_flagship_price(self):
        assert counterfactual_cost_usd(1000, 500) == pytest.approx(
            cost_usd("openai/gpt-4o", 1000, 500)
        )

    def test_unknown_model_rejected(self):
        with pytest.raises(KeyError):
            cost_usd("nope/unknown", 1, 1)

    def test_real_spend_taxonomy(self):
        assert is_real_spend("openai/gpt-4o")
        assert is_real_spend("anthropic/claude-haiku")
        assert not is_real_spend("mock/cheap-a")
        assert not is_real_spend("ollama/llama3.1-8b")


class TestSpendGuard:
    async def test_kill_switch_blocks_paid_calls_at_cap(self):
        guard = SpendGuard(cap_usd=0.01)
        await guard.charge("openai/gpt-4o", 0.02)
        with pytest.raises(SpendCapExceeded):
            await guard.check("openai/gpt-4o")

    async def test_free_providers_never_blocked(self):
        guard = SpendGuard(cap_usd=0.0)
        await guard.check("mock/cheap-a")  # no raise
        await guard.check("ollama/llama3.1-8b")  # no raise

    async def test_free_spend_not_recorded_as_real(self):
        guard = SpendGuard(cap_usd=1.0)
        await guard.charge("mock/top-c", 5.0)  # simulated dollars
        assert await guard.spent_today() == 0.0


class TestOpenAIAdapterKillSwitch:
    async def test_precall_gate_fires_before_any_http(self):
        def explode(_request):
            raise AssertionError("HTTP request should never be sent past the kill-switch")

        transport = httpx.MockTransport(explode)
        guard = SpendGuard(cap_usd=0.0)
        await guard.charge("openai/gpt-4o", 0.01)
        async with httpx.AsyncClient(transport=transport) as http:
            adapter = OpenAIAdapter(http, guard, base_url="https://api.openai.com", api_key="k")
            with pytest.raises(SpendCapExceeded):
                await adapter.chat(request_for("hi"), "gpt-4o")


class TestMockAdapter:
    async def test_chat_returns_cost_and_usage(self, mock_adapter):
        result = await mock_adapter.chat(request_for("What is Python?"), "cheap-a")
        assert result.provider == "mock"
        assert result.finish_reason == "stop"
        assert result.usage.total_tokens > 0
        expected = cost_usd("mock/cheap-a", result.usage.prompt_tokens,
                            result.usage.completion_tokens)
        assert result.cost_usd == pytest.approx(expected)

    async def test_simulated_cost_never_hits_real_spend(self, mock_adapter):
        await mock_adapter.chat(request_for("What is Python?"), "cheap-a")
        assert await mock_adapter.spend_guard.spent_today() == 0.0

    async def test_upstream_500_is_retryable_error(self, mock_app, mock_adapter):
        mock_app.state.chaos.error_rate = 1.0
        with pytest.raises(AdapterError) as exc:
            await mock_adapter.chat(request_for("x"), "cheap-a")
        assert exc.value.retryable
        assert exc.value.status_code == 500

    async def test_401_is_not_retryable(self, mock_app, mock_adapter):
        # simulate a non-retryable auth failure via a custom transport
        def unauthorized(_request):
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(unauthorized)) as http:
            adapter = MockAdapter(http, SpendGuard(5.0), base_url="http://mock")
            with pytest.raises(AdapterError) as exc:
                await adapter.chat(request_for("x"), "cheap-a")
        assert not exc.value.retryable

    async def test_stream_collector_tees_full_content(self, mock_app, mock_adapter):
        mock_app.state.chaos.stream_tokens_per_sec = 100_000
        mock_app.state.chaos.base_latency_ms = 0

        non_stream = await mock_adapter.chat(request_for("stream me"), "cheap-a")
        expected = non_stream.response["choices"][0]["message"]["content"]

        collector = StreamCollector()
        chunks = [
            c async for c in mock_adapter.stream_chat(request_for("stream me", stream=True),
                                                      "cheap-a", collector)
        ]
        assert len(chunks) > 2
        assert collector.complete()
        assert collector.content == expected

        result = await mock_adapter.finalize_stream(
            request_for("stream me"), "cheap-a", collector, latency_ms=5
        )
        assert result.usage.completion_tokens == non_stream.usage.completion_tokens
        assert result.cost_usd == pytest.approx(non_stream.cost_usd)
