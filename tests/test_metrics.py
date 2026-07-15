"""C9: the pipeline drives the contractual metric names end-to-end."""

from __future__ import annotations

import httpx
import pytest
from asgi_lifespan import LifespanManager

from gateway.obs.metrics import RelayMetrics
from tests.conftest import auth, build_harness, chat_body


@pytest.fixture
async def metered_harness():
    metrics = RelayMetrics()
    h = await build_harness(pipeline_kwargs={"metrics": metrics})
    h.deps.extras["metrics"] = metrics
    async with LifespanManager(h.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=h.app), base_url="http://relay"
        ) as client:
            h.client = client
            yield h, metrics


async def test_request_metrics_flow(metered_harness):
    h, metrics = metered_harness
    r = await h.client.post("/v1/chat/completions", json=chat_body("What is Python?"),
                            headers=auth())
    assert r.status_code == 200
    await h.drain()

    exported = metrics.export().decode()
    assert 'relay_requests_total{cache="miss"' in exported
    assert "relay_overhead_seconds_bucket" in exported
    assert 'relay_cost_usd_total{attribution="actual",team="demo"}' in exported
    assert 'relay_cost_usd_total{attribution="counterfactual",team="demo"}' in exported
    assert "relay_latency_seconds_bucket" in exported

    # /metrics endpoint serves the same registry
    scraped = (await h.client.get("/metrics")).text
    assert "relay_requests_total" in scraped


async def test_breaker_gauge_names(metered_harness):
    _, metrics = metered_harness
    metrics.set_breaker("mock/cheap-a", "open")
    exported = metrics.export().decode()
    assert 'relay_breaker_state{model="cheap-a",provider="mock"} 2.0' in exported
    metrics.verification_disagreement(1)
    assert 'relay_verification_disagreements_total{tier="1"}' in metrics.export().decode()
