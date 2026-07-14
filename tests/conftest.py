"""Shared test harness: gateway wired to an in-process chaos mock provider.

No network, no Docker — the mock provider is mounted via httpx.ASGITransport.
Integration tests that need real Redis/Postgres are marked `integration` and
build their own containers.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from gateway.adapters import MockAdapter
from gateway.main import Deps, create_app
from gateway.middleware.auth import InMemoryTeamStore
from gateway.models import TeamContext
from gateway.obs.logging import NullRequestLogger
from gateway.pipeline import Pipeline
from gateway.registry import ProviderRegistry
from gateway.spend import SpendGuard
from mockprovider.app import create_app as create_mock_app

DEMO_KEY = "relay-test-demo-key"
RESTRICTED_KEY = "relay-test-restricted-key"


@dataclass
class Harness:
    app: FastAPI
    client: httpx.AsyncClient
    mock_app: FastAPI
    deps: Deps
    request_log: NullRequestLogger

    @property
    def chaos(self):
        return self.mock_app.state.chaos

    async def drain(self) -> None:
        assert self.deps.pipeline is not None
        await self.deps.pipeline.drain()


def default_teams() -> dict[str, TeamContext]:
    return {
        DEMO_KEY: TeamContext(
            team_id=1, name="demo", rpm=1000, tpm=1_000_000,
            daily_budget_usd=100.0, monthly_budget_usd=1000.0,
            allowed_models=["*"], cache_scope="team",
        ),
        RESTRICTED_KEY: TeamContext(
            team_id=2, name="restricted", rpm=1000, tpm=1_000_000,
            daily_budget_usd=100.0, monthly_budget_usd=1000.0,
            allowed_models=["mock/cheap-a"], cache_scope="team",
        ),
    }


async def build_harness(pipeline_kwargs: dict | None = None,
                        teams: dict[str, TeamContext] | None = None) -> Harness:
    mock_app = create_mock_app()
    mock_app.state.chaos.base_latency_ms = 0
    mock_app.state.chaos.latency_jitter_ms = 0
    mock_app.state.chaos.stream_tokens_per_sec = 100_000

    provider_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app))
    registry = ProviderRegistry()
    registry.register(MockAdapter(provider_http, SpendGuard(5.0), base_url="http://mockprov"))

    store = InMemoryTeamStore(teams or default_teams())
    request_log = NullRequestLogger()
    pipeline = Pipeline(registry, request_log, **(pipeline_kwargs or {}))
    deps = Deps(
        registry=registry, team_store=store, request_logger=request_log,
        pipeline=pipeline, http_client=provider_http,
    )
    app = create_app(deps)
    return Harness(app=app, client=None, mock_app=mock_app, deps=deps,  # type: ignore[arg-type]
                   request_log=request_log)


@pytest.fixture
async def harness():
    h = await build_harness()
    async with LifespanManager(h.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=h.app), base_url="http://relay"
        ) as client:
            h.client = client
            yield h


def auth(key: str = DEMO_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def chat_body(prompt: str, model: str = "mock/cheap-a", stream: bool = False, **extra) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        **extra,
    }
