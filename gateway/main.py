"""Relay gateway — FastAPI app factory + production wiring.

`create_app(deps=...)` lets tests inject an in-memory team store, a mock
adapter bound to an in-process chaos provider, and a null request logger.
Production wiring (`build_production_deps`) connects Redis, Postgres, and all
configured providers; components that land in later milestones (cache, router,
resilience, verifier) attach to the same Deps object.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from gateway.config import Settings, settings
from gateway.middleware.auth import TeamStore, authenticate, check_model_allowed
from gateway.models import ChatCompletionRequest, TeamContext
from gateway.obs.logging import NullRequestLogger, RequestLogger
from gateway.pipeline import AUTO_MODELS, Pipeline
from gateway.registry import ProviderRegistry

logger = logging.getLogger("relay")


@dataclass
class Deps:
    registry: ProviderRegistry
    team_store: TeamStore
    request_logger: RequestLogger
    pipeline: Pipeline | None = None
    http_client: httpx.AsyncClient | None = None
    redis: Any = None
    engine: Any = None
    extras: dict[str, Any] = field(default_factory=dict)
    background_tasks: list[Any] = field(default_factory=list)


def build_production_deps(cfg: Settings) -> Deps:
    """Wire real infrastructure. Called from the lifespan, once per process."""
    from redis.asyncio import Redis

    from gateway.adapters import AnthropicAdapter, MockAdapter, OllamaAdapter, OpenAIAdapter
    from gateway.db import make_engine, make_session_factory
    from gateway.middleware.auth import PostgresTeamStore
    from gateway.obs.logging import PostgresRequestLogger
    from gateway.spend import SpendGuard

    http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    redis = Redis.from_url(cfg.redis_url, decode_responses=False)
    engine = make_engine(cfg.database_url)
    session_factory = make_session_factory(engine)
    spend_guard = SpendGuard(cfg.max_daily_spend_usd, redis=redis)

    registry = ProviderRegistry()
    registry.register(MockAdapter(http_client, spend_guard, base_url=cfg.mock_provider_url))
    registry.register(OllamaAdapter(http_client, spend_guard, base_url=cfg.ollama_base_url))
    if cfg.openai_api_key:
        registry.register(OpenAIAdapter(
            http_client, spend_guard, base_url=cfg.openai_base_url, api_key=cfg.openai_api_key
        ))
    if cfg.anthropic_api_key:
        registry.register(AnthropicAdapter(
            http_client, spend_guard, base_url=cfg.anthropic_base_url,
            api_key=cfg.anthropic_api_key,
        ))

    deps = Deps(
        registry=registry,
        team_store=PostgresTeamStore(session_factory),
        request_logger=PostgresRequestLogger(session_factory),
        http_client=http_client,
        redis=redis,
        engine=engine,
    )
    deps.extras["session_factory"] = session_factory
    deps.extras["spend_guard"] = spend_guard
    deps.extras["settings"] = cfg
    return deps


def create_app(deps: Deps | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        d = deps
        if d is None:
            d = build_production_deps(settings)
            attach_runtime_components(d)
        if d.pipeline is None:
            d.pipeline = Pipeline(d.registry, d.request_logger)
        app.state.deps = d
        app.state.team_store = d.team_store
        app.state.pipeline = d.pipeline
        try:
            yield
        finally:
            await d.pipeline.drain()
            for task in d.background_tasks:
                task.cancel()
            if d.http_client is not None:
                await d.http_client.aclose()
            if d.redis is not None:
                await d.redis.aclose()
            if d.engine is not None:
                await d.engine.dispose()

    app = FastAPI(title="relay", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(team: TeamContext = Depends(authenticate)) -> dict[str, Any]:
        registry: ProviderRegistry = app.state.deps.registry
        data = [
            {"id": key, "object": "model", "created": 0, "owned_by": "relay"}
            for key in sorted(registry.available_models())
        ]
        data.append({"id": "relay-auto", "object": "model", "created": 0, "owned_by": "relay"})
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request, team: TeamContext = Depends(authenticate)
    ) -> Response:
        started = time.perf_counter()
        try:
            body = await request.json()
            chat_request = ChatCompletionRequest.model_validate(body)
        except Exception as e:
            return JSONResponse(
                {"error": {"message": f"Invalid request body: {e}", "type":
                           "invalid_request_error", "param": None, "code": "invalid_body"}},
                status_code=400,
            )
        if chat_request.model not in AUTO_MODELS:
            check_model_allowed(team, chat_request.model)
        pipeline: Pipeline = app.state.pipeline
        response = await pipeline.handle(chat_request, team)
        response.headers.setdefault(
            "x-relay-overhead-ms", str(int((time.perf_counter() - started) * 1000))
        )
        return response

    return app


def attach_runtime_components(deps: Deps) -> None:
    """Attach M2/M3 components (limits, budgets, cache, router, resilience).

    Implemented incrementally; safe to call when subsystems are unavailable —
    each component degrades to None and the pipeline skips it.
    """
    # Populated in later milestones (see gateway/bootstrap.py once it exists).


app = create_app()
