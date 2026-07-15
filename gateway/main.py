"""Relay gateway — FastAPI app factory + production wiring.

`create_app(deps=...)` lets tests inject an in-memory team store, a mock
adapter bound to an in-process chaos provider, and a null request logger.
With no injected deps, full production wiring happens at construction time
(gateway/bootstrap.py); anything needing a running loop initializes in the
lifespan.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

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
    """Wire real infrastructure. Clients are lazy — nothing connects here."""
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
    production = deps is None
    if production:
        from gateway.bootstrap import attach_runtime_components
        from gateway.obs.otel import setup_tracing

        setup_tracing()
        deps = build_production_deps(settings)
        attach_runtime_components(deps, settings)
    if deps.pipeline is None:
        deps.pipeline = Pipeline(deps.registry, deps.request_logger)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for coro in deps.extras.get("async_init", []):
            await coro
        for factory in deps.extras.get("task_factories", []):
            deps.background_tasks.append(factory())
        app.state.deps = deps
        app.state.team_store = deps.team_store
        app.state.pipeline = deps.pipeline
        try:
            yield
        finally:
            await deps.pipeline.drain()
            for task in deps.background_tasks:
                task.cancel()
            if deps.http_client is not None:
                await deps.http_client.aclose()
            if deps.redis is not None:
                await deps.redis.aclose()
            if deps.engine is not None:
                await deps.engine.dispose()

    app = FastAPI(title="relay", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        relay_metrics = deps.extras.get("metrics")
        if relay_metrics is None:
            return Response("# no metrics wired\n", media_type="text/plain")
        return Response(relay_metrics.export(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models(team: TeamContext = Depends(authenticate)) -> dict[str, Any]:
        registry: ProviderRegistry = deps.registry
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

    if "routing_store" in deps.extras:
        from admin.api import create_admin_router

        session_factory = deps.extras.get("session_factory")

        async def audit(actor: str, path: str, old: Any, new: Any) -> None:
            if session_factory is None:
                logger.info("audit (no db): %s %s", actor, path)
                return
            from gateway.db import ConfigAudit

            async with session_factory() as session:
                session.add(ConfigAudit(actor=actor, path=path, old=old, new=new))
                await session.commit()

        app.include_router(create_admin_router(
            admin_key=deps.extras.get("admin_key", settings.admin_key),
            routing_store=deps.extras["routing_store"],
            audit=deps.extras.get("audit_writer", audit),
            cache=deps.extras.get("cache"),
            session_factory=session_factory,
        ))

    return app


# Run with: uvicorn "gateway.main:create_app" --factory  (docker-compose does)
