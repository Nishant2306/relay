"""The request pipeline (SPEC §2) — one linear lifecycle per request.

    auth (done in the endpoint) -> rate limit -> budget -> cache lookup ->
    classify -> route -> provider call (resilience) -> respond ->
    async: verify sample, cache write, request log

Stages are pluggable: M1 ships auth + provider call + logging; rate limits,
budgets, cache, router, and the resilient caller land in M2/M3 by injection,
so this file is the stable spine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol

from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from gateway.adapters.base import AdapterError, StreamCollector
from gateway.models import AdapterResult, ChatCompletionRequest, TeamContext, approx_tokens
from gateway.obs.logging import RequestLogEntry, RequestLogger
from gateway.pricing import counterfactual_cost_usd
from gateway.registry import ProviderRegistry
from gateway.spend import SpendCapExceeded

logger = logging.getLogger("relay.pipeline")

AUTO_MODELS = {"auto", "relay-auto"}


class CallMeta(BaseModel):
    model_key: str
    retries: int = 0
    fallback_used: bool = False
    breaker_open: bool = False


class RouteDecision(BaseModel):
    tier: int
    chain: list[str]
    confidence: float
    promoted: bool = False


class LimitDecision(BaseModel):
    allowed: bool
    kind: str | None = None  # 'rpm' | 'tpm'
    retry_after_s: float | None = None


class BudgetDecision(BaseModel):
    allowed: bool
    reason: str | None = None


class CacheHit(BaseModel):
    response: dict[str, Any]
    kind: str  # 'exact' | 'semantic'
    similarity: float | None = None
    model_key: str
    tokens_in: int
    tokens_out: int


class Limiter(Protocol):
    async def check(self, team: TeamContext, est_tokens: int) -> LimitDecision: ...
    async def reconcile(self, team: TeamContext, est_tokens: int, actual_tokens: int) -> None: ...


class BudgetGuard(Protocol):
    async def check(self, team: TeamContext) -> BudgetDecision: ...
    async def charge(self, team: TeamContext, cost_usd: float) -> None: ...


class SemanticCacheProtocol(Protocol):
    async def lookup(self, request: ChatCompletionRequest, team: TeamContext,
                     tier: int | None) -> CacheHit | None: ...
    async def write(self, request: ChatCompletionRequest, team: TeamContext,
                    result: AdapterResult, tier: int | None) -> None: ...
    async def release(self, request: ChatCompletionRequest, team: TeamContext) -> None: ...


class Router(Protocol):
    def decide(self, request: ChatCompletionRequest) -> RouteDecision: ...


class Caller(Protocol):
    async def call(self, chain: list[str],
                   request: ChatCompletionRequest) -> tuple[AdapterResult, CallMeta]: ...

    async def open_stream(self, chain: list[str], request: ChatCompletionRequest) -> tuple[
        AsyncIterator[dict[str, Any]], StreamCollector, CallMeta, Any, str
    ]: ...


class VerifierQueue(Protocol):
    async def maybe_enqueue(self, request: ChatCompletionRequest, result: AdapterResult,
                            tier: int | None, request_log_id: int | None) -> None: ...


class DirectCaller:
    """M1 caller: first model in the chain, no retry/fallback/breaker."""

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    async def call(self, chain, request):
        adapter, model = self.registry.resolve(chain[0])
        result = await adapter.chat(request, model)
        return result, CallMeta(model_key=chain[0])

    async def open_stream(self, chain, request):
        adapter, model = self.registry.resolve(chain[0])
        collector = StreamCollector()
        aiter = adapter.stream_chat(request, model, collector)
        return aiter, collector, CallMeta(model_key=chain[0]), adapter, model


def openai_error(status: int, message: str, err_type: str, code: str,
                 headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": err_type, "param": None, "code": code}},
        status_code=status,
        headers=headers or {},
    )


class Pipeline:
    def __init__(
        self,
        registry: ProviderRegistry,
        request_logger: RequestLogger,
        *,
        limiter: Limiter | None = None,
        budget: BudgetGuard | None = None,
        cache: SemanticCacheProtocol | None = None,
        router: Router | None = None,
        caller: Caller | None = None,
        verifier_queue: VerifierQueue | None = None,
        metrics: Any = None,
    ):
        self.registry = registry
        self.request_logger = request_logger
        self.limiter = limiter
        self.budget = budget
        self.cache = cache
        self.router = router
        self.caller = caller or DirectCaller(registry)
        self.verifier_queue = verifier_queue
        self.metrics = metrics
        self._pending: set[asyncio.Task] = set()

    # -- background bookkeeping ------------------------------------------------
    def _post(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        """Await all post-response tasks (tests + graceful shutdown)."""
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    # -- the lifecycle ----------------------------------------------------------
    async def handle(self, request: ChatCompletionRequest, team: TeamContext) -> Response:
        started = time.perf_counter()
        est_tokens = approx_tokens(
            request.full_prompt_text() + request.system_prompt()
        ) + (request.max_tokens or 512)

        # [2] rate limit
        if self.limiter is not None:
            decision = await self.limiter.check(team, est_tokens)
            if not decision.allowed:
                if self.metrics:
                    self.metrics.ratelimit_rejection(team.name, decision.kind or "rpm")
                self._post(self.request_logger.log(RequestLogEntry(
                    team_id=team.team_id, model_requested=request.model, status=429,
                    error_class=f"rate_limit_{decision.kind}", cache="bypass",
                )))
                retry_after = max(1, int(decision.retry_after_s or 1))
                return openai_error(
                    429,
                    f"Rate limit exceeded ({decision.kind}). Retry after {retry_after}s.",
                    "rate_limit_error", "rate_limit_exceeded",
                    headers={"Retry-After": str(retry_after)},
                )

        # [3] budget
        if self.budget is not None:
            verdict = await self.budget.check(team)
            if not verdict.allowed:
                self._post(self.request_logger.log(RequestLogEntry(
                    team_id=team.team_id, model_requested=request.model, status=429,
                    error_class="budget_exhausted", cache="bypass",
                )))
                return openai_error(
                    429,
                    verdict.reason or "Team budget exhausted.",
                    "rate_limit_error", "budget_exhausted",
                )

        # [5] classify (cheap; also feeds the cache threshold + verification)
        route: RouteDecision | None = self.router.decide(request) if self.router else None
        tier = route.tier if route else None

        # [4] cache lookup
        if self.cache is not None and team.cache_scope != "off":
            hit = await self.cache.lookup(request, team, tier)
            if hit is not None:
                overhead_ms = int((time.perf_counter() - started) * 1000)
                counterfactual = counterfactual_cost_usd(hit.tokens_in, hit.tokens_out)
                if self.metrics:
                    self.metrics.request(team.name, hit.model_key, tier, "hit", 200)
                    self.metrics.cache_hit(hit.kind, hit.similarity)
                    self.metrics.cost(team.name, 0.0, counterfactual)
                    self.metrics.overhead(overhead_ms / 1000)
                self._post(self.request_logger.log(RequestLogEntry(
                    team_id=team.team_id, model_requested=request.model,
                    model_served=hit.model_key, provider=hit.model_key.split("/", 1)[0],
                    tier=tier, cache="hit", cache_kind=hit.kind,
                    cache_similarity=hit.similarity, tokens_in=hit.tokens_in,
                    tokens_out=hit.tokens_out, actual_cost_usd=0.0,
                    counterfactual_cost_usd=counterfactual,
                    latency_ms=overhead_ms, overhead_ms=overhead_ms, status=200,
                )))
                headers = {
                    "x-relay-cache": "hit",
                    "x-relay-cache-kind": hit.kind,
                    "x-relay-model": hit.model_key,
                    "x-relay-cost": "0.00000000",
                    "x-relay-fallback": "false",
                    "x-relay-overhead-ms": str(overhead_ms),
                }
                if hit.similarity is not None:
                    headers["x-relay-cache-similarity"] = f"{hit.similarity:.4f}"
                if tier is not None:
                    headers["x-relay-tier"] = str(tier)
                if request.stream:
                    return StreamingResponse(
                        _replay_as_stream(hit.response), media_type="text/event-stream",
                        headers=headers,
                    )
                return JSONResponse(hit.response, headers=headers)

        # [6] route: 'auto' -> tier chain; concrete model -> that model
        if request.model in AUTO_MODELS:
            if route is None:
                return openai_error(
                    400, "model 'auto' requires the router; it is not enabled",
                    "invalid_request_error", "router_disabled",
                )
            chain = route.chain
        else:
            chain = [request.model]

        # [7] provider call + [8] respond
        try:
            if request.stream:
                return await self._respond_stream(request, team, chain, tier, started, est_tokens)
            return await self._respond_json(request, team, chain, tier, started, est_tokens)
        except HTTPException as e:
            await self._release_singleflight(request, team)
            return openai_error(e.status_code, str(e.detail), "invalid_request_error",
                                "invalid_model")
        except SpendCapExceeded as e:
            await self._release_singleflight(request, team)
            self._post(self.request_logger.log(RequestLogEntry(
                team_id=team.team_id, model_requested=request.model, tier=tier,
                status=503, error_class="spend_cap", cache="miss",
            )))
            return openai_error(503, str(e), "server_error", "relay_spend_cap")
        except AdapterError as e:
            await self._release_singleflight(request, team)
            status = 502 if (e.status_code is None or e.status_code >= 500) else e.status_code
            if self.metrics:
                self.metrics.request(team.name, f"{e.provider}/{e.model}", tier, "miss", status)
            self._post(self.request_logger.log(RequestLogEntry(
                team_id=team.team_id, model_requested=request.model,
                model_served=f"{e.provider}/{e.model}", provider=e.provider, tier=tier,
                status=status, error_class=f"upstream_{e.status_code or 'transport'}",
                cache="miss",
            )))
            return openai_error(status, str(e), "server_error", "upstream_error")

    async def _release_singleflight(self, request: ChatCompletionRequest,
                                    team: TeamContext) -> None:
        if self.cache is not None:
            try:
                await self.cache.release(request, team)
            except Exception:
                logger.exception("singleflight release failed")

    def _headers(self, meta: CallMeta, tier: int | None, cost: float,
                 overhead_ms: int) -> dict[str, str]:
        headers = {
            "x-relay-cache": "miss",
            "x-relay-model": meta.model_key,
            "x-relay-cost": f"{cost:.8f}",
            "x-relay-fallback": "true" if meta.fallback_used else "false",
            "x-relay-overhead-ms": str(overhead_ms),
        }
        if tier is not None:
            headers["x-relay-tier"] = str(tier)
        return headers

    async def _finish(self, request: ChatCompletionRequest, team: TeamContext,
                      result: AdapterResult, meta: CallMeta, tier: int | None,
                      overhead_ms: int, est_tokens: int) -> None:
        """Post-response bookkeeping shared by stream + non-stream paths."""
        counterfactual = counterfactual_cost_usd(
            result.usage.prompt_tokens, result.usage.completion_tokens
        )
        if self.limiter is not None:
            await self.limiter.reconcile(team, est_tokens, result.usage.total_tokens)
        if self.budget is not None:
            await self.budget.charge(team, result.cost_usd)
        if self.metrics:
            self.metrics.request(team.name, meta.model_key, tier, "miss", 200)
            self.metrics.cost(team.name, result.cost_usd, counterfactual)
            self.metrics.overhead(overhead_ms / 1000)
            if meta.fallback_used:
                self.metrics.fallback(meta.model_key)
        log_id = await self.request_logger.log(RequestLogEntry(
            team_id=team.team_id, model_requested=request.model, model_served=meta.model_key,
            provider=result.provider, tier=tier, cache="miss",
            tokens_in=result.usage.prompt_tokens, tokens_out=result.usage.completion_tokens,
            actual_cost_usd=result.cost_usd, counterfactual_cost_usd=counterfactual,
            latency_ms=result.latency_ms, overhead_ms=overhead_ms, status=200,
            retries=meta.retries, fallback_used=meta.fallback_used,
            breaker_open=meta.breaker_open,
            verified="pending" if (self.verifier_queue and tier in (1, 2)) else None,
        ))
        # [10] cache write — complete successful responses only (ADR-0006)
        if self.cache is not None and team.cache_scope != "off" \
                and result.finish_reason == "stop":
            try:
                await self.cache.write(request, team, result, tier)
            except Exception:
                logger.exception("cache write failed")
        # [9] verification sampling
        if self.verifier_queue is not None and tier in (1, 2):
            try:
                await self.verifier_queue.maybe_enqueue(request, result, tier, log_id)
            except Exception:
                logger.exception("verifier enqueue failed")

    async def _respond_json(self, request, team, chain, tier, started, est_tokens) -> Response:
        result, meta = await self.caller.call(chain, request)
        total_ms = int((time.perf_counter() - started) * 1000)
        overhead_ms = max(0, total_ms - result.latency_ms)
        self._post(self._finish(request, team, result, meta, tier, overhead_ms, est_tokens))
        return JSONResponse(
            result.response, headers=self._headers(meta, tier, result.cost_usd, overhead_ms)
        )

    async def _respond_stream(self, request, team, chain, tier, started, est_tokens) -> Response:
        aiter, collector, meta, adapter, model = await self.caller.open_stream(chain, request)
        # Pull the first chunk before committing to 200 so upstream failures
        # can still fall back / surface as errors.
        try:
            first = await anext(aiter)
        except StopAsyncIteration:
            first = None
        overhead_ms = int((time.perf_counter() - started) * 1000)

        async def body() -> AsyncIterator[bytes]:
            provider_start = time.perf_counter()
            try:
                if first is not None:
                    yield f"data: {json.dumps(first)}\n\n".encode()
                async for chunk in aiter:
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            except asyncio.CancelledError:
                # client disconnected mid-stream: discard buffer (ADR-0006)
                await self._release_singleflight(request, team)
                self._post(self.request_logger.log(RequestLogEntry(
                    team_id=team.team_id, model_requested=request.model,
                    model_served=meta.model_key, provider=meta.model_key.split("/", 1)[0],
                    tier=tier, cache="miss", status=499, error_class="client_disconnect",
                )))
                raise
            except AdapterError as e:
                # mid-stream provider failure: emit an SSE error event, discard buffer
                await self._release_singleflight(request, team)
                yield (
                    b'data: {"error": {"message": "upstream stream failure", '
                    b'"type": "server_error"}}\n\n'
                )
                self._post(self.request_logger.log(RequestLogEntry(
                    team_id=team.team_id, model_requested=request.model,
                    model_served=meta.model_key, provider=e.provider, tier=tier,
                    cache="miss", status=502, error_class="upstream_stream_failure",
                )))
                return
            latency_ms = int((time.perf_counter() - provider_start) * 1000)
            if collector.complete():
                result = await adapter.finalize_stream(request, model, collector, latency_ms)
                self._post(self._finish(
                    request, team, result, meta, tier, overhead_ms, est_tokens
                ))
            else:
                await self._release_singleflight(request, team)
                self._post(self.request_logger.log(RequestLogEntry(
                    team_id=team.team_id, model_requested=request.model,
                    model_served=meta.model_key, provider=meta.model_key.split("/", 1)[0],
                    tier=tier, cache="miss", status=200,
                    error_class="incomplete_stream",
                )))

        return StreamingResponse(
            body(), media_type="text/event-stream",
            headers=self._headers(meta, tier, 0.0, overhead_ms),
        )


async def _replay_as_stream(response: dict[str, Any]) -> AsyncIterator[bytes]:
    """Serve a cached non-stream response body as a minimal SSE stream."""
    content = ""
    if response.get("choices"):
        content = response["choices"][0].get("message", {}).get("content", "") or ""
    base = {
        "id": response.get("id", "chatcmpl-cache"),
        "object": "chat.completion.chunk",
        "created": response.get("created", int(time.time())),
        "model": response.get("model", ""),
    }
    first = {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                                  "finish_reason": None}]}
    final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": response.get("usage")}
    yield f"data: {json.dumps(first)}\n\n".encode()
    yield f"data: {json.dumps(final)}\n\n".encode()
    yield b"data: [DONE]\n\n"
