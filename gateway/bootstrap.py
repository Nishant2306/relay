"""Production wiring: attach every runtime component to the Deps container.

Called synchronously at app-construction time; anything requiring a running
event loop (index creation, file watchers, prober loop) is deferred into
`deps.extras["async_init"]` / `deps.extras["task_factories"]`, which the
lifespan executes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from gateway.cache.service import SemanticCache
from gateway.config import Settings
from gateway.main import Deps
from gateway.middleware.budget import RedisBudgetGuard
from gateway.middleware.ratelimit import RedisRateLimiter
from gateway.obs.metrics import RelayMetrics
from gateway.obs.slack import SlackNotifier
from gateway.pipeline import Pipeline
from gateway.resilience.breaker import BreakerRegistry, BreakerState
from gateway.resilience.fallback import ResilientCaller
from gateway.resilience.health import HealthTracker, prober_loop
from gateway.router.classifier import ComplexityClassifier, HeuristicClassifier
from gateway.router.routes import RoutingConfigStore, TierRouter
from verifier.queue import RedisVerifyQueue, VerifierSampler

logger = logging.getLogger("relay.bootstrap")


def attach_runtime_components(deps: Deps, cfg: Settings) -> None:
    metrics = RelayMetrics()
    slack = SlackNotifier(cfg.slack_webhook_url, http=deps.http_client)
    store = RoutingConfigStore(cfg.routing_config_path)

    limiter = RedisRateLimiter(deps.redis)
    budget = RedisBudgetGuard(deps.redis, slack=slack)
    cache = SemanticCache(deps.redis, thresholds=store.thresholds,
                          ttl_config=store.ttl_config)

    classifier: ComplexityClassifier | HeuristicClassifier
    try:
        classifier = ComplexityClassifier()
        logger.info("loaded trained complexity classifier")
    except Exception:
        classifier = HeuristicClassifier()
        logger.warning(
            "models/classifier.joblib not found — using the heuristic fallback "
            "classifier (run `make train`); routing quality is reduced"
        )
    router = TierRouter(classifier, store)

    session_factory = deps.extras.get("session_factory")

    def on_breaker_transition(name: str, old: BreakerState, new: BreakerState,
                              reason: str) -> None:
        metrics.set_breaker(name, new.value)
        logger.warning("breaker %s: %s -> %s (%s)", name, old.value, new.value, reason)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(slack.send(
            f":electric_plug: breaker *{name}*: {old.value} -> {new.value} ({reason})"
        ))
        if session_factory is not None:
            provider, model = name.split("/", 1)
            loop.create_task(_record_health_event(
                session_factory, provider, model, old.value, new.value, reason
            ))

    breakers = BreakerRegistry(on_transition=on_breaker_transition)
    health = HealthTracker()
    caller = ResilientCaller(deps.registry, breakers, health=health)

    sampler = VerifierSampler(
        RedisVerifyQueue(deps.redis),
        sample_rate=lambda: store.config.verification.sample_rate,
    )

    deps.pipeline = Pipeline(
        deps.registry, deps.request_logger,
        limiter=limiter, budget=budget, cache=cache, router=router,
        caller=caller, verifier_queue=sampler, metrics=metrics,
    )
    deps.extras.update({
        "metrics": metrics, "routing_store": store, "cache": cache,
        "breakers": breakers, "health": health, "slack": slack,
        "dev_curve_path": Path("models/trap_dev_curve.json"),
    })

    async def on_health_change(model_key: str, old: str, new: str) -> None:
        logger.warning("health %s: %s -> %s", model_key, old, new)
        await slack.send(f":stethoscope: provider *{model_key}*: {old} -> {new}")
        if session_factory is not None:
            provider, model = model_key.split("/", 1)
            await _record_health_event(session_factory, provider, model, old, new,
                                       "rolling-window evaluation")

    deps.extras["async_init"] = [cache.start()]
    deps.extras["task_factories"] = [
        lambda: asyncio.create_task(store.watch()),
        lambda: asyncio.create_task(prober_loop(health, on_health_change)),
    ]


async def _record_health_event(session_factory, provider: str, model: str,
                               from_state: str, to_state: str, reason: str) -> None:
    from gateway.db import ProviderHealthEvent

    try:
        async with session_factory() as session:
            session.add(ProviderHealthEvent(
                provider=provider, model=model, from_state=from_state,
                to_state=to_state, reason=reason,
            ))
            await session.commit()
    except Exception:
        logger.exception("failed to record provider_health_event")
