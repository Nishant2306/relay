"""Verifier worker (SPEC C7): pop sampled tier-1/2 responses, re-run the
prompt on the top-tier model, LLM-judge agreement 1-5, log disagreements to
routing_failures with extracted features -> the classifier retrain set.

Run standalone: python -m verifier.worker (docker-compose service `verifier`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from gateway.models import ChatCompletionRequest
from gateway.registry import ProviderRegistry
from gateway.router.features import extract_feature_dict
from verifier.judge import judge_messages, parse_score
from verifier.queue import VerifyQueue
from verifier.repo import VerifierRepo

logger = logging.getLogger("relay.verifier")


class VerifierWorker:
    def __init__(
        self,
        queue: VerifyQueue,
        repo: VerifierRepo,
        registry: ProviderRegistry,
        top_model_key=lambda: "mock/top-c",
        judge_model_key=lambda: "mock/top-c",
        agree_threshold=lambda: 4,
        metrics: Any = None,
    ):
        self.queue = queue
        self.repo = repo
        self.registry = registry
        self.top_model_key = top_model_key
        self.judge_model_key = judge_model_key
        self.agree_threshold = agree_threshold
        self.metrics = metrics
        self.processed = 0
        self.disagreements = 0

    async def process_item(self, item: dict[str, Any]) -> str:
        """Returns the verdict ('agree' | 'disagree')."""
        top_key = self.top_model_key()
        adapter, top_model = self.registry.resolve(top_key)
        shadow_request = ChatCompletionRequest(model=top_key, messages=item["messages"])
        shadow = await adapter.chat(shadow_request, top_model)
        top_answer = ""
        if shadow.response.get("choices"):
            top_answer = shadow.response["choices"][0].get("message", {}).get("content", "") or ""

        judge_key = self.judge_model_key()
        judge_adapter, judge_model = self.registry.resolve(judge_key)
        judge_request = ChatCompletionRequest(
            model=judge_key,
            messages=judge_messages(item["prompt_text"], item["response_content"], top_answer),
            max_tokens=8,
        )
        judge_result = await judge_adapter.chat(judge_request, judge_model)
        judge_reply = ""
        if judge_result.response.get("choices"):
            judge_reply = (
                judge_result.response["choices"][0].get("message", {}).get("content", "") or ""
            )

        score, source = parse_score(judge_reply, item["response_content"], top_answer)
        verdict = "agree" if score >= self.agree_threshold() else "disagree"
        log_id = item.get("log_id")
        if log_id is not None:
            await self.repo.mark_verified(log_id, verdict)
        if verdict == "disagree":
            self.disagreements += 1
            await self.repo.record_failure(
                log_id=log_id or 0,
                tier=item["tier"],
                judge_agreement=score,
                cheap_model=item["model_served"],
                top_model=top_key,
                prompt_features={
                    "prompt": item["prompt_text"],
                    "score_source": source,
                    **extract_feature_dict(item["prompt_text"]),
                },
            )
        if self.metrics and verdict == "disagree":
            self.metrics.verification_disagreement(item["tier"])
        self.processed += 1
        return verdict

    async def run_forever(self) -> None:
        logger.info("verifier worker started")
        while True:
            try:
                item = await self.queue.pop(timeout_s=5.0)
                if item is not None:
                    await self.process_item(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("verification item failed — continuing")


def main() -> None:
    import httpx
    from redis.asyncio import Redis

    from gateway.adapters import AnthropicAdapter, MockAdapter, OllamaAdapter, OpenAIAdapter
    from gateway.config import settings
    from gateway.db import make_engine, make_session_factory
    from gateway.router.routes import RoutingConfigStore
    from gateway.spend import SpendGuard
    from verifier.queue import RedisVerifyQueue
    from verifier.repo import PostgresVerifierRepo

    logging.basicConfig(level=settings.log_level)

    async def _run() -> None:
        redis = Redis.from_url(settings.redis_url)
        http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        spend = SpendGuard(settings.max_daily_spend_usd, redis=redis)
        registry = ProviderRegistry()
        registry.register(MockAdapter(http, spend, base_url=settings.mock_provider_url))
        registry.register(OllamaAdapter(http, spend, base_url=settings.ollama_base_url))
        if settings.openai_api_key:
            registry.register(OpenAIAdapter(http, spend, base_url=settings.openai_base_url,
                                            api_key=settings.openai_api_key))
        if settings.anthropic_api_key:
            registry.register(AnthropicAdapter(http, spend, base_url=settings.anthropic_base_url,
                                               api_key=settings.anthropic_api_key))
        store = RoutingConfigStore(settings.routing_config_path)
        engine = make_engine(settings.database_url)
        worker = VerifierWorker(
            queue=RedisVerifyQueue(redis),
            repo=PostgresVerifierRepo(make_session_factory(engine)),
            registry=registry,
            top_model_key=lambda: store.config.tiers[3].chain[0],
            judge_model_key=lambda: store.config.verification.judge,
            agree_threshold=lambda: store.config.verification.agree_threshold,
        )
        watch = asyncio.create_task(store.watch())
        try:
            await worker.run_forever()
        finally:
            watch.cancel()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
