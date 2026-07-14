"""C7: judge parsing, sampling behavior, and the worker's verdict/failure path."""

from __future__ import annotations

import random

import httpx
import pytest

from gateway.adapters import MockAdapter
from gateway.models import AdapterResult, ChatCompletionRequest, Usage
from gateway.registry import ProviderRegistry
from gateway.spend import SpendGuard
from mockprovider.app import create_app as create_mock_app
from verifier.judge import parse_score
from verifier.queue import InMemoryVerifyQueue, VerifierSampler
from verifier.repo import InMemoryVerifierRepo
from verifier.worker import VerifierWorker


class TestJudgeParsing:
    def test_clean_digit(self):
        assert parse_score("4", "a", "b") == (4, "judge")
        assert parse_score(" 5 ", "a", "b") == (5, "judge")
        assert parse_score("Score: 3", "a", "b") == (3, "judge")

    def test_fallback_similarity(self):
        same = "the answer is forty-two, obviously"
        score, source = parse_score("word salad no digits", same, same)
        assert source == "similarity_fallback" and score == 5
        score, source = parse_score("word salad", "completely different", "nothing alike zzz")
        assert source == "similarity_fallback" and score <= 2


def adapter_result(content: str, model: str = "cheap-a") -> AdapterResult:
    return AdapterResult(
        response={"choices": [{"message": {"role": "assistant", "content": content},
                               "finish_reason": "stop"}]},
        usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        provider="mock", model=model, cost_usd=0.0001, latency_ms=5, finish_reason="stop",
    )


def request_for(prompt: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="relay-auto",
                                 messages=[{"role": "user", "content": prompt}])


class TestSampler:
    async def test_only_tiers_1_and_2_and_rate(self):
        queue = InMemoryVerifyQueue()
        sampler = VerifierSampler(queue, sample_rate=lambda: 1.0, rng=random.Random(7))
        await sampler.maybe_enqueue(request_for("p"), adapter_result("a"), tier=3,
                                    request_log_id=1)
        assert await queue.pop(timeout_s=0.05) is None  # tier 3 never sampled
        await sampler.maybe_enqueue(request_for("p"), adapter_result("a"), tier=1,
                                    request_log_id=2)
        item = await queue.pop(timeout_s=0.5)
        assert item is not None and item["log_id"] == 2 and item["tier"] == 1
        assert item["model_served"] == "mock/cheap-a"

    async def test_zero_rate_never_samples(self):
        queue = InMemoryVerifyQueue()
        sampler = VerifierSampler(queue, sample_rate=lambda: 0.0, rng=random.Random(7))
        for i in range(50):
            await sampler.maybe_enqueue(request_for("p"), adapter_result("a"), tier=1,
                                        request_log_id=i)
        assert await queue.pop(timeout_s=0.05) is None


class TestWorker:
    @pytest.fixture
    async def worker_setup(self):
        mock_app = create_mock_app()
        mock_app.state.chaos.base_latency_ms = 0
        mock_app.state.chaos.latency_jitter_ms = 0
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app))
        registry = ProviderRegistry()
        registry.register(MockAdapter(http, SpendGuard(5.0), base_url="http://mockprov"))
        repo = InMemoryVerifierRepo()
        worker = VerifierWorker(
            queue=InMemoryVerifyQueue(), repo=repo, registry=registry,
            top_model_key=lambda: "mock/top-c", judge_model_key=lambda: "mock/top-c",
            agree_threshold=lambda: 4,
        )
        yield worker, repo
        await http.aclose()

    async def test_disagreement_records_failure_with_features(self, worker_setup):
        worker, repo = worker_setup
        # mock top-c's answer will be deterministic gibberish unrelated to the
        # cheap answer -> similarity fallback -> low score -> disagree
        verdict = await worker.process_item({
            "log_id": 42, "tier": 1,
            "messages": [{"role": "user", "content": "Analyze the trade-offs of caching."}],
            "prompt_text": "Analyze the trade-offs of caching.",
            "model_served": "mock/cheap-a",
            "response_content": "completely unrelated cheap answer",
        })
        assert verdict == "disagree"
        assert repo.verified[42] == "disagree"
        assert len(repo.failures) == 1
        failure = repo.failures[0]
        assert failure["cheap_model"] == "mock/cheap-a"
        assert failure["top_model"] == "mock/top-c"
        assert failure["judge_agreement"] < 4
        assert failure["prompt_features"]["heavy_verb_count"] >= 1
        assert failure["prompt_features"]["prompt"]  # harvested for retraining

    async def test_agreement_marks_agree_without_failure_row(self, worker_setup):
        worker, repo = worker_setup
        # make the cheap answer identical to what top-c will deterministically say
        registry = worker.registry
        adapter, model = registry.resolve("mock/top-c")
        shadow = await adapter.chat(
            ChatCompletionRequest(
                model="mock/top-c",
                messages=[{"role": "user", "content": "What is DNS?"}],
            ),
            model,
        )
        top_answer = shadow.response["choices"][0]["message"]["content"]
        verdict = await worker.process_item({
            "log_id": 43, "tier": 2,
            "messages": [{"role": "user", "content": "What is DNS?"}],
            "prompt_text": "What is DNS?",
            "model_served": "mock/cheap-a",
            "response_content": top_answer,  # perfect agreement
        })
        assert verdict == "agree"
        assert repo.verified[43] == "agree"
        assert repo.failures == []
