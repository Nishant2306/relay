"""C6: feature extraction, tier routing + fail-safe promotion, config
validation, and hot reload within 2s."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from gateway.models import ChatCompletionRequest
from gateway.router.features import FEATURE_NAMES, extract_feature_dict, extract_features
from gateway.router.routes import RoutingConfigStore, TierRouter, validate_config

ROUTING_YAML = Path(__file__).resolve().parent.parent / "config" / "routing.yaml"


def request(prompt: str, model: str = "relay-auto") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[{"role": "user", "content": prompt}])


class FakeClassifier:
    def __init__(self, tier: int, confidence: float):
        self.tier, self.confidence = tier, confidence

    def classify(self, text: str) -> tuple[int, float]:
        return self.tier, self.confidence


class TestFeatures:
    def test_shape_and_names_aligned(self):
        vec = extract_features("Analyze the trade-offs of caching; return JSON.")
        assert vec.shape == (len(FEATURE_NAMES),)

    def test_cue_features_fire(self):
        f = extract_feature_dict(
            "Analyze and compare the trade-offs; design a fallback plan. "
            "Must keep under 200 words, at least 3 options, format as JSON."
        )
        assert f["heavy_verb_count"] >= 3
        assert f["constraint_count"] >= 2
        assert f["requests_json"] == 1.0
        assert f["format_directive_count"] >= 1

    def test_simple_lookup_profile(self):
        f = extract_feature_dict("What is the capital of Kansas?")
        assert f["short_simple_question"] == 1.0
        assert f["heavy_verb_count"] == 0.0

    def test_length_is_a_feature_not_a_proxy(self):
        # rubric line 7: long deterministic text has low cue counts even
        # though it is long
        long_tier1 = "Convert this list to lowercase: " + ", ".join(f"Item{i}" for i in range(300))
        f = extract_feature_dict(long_tier1)
        assert f["approx_tokens"] > 500
        assert f["heavy_verb_count"] == 0.0
        assert f["judgment_cue_count"] == 0.0


class TestRoutingConfig:
    def test_repo_config_is_valid(self):
        config = validate_config(yaml.safe_load(ROUTING_YAML.read_text(encoding="utf-8")))
        assert set(config.tiers) == {1, 2, 3}
        assert config.tiers[3].cache_threshold is None  # 'disabled'
        assert config.tiers[1].cache_threshold == pytest.approx(0.755)

    def test_invalid_configs_rejected(self):
        good = yaml.safe_load(ROUTING_YAML.read_text(encoding="utf-8"))
        for mutation in (
            lambda c: c["tiers"].pop(2),
            lambda c: c["tiers"][1].update(chain=[]),
            lambda c: c["tiers"][1].update(chain=["nomodel"]),
            lambda c: c["tiers"][1].update(cache_threshold="sometimes"),
            lambda c: c["verification"].update(sample_rate=3.0),
        ):
            broken = yaml.safe_load(ROUTING_YAML.read_text(encoding="utf-8"))
            mutation(broken)
            with pytest.raises(Exception):
                validate_config(broken)
        validate_config(good)  # unmutated passes


class TestTierRouter:
    @pytest.fixture
    def store(self, tmp_path):
        path = tmp_path / "routing.yaml"
        path.write_text(ROUTING_YAML.read_text(encoding="utf-8"), encoding="utf-8")
        return RoutingConfigStore(path)

    def test_confident_decision_uses_tier_chain(self, store):
        router = TierRouter(FakeClassifier(tier=1, confidence=0.9), store)
        decision = router.decide(request("What is DNS?"))
        assert decision.tier == 1
        assert not decision.promoted
        assert decision.chain == store.config.tiers[1].chain

    def test_low_confidence_promotes_one_tier(self, store):
        router = TierRouter(FakeClassifier(tier=1, confidence=0.45), store)
        decision = router.decide(request("ambiguous prompt"))
        assert decision.tier == 2
        assert decision.promoted

    def test_tier3_never_promotes_past_top(self, store):
        router = TierRouter(FakeClassifier(tier=3, confidence=0.2), store)
        assert router.decide(request("hard prompt")).tier == 3

    async def test_hot_reload_applies_within_2s(self, store):
        watch_task = asyncio.create_task(store.watch())
        try:
            await asyncio.sleep(0.2)  # let the watcher attach
            raw = yaml.safe_load(store.path.read_text(encoding="utf-8"))
            raw["tiers"][1]["chain"] = ["mock/mid-b"]
            store.path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            deadline = asyncio.get_event_loop().time() + 2.0
            while asyncio.get_event_loop().time() < deadline:
                if store.config.tiers[1].chain == ["mock/mid-b"]:
                    break
                await asyncio.sleep(0.05)
            assert store.config.tiers[1].chain == ["mock/mid-b"]
        finally:
            watch_task.cancel()

    async def test_invalid_edit_keeps_previous_config(self, store):
        before = store.config.tiers[1].chain
        store.path.write_text("tiers: {1: {chain: []}}", encoding="utf-8")
        store.reload()
        assert store.config.tiers[1].chain == before


class TestTrainedModel:
    """Uses the committed trained model; regression-gates DEV accuracy only
    (test-split numbers live in models/classifier_meta.json, not pytest)."""

    @pytest.mark.slow
    def test_trained_model_beats_control_on_dev(self):
        import numpy as np

        from gateway.datasets import load_complexity_labels
        from gateway.router.classifier import ComplexityClassifier

        clf = ComplexityClassifier()
        dev = [r for r in load_complexity_labels() if r["split"] == "dev"]
        correct = sum(1 for r in dev if clf.classify(r["prompt"])[0] == r["tier"])
        accuracy = correct / len(dev)
        assert accuracy > 0.60, f"dev accuracy {accuracy:.1%} regressed"
        # sanity: confidence is a probability
        tier, conf = clf.classify("What is DNS?")
        assert tier in (1, 2, 3) and 0 < conf <= 1
        assert isinstance(np.float64(conf), np.float64)
