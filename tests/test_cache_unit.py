"""C5 unit layer — no Redis, no embeddings: key derivation, TTL heuristic
(validated against the frozen labeled data), and the answer-determining guards.

Guard/dataset assertions use the DEV split only: pytest must never become a
tuning loop against the held-out test pairs (ADR-0009).
"""

from __future__ import annotations

from collections import Counter

from gateway.cache.guards import answer_determining_conflict
from gateway.cache.keys import exact_key, namespace_for, normalize_prompt, param_bucket
from gateway.cache.ttl import classify_cache_class, ttl_for
from gateway.datasets import load_cache_traps, load_complexity_labels
from gateway.models import ChatCompletionRequest, TeamContext


def team(team_id: int = 1, scope: str = "team") -> TeamContext:
    return TeamContext(
        team_id=team_id, name=f"t{team_id}", rpm=60, tpm=100_000,
        daily_budget_usd=5, monthly_budget_usd=50, allowed_models=["*"],
        cache_scope=scope,  # type: ignore[arg-type]
    )


def request(prompt: str, system: str = "", model: str = "mock/cheap-a",
            temperature: float | None = None) -> ChatCompletionRequest:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return ChatCompletionRequest(model=model, messages=messages, temperature=temperature)


class TestKeys:
    def test_normalization_conservative(self):
        assert normalize_prompt("  What   is\tPython? ") == "what is python?"
        # punctuation, numbers, word order stay significant
        assert normalize_prompt("a to b") != normalize_prompt("b to a")
        assert normalize_prompt("top 5") != normalize_prompt("top 10")

    def test_param_buckets(self):
        assert param_bucket(None) == "na"
        assert param_bucket(0.65) == param_bucket(0.7)
        assert param_bucket(0.1) != param_bucket(0.5)

    def test_namespace_isolation_boundaries(self):
        base = namespace_for(request("hi", system="You are helpful"), team())
        assert namespace_for(request("hi", system="You are terse"), team()) != base
        assert namespace_for(request("hi", system="You are helpful",
                                     model="mock/top-c"), team()) != base
        assert namespace_for(request("hi", system="You are helpful",
                                     temperature=0.9), team()) != base
        assert namespace_for(request("hi", system="You are helpful"), team(2)) != base
        # same everything -> same namespace
        assert namespace_for(request("hi", system="You are helpful"), team()) == base

    def test_global_scope_shares_namespace_across_teams(self):
        a = namespace_for(request("hi"), team(1, scope="global"))
        b = namespace_for(request("hi"), team(2, scope="global"))
        assert a == b

    def test_exact_key_normalizes_surface_form(self):
        ns = "abc"
        assert exact_key(ns, "What is Python?") == exact_key(ns, "  what is   python? ")
        assert exact_key(ns, "What is Python?") != exact_key(ns, "What is Rust?")


class TestTtlHeuristic:
    """The runtime heuristic must agree with the frozen labels (SPEC C5)."""

    def test_agreement_with_labeled_data(self):
        rows = load_complexity_labels()
        confusion: Counter = Counter()
        for r in rows:
            confusion[(r["cache_class"], classify_cache_class(r["prompt"]))] += 1

        def recall(label: str) -> float:
            total = sum(v for (lab, _), v in confusion.items() if lab == label)
            return confusion[(label, label)] / total

        overall = sum(confusion[(x, x)] for x in ("stable", "temporal", "no_cache")) / len(rows)
        assert overall >= 0.93, f"overall agreement {overall:.1%} regressed"
        assert recall("stable") >= 0.95
        assert recall("no_cache") >= 0.70  # measured 79.5% at freeze
        assert recall("temporal") >= 0.70  # measured 85.7% at freeze

    def test_high_temperature_bypasses(self):
        assert classify_cache_class("What is 2+2?", temperature=1.2) == "no_cache"

    def test_ttl_mapping(self):
        assert ttl_for("stable") == 86_400
        assert ttl_for("temporal") == 3_600
        assert ttl_for("no_cache") == 0
        assert ttl_for("stable", {"stable": 60}) == 60


class TestGuards:
    def test_number_mismatch(self):
        assert answer_determining_conflict("top 5 results", "top 10 results")
        assert answer_determining_conflict("List primes below 20", "List primes below 30")

    def test_number_format_equivalence_passes(self):
        assert answer_determining_conflict("Sum 1,000 and 5", "Sum 1000 and 5") is None
        assert answer_determining_conflict("Meet at 5 PM", "Meet at 17:00") is None
        assert answer_determining_conflict("pick ten items", "pick 10 items") is None

    def test_negation(self):
        assert answer_determining_conflict(
            "Only include rows with errors", "Only include rows without errors"
        )

    def test_inverse_pairs(self):
        assert answer_determining_conflict("Find the minimum value", "Find the maximum value")
        assert answer_determining_conflict("encrypt this file", "decrypt this file")

    def test_direction_swap(self):
        assert answer_determining_conflict(
            "Convert 20 Celsius to Fahrenheit.", "Convert 20 Fahrenheit to Celsius."
        )
        assert answer_determining_conflict(
            "Translate from English to French: hello", "Translate from French to English: hello"
        )

    def test_content_substitution(self):
        assert answer_determining_conflict(
            "Plan a three-day trip to Paris.", "Plan a three-day trip to London."
        )
        assert answer_determining_conflict(
            "Implement this endpoint with HTTP GET.", "Implement this endpoint with HTTP POST."
        )

    def test_paraphrases_pass(self):
        assert answer_determining_conflict(
            "What is Python?", "Could you explain what Python is?"
        ) is None
        assert answer_determining_conflict(
            "Explain photosynthesis in simple terms.",
            "In plain language, what is photosynthesis?",
        ) is None
        # morphology
        assert answer_determining_conflict(
            "reverse the list of values", "reversing the lists of value"
        ) is None

    def test_dev_split_regression(self):
        """Guards were tuned on dev: lock in that behavior. (dev only — the
        held-out test numbers live in results/, not in pytest.)"""
        dev = [r for r in load_cache_traps() if r["split"] == "dev"]
        miss_caught = sum(
            1 for r in dev if r["expected"] == "miss"
            and answer_determining_conflict(r["prompt_a"], r["prompt_b"])
        )
        hit_blocked = sum(
            1 for r in dev if r["expected"] == "hit"
            and answer_determining_conflict(r["prompt_a"], r["prompt_b"])
        )
        assert miss_caught >= 19  # 19/20 at tuning time
        assert hit_blocked <= 1  # 1/20 collateral at tuning time
