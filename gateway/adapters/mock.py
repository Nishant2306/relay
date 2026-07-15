"""Mock adapter — the chaos provider. $0 spend, simulated pricing for savings math."""

from __future__ import annotations

from gateway.adapters.base import OpenAIStyleAdapter


class MockAdapter(OpenAIStyleAdapter):
    provider = "mock"


class MockBAdapter(OpenAIStyleAdapter):
    """Second chaos-mock instance — the same-tier alternate provider that
    fallback chains walk to during outage drills (ADR-0007)."""

    provider = "mockb"
