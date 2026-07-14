"""Mock adapter — the chaos provider. $0 spend, simulated pricing for savings math."""

from __future__ import annotations

from gateway.adapters.base import OpenAIStyleAdapter


class MockAdapter(OpenAIStyleAdapter):
    provider = "mock"
