"""OpenAI adapter — real-dollar provider, gated by the spend kill-switch."""

from __future__ import annotations

from gateway.adapters.base import OpenAIStyleAdapter


class OpenAIAdapter(OpenAIStyleAdapter):
    provider = "openai"
