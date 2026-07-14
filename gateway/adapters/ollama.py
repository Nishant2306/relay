"""Ollama adapter — local models via Ollama's OpenAI-compatible /v1 endpoint. Free."""

from __future__ import annotations

from gateway.adapters.base import OpenAIStyleAdapter


class OllamaAdapter(OpenAIStyleAdapter):
    provider = "ollama"
