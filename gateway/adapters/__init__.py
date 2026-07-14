"""Provider adapters. ALL provider calls go through here (cost tracking + kill-switch)."""

from gateway.adapters.anthropic import AnthropicAdapter
from gateway.adapters.base import AdapterError, OpenAIStyleAdapter, ProviderAdapter
from gateway.adapters.mock import MockAdapter
from gateway.adapters.ollama import OllamaAdapter
from gateway.adapters.openai import OpenAIAdapter

__all__ = [
    "AdapterError",
    "AnthropicAdapter",
    "MockAdapter",
    "OllamaAdapter",
    "OpenAIAdapter",
    "OpenAIStyleAdapter",
    "ProviderAdapter",
]
