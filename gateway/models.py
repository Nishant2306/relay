"""OpenAI-compatible request/response models.

We validate only the fields the pipeline needs and pass everything else
through untouched (`extra="allow"`) so unknown client fields survive the trip
to the provider.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[Any] | None = None

    def text(self) -> str:
        """Flatten content to plain text (multimodal parts contribute their text)."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts = []
            for p in self.content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
            return "\n".join(parts)
        return ""


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    n: int | None = None
    stream: bool = False
    user: str | None = None

    def system_prompt(self) -> str:
        return "\n".join(m.text() for m in self.messages if m.role == "system")

    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.text()
        return ""

    def full_prompt_text(self) -> str:
        """All non-system message text, used for embedding + feature extraction."""
        return "\n".join(m.text() for m in self.messages if m.role != "system")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AdapterResult(BaseModel):
    """Outcome of one provider call, normalized to the OpenAI response shape."""

    response: dict[str, Any]
    usage: Usage
    provider: str
    model: str
    cost_usd: float
    latency_ms: int
    finish_reason: str | None = None


def approx_tokens(text: str) -> int:
    """Cheap token estimate (chars/4) for pre-charging and the mock provider."""
    return max(1, len(text) // 4)


def openai_response(
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: Literal["stop", "length"] = "stop",
) -> dict[str, Any]:
    """Build a byte-shape-compatible chat.completion body."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class TeamContext(BaseModel):
    """Attached to a request after auth."""

    team_id: int
    name: str
    rpm: int
    tpm: int
    daily_budget_usd: float
    monthly_budget_usd: float
    allowed_models: list[str] = Field(default_factory=list)
    cache_scope: Literal["team", "global", "off"] = "team"
