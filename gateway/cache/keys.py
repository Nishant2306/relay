"""Cache key derivation (ADR-0002, ADR-0003).

namespace = sha256(system_prompt) + model + temp_bucket + top_p_bucket + team_scope
exact_key = sha256(namespace + normalized_prompt)

Vector search happens only WITHIN a namespace: a different system prompt,
model, sampling-parameter bucket, or team can never share an entry. This is a
correctness boundary, not an optimization.
"""

from __future__ import annotations

import hashlib
import re

from gateway.models import ChatCompletionRequest, TeamContext

_WS = re.compile(r"\s+")


def normalize_prompt(text: str) -> str:
    """Conservative normalization: trim, collapse whitespace, casefold.

    Deliberately does NOT touch punctuation, numbers, or word order — anything
    that could be answer-determining stays significant.
    """
    return _WS.sub(" ", text).strip().casefold()


def param_bucket(value: float | None, step: float = 0.2, default: str = "na") -> str:
    if value is None:
        return default
    bucketed = round(round(value / step) * step, 1)
    return f"{bucketed:.1f}"


def team_scope(team: TeamContext) -> str:
    return "global" if team.cache_scope == "global" else f"team:{team.team_id}"


def namespace_for(request: ChatCompletionRequest, team: TeamContext) -> str:
    system_hash = hashlib.sha256(request.system_prompt().encode()).hexdigest()
    raw = "|".join([
        system_hash,
        request.model,
        param_bucket(request.temperature),
        param_bucket(request.top_p),
        team_scope(team),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def exact_key(namespace: str, prompt_text: str) -> str:
    return hashlib.sha256((namespace + "|" + normalize_prompt(prompt_text)).encode()).hexdigest()
