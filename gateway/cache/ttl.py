"""Runtime cache-class heuristic + TTL mapping (SPEC C5).

The frozen complexity dataset labels every prompt stable/temporal/no_cache
(554/39/7). These regexes were written against that labeled data — see
tests/test_cache_unit.py::TestTtlHeuristic, which measures agreement against
all 600 rows and gates regressions — rather than invented blind.

Mapping (config/routing.yaml `cache_ttl`): stable -> 24h, temporal -> 1h,
no_cache -> bypass entirely (never written, never served).
"""

from __future__ import annotations

import re
from typing import Literal

CacheClass = Literal["stable", "temporal", "no_cache"]

# Answer changes with when you ask: explicit deictic time references.
_TEMPORAL = re.compile(
    r"\b(today|yesterday|tomorrow|tonight|right now|currently|as of now|"
    r"this (week|month|year|morning|afternoon)|next (week|month|year)|"
    r"latest|breaking|current(ly)? (news|price|weather|version|date))\b",
    re.IGNORECASE,
)

# No canonical answer: creative writing, subjective judgment, open-ended
# recommendation/strategy — serving a cached answer is wrong even if similar.
_NO_CACHE = re.compile(
    r"\b(write a (story|poem|song|letter|diary|memo|pamphlet|preface|haiku)|"
    r"create (a |an )?(story|satirical|fictional|children'?s|blank verse|content|two-timeline)|"
    r"recommend between|develop a strategy|choose (whether|between|strong or)|"
    r"imagine|brainstorm|come up with|invent|"
    r"top \d+|most popular|best \w+ of all time|"
    r"should i\b|is .{1,40} important|will ai\b|"
    r"from the perspective of|tell a\b)\b",
    re.IGNORECASE,
)

HIGH_TEMPERATURE_NO_CACHE = 1.0


def classify_cache_class(prompt_text: str, temperature: float | None = None) -> CacheClass:
    if temperature is not None and temperature >= HIGH_TEMPERATURE_NO_CACHE:
        return "no_cache"
    if _NO_CACHE.search(prompt_text):
        return "no_cache"
    if _TEMPORAL.search(prompt_text):
        return "temporal"
    return "stable"


DEFAULT_TTLS: dict[str, int] = {"stable": 86_400, "temporal": 3_600, "no_cache": 0}


def ttl_for(cache_class: CacheClass, ttl_config: dict[str, int] | None = None) -> int:
    """Seconds to live; 0 means bypass the cache entirely."""
    config = ttl_config or DEFAULT_TTLS
    return int(config.get(cache_class, DEFAULT_TTLS[cache_class]))
