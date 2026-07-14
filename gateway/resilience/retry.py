"""Retry policy (SPEC C8): retryable errors only, exponential backoff + jitter.

Non-retryable failures (400, 401, 403, content policy) surface immediately —
retrying a request that is wrong by construction just burns quota.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class RetryConfig:
    max_retries: int = 3
    backoffs_s: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    jitter_fraction: float = 0.5  # +/-50%


def backoff_delay(config: RetryConfig, attempt: int,
                  rng: random.Random | None = None) -> float:
    """Delay before retry `attempt` (0-based). Jitter prevents retry alignment
    across gateway workers hammering a recovering provider in lockstep."""
    base = config.backoffs_s[min(attempt, len(config.backoffs_s) - 1)]
    r = rng or random
    jitter = 1.0 + config.jitter_fraction * (2 * r.random() - 1)
    return base * jitter
