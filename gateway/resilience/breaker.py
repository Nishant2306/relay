"""Circuit breaker per provider/model (SPEC C8).

FSM:
  closed     -- >= N retryable failures within the rolling window --> open
  open       -- cooldown elapsed --> half_open (exactly one probe allowed)
  half_open  -- probe success --> closed | probe failure --> open (new cooldown)

Hysteresis against flapping: the failure *window* stops one bad blip from
opening the breaker, the *cooldown* stops a dead provider from being hammered,
and the *single half-open probe* stops a recovering provider from being
stampeded. Every transition is reported (Prometheus + Slack +
provider_health_events).

The clock is injectable so the FSM is table-testable without sleeping.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class BreakerConfig:
    failure_threshold: int = 5  # N failures ...
    window_s: float = 30.0  # ... within M seconds -> open
    cooldown_s: float = 30.0  # open -> half_open after this


TransitionHook = Callable[[str, BreakerState, BreakerState, str], None]


@dataclass
class CircuitBreaker:
    name: str  # "provider/model"
    config: BreakerConfig = field(default_factory=BreakerConfig)
    clock: Callable[[], float] = time.monotonic
    on_transition: TransitionHook | None = None

    def __post_init__(self) -> None:
        self._state = BreakerState.CLOSED
        self._failures: deque[float] = deque()
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        return self._state

    def _transition(self, to: BreakerState, reason: str) -> None:
        if to is self._state:
            return
        old, self._state = self._state, to
        if self.on_transition is not None:
            self.on_transition(self.name, old, to, reason)

    def allow(self) -> bool:
        """May a call be attempted right now? (May transition open->half_open.)"""
        if self._state is BreakerState.CLOSED:
            return True
        if self._state is BreakerState.OPEN:
            if self.clock() - self._opened_at >= self.config.cooldown_s:
                self._transition(BreakerState.HALF_OPEN, "cooldown elapsed")
                self._probe_in_flight = True
                return True  # the single probe
            return False
        # HALF_OPEN: exactly one probe at a time
        if not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        if self._state is BreakerState.HALF_OPEN:
            self._probe_in_flight = False
            self._failures.clear()
            self._transition(BreakerState.CLOSED, "probe succeeded")
        elif self._state is BreakerState.CLOSED:
            self._failures.clear()

    def record_failure(self) -> None:
        now = self.clock()
        if self._state is BreakerState.HALF_OPEN:
            self._probe_in_flight = False
            self._opened_at = now
            self._transition(BreakerState.OPEN, "probe failed")
            return
        if self._state is BreakerState.CLOSED:
            self._failures.append(now)
            cutoff = now - self.config.window_s
            while self._failures and self._failures[0] < cutoff:
                self._failures.popleft()
            if len(self._failures) >= self.config.failure_threshold:
                self._opened_at = now
                self._failures.clear()
                self._transition(
                    BreakerState.OPEN,
                    f">={self.config.failure_threshold} failures in "
                    f"{self.config.window_s:.0f}s",
                )


class BreakerRegistry:
    def __init__(self, config: BreakerConfig | None = None,
                 on_transition: TransitionHook | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.config = config or BreakerConfig()
        self.on_transition = on_transition
        self.clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, model_key: str) -> CircuitBreaker:
        if model_key not in self._breakers:
            self._breakers[model_key] = CircuitBreaker(
                name=model_key, config=self.config, clock=self.clock,
                on_transition=self.on_transition,
            )
        return self._breakers[model_key]

    def states(self) -> dict[str, str]:
        return {name: b.state.value for name, b in self._breakers.items()}
