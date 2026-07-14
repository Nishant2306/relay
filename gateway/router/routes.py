"""Routing configuration (config/routing.yaml) with hot reload (SPEC C6).

`RoutingConfigStore.watch()` runs on watchfiles; edits apply to live traffic
within ~2s without a restart. `validate_config` is shared with the admin PUT
endpoint so invalid YAML is rejected with a precise error, never applied.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from gateway.models import ChatCompletionRequest
from gateway.pipeline import RouteDecision, Router

logger = logging.getLogger("relay.routing")


class TierConfig(BaseModel):
    chain: list[str] = Field(min_length=1)
    cache_threshold: float | None = None

    @field_validator("cache_threshold", mode="before")
    @classmethod
    def _disabled_is_none(cls, v: Any) -> Any:
        if isinstance(v, str):
            if v.lower() in ("disabled", "off", "none"):
                return None
            raise ValueError(f"cache_threshold must be a float or 'disabled', got {v!r}")
        return v

    @field_validator("chain")
    @classmethod
    def _chain_models_qualified(cls, v: list[str]) -> list[str]:
        for m in v:
            if "/" not in m:
                raise ValueError(f"chain entries must be 'provider/model', got {m!r}")
        return v


class ClassifierConfig(BaseModel):
    min_confidence: float = 0.6
    on_low_confidence: Literal["promote_one_tier", "keep"] = "promote_one_tier"


class VerificationConfig(BaseModel):
    sample_rate: float = Field(0.15, ge=0.0, le=1.0)
    judge: str = "mock/top-c"
    agree_threshold: int = Field(4, ge=1, le=5)


class RoutingConfig(BaseModel):
    tiers: dict[int, TierConfig]
    classifier: ClassifierConfig = ClassifierConfig()
    verification: VerificationConfig = VerificationConfig()
    cache_ttl: dict[str, int] = Field(
        default_factory=lambda: {"stable": 86_400, "temporal": 3_600, "no_cache": 0}
    )

    @field_validator("tiers")
    @classmethod
    def _tiers_complete(cls, v: dict[int, TierConfig]) -> dict[int, TierConfig]:
        if set(v) != {1, 2, 3}:
            raise ValueError(f"tiers must be exactly {{1, 2, 3}}, got {sorted(v)}")
        return v


def validate_config(raw: dict[str, Any]) -> RoutingConfig:
    """Raises pydantic.ValidationError with field-level detail on bad input."""
    return RoutingConfig.model_validate(raw)


class RoutingConfigStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._config = validate_config(yaml.safe_load(self.path.read_text(encoding="utf-8")))

    @property
    def config(self) -> RoutingConfig:
        return self._config

    def reload(self) -> None:
        try:
            self._config = validate_config(yaml.safe_load(self.path.read_text(encoding="utf-8")))
            logger.info("routing config reloaded from %s", self.path)
        except Exception:
            logger.exception("invalid routing config — keeping the previous one")

    def apply(self, raw: dict[str, Any]) -> RoutingConfig:
        """Validate + hot-apply + persist (admin PUT path)."""
        config = validate_config(raw)
        self.path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        self._config = config
        return config

    def thresholds(self) -> dict[int, float | None]:
        return {tier: tc.cache_threshold for tier, tc in self._config.tiers.items()}

    def ttl_config(self) -> dict[str, int]:
        return self._config.cache_ttl

    async def watch(self) -> None:
        """Hot reload on file change (run as a background task)."""
        import watchfiles

        async for _changes in watchfiles.awatch(self.path):
            self.reload()


class TierRouter(Router):
    """features -> tier (+ fail-safe promotion) -> model chain."""

    def __init__(self, classifier: Any, store: RoutingConfigStore):
        self.classifier = classifier
        self.store = store

    def decide(self, request: ChatCompletionRequest) -> RouteDecision:
        config = self.store.config
        tier, confidence = self.classifier.classify(request.full_prompt_text())
        promoted = False
        if (
            confidence < config.classifier.min_confidence
            and config.classifier.on_low_confidence == "promote_one_tier"
            and tier < 3
        ):
            tier += 1
            promoted = True
        return RouteDecision(
            tier=tier, chain=list(config.tiers[tier].chain),
            confidence=confidence, promoted=promoted,
        )
