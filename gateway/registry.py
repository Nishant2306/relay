"""Provider registry: resolves 'provider/model' keys to adapters."""

from __future__ import annotations

from fastapi import HTTPException

from gateway.adapters.base import ProviderAdapter
from gateway.pricing import PRICING_PER_MTOK


class ProviderRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}

    def register(self, adapter: ProviderAdapter) -> None:
        self._adapters[adapter.provider] = adapter

    def get(self, provider: str) -> ProviderAdapter | None:
        return self._adapters.get(provider)

    def resolve(self, model_key: str) -> tuple[ProviderAdapter, str]:
        """'mock/cheap-a' -> (MockAdapter, 'cheap-a'). 400 on unknown."""
        if "/" not in model_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model must be 'provider/model' (e.g. 'mock/cheap-a'); got {model_key!r}. "
                    f"Available: {sorted(self.available_models())}"
                ),
            )
        provider, model = model_key.split("/", 1)
        adapter = self._adapters.get(provider)
        if adapter is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider {provider!r}. Registered: {sorted(self._adapters)}",
            )
        return adapter, model

    def available_models(self) -> list[str]:
        return [
            key for key in PRICING_PER_MTOK
            if key.split("/", 1)[0] in self._adapters
        ]
