"""Verifier persistence: update request_log.verified, insert routing_failures."""

from __future__ import annotations

from typing import Any, Protocol

from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker

from gateway.db import RequestLog, RoutingFailure


class VerifierRepo(Protocol):
    async def mark_verified(self, log_id: int, verdict: str) -> None: ...

    async def record_failure(self, log_id: int, tier: int, judge_agreement: int,
                             cheap_model: str, top_model: str,
                             prompt_features: dict[str, Any]) -> None: ...


class PostgresVerifierRepo:
    def __init__(self, session_factory: async_sessionmaker):
        self._sessions = session_factory

    async def mark_verified(self, log_id: int, verdict: str) -> None:
        async with self._sessions() as session:
            await session.execute(
                update(RequestLog).where(RequestLog.id == log_id).values(verified=verdict)
            )
            await session.commit()

    async def record_failure(self, log_id: int, tier: int, judge_agreement: int,
                             cheap_model: str, top_model: str,
                             prompt_features: dict[str, Any]) -> None:
        async with self._sessions() as session:
            session.add(RoutingFailure(
                request_id=log_id, tier=tier, judge_agreement=judge_agreement,
                cheap_model=cheap_model, top_model=top_model,
                prompt_features=prompt_features,
            ))
            await session.commit()


class InMemoryVerifierRepo:
    def __init__(self) -> None:
        self.verified: dict[int, str] = {}
        self.failures: list[dict[str, Any]] = []

    async def mark_verified(self, log_id: int, verdict: str) -> None:
        self.verified[log_id] = verdict

    async def record_failure(self, log_id: int, tier: int, judge_agreement: int,
                             cheap_model: str, top_model: str,
                             prompt_features: dict[str, Any]) -> None:
        self.failures.append({
            "log_id": log_id, "tier": tier, "judge_agreement": judge_agreement,
            "cheap_model": cheap_model, "top_model": top_model,
            "prompt_features": prompt_features,
        })
