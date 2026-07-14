"""Durable request logging to Postgres (request_log table).

Writes happen post-response in a background task so they never add gateway
overhead. `NullRequestLogger` keeps unit tests infrastructure-free.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

from gateway.db import RequestLog

logger = logging.getLogger("relay.request_log")


class RequestLogEntry(BaseModel):
    team_id: int
    model_requested: str
    model_served: str | None = None
    provider: str | None = None
    tier: int | None = None
    cache: str = "miss"
    cache_kind: str | None = None
    cache_similarity: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    actual_cost_usd: float = 0.0
    counterfactual_cost_usd: float = 0.0
    latency_ms: int = 0
    overhead_ms: int = 0
    status: int = 200
    error_class: str | None = None
    retries: int = 0
    fallback_used: bool = False
    breaker_open: bool = False
    verified: str | None = None


class RequestLogger(Protocol):
    async def log(self, entry: RequestLogEntry) -> int | None: ...


class NullRequestLogger:
    """Unit tests / offline mode: remembers entries, writes nothing."""

    def __init__(self) -> None:
        self.entries: list[RequestLogEntry] = []

    async def log(self, entry: RequestLogEntry) -> int | None:
        self.entries.append(entry)
        return len(self.entries)


class PostgresRequestLogger:
    def __init__(self, session_factory: async_sessionmaker):
        self._sessions = session_factory

    async def log(self, entry: RequestLogEntry) -> int | None:
        try:
            async with self._sessions() as session:
                row = RequestLog(**entry.model_dump())
                session.add(row)
                await session.commit()
                return row.id
        except Exception:  # logging must never break serving
            logger.exception("failed to write request_log row")
            return None
