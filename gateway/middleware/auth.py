"""Team auth (SPEC C3): API key -> team context.

Keys are compared as sha256 hashes and never logged. Outcomes:
  - missing/unknown key -> 401
  - disallowed model    -> 403 with a clear message (checked at dispatch time)

`TeamStore` is a protocol so unit tests run against an in-memory store and
production uses Postgres.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from gateway.db import Team, TeamLimits
from gateway.models import TeamContext


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class TeamStore(Protocol):
    async def by_key_hash(self, key_hash: str) -> TeamContext | None: ...


class InMemoryTeamStore:
    """For unit tests and offline development."""

    def __init__(self, teams: dict[str, TeamContext] | None = None):
        self._by_hash: dict[str, TeamContext] = {}
        for api_key, ctx in (teams or {}).items():
            self._by_hash[hash_key(api_key)] = ctx

    def add(self, api_key: str, ctx: TeamContext) -> None:
        self._by_hash[hash_key(api_key)] = ctx

    async def by_key_hash(self, key_hash: str) -> TeamContext | None:
        return self._by_hash.get(key_hash)


class PostgresTeamStore:
    def __init__(self, session_factory: async_sessionmaker):
        self._sessions = session_factory

    async def by_key_hash(self, key_hash: str) -> TeamContext | None:
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(Team, TeamLimits)
                    .join(TeamLimits, TeamLimits.team_id == Team.id)
                    .where(Team.api_key_hash == key_hash)
                )
            ).first()
            if row is None:
                return None
            team, limits = row
            return TeamContext(
                team_id=team.id,
                name=team.name,
                rpm=limits.rpm,
                tpm=limits.tpm,
                daily_budget_usd=float(limits.daily_budget_usd),
                monthly_budget_usd=float(limits.monthly_budget_usd),
                allowed_models=list(limits.allowed_models or []),
                cache_scope=limits.cache_scope,
            )


async def authenticate(request: Request) -> TeamContext:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    api_key = auth[7:].strip()
    store: TeamStore = request.app.state.team_store
    team = await store.by_key_hash(hash_key(api_key))
    if team is None:
        raise HTTPException(status_code=401, detail="Unknown API key")
    return team


def check_model_allowed(team: TeamContext, model: str) -> None:
    allowed = team.allowed_models
    if not allowed or "*" in allowed or model in allowed:
        return
    raise HTTPException(
        status_code=403,
        detail=(
            f"Team '{team.name}' is not allowed to use model '{model}'. "
            f"Allowed: {sorted(allowed)}"
        ),
    )
