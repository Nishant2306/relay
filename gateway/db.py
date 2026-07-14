"""Postgres schema (SPEC §7) — SQLAlchemy 2.0 typed models + engine helpers.

Enums use native_enum=False (VARCHAR + CHECK) so the schema stays portable
and Alembic migrations stay simple.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

CACHE_ENUM = Enum("hit", "miss", "bypass", name="cache_outcome", native_enum=False)
CACHE_KIND_ENUM = Enum("exact", "semantic", name="cache_kind", native_enum=False)
VERIFIED_ENUM = Enum("pending", "agree", "disagree", name="verified_state", native_enum=False)
CACHE_SCOPE_ENUM = Enum("team", "global", "off", name="cache_scope", native_enum=False)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    plan: Mapped[str] = mapped_column(String(40), default="dev")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TeamLimits(Base):
    __tablename__ = "team_limits"

    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), primary_key=True)
    rpm: Mapped[int] = mapped_column(Integer, default=60)
    tpm: Mapped[int] = mapped_column(Integer, default=100_000)
    daily_budget_usd: Mapped[float] = mapped_column(Numeric(12, 4), default=5.0)
    monthly_budget_usd: Mapped[float] = mapped_column(Numeric(12, 4), default=50.0)
    allowed_models: Mapped[dict[str, Any]] = mapped_column(JSON, default=list)
    cache_scope: Mapped[str] = mapped_column(CACHE_SCOPE_ENUM, default="team")


class RequestLog(Base):
    __tablename__ = "request_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    model_requested: Mapped[str] = mapped_column(String(120))
    model_served: Mapped[str | None] = mapped_column(String(120), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(40), nullable=True)
    tier: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    cache: Mapped[str] = mapped_column(CACHE_ENUM, default="miss")
    cache_kind: Mapped[str | None] = mapped_column(CACHE_KIND_ENUM, nullable=True)
    cache_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    actual_cost_usd: Mapped[float] = mapped_column(Numeric(12, 8), default=0)
    counterfactual_cost_usd: Mapped[float] = mapped_column(Numeric(12, 8), default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    overhead_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[int] = mapped_column(SmallInteger, default=200)
    error_class: Mapped[str | None] = mapped_column(Text, nullable=True)
    retries: Mapped[int] = mapped_column(SmallInteger, default=0)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    breaker_open: Mapped[bool] = mapped_column(Boolean, default=False)
    verified: Mapped[str | None] = mapped_column(VERIFIED_ENUM, nullable=True)


class SpendLedger(Base):
    __tablename__ = "spend_ledger"

    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), primary_key=True)
    date: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0)


class RoutingFailure(Base):
    __tablename__ = "routing_failures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("request_log.id"))
    tier: Mapped[int] = mapped_column(SmallInteger)
    judge_agreement: Mapped[int] = mapped_column(SmallInteger)
    cheap_model: Mapped[str] = mapped_column(String(120))
    top_model: Mapped[str] = mapped_column(String(120))
    prompt_features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ConfigAudit(Base):
    __tablename__ = "config_audit"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(120))
    path: Mapped[str] = mapped_column(String(255))
    old: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    new: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProviderHealthEvent(Base):
    __tablename__ = "provider_health_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(120))
    from_state: Mapped[str] = mapped_column(String(20))
    to_state: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def make_engine(url: str):
    return create_async_engine(url, pool_size=10, max_overflow=20)


def make_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
