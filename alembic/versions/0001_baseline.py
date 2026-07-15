"""Baseline schema (SPEC §7): teams, limits, request_log, spend_ledger,
routing_failures, config_audit, provider_health_events.

The baseline creates from gateway.db metadata (single source of truth);
subsequent migrations must use explicit op.* calls.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-07-14
"""
from __future__ import annotations

from alembic import op
from gateway.db import Base

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(op.get_bind())
