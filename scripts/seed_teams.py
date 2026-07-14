"""Seed demo teams + limits from config/teams.seed.yaml into Postgres (make seed)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.config import settings
from gateway.db import Team, TeamLimits
from gateway.middleware.auth import hash_key

SEED_PATH = Path(__file__).resolve().parent.parent / "config" / "teams.seed.yaml"


def main() -> int:
    seed = yaml.safe_load(SEED_PATH.read_text(encoding="utf-8"))
    engine = create_engine(settings.database_url_sync)
    created, updated = 0, 0
    with Session(engine) as session:
        for spec in seed["teams"]:
            team = session.scalar(select(Team).where(Team.name == spec["name"]))
            if team is None:
                team = Team(
                    name=spec["name"],
                    api_key_hash=hash_key(spec["api_key"]),
                    plan=spec.get("plan", "dev"),
                )
                session.add(team)
                session.flush()
                created += 1
            else:
                team.api_key_hash = hash_key(spec["api_key"])
                updated += 1
            limits = session.get(TeamLimits, team.id) or TeamLimits(team_id=team.id)
            limits.rpm = spec.get("rpm", 60)
            limits.tpm = spec.get("tpm", 100_000)
            limits.daily_budget_usd = spec.get("daily_budget_usd", 5.0)
            limits.monthly_budget_usd = spec.get("monthly_budget_usd", 50.0)
            limits.allowed_models = spec.get("allowed_models", ["*"])
            limits.cache_scope = spec.get("cache_scope", "team")
            session.merge(limits)
        session.commit()
    print(f"Seeded teams: {created} created, {updated} updated (keys hashed, never stored raw).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
