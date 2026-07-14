"""Nightly reconcile: Redis budget counters -> Postgres spend_ledger (SPEC C4).

Run via cron/scheduler (or manually). Idempotent upsert per (team_id, date).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redis.asyncio import Redis
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from gateway.config import settings
from gateway.db import SpendLedger, make_engine


async def main() -> int:
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    engine = make_engine(settings.database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    count = 0
    async with sessions() as session:
        async for key in redis.scan_iter(match="budget:*"):
            parts = key.split(":")
            # budget:{team_id}:{yyyy-mm-dd} — skip monthly + sentinel keys
            if len(parts) != 3 or len(parts[2]) != 10:
                continue
            team_id, date_str = int(parts[1]), parts[2]
            usd = float(await redis.get(key) or 0)
            stmt = pg_insert(SpendLedger).values(
                team_id=team_id, date=dt.date.fromisoformat(date_str), usd=usd
            ).on_conflict_do_update(
                index_elements=[SpendLedger.team_id, SpendLedger.date], set_={"usd": usd}
            )
            await session.execute(stmt)
            count += 1
        await session.commit()

    await redis.aclose()
    await engine.dispose()
    print(f"Reconciled {count} team-day spend counters into spend_ledger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
