"""Per-team dollar budgets (SPEC C4): warn at 80% via Slack, block at 100%.

Fast path is a Redis counter per team/day and team/month
(`budget:{team}:{yyyy-mm-dd}` / `budget:{team}:{yyyy-mm}`); a nightly script
reconciles Redis into the Postgres spend_ledger (scripts/reconcile_spend.py).
The 80% warning fires at most once per team per day (SET NX sentinel).
"""

from __future__ import annotations

import datetime as dt

from redis.asyncio import Redis

from gateway.models import TeamContext
from gateway.obs.slack import SlackNotifier
from gateway.pipeline import BudgetDecision

WARN_FRACTION = 0.8


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _month() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m")


class RedisBudgetGuard:
    def __init__(self, redis: Redis, slack: SlackNotifier | None = None):
        self.redis = redis
        self.slack = slack

    async def _spent(self, key: str) -> float:
        val = await self.redis.get(key)
        return float(val) if val else 0.0

    async def check(self, team: TeamContext) -> BudgetDecision:
        daily_key = f"budget:{team.team_id}:{_today()}"
        monthly_key = f"budget:{team.team_id}:{_month()}"
        daily = await self._spent(daily_key)
        monthly = await self._spent(monthly_key)

        if team.daily_budget_usd > 0 and daily >= team.daily_budget_usd:
            return BudgetDecision(
                allowed=False,
                reason=(
                    f"Daily budget exhausted: ${daily:.4f} of ${team.daily_budget_usd:.2f} "
                    f"spent today by team '{team.name}'. Resets at midnight UTC."
                ),
            )
        if team.monthly_budget_usd > 0 and monthly >= team.monthly_budget_usd:
            return BudgetDecision(
                allowed=False,
                reason=(
                    f"Monthly budget exhausted: ${monthly:.4f} of "
                    f"${team.monthly_budget_usd:.2f} spent this month by team '{team.name}'."
                ),
            )
        return BudgetDecision(allowed=True)

    async def charge(self, team: TeamContext, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        daily_key = f"budget:{team.team_id}:{_today()}"
        monthly_key = f"budget:{team.team_id}:{_month()}"
        pipe = self.redis.pipeline()
        pipe.incrbyfloat(daily_key, cost_usd)
        pipe.expire(daily_key, 60 * 60 * 48)
        pipe.incrbyfloat(monthly_key, cost_usd)
        pipe.expire(monthly_key, 60 * 60 * 24 * 62)
        results = await pipe.execute()
        daily_total = float(results[0])

        if (
            self.slack is not None
            and team.daily_budget_usd > 0
            and daily_total >= WARN_FRACTION * team.daily_budget_usd
        ):
            sentinel = f"budget:warned:{team.team_id}:{_today()}"
            first = await self.redis.set(sentinel, "1", nx=True, ex=60 * 60 * 24)
            if first:
                pct = 100 * daily_total / team.daily_budget_usd
                await self.slack.send(
                    f":warning: Team *{team.name}* is at {pct:.0f}% of its daily budget "
                    f"(${daily_total:.4f} / ${team.daily_budget_usd:.2f})."
                )
