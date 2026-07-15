"""Admin API (SPEC C10): teams/limits CRUD, routing get/put (validate +
hot-apply), cache invalidation, near-miss tuning, savings stats.

Every mutating call writes a config_audit row. Admin auth is a shared key
(x-admin-key) — deliberately simpler than team auth; JWT/OIDC is v2 scope.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ValidationError

from gateway.middleware.auth import hash_key
from gateway.router.routes import RoutingConfigStore

logger = logging.getLogger("relay.admin")

AuditWriter = Callable[[str, str, Any, Any], Any]  # (actor, path, old, new) -> awaitable


class TeamCreate(BaseModel):
    name: str
    api_key: str
    plan: str = "dev"
    rpm: int = 60
    tpm: int = 100_000
    daily_budget_usd: float = 5.0
    monthly_budget_usd: float = 50.0
    allowed_models: list[str] = ["*"]
    cache_scope: str = "team"


class LimitsUpdate(BaseModel):
    rpm: int | None = None
    tpm: int | None = None
    daily_budget_usd: float | None = None
    monthly_budget_usd: float | None = None
    allowed_models: list[str] | None = None
    cache_scope: str | None = None


class InvalidateRequest(BaseModel):
    scope: str  # 'namespace' | 'model' | 'all'
    value: str = ""


def create_admin_router(
    *,
    admin_key: str,
    routing_store: RoutingConfigStore,
    audit: AuditWriter,
    cache: Any = None,
    session_factory: Any = None,
    dev_curve_path: Path = Path("models/trap_dev_curve.json"),
) -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])

    async def require_admin(request: Request) -> str:
        provided = request.headers.get("x-admin-key", "")
        if not admin_key or provided != admin_key:
            raise HTTPException(status_code=401, detail="admin key required")
        return "admin"

    def need_db() -> Any:
        if session_factory is None:
            raise HTTPException(status_code=503, detail="database not configured")
        return session_factory

    # -- teams / limits ---------------------------------------------------------
    @router.get("/teams")
    async def list_teams(actor: str = Depends(require_admin)) -> list[dict[str, Any]]:
        from sqlalchemy import select

        from gateway.db import Team, TeamLimits

        async with need_db()() as session:
            rows = (
                await session.execute(
                    select(Team, TeamLimits).join(TeamLimits, TeamLimits.team_id == Team.id)
                )
            ).all()
        return [
            {
                "id": t.id, "name": t.name, "plan": t.plan, "rpm": lim.rpm, "tpm": lim.tpm,
                "daily_budget_usd": float(lim.daily_budget_usd),
                "monthly_budget_usd": float(lim.monthly_budget_usd),
                "allowed_models": lim.allowed_models, "cache_scope": lim.cache_scope,
            }
            for t, lim in rows
        ]

    @router.post("/teams", status_code=201)
    async def create_team(body: TeamCreate,
                          actor: str = Depends(require_admin)) -> dict[str, Any]:
        from gateway.db import Team, TeamLimits

        async with need_db()() as session:
            team = Team(name=body.name, api_key_hash=hash_key(body.api_key), plan=body.plan)
            session.add(team)
            await session.flush()
            session.add(TeamLimits(
                team_id=team.id, rpm=body.rpm, tpm=body.tpm,
                daily_budget_usd=body.daily_budget_usd,
                monthly_budget_usd=body.monthly_budget_usd,
                allowed_models=body.allowed_models, cache_scope=body.cache_scope,
            ))
            await session.commit()
            team_id = team.id
        await audit(actor, "/admin/teams", None,
                    {"name": body.name, "plan": body.plan})  # never audit the key
        return {"id": team_id, "name": body.name}

    @router.put("/limits/{team_id}")
    async def update_limits(team_id: int, body: LimitsUpdate,
                            actor: str = Depends(require_admin)) -> dict[str, Any]:
        from gateway.db import TeamLimits

        async with need_db()() as session:
            limits = await session.get(TeamLimits, team_id)
            if limits is None:
                raise HTTPException(status_code=404, detail=f"team {team_id} not found")
            old = {
                "rpm": limits.rpm, "tpm": limits.tpm,
                "daily_budget_usd": float(limits.daily_budget_usd),
                "monthly_budget_usd": float(limits.monthly_budget_usd),
                "allowed_models": limits.allowed_models, "cache_scope": limits.cache_scope,
            }
            changes = {k: v for k, v in body.model_dump().items() if v is not None}
            for field, value in changes.items():
                setattr(limits, field, value)
            await session.commit()
        await audit(actor, f"/admin/limits/{team_id}", old, changes)
        return {"team_id": team_id, "applied": changes}

    # -- routing config ---------------------------------------------------------
    @router.get("/routing")
    async def get_routing(actor: str = Depends(require_admin)) -> dict[str, Any]:
        return yaml.safe_load(routing_store.path.read_text(encoding="utf-8"))

    @router.put("/routing")
    async def put_routing(body: dict[str, Any],
                          actor: str = Depends(require_admin)) -> dict[str, Any]:
        old = yaml.safe_load(routing_store.path.read_text(encoding="utf-8"))
        try:
            routing_store.apply(body)
        except ValidationError as e:
            # diff-style error: which field, what's wrong
            detail = [
                {"path": ".".join(str(p) for p in err["loc"]), "error": err["msg"]}
                for err in e.errors()
            ]
            raise HTTPException(status_code=422, detail=detail) from e
        await audit(actor, "/admin/routing", old, body)
        return {"applied": True}

    # -- cache ------------------------------------------------------------------
    @router.post("/cache/invalidate")
    async def invalidate(body: InvalidateRequest,
                         actor: str = Depends(require_admin)) -> dict[str, Any]:
        if cache is None:
            raise HTTPException(status_code=503, detail="cache not configured")
        if body.scope == "namespace":
            deleted = await cache.invalidate_namespace(body.value)
        elif body.scope == "model":
            deleted = await cache.invalidate_model(body.value)
        elif body.scope == "all":
            deleted = await cache.flush_all()
        else:
            raise HTTPException(status_code=422,
                                detail="scope must be namespace | model | all")
        await audit(actor, "/admin/cache/invalidate", None,
                    {"scope": body.scope, "value": body.value, "deleted": deleted})
        return {"deleted": deleted}

    @router.get("/cache/tuning")
    async def cache_tuning(thresholds: str = "0.75,0.80,0.85,0.90,0.94",
                           actor: str = Depends(require_admin)) -> list[dict[str, Any]]:
        if cache is None:
            raise HTTPException(status_code=503, detail="cache not configured")
        grid = [float(t) for t in thresholds.split(",")]
        dev_curve = None
        if dev_curve_path.exists():
            dev_curve = json.loads(dev_curve_path.read_text())
        return await cache.tuning_table(grid, dev_curve)

    # -- stats -------------------------------------------------------------------
    @router.get("/stats")
    async def stats(actor: str = Depends(require_admin)) -> dict[str, Any]:
        from sqlalchemy import case, func, select

        from gateway.db import RequestLog

        async with need_db()() as session:
            totals = (
                await session.execute(
                    select(
                        func.count(RequestLog.id),
                        func.sum(RequestLog.actual_cost_usd),
                        func.sum(RequestLog.counterfactual_cost_usd),
                        func.sum(case((RequestLog.cache == "hit", 1), else_=0)),
                        func.sum(case((RequestLog.fallback_used, 1), else_=0)),
                        func.avg(RequestLog.overhead_ms),
                    ).where(RequestLog.status == 200)
                )
            ).one()
            # savings attribution (ADR-0008): cache rows vs down-routed rows
            cache_saved = (
                await session.execute(
                    select(func.sum(RequestLog.counterfactual_cost_usd)).where(
                        RequestLog.cache == "hit", RequestLog.status == 200
                    )
                )
            ).scalar() or 0
            routing_saved = (
                await session.execute(
                    select(
                        func.sum(RequestLog.counterfactual_cost_usd - RequestLog.actual_cost_usd)
                    ).where(RequestLog.cache != "hit", RequestLog.status == 200)
                )
            ).scalar() or 0

        n, actual, counterfactual, hits, fallbacks, avg_overhead = totals
        actual = float(actual or 0)
        counterfactual = float(counterfactual or 0)
        return {
            "requests_ok": int(n or 0),
            "cache_hit_rate": round((hits or 0) / n, 4) if n else None,
            "actual_cost_usd": round(actual, 6),
            "counterfactual_cost_usd": round(counterfactual, 6),
            "savings_pct": round(1 - actual / counterfactual, 4) if counterfactual else None,
            "savings_attribution_usd": {
                "cache_hits": round(float(cache_saved), 6),
                "down_routing": round(float(routing_saved), 6),
            },
            "fallback_rate": round((fallbacks or 0) / n, 4) if n else None,
            "avg_overhead_ms": round(float(avg_overhead or 0), 2),
        }

    return router
