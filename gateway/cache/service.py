"""SemanticCache facade — the pipeline's cache stage (SPEC C5).

Lookup order (ADR-0003): exact hash within namespace -> vector KNN within
namespace at the tier's threshold -> miss (with singleflight so concurrent
identical misses collapse to one upstream call).

Write policy (ADR-0006): only complete successful responses, only when the
prompt's cache_class allows it; TTL by cache_class.
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
from redis.asyncio import Redis

from gateway.cache.embedding import embed
from gateway.cache.exact import ExactStore
from gateway.cache.guards import answer_determining_conflict
from gateway.cache.keys import exact_key, namespace_for
from gateway.cache.semantic import VectorIndex
from gateway.cache.singleflight import Singleflight
from gateway.cache.ttl import classify_cache_class, ttl_for
from gateway.models import AdapterResult, ChatCompletionRequest, TeamContext
from gateway.pipeline import CacheHit

logger = logging.getLogger("relay.cache")

# Dev-tuned (scripts/eval_cache_traps.py); routing.yaml overrides these at runtime.
DEFAULT_THRESHOLDS: dict[int, float | None] = {1: 0.755, 2: 0.755, 3: None}
DEFAULT_THRESHOLD_UNTIERED = 0.94  # no tier info -> stay conservative
NEAR_MISS_FLOOR = 0.85
SIMLOG_KEY = "cache:simlog"
SIMLOG_CAP = 20_000
STATS_KEY = "cache:stats"


class _EmbeddingMemo:
    """Tiny LRU so a lookup's embedding is reused by the write that follows."""

    def __init__(self, cap: int = 2048):
        self.cap = cap
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> np.ndarray | None:
        vec = self._data.get(key)
        if vec is not None:
            self._data.move_to_end(key)
        return vec

    def put(self, key: str, vec: np.ndarray) -> None:
        self._data[key] = vec
        self._data.move_to_end(key)
        while len(self._data) > self.cap:
            self._data.popitem(last=False)


class SemanticCache:
    def __init__(
        self,
        redis: Redis,
        *,
        thresholds: Callable[[], dict[int, float | None]] | None = None,
        ttl_config: Callable[[], dict[str, int]] | None = None,
        singleflight_wait_s: float = 10.0,
    ):
        self.redis = redis
        self.exact = ExactStore(redis)
        self.index = VectorIndex(redis)
        self.singleflight = Singleflight(redis, wait_timeout_s=singleflight_wait_s)
        self._thresholds = thresholds or (lambda: DEFAULT_THRESHOLDS)
        self._ttl_config = ttl_config or (lambda: None)  # type: ignore[return-value]
        self._memo = _EmbeddingMemo()

    async def start(self) -> None:
        await self.index.ensure_index()

    # -- helpers ---------------------------------------------------------------
    def _threshold_for(self, tier: int | None) -> float | None:
        table = self._thresholds()
        if tier is None:
            return DEFAULT_THRESHOLD_UNTIERED
        return table.get(tier, DEFAULT_THRESHOLD_UNTIERED)

    @staticmethod
    def _hit_from_entry(entry: dict[str, Any], kind: str,
                        similarity: float | None) -> CacheHit:
        return CacheHit(
            response=entry["response"], kind=kind, similarity=similarity,
            model_key=entry["model_key"], tokens_in=entry.get("tokens_in", 0),
            tokens_out=entry.get("tokens_out", 0),
        )

    async def _bump(self, field: str) -> None:
        try:
            await self.redis.hincrby(STATS_KEY, field, 1)
        except Exception:
            pass

    async def _log_similarity(self, similarity: float, tier: int | None,
                              threshold: float) -> None:
        try:
            payload = json.dumps({
                "sim": round(similarity, 6), "tier": tier,
                "threshold": threshold, "ts": int(time.time()),
            })
            pipe = self.redis.pipeline()
            pipe.lpush(SIMLOG_KEY, payload)
            pipe.ltrim(SIMLOG_KEY, 0, SIMLOG_CAP - 1)
            await pipe.execute()
        except Exception:
            pass

    async def _embedding_for(self, ek: str, prompt: str) -> np.ndarray:
        vec = self._memo.get(ek)
        if vec is None:
            vec = (await embed([prompt]))[0]
            self._memo.put(ek, vec)
        return vec

    # -- pipeline protocol -------------------------------------------------------
    async def lookup(self, request: ChatCompletionRequest, team: TeamContext,
                     tier: int | None) -> CacheHit | None:
        prompt = request.full_prompt_text()
        if classify_cache_class(prompt, request.temperature) == "no_cache":
            await self._bump("bypass")
            return None

        ns = namespace_for(request, team)
        ek = exact_key(ns, prompt)

        entry = await self.exact.get(ns, ek)
        if entry is not None:
            await self._bump("hit_exact")
            return self._hit_from_entry(entry, "exact", None)

        threshold = self._threshold_for(tier)
        if threshold is not None:
            vec = await self._embedding_for(ek, prompt)
            neighbors = await self.index.knn(ns, vec, k=1)
            if neighbors:
                neighbor_key, similarity = neighbors[0]
                await self._log_similarity(similarity, tier, threshold)
                if similarity >= threshold:
                    entry = await self.exact.get(ns, neighbor_key)
                    if entry is not None:
                        # similarity alone is not sufficient: run the
                        # answer-determining guards against the stored prompt
                        conflict = answer_determining_conflict(
                            prompt, entry.get("prompt", "")
                        )
                        if conflict is None:
                            await self._bump("hit_semantic")
                            return self._hit_from_entry(entry, "semantic", similarity)
                        await self._bump("guard_reject")
                        logger.debug("guard rejected semantic candidate: %s", conflict)

        # miss -> singleflight (ADR-0004)
        await self._bump("miss")
        if not await self.singleflight.acquire(ns, ek):
            entry = await self.singleflight.wait_for_result(
                ns, ek, lambda: self.exact.get(ns, ek)
            )
            if entry is not None:
                await self._bump("hit_singleflight")
                return self._hit_from_entry(entry, "exact", None)
        return None

    async def write(self, request: ChatCompletionRequest, team: TeamContext,
                    result: AdapterResult, tier: int | None) -> None:
        prompt = request.full_prompt_text()
        ns = namespace_for(request, team)
        ek = exact_key(ns, prompt)
        try:
            cache_class = classify_cache_class(prompt, request.temperature)
            ttl_s = ttl_for(cache_class, self._ttl_config())
            if ttl_s <= 0 or result.finish_reason != "stop":
                return
            entry = {
                "response": result.response,
                "model_key": f"{result.provider}/{result.model}",
                "tokens_in": result.usage.prompt_tokens,
                "tokens_out": result.usage.completion_tokens,
                "cache_class": cache_class,
                "prompt": prompt,  # guards compare query vs stored prompt on semantic hits
            }
            await self.exact.set(ns, ek, entry, ttl_s)
            vec = await self._embedding_for(ek, prompt)
            await self.index.add(ns, ek, vec, str(entry["model_key"]), ttl_s)
        finally:
            await self.singleflight.release(ns, ek)

    async def release(self, request: ChatCompletionRequest, team: TeamContext) -> None:
        """Free the singleflight lock when the upstream call failed."""
        ns = namespace_for(request, team)
        await self.singleflight.release(ns, exact_key(ns, request.full_prompt_text()))

    # -- admin -------------------------------------------------------------------
    async def invalidate_namespace(self, namespace: str) -> int:
        return (await self.exact.delete_namespace(namespace)
                + await self.index.delete_namespace(namespace))

    async def invalidate_model(self, model_key: str) -> int:
        return await self.index.delete_model(model_key)

    async def flush_all(self) -> int:
        deleted = 0
        for pattern in ("ce:*", "cv:*", "lock:*"):
            async for key in self.redis.scan_iter(match=pattern, count=500):
                await self.redis.delete(key)
                deleted += 1
        return deleted

    async def stats(self) -> dict[str, int]:
        raw = await self.redis.hgetall(STATS_KEY)
        return {
            (k.decode() if isinstance(k, bytes) else k):
            int(v) for k, v in raw.items()
        }

    async def logged_similarities(self, limit: int = SIMLOG_CAP) -> list[dict[str, Any]]:
        raw = await self.redis.lrange(SIMLOG_KEY, 0, limit - 1)
        return [json.loads(r) for r in raw]

    async def tuning_table(self, thresholds: list[float],
                           dev_curve: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Replay logged lookup similarities at hypothetical thresholds.

        `dev_curve` (from scripts/eval_cache_traps.py --write-curve) supplies
        the estimated wrong-answer rate per threshold, measured on the DEV trap
        pairs only — never the held-out test split (ADR-0009).
        """
        sims = [s["sim"] for s in await self.logged_similarities()]
        table = []
        curve = {round(float(t), 3): v for t, v in (dev_curve or {}).items()}
        for threshold in thresholds:
            would_hit = sum(1 for s in sims if s >= threshold)
            row: dict[str, Any] = {
                "threshold": threshold,
                "observed_lookups": len(sims),
                "would_hit": would_hit,
                "would_hit_rate": round(would_hit / len(sims), 4) if sims else None,
            }
            if curve:
                est = curve.get(round(threshold, 3))
                row["estimated_false_hit_rate_dev"] = est.get("false_hit_rate") if est else None
            table.append(row)
        return table
