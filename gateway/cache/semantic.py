"""Vector index over cache entries — Redis query engine (FT.*), HNSW, cosine.

One global index `idx:cache` over hashes cv:{namespace}:{exact_key} with the
namespace as a TAG field; every KNN query filters on `@ns:{namespace}` so
similarity search can never cross a namespace boundary (ADR-0002).

Implementation note: SPEC names RedisVL; we speak the same FT.* protocol
directly through redis-py to keep the dependency surface small (documented in
ADR-0001). The engine and index structure are identical.
"""

from __future__ import annotations

import numpy as np
from redis.asyncio import Redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

INDEX_NAME = "idx:cache"
VECTOR_PREFIX = "cv:"
EMBEDDING_DIM = 384


def vector_key(namespace: str, exact: str) -> str:
    return f"{VECTOR_PREFIX}{namespace}:{exact}"


class VectorIndex:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def ensure_index(self) -> None:
        try:
            await self.redis.ft(INDEX_NAME).info()
        except Exception:
            schema = [
                TagField("ns"),
                TagField("model_key"),
                VectorField(
                    "emb",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE",
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                ),
            ]
            definition = IndexDefinition(prefix=[VECTOR_PREFIX], index_type=IndexType.HASH)
            await self.redis.ft(INDEX_NAME).create_index(schema, definition=definition)

    async def add(self, namespace: str, exact: str, embedding: np.ndarray,
                  model_key: str, ttl_s: int) -> None:
        key = vector_key(namespace, exact)
        await self.redis.hset(key, mapping={
            "ns": namespace,
            "model_key": model_key,
            "exact": exact,
            "emb": embedding.astype(np.float32).tobytes(),
        })
        if ttl_s > 0:
            await self.redis.expire(key, ttl_s)

    async def knn(self, namespace: str, embedding: np.ndarray,
                  k: int = 1) -> list[tuple[str, float]]:
        """Top-k (exact_key, cosine_similarity) within the namespace only."""
        query = (
            Query(f"(@ns:{{{namespace}}})=>[KNN {k} @emb $vec AS dist]")
            .sort_by("dist")
            .return_fields("exact", "dist")
            .dialect(2)
        )
        try:
            res = await self.redis.ft(INDEX_NAME).search(
                query, query_params={"vec": embedding.astype(np.float32).tobytes()}
            )
        except Exception as e:
            if "not found" in str(e).lower():
                # index vanished (FLUSHALL / redis restart) — recreate; FT.CREATE
                # re-indexes existing cv:* keys, so treat this lookup as a miss
                await self.ensure_index()
                return []
            raise
        out: list[tuple[str, float]] = []
        for doc in res.docs:
            similarity = 1.0 - float(doc.dist)
            out.append((doc.exact, similarity))
        return out

    async def delete_namespace(self, namespace: str) -> int:
        deleted = 0
        async for key in self.redis.scan_iter(match=f"{VECTOR_PREFIX}{namespace}:*", count=500):
            await self.redis.delete(key)
            deleted += 1
        return deleted

    async def delete_model(self, model_key: str) -> int:
        """Invalidate every entry served by a model (uses the TAG field)."""
        deleted = 0
        escaped = model_key.replace("/", "\\/").replace("-", "\\-").replace(".", "\\.")
        while True:
            res = await self.redis.ft(INDEX_NAME).search(
                Query(f"@model_key:{{{escaped}}}").return_fields("ns", "exact").paging(0, 500)
            )
            if not res.docs:
                return deleted
            for doc in res.docs:
                ns, exact = doc.ns, doc.exact
                await self.redis.delete(vector_key(ns, exact))
                await self.redis.delete(f"ce:{ns}:{exact}")
                deleted += 1
