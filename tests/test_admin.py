"""C10: admin auth, routing validate+hot-apply+audit, cache invalidation."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml
from asgi_lifespan import LifespanManager

from gateway.router.routes import RoutingConfigStore
from tests.conftest import build_harness

ROUTING_YAML = Path(__file__).resolve().parent.parent / "config" / "routing.yaml"
ADMIN_KEY = "test-admin-key"


class FakeCache:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def invalidate_namespace(self, ns: str) -> int:
        self.calls.append(("namespace", ns))
        return 3

    async def invalidate_model(self, model: str) -> int:
        self.calls.append(("model", model))
        return 5

    async def flush_all(self) -> int:
        self.calls.append(("all", ""))
        return 42

    async def tuning_table(self, thresholds, dev_curve=None):
        return [{"threshold": t, "observed_lookups": 0} for t in thresholds]


@pytest.fixture
async def admin_harness(tmp_path):
    routing_path = tmp_path / "routing.yaml"
    routing_path.write_text(ROUTING_YAML.read_text(encoding="utf-8"), encoding="utf-8")
    store = RoutingConfigStore(routing_path)
    audits: list[tuple[str, str]] = []

    async def audit(actor: str, path: str, old, new) -> None:
        audits.append((actor, path))

    h = await build_harness()
    h.deps.extras.update({
        "routing_store": store, "admin_key": ADMIN_KEY,
        "audit_writer": audit, "cache": FakeCache(),
    })
    from gateway.main import create_app

    app = create_app(h.deps)
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay"
        ) as client:
            yield client, store, audits, h.deps.extras["cache"]


def admin_headers() -> dict[str, str]:
    return {"x-admin-key": ADMIN_KEY}


class TestAdminAuth:
    async def test_no_key_is_401(self, admin_harness):
        client, *_ = admin_harness
        assert (await client.get("/admin/routing")).status_code == 401
        assert (
            await client.get("/admin/routing", headers={"x-admin-key": "wrong"})
        ).status_code == 401


class TestRoutingAdmin:
    async def test_get_and_put_roundtrip_with_audit(self, admin_harness):
        client, store, audits, _ = admin_harness
        current = (await client.get("/admin/routing", headers=admin_headers())).json()
        assert current["tiers"][1 if 1 in current["tiers"] else "1"]

        current["tiers"]["1"] = current["tiers"].pop(1, current["tiers"].get("1"))
        raw = yaml.safe_load(store.path.read_text(encoding="utf-8"))
        raw["tiers"][1]["chain"] = ["mock/mid-b", "mock/cheap-a"]
        r = await client.put("/admin/routing", json=raw, headers=admin_headers())
        assert r.status_code == 200
        assert store.config.tiers[1].chain == ["mock/mid-b", "mock/cheap-a"]  # hot-applied
        assert ("admin", "/admin/routing") in audits

    async def test_invalid_routing_rejected_with_field_errors(self, admin_harness):
        client, store, audits, _ = admin_harness
        before = store.config.tiers[1].chain
        raw = yaml.safe_load(store.path.read_text(encoding="utf-8"))
        raw["tiers"][1]["chain"] = []
        raw["verification"]["sample_rate"] = 9
        r = await client.put("/admin/routing", json=raw, headers=admin_headers())
        assert r.status_code == 422
        paths = {err["path"] for err in r.json()["detail"]}
        assert any("chain" in p for p in paths)
        assert any("sample_rate" in p for p in paths)
        assert store.config.tiers[1].chain == before  # not applied
        assert ("admin", "/admin/routing") not in audits


class TestCacheAdmin:
    async def test_invalidate_scopes_and_audit(self, admin_harness):
        client, _, audits, cache = admin_harness
        r = await client.post("/admin/cache/invalidate",
                              json={"scope": "model", "value": "mock/cheap-a"},
                              headers=admin_headers())
        assert r.json()["deleted"] == 5
        r = await client.post("/admin/cache/invalidate", json={"scope": "all"},
                              headers=admin_headers())
        assert r.json()["deleted"] == 42
        assert cache.calls == [("model", "mock/cheap-a"), ("all", "")]
        assert audits.count(("admin", "/admin/cache/invalidate")) == 2

    async def test_bad_scope_rejected(self, admin_harness):
        client, *_ = admin_harness
        r = await client.post("/admin/cache/invalidate", json={"scope": "everything"},
                              headers=admin_headers())
        assert r.status_code == 422

    async def test_tuning_endpoint(self, admin_harness):
        client, *_ = admin_harness
        r = await client.get("/admin/cache/tuning?thresholds=0.8,0.9",
                             headers=admin_headers())
        assert [row["threshold"] for row in r.json()] == [0.8, 0.9]


class TestMetricsEndpoint:
    async def test_metrics_exposed(self, admin_harness):
        client, *_ = admin_harness
        from gateway.obs.metrics import RelayMetrics

        # not wired in this harness -> placeholder; wire one and check export
        r = await client.get("/metrics")
        assert r.status_code == 200
