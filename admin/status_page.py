"""Minimal Streamlit status page rendering /admin/stats (SPEC C10).

Grafana is the real UI; this is the at-a-glance savings summary.
Run: streamlit run admin/status_page.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

GATEWAY = os.environ.get("RELAY_GATEWAY_URL", "http://localhost:8080")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "relay-admin-dev-key")

st.set_page_config(page_title="Relay status", page_icon="R", layout="centered")
st.title("Relay — savings summary")

try:
    stats = httpx.get(
        f"{GATEWAY}/admin/stats", headers={"x-admin-key": ADMIN_KEY}, timeout=10
    ).raise_for_status().json()
except Exception as e:  # pragma: no cover - display-only
    st.error(f"Could not reach the gateway at {GATEWAY}: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Requests (ok)", f"{stats['requests_ok']:,}")
hit_rate = stats["cache_hit_rate"]
col2.metric("Cache hit rate", f"{hit_rate:.1%}" if hit_rate is not None else "n/a")
savings = stats["savings_pct"]
col3.metric("Savings vs flagship", f"{savings:.1%}" if savings is not None else "n/a")

st.subheader("Savings attribution (ADR-0008 — never blended)")
attribution = stats["savings_attribution_usd"]
st.bar_chart({
    "USD saved": {
        "cache hits": attribution["cache_hits"],
        "down-routing": attribution["down_routing"],
    }
})

col4, col5, col6 = st.columns(3)
col4.metric("Actual spend", f"${stats['actual_cost_usd']:.4f}")
col5.metric("Counterfactual", f"${stats['counterfactual_cost_usd']:.4f}")
col6.metric("Avg overhead", f"{stats['avg_overhead_ms']:.1f} ms")

st.caption("Counterfactual = every request priced at the flagship model. "
           "Full dashboards live in Grafana (:3000).")
