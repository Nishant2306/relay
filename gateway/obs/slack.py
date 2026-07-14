"""Slack webhook alerts (budget warnings, breaker transitions). No-op when unset."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("relay.slack")


class SlackNotifier:
    def __init__(self, webhook_url: str, http: httpx.AsyncClient | None = None):
        self.webhook_url = webhook_url
        self._http = http
        self.sent: list[str] = []  # kept for tests/inspection

    async def send(self, text: str) -> None:
        self.sent.append(text)
        if not self.webhook_url:
            logger.info("slack (disabled): %s", text)
            return
        try:
            client = self._http or httpx.AsyncClient(timeout=5.0)
            resp = await client.post(self.webhook_url, json={"text": text})
            resp.raise_for_status()
            if self._http is None:
                await client.aclose()
        except Exception:
            logger.exception("slack notification failed (never blocks serving)")
