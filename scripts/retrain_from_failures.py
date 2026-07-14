"""Fold verification failures back into classifier training (SPEC C7).

A disagreement on a tier-T request means the prompt needed more capability
than tier T delivered -> harvest it labeled tier T+1 (capped at 3), then
retrain and print the before/after dev confusion so the README can show the
loop measurably shifting the planted prompts. Manual/weekly by design —
automation is v2.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from gateway.config import settings
from gateway.db import RoutingFailure, make_engine

HARVEST_PATH = Path("models/harvested_failures.jsonl")


async def harvest() -> int:
    engine = make_engine(settings.database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    rows = []
    async with sessions() as session:
        failures = (await session.execute(select(RoutingFailure))).scalars().all()
        for f in failures:
            prompt = (f.prompt_features or {}).get("prompt", "")
            if prompt:
                rows.append({"prompt": prompt, "tier": min(3, f.tier + 1),
                             "source": "verification_failure", "failure_id": f.id})
    await engine.dispose()

    HARVEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HARVEST_PATH.open("w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    print(f"Harvested {len(rows)} failure prompts -> {HARVEST_PATH}")
    return len(rows)


def main() -> int:
    n = asyncio.run(harvest())
    if n == 0:
        print("No failures to fold in; nothing to retrain.")
        return 0
    print("\nRetraining with harvested rows (before/after lives in "
          "models/classifier_meta.json — rerun scripts/train_classifier.py "
          "without --extra-training-rows to compare):")
    import subprocess

    return subprocess.call([
        sys.executable, "scripts/train_classifier.py",
        "--extra-training-rows", str(HARVEST_PATH),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
