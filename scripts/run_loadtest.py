"""Load-test orchestrator (make loadtest / make drill).

Usage: python scripts/run_loadtest.py steady repeats storm budget outage
Runs each scenario headless against the running stack (make up first),
writes locust CSVs + scenario JSONs into results/, and after `steady`
replays the held-out trap pairs against the live gateway to measure the
false-hit rate under production routing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GATEWAY = "http://localhost:8080"
RESULTS = Path("results")

SCENARIOS: dict[str, dict] = {
    "steady": {"users": 30, "rate": 5, "time": "120s"},
    "repeats": {"users": 15, "rate": 5, "time": "90s"},
    "storm": {"users": 10, "rate": 5, "time": "60s"},
    "budget": {"users": 5, "rate": 5, "time": "90s"},
    "outage": {"users": 20, "rate": 5, "time": "300s"},
}


def check_stack() -> None:
    try:
        httpx.get(f"{GATEWAY}/health", timeout=5).raise_for_status()
    except Exception as e:
        raise SystemExit(f"gateway not reachable at {GATEWAY} — run `make up && make seed` "
                         f"first ({e})") from e


def ensure_corpus() -> None:
    if not (RESULTS / "loadtest_corpus.jsonl").exists():
        subprocess.check_call([sys.executable, "scripts/build_loadtest_corpus.py"])


def run_scenario(name: str) -> int:
    spec = SCENARIOS[name]
    print(f"\n=== scenario: {name} ({spec['time']}, {spec['users']} users) ===")
    return subprocess.call([
        sys.executable, "-m", "locust",
        "-f", f"loadtest/{name}.py", "--headless",
        "-u", str(spec["users"]), "-r", str(spec["rate"]), "-t", spec["time"],
        "--host", GATEWAY,
        "--csv", str(RESULTS / name), "--csv-full-history",
        "--only-summary",
    ])


def replay_traps() -> None:
    """Measure the held-out trap corpus against the LIVE gateway (demo team,
    fresh namespace): prime with prompt_a, then probe with prompt_b."""
    from gateway.datasets import load_cache_traps

    print("\n=== trap replay under production routing (held-out test pairs) ===")
    key = {"Authorization": "Bearer relay-demo-key"}
    stats = {"hit_pairs": 0, "hit_collided": 0, "miss_pairs": 0, "miss_false_hits": 0,
             "bypassed": 0, "false_hit_ids": []}
    with httpx.Client(timeout=60) as client:
        for trap in load_cache_traps():
            if trap["split"] != "test":
                continue
            prime = client.post(f"{GATEWAY}/v1/chat/completions", headers=key,
                                json={"model": "relay-auto", "messages":
                                      [{"role": "user", "content": trap["prompt_a"]}]})
            if prime.status_code != 200:
                continue
            time.sleep(0.15)  # let the async cache write land
            probe = client.post(f"{GATEWAY}/v1/chat/completions", headers=key,
                                json={"model": "relay-auto", "messages":
                                      [{"role": "user", "content": trap["prompt_b"]}]})
            if probe.status_code != 200:
                continue
            collided = probe.headers.get("x-relay-cache") == "hit"
            if trap["expected"] == "hit":
                stats["hit_pairs"] += 1
                stats["hit_collided"] += int(collided)
            else:
                stats["miss_pairs"] += 1
                if collided:
                    stats["miss_false_hits"] += 1
                    stats["false_hit_ids"].append(trap["id"])
    if stats["hit_pairs"]:
        stats["hit_rate"] = round(stats["hit_collided"] / stats["hit_pairs"], 4)
    if stats["miss_pairs"]:
        stats["false_hit_rate"] = round(stats["miss_false_hits"] / stats["miss_pairs"], 4)
    (RESULTS / "traps_under_load.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main() -> int:
    names = sys.argv[1:] or ["steady"]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenarios {unknown}; choose from {list(SCENARIOS)}")
    check_stack()
    ensure_corpus()
    RESULTS.mkdir(exist_ok=True)
    failed = 0
    for name in names:
        failed |= run_scenario(name)
        if name == "steady":
            replay_traps()
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
