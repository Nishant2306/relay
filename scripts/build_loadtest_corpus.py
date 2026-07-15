"""Build the load-test corpus (SPEC C11) — leakage-safe by construction.

- 1,000 unique prompts sampled from the TRAIN split ONLY of the complexity
  dataset (never dev/test — reported numbers must not see tuning data);
  train has 429 uniques, so uniqueness comes from deterministic surface
  variants that preserve the task
- 300 exact repeats of corpus prompts
- 300 paraphrased variants (template paraphrases; deterministic, $0)
- the trap prompts from the held-out TEST split, included to measure the
  false-hit rate under realistic load

Output: results/loadtest_corpus.jsonl with {kind, prompt, ref} rows.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.datasets import load_cache_traps, load_complexity_labels

OUT = Path("results/loadtest_corpus.jsonl")
RNG = random.Random(42)

PREFIXES = ["", "Please ", "Could you ", "Quick request: ", "Hey — ", "Task: "]
SUFFIXES = ["", " Thanks!", " Keep it brief.", " Respond clearly."]

PARAPHRASE_WRAPPERS = [
    "Could you {p}",
    "I'd like you to {p}",
    "Please help me with this: {p}",
    "Here's what I need — {p}",
    "{p} (answer concisely)",
]


def surface_variant(prompt: str, i: int) -> str:
    """Unique-but-equivalent surface form (prefix/suffix padding only)."""
    prefix = PREFIXES[i % len(PREFIXES)]
    suffix = SUFFIXES[(i // len(PREFIXES)) % len(SUFFIXES)]
    body = prompt[0].lower() + prompt[1:] if prefix and prompt else prompt
    return f"{prefix}{body}{suffix}"


def paraphrase(prompt: str, i: int) -> str:
    body = prompt[0].lower() + prompt[1:] if prompt else prompt
    return PARAPHRASE_WRAPPERS[i % len(PARAPHRASE_WRAPPERS)].format(p=body)


def main() -> int:
    labels = load_complexity_labels()
    train_prompts = [r["prompt"] for r in labels if r["split"] == "train"]
    assert all(r["split"] == "train" for r in labels if r["prompt"] in set(train_prompts))

    rows: list[dict] = []

    # 1,000 uniques from train only
    uniques: list[str] = []
    i = 0
    while len(uniques) < 1000:
        base = train_prompts[i % len(train_prompts)]
        uniques.append(base if i < len(train_prompts) else surface_variant(base, i))
        i += 1
    uniques = list(dict.fromkeys(uniques))[:1000]
    rows += [{"kind": "unique", "prompt": p} for p in uniques]

    # 300 exact repeats
    for p in RNG.sample(uniques, 300):
        rows.append({"kind": "repeat", "prompt": p, "ref": p})

    # 300 paraphrases
    for j, p in enumerate(RNG.sample(uniques, 300)):
        rows.append({"kind": "paraphrase", "prompt": paraphrase(p, j), "ref": p})

    # held-out trap pairs (the false-hit measurement under load)
    for trap in load_cache_traps():
        if trap["split"] == "test":
            rows.append({
                "kind": f"trap_{trap['expected']}", "prompt": trap["prompt_a"],
                "ref": trap["prompt_b"], "trap_id": trap["id"],
                "expected": trap["expected"],
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(f"wrote {OUT} — {len(rows)} rows: {kinds}")
    print("corpus draws from the TRAIN split only; traps from held-out TEST (SPEC C11)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
