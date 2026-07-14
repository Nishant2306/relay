"""LLM-judge scoring (SPEC C7): does the cheap answer agree with the top
model's answer, 1-5?

When the judge's reply contains no parseable score (the chaos mock returns
deterministic word salad), a difflib similarity fallback keeps the loop
functional in $0 dev/load-test environments — real judges (routing.yaml
verification.judge) produce real scores in demos.
"""

from __future__ import annotations

import difflib
import re

JUDGE_SYSTEM = (
    "You grade agreement between two answers to the same prompt. "
    "5 = same substance and correctness, 1 = contradictory or wrong. "
    "Respond with ONLY one digit, 1-5."
)

_SCORE_RE = re.compile(r"\b([1-5])\b")


def judge_messages(prompt: str, cheap_answer: str, top_answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"PROMPT:\n{prompt}\n\n"
                f"REFERENCE ANSWER (top-tier model):\n{top_answer}\n\n"
                f"CANDIDATE ANSWER (cheaper model):\n{cheap_answer}\n\n"
                "Agreement score (1-5):"
            ),
        },
    ]


def parse_score(judge_reply: str, cheap_answer: str, top_answer: str) -> tuple[int, str]:
    """(score, source) where source is 'judge' or 'similarity_fallback'."""
    match = _SCORE_RE.search(judge_reply.strip()[:40])
    if match:
        return int(match.group(1)), "judge"
    ratio = difflib.SequenceMatcher(a=cheap_answer, b=top_answer, autojunk=False).ratio()
    if ratio > 0.9:
        return 5, "similarity_fallback"
    if ratio > 0.75:
        return 4, "similarity_fallback"
    if ratio > 0.6:
        return 3, "similarity_fallback"
    if ratio > 0.4:
        return 2, "similarity_fallback"
    return 1, "similarity_fallback"
