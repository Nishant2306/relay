"""Answer-determining guards for semantic candidates (SPEC C5).

Raw cosine similarity cannot separate paraphrases from traps on the frozen
corpus: hit and miss pairs have nearly identical similarity distributions
(median 0.911 vs 0.913 — see results/cache_trap_eval.json). Trap pairs are
*designed* lookalikes: "convert miles to km" vs "convert km to miles" embeds
at 0.99. So a KNN candidate must also survey cheap deterministic checks on the
things that change an answer:

  1. numbers        — normalized numeric multisets must match (handles
                      "1,000" vs "1000", "5 PM" vs "17:00", "ten" vs "10")
  2. negation       — negation presence must match on both sides
  3. inverse pairs  — min/max, encrypt/decrypt, sum/average, before/after, ...
  4. direction      — "X to Y" vs "Y to X" argument swaps

Guards were tuned exclusively against the 40 DEV pairs (ADR-0009); the
held-out test numbers in the README come from running the full matcher
(similarity threshold + guards) on the 80 test pairs once.
"""

from __future__ import annotations

import difflib
import re

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
    "thousand": 1000, "million": 1_000_000,
}

_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_CLOCK_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")

_NEGATION = frozenset({
    "not", "no", "never", "without", "none", "cannot", "can't", "don't",
    "doesn't", "isn't", "aren't", "won't", "shouldn't", "exclude", "excluding",
    "excluded", "disable", "disabled", "disallow", "disallowed", "deny",
    "denied", "forbid", "forbidden", "except", "non",
})

_INVERSE_PAIRS: list[tuple[str, str]] = [
    ("minimum", "maximum"), ("min", "max"), ("highest", "lowest"),
    ("largest", "smallest"), ("biggest", "smallest"), ("most", "least"),
    ("ascending", "descending"), ("asc", "desc"),
    ("encrypt", "decrypt"), ("encode", "decode"), ("encoding", "decoding"),
    ("compress", "decompress"), ("serialize", "deserialize"),
    ("add", "remove"), ("adding", "removing"), ("insert", "delete"),
    ("enable", "disable"), ("enabled", "disabled"),
    ("before", "after"), ("above", "below"), ("over", "under"),
    ("import", "export"), ("upload", "download"),
    ("inclusive", "exclusive"), ("include", "exclude"),
    ("sum", "average"), ("sum", "mean"), ("total", "average"),
    ("count", "average"), ("median", "mean"),
    ("first", "last"), ("oldest", "newest"), ("earliest", "latest"),
    ("read", "write"), ("reads", "writes"), ("get", "set"),
    ("open", "close"), ("start", "stop"), ("starts", "stops"),
    ("allow", "deny"), ("grant", "revoke"), ("allowed", "denied"),
    ("rows", "columns"), ("row", "column"), ("width", "height"),
    ("input", "output"), ("inputs", "outputs"), ("source", "target"),
    ("left", "right"), ("inner", "outer"), ("union", "intersection"),
    ("increase", "decrease"), ("increases", "decreases"),
    ("gain", "loss"), ("profit", "loss"), ("buy", "sell"),
    ("synchronous", "asynchronous"), ("sync", "async"),
    ("horizontal", "vertical"), ("north", "south"), ("east", "west"),
    ("open", "closed"), ("public", "private"), ("internal", "external"),
    ("client", "server"), ("push", "pull"), ("local", "remote"),
    ("shortest", "longest"), ("fastest", "slowest"), ("newest", "oldest"),
]

_DIR_RE = re.compile(r"\b([\w.+#-]+)\s+(?:to|into)\s+([\w.+#-]+)")
_FROM_TO_RE = re.compile(r"\bfrom\s+([\w.+#-]+)\b.{0,50}?\bto\s+([\w.+#-]+)")

_TOKEN_RE = re.compile(r"[a-z][a-z'+-]*|\d+")


def _normalize_times(text: str) -> str:
    """'5 PM' -> '17', '17:00' -> '17' so clock formats compare as hours."""

    def ampm(m: re.Match) -> str:
        hour = int(m.group(1)) % 12
        if m.group(3).lower() == "pm":
            hour += 12
        minutes = m.group(2)
        return f" {hour} {minutes} " if minutes and minutes != "00" else f" {hour} "

    def clock(m: re.Match) -> str:
        return f" {int(m.group(1))} " if m.group(2) == "00" \
            else f" {int(m.group(1))} {m.group(2)} "

    return _CLOCK_RE.sub(clock, _TIME_RE.sub(ampm, text))


def _numbers(text: str) -> tuple[float, ...]:
    text = _normalize_times(text.casefold())
    nums = [float(n.replace(",", "")) for n in _NUM_RE.findall(text)]
    for word, value in _WORD_NUMBERS.items():
        nums.extend([float(value)] * len(re.findall(rf"\b{word}\b", text)))
    return tuple(sorted(nums))


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.casefold()))


def _has_negation(tokens: set[str]) -> bool:
    return bool(tokens & _NEGATION)


def _direction_pairs(text: str) -> set[tuple[str, str]]:
    text = text.casefold()
    raw = _DIR_RE.findall(text) + _FROM_TO_RE.findall(text)
    # the token class allows '.' (node.js, 3.5) — trim sentence punctuation
    return {
        (a2, b2)
        for a, b in raw
        if (a2 := a.strip(".,;:!?")) != (b2 := b.strip(".,;:!?"))
    }


_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "isn't": "is not",
    "aren't": "are not", "can't": "cannot", "won't": "will not",
    "shouldn't": "should not", "couldn't": "could not", "wouldn't": "would not",
    "it's": "it is", "what's": "what is", "i'm": "i am",
}

_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")


def _seq_tokens(text: str) -> list[str]:
    """Ordered token list for structural alignment: casefolded, times
    normalized, contractions expanded, hyphens fused, punctuation stripped."""
    text = _normalize_times(text.casefold())
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)  # e-mail -> email, comma-separated -> commaseparated
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)
    # word numbers align with digits ("ten items" == "10 items")
    return [str(_WORD_NUMBERS[t]) if t in _WORD_NUMBERS else t for t in tokens]


def _equivalent_tokens(x: str, y: str) -> bool:
    """Same word modulo morphology (value/values, reverse/reversed) or both
    numeric (the global number guard already compared multisets)."""
    if x == y:
        return True
    if _NUMERIC_RE.match(x) and _NUMERIC_RE.match(y):
        return True
    prefix = 0
    for cx, cy in zip(x, y):
        if cx != cy:
            break
        prefix += 1
    shorter = min(len(x), len(y))
    return prefix >= 4 and prefix >= shorter - 3 and max(len(x), len(y)) <= prefix + 4


def _substitution_conflict(query: str, candidate: str) -> str | None:
    """Structural near-duplicates that differ by a small content-word swap
    ("Paris" -> "London", "GET" -> "POST", "sentence" -> "page") are traps:
    the surrounding template is identical, so embedding similarity is sky-high,
    but the swapped word determines the answer.

    Real paraphrases restructure (inserts/deletes dominate their diff), so
    requiring pure same-length replace spans keeps them out of this guard's
    blast radius. A false reject only costs one upstream call; a false accept
    serves a wrong answer — so ambiguity resolves toward rejection."""
    qt, ct = _seq_tokens(query), _seq_tokens(candidate)
    if not qt or not ct:
        return None
    matcher = difflib.SequenceMatcher(a=qt, b=ct, autojunk=False)
    substitutions: list[tuple[list[str], list[str]]] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and (i2 - i1) <= 2 and (i2 - i1) == (j2 - j1):
            substitutions.append((qt[i1:i2], ct[j1:j2]))
            continue
        return None  # restructuring edit -> paraphrase-shaped, not a swap
    if not substitutions or sum(len(a) for a, _ in substitutions) > 3:
        return None
    for q_span, c_span in substitutions:
        for x, y in zip(q_span, c_span):
            if not _equivalent_tokens(x, y):
                return f"content_substitution:{x}/{y}"
    return None


def answer_determining_conflict(query: str, candidate: str) -> str | None:
    """Return the reason a candidate must NOT be served for this query,
    or None if no guard fires. Both inputs are raw prompt texts."""
    if _numbers(query) != _numbers(candidate):
        return "number_mismatch"

    q_tokens, c_tokens = _tokens(query), _tokens(candidate)
    if _has_negation(q_tokens) != _has_negation(c_tokens):
        return "negation_mismatch"

    only_q, only_c = q_tokens - c_tokens, c_tokens - q_tokens
    for x, y in _INVERSE_PAIRS:
        if (x in only_q and y in only_c) or (y in only_q and x in only_c):
            return f"inverse_pair:{x}/{y}"

    # uppercase boolean operators (AND/OR swaps in query-language prompts)
    q_bool = {t for t in ("AND", "OR") if re.search(rf"\b{t}\b", query)}
    c_bool = {t for t in ("AND", "OR") if re.search(rf"\b{t}\b", candidate)}
    if q_bool and c_bool and q_bool != c_bool:
        return "boolean_operator_swap"

    q_dirs, c_dirs = _direction_pairs(query), _direction_pairs(candidate)
    for a, b in q_dirs:
        if (b, a) in c_dirs and (a, b) not in c_dirs:
            return f"direction_swap:{a}->{b}"

    return _substitution_conflict(query, candidate)
