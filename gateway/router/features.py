"""Hand-built feature extraction for the complexity classifier (SPEC C6).

Rubric line 7 is a hard constraint here: tier is a property of the required
WORK, not of prompt length. Length appears as *a* feature but the 45.3%
length-only control exists to prove the classifier learned more than
character counting. Every feature is derived from raw text only — the
dataset's features_for_audit fields are never read at runtime, so training
and serving share this exact code path.
"""

from __future__ import annotations

import re

import numpy as np

# Cues for reasoning-intensive work (tier 3 direction). Rubric line 8: these
# are cues, not labels — the classifier weighs them against everything else.
_HEAVY_VERBS = re.compile(
    r"\b(analy[sz]e|compare|design|prove|derive|optimi[sz]e|architect|evaluate|"
    r"justify|debug|diagnose|isolate|synthesi[sz]e|critique|assess|threat.?model|"
    r"trade.?offs?|strategy|strategi[sz]e|reconcile|refactor|formali[sz]e|"
    r"model|simulate|plan|prioriti[sz]e|decide|recommend|choose|weigh)\b",
    re.IGNORECASE,
)

# Bounded-work cues (tier 1 direction).
_LIGHT_VERBS = re.compile(
    r"\b(extract|list|convert|translate|reformat|rename|capitali[sz]e|count|"
    r"look ?up|classify|label|sort|dedupe|normalize|parse|sum|multiply|"
    r"return only|rewrite|correct|fix)\b",
    re.IGNORECASE,
)

_CONSTRAINT_MARKERS = re.compile(
    r"\b(must|at least|at most|no more than|under \d+|within \d+|keep under|"
    r"exactly|only|except|unless|limit(ed)? to|between \d+|per \w+)\b",
    re.IGNORECASE,
)

_FORMAT_DIRECTIVES = re.compile(
    r"\b(format as|return (only|as|no)|respond (with|in)|output (as|only)|"
    r"in (json|yaml|csv|xml|markdown|bullets?)|as a (table|list)|"
    r"bullet(s| points?)?|numbered list|sections?|fields?|headers?|columns?)\b",
    re.IGNORECASE,
)

_CODE_CUES = re.compile(
    r"\b(code|function|script|sql|regex|endpoint|implement|class|method|api|"
    r"python|javascript|typescript|postgresql|query|algorithm|compile)\b",
    re.IGNORECASE,
)

_JSON_CUES = re.compile(r"\b(json|yaml|csv|xml|schema)\b", re.IGNORECASE)

_MULTI_STEP = re.compile(
    r"\b(then|after that|first\b.*\bthen|step(s| by step)?|finally|next,|"
    r"followed by|and (also|then))\b",
    re.IGNORECASE | re.DOTALL,
)

_JUDGMENT_CUES = re.compile(
    r"\b(should|whether|risks?|ethic(s|al)|fair(ness)?|nuance[ds]?|ambiguous|"
    r"uncertain|acceptable|appropriate|policy|safety|caveats?|assumptions?|"
    r"sensitivity|qualitative)\b",
    re.IGNORECASE,
)

_CREATIVE_CUES = re.compile(
    r"\b(story|poem|song|satir(e|ical)|fiction(al)?|diary|letter|haiku|"
    r"blank verse|memo from|children'?s|imagine|persona)\b",
    re.IGNORECASE,
)

_QUESTION_START = re.compile(
    r"^\s*(what|who|when|where|why|how|is|are|does|do|can|will|which|should)\b",
    re.IGNORECASE,
)

_ATTACHED_CONTEXT = re.compile(
    r"(:\s*['\"]|:\s*\n|```|this (report|text|code|file|column|dataset|table|"
    r"function|email|list|log))",
    re.IGNORECASE,
)

# Structured-synthesis cues (tier 2): reorganize supplied material under
# fidelity constraints — surface-similar to tier-1 extraction, but the work
# is synthesis ("turn these notes into...", "do not invent").
_SYNTHESIS_CUES = re.compile(
    r"\b(turn (these|this|the)|organi[sz]e|consolidate|structure (these|this)|"
    r"do not invent|synthesi[sz]e|distill)\b",
    re.IGNORECASE,
)

# Heavily-constrained creation is tier 3 by rubric line 5 ("constrained
# creation"), unlike freeform creative asks.
_CREATIVE_CONSTRAINTS = re.compile(
    r"\b(maintain|consistent|consistency|coheren(t|ce)|converging|arc|"
    r"technically|preserving|while (keeping|preserving|avoiding))\b",
    re.IGNORECASE,
)

FEATURE_NAMES: list[str] = [
    "approx_tokens",
    "word_count",
    "sentence_count",
    "heavy_verb_count",
    "light_verb_count",
    "constraint_count",
    "format_directive_count",
    "requests_code",
    "requests_json",
    "multi_step",
    "judgment_cue_count",
    "creative_cue_count",
    "is_question",
    "has_attached_context",
    "comma_density",
    "digit_count",
    "synthesis_cue_count",
    "creative_constrained",
    "short_simple_question",
    "attached_context_chars",
]


def extract_features(text: str) -> np.ndarray:
    words = text.split()
    word_count = len(words)
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))
    heavy = len(_HEAVY_VERBS.findall(text))
    constraints = len(_CONSTRAINT_MARKERS.findall(text))
    creative = len(_CREATIVE_CUES.findall(text))
    context_match = _ATTACHED_CONTEXT.search(text)
    attached_chars = float(len(text) - context_match.start()) if context_match else 0.0
    short_simple_q = float(
        bool(_QUESTION_START.match(text)) and word_count <= 12
        and heavy == 0 and constraints == 0
    )
    values = [
        len(text) / 4.0,
        float(word_count),
        float(sentences),
        float(heavy),
        float(len(_LIGHT_VERBS.findall(text))),
        float(constraints),
        float(len(_FORMAT_DIRECTIVES.findall(text))),
        float(bool(_CODE_CUES.search(text))),
        float(bool(_JSON_CUES.search(text))),
        float(bool(_MULTI_STEP.search(text))),
        float(len(_JUDGMENT_CUES.findall(text))),
        float(creative),
        float(bool(_QUESTION_START.match(text))),
        float(bool(context_match)),
        text.count(",") / max(1, sentences),
        float(sum(c.isdigit() for c in text)),
        float(len(_SYNTHESIS_CUES.findall(text))),
        float(creative > 0 and len(_CREATIVE_CONSTRAINTS.findall(text)) >= 1),
        short_simple_q,
        attached_chars,
    ]
    return np.array(values, dtype=np.float64)


def extract_feature_dict(text: str) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, extract_features(text), strict=True))
