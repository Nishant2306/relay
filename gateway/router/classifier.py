"""Runtime tier classification (SPEC C6): <1 ms, interpretable, retrainable.

Confidence below `min_confidence` promotes one tier — fail-safe toward
quality, never toward cost.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from gateway.router.features import extract_features

DEFAULT_MODEL_PATH = Path("models/classifier.joblib")


class ComplexityClassifier:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self._model = joblib.load(model_path)
        self._classes: list[int] = [int(c) for c in self._model.classes_]

    def classify(self, text: str) -> tuple[int, float]:
        """(tier, confidence) — confidence is the winning class probability."""
        proba = self._model.predict_proba(extract_features(text).reshape(1, -1))[0]
        best = int(proba.argmax())
        return self._classes[best], float(proba[best])


class HeuristicClassifier:
    """Fallback when no trained model exists (SPEC cut order #4): a blunt
    cue-count rule, clearly weaker — the gateway logs a warning when active."""

    def classify(self, text: str) -> tuple[int, float]:
        from gateway.router.features import extract_feature_dict

        f = extract_feature_dict(text)
        score = (
            f["heavy_verb_count"] + f["judgment_cue_count"] + f["multi_step"]
            + f["creative_constrained"] - f["light_verb_count"] - f["short_simple_question"]
        )
        if score >= 3:
            return 3, 0.5
        if score >= 1:
            return 2, 0.5
        return 1, 0.5
