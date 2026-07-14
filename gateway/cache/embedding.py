"""Local embeddings for cache lookups (ADR-0001).

fastembed / bge-small-en-v1.5, 384-d, CPU, ~5-15 ms per prompt. Lookups need
*consistency*, not SOTA retrieval quality — and a paid embedding API on every
request would add cost and 20-60 ms latency to the thing meant to save both.

Similarity is symmetric sentence-to-sentence, so no bge query instruction
prefix is applied to either side.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding

            _model = TextEmbedding(model_name=_MODEL_NAME)
    return _model


def embed_sync(texts: list[str]) -> np.ndarray:
    """L2-normalized float32 embeddings, shape (n, 384)."""
    model = _get_model()
    vectors = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


async def embed(texts: list[str]) -> np.ndarray:
    """Async wrapper — the ONNX runtime is sync, so run it off the event loop."""
    return await asyncio.to_thread(embed_sync, texts)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
