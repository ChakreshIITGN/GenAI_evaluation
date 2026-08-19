"""Semantic similarity baseline — Layer 1a (plan §3, 1a).

Cosine similarity between each human/AI answer pair using a sentence
embedding model. Reported as a baseline ONLY: it is topic-dominated and
stance-blind (a polite opposite scores high), so it is never a KPI.
The model is loaded once, on first use.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

_CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())
_EMBED_MODEL_NAME: str = _CONFIG["eval"]["embedding_model"]

# Loaded lazily on first use, then cached (module import stays light).
_MODEL: SentenceTransformer | None = None


def _model() -> SentenceTransformer:
    """Return the shared embedding model, loading it on first call only."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    return _MODEL


def pair_similarity(human_texts: list[str], ai_texts: list[str]) -> list[float]:
    """Cosine similarity per (human, AI) answer pair.

    Args:
        human_texts: human answers, one per pair.
        ai_texts: AI answers, aligned to human_texts by position.

    Returns:
        One cosine similarity in [-1, 1] per pair.

    Raises:
        ValueError: if the two lists differ in length.
    """
    if len(human_texts) != len(ai_texts):
        raise ValueError("human_texts and ai_texts must be the same length")
    # normalize_embeddings=True makes the dot product the cosine similarity.
    human_vecs = _model().encode(human_texts, normalize_embeddings=True)
    ai_vecs = _model().encode(ai_texts, normalize_embeddings=True)
    return [float(np.dot(h, a)) for h, a in zip(human_vecs, ai_vecs)]
