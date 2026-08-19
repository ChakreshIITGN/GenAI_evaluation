"""Sentence segmentation for Layer 2 (plan §2a).

The unit of content checking is the SENTENCE: spaCy's rule-based sentence
splitter, nothing else (no clause splitting, no coreference, no LLM
decomposition) — the same recipe SummaC-style NLI consistency work uses.
The spaCy pipeline is loaded once and reused for every call.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import spacy
import yaml

_CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())
_SPACY_MODEL: str = _CONFIG["eval"]["spacy_model"]

# Loaded lazily on first use, then cached — importing this module must not
# require the spaCy model to be downloaded (keeps unrelated tests runnable).
_NLP: spacy.language.Language | None = None


def _nlp() -> spacy.language.Language:
    """Return the shared spaCy pipeline, loading it on first call only."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(_SPACY_MODEL, disable=["ner", "lemmatizer"])
    return _NLP


def split_sentences(text: str) -> list[str]:
    """Split one answer into sentences.

    Args:
        text: raw answer text (any case, any length).

    Returns:
        Non-empty sentences, whitespace-trimmed, in order. Empty or
        whitespace-only input returns [].
    """
    if not text.strip():
        return []
    return [sent.text.strip() for sent in _nlp()(text).sents if sent.text.strip()]
