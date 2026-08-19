"""Tests for src/segment.py — sentence splitting on known inputs.

Skips cleanly if the spaCy model is not downloaded yet
(uv run python -m spacy download en_core_web_sm).

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import pytest

spacy = pytest.importorskip("spacy")

try:  # skip the whole module if the model is missing, not error
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("spaCy model en_core_web_sm not downloaded", allow_module_level=True)

from src.segment import split_sentences  # noqa: E402  (guarded import above)


def test_splits_two_sentences() -> None:
    """Two clear sentences come back as two trimmed units."""
    assert split_sentences("I buy online. It is cheaper.") == [
        "I buy online.",
        "It is cheaper.",
    ]


def test_handles_questions_and_exclamations() -> None:
    """?, ! and . all end sentences."""
    result = split_sentences("Why switch? I love it! Price matters.")
    assert len(result) == 3


def test_single_sentence_stays_whole() -> None:
    """No sentence-ending punctuation -> one unit, not zero."""
    assert split_sentences("i usually buy my products online") == [
        "i usually buy my products online"
    ]


def test_empty_text_gives_empty_list() -> None:
    """Empty and whitespace-only input never crash."""
    assert split_sentences("") == []
    assert split_sentences("   \n ") == []
