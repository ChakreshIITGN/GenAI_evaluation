"""Tests for src/stylometric.py — known inputs -> known outputs per feature.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import pytest

from src.stylometric import Stylometric

FEATURE_KEYS = [
    "length",
    "lexical_diversity",
    "mean_word_length",
    "contraction_rate",
    "hedging_density",
    "sentiment_mean",
    "sentiment_std",
    "stance_rate",
]


def test_length_counts_words() -> None:
    """Three words -> 3; contractions count as one word."""
    assert Stylometric("one two three").length() == 3
    assert Stylometric("I can't go").length() == 3


def test_mean_word_length_known_value() -> None:
    """'aa bbbb' -> (2 + 4) / 2 = 3 letters per word."""
    assert Stylometric("aa bbbb").mean_word_length() == 3.0


def test_contraction_rate_two_in_ten() -> None:
    """Two contractions in ten words -> 0.2 (the plan's own example)."""
    text = "i can't say i won't go to the shop today"
    assert Stylometric(text).length() == 10
    assert Stylometric(text).contraction_rate() == pytest.approx(0.2)


def test_hedging_density_per_100_words() -> None:
    """One hedge ('maybe') in four words -> 25 per 100 words."""
    assert Stylometric("maybe it works fine").hedging_density() == pytest.approx(25.0)


def test_hedging_counts_multiword_phrases() -> None:
    """'kind of' is one hedge; 'may' must not match inside 'maybe'."""
    assert Stylometric("it is kind of nice").hedging_density() == pytest.approx(20.0)


def test_stance_rate_per_100_words() -> None:
    """One stance marker ('i think') in five words -> 20 per 100 words."""
    assert Stylometric("i think this is fine").stance_rate() == pytest.approx(20.0)


def test_sentiment_sign_and_spread() -> None:
    """Positive text scores > 0; mixed sentences produce a non-zero spread."""
    assert Stylometric("I love this, it is wonderful.").sentiment()["mean"] > 0
    mixed = Stylometric("I love this. I hate this.").sentiment()
    assert mixed["std"] > 0


def test_sentiment_single_sentence_has_zero_std() -> None:
    """Population std of one sentence is exactly 0."""
    assert Stylometric("This is a table.").sentiment()["std"] == 0.0


def test_lexical_diversity_orders_varied_above_repetitive() -> None:
    """Varied vocabulary scores higher MTLD than one word repeated."""
    varied = Stylometric("the quick brown fox jumps over a lazy dog near the river bank")
    repetitive = Stylometric("nice nice nice nice nice nice nice nice nice nice nice nice")
    assert varied.lexical_diversity() > repetitive.lexical_diversity()


def test_empty_text_is_all_zeros() -> None:
    """Empty input never crashes; every feature is zero."""
    features = Stylometric("").all_features()
    assert all(value == 0 for value in features.values())


def test_all_features_keys_and_determinism() -> None:
    """all_features returns exactly the documented keys, deterministically."""
    text = "I think it's probably great. Maybe I love it!"
    first = Stylometric(text).all_features()
    assert list(first.keys()) == FEATURE_KEYS
    assert first == Stylometric(text).all_features()
