"""Tests for src/entail.py — label mapping and coverage/fidelity maths.

The cross-encoder is mocked throughout: these tests check OUR logic
(argmax -> label, ratios, empty guards), not the NLI model itself.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import numpy as np
import pytest

from src import entail


class FakeNLI:
    """Stands in for the CrossEncoder: returns scripted score rows."""

    def __init__(self, rows: list[list[float]]):
        self.rows = rows

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        # One scripted row per (premise, hypothesis) pair, in order.
        return np.array(self.rows[: len(pairs)])


def test_check_maps_argmax_to_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Highest score per row picks the label, in the model's fixed order."""
    fake = FakeNLI([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    monkeypatch.setattr(entail, "_NLI", fake)
    labels = entail.check("reference", ["s1", "s2", "s3"])
    assert labels == ["contradiction", "entailment", "neutral"]


def test_check_empty_sentences_is_empty() -> None:
    """No sentences -> no labels, and the model is never touched."""
    assert entail.check("reference", []) == []


def test_coverage_fidelity_maths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratios and contradiction count follow the plan's formulas exactly."""

    def scripted_check(reference: str, sentences: list[str]) -> list[str]:
        # coverage direction: 2 human units -> 1 entailed = 0.5 coverage
        if reference == "AI ANSWER":
            return ["entailment", "neutral"]
        # fidelity direction: 4 AI units -> 1 entailed, 2 contradictions
        return ["entailment", "contradiction", "contradiction", "neutral"]

    monkeypatch.setattr(entail, "check", scripted_check)
    result = entail.coverage_fidelity(
        "HUMAN ANSWER", "AI ANSWER", ["h1", "h2"], ["a1", "a2", "a3", "a4"]
    )
    assert result["coverage"] == pytest.approx(0.5)
    assert result["fidelity"] == pytest.approx(0.25)
    assert result["contradictions"] == 2
    assert result["n_human_units"] == 2
    assert result["n_ai_units"] == 4


def test_coverage_fidelity_empty_units_are_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty unit lists give 0.0, never a ZeroDivisionError."""
    monkeypatch.setattr(entail, "check", lambda reference, sentences: [])
    result = entail.coverage_fidelity("h", "a", [], [])
    assert result["coverage"] == 0.0
    assert result["fidelity"] == 0.0
    assert result["contradictions"] == 0
