"""Tests for src/judges.py — prompt wiring, alpha, discrimination rate.

llm.ask is mocked in the wiring tests; alpha and discrimination are pure
maths on hand-built frames.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import pandas as pd
import pytest

from src import judges, llm
from src.judges import (
    BelievabilityScore,
    IndividuationScore,
    discrimination_rate,
    judge_believability,
    judge_individuation,
    krippendorff_alpha,
)


@pytest.fixture
def captured_asks(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, type, str]]:
    """Replace llm.ask with a fake recording (prompt, schema, model)."""
    calls: list[tuple[str, type, str]] = []

    def fake_ask(prompt: str, schema: type, model: str | None = None):
        calls.append((prompt, schema, model))
        return schema(score=0.5, reason="stub")

    monkeypatch.setattr(llm, "ask", fake_ask)
    return calls


def test_individuation_wiring(captured_asks: list) -> None:
    """Judge A gets the numbered persona sketch, both answers, right schema."""
    judge_individuation(["PERSONA ONE", "PERSONA TWO"], "HUMAN TEXT", "AI TEXT", "m1")
    prompt, schema, model = captured_asks[0]
    assert "Answer 1:\nPERSONA ONE" in prompt
    assert "Answer 2:\nPERSONA TWO" in prompt
    assert "HUMAN TEXT" in prompt and "AI TEXT" in prompt
    assert schema is IndividuationScore
    assert model == "m1"


def test_believability_wiring_is_blind(captured_asks: list) -> None:
    """Judge B sees the answer and nothing else — no labels, no context."""
    judge_believability("JUST THE ANSWER", "m2")
    prompt, schema, model = captured_asks[0]
    assert "JUST THE ANSWER" in prompt
    assert "human answer" not in prompt.lower().replace("human-written", "")
    assert schema is BelievabilityScore
    assert model == "m2"


def test_alpha_perfect_agreement_is_one() -> None:
    """Identical scores across judges -> alpha 1.0."""
    scores = pd.DataFrame({"j1": [0.5, -1.0, 0.0, 1.0], "j2": [0.5, -1.0, 0.0, 1.0]})
    assert krippendorff_alpha(scores) == pytest.approx(1.0)


def test_alpha_disagreement_below_one() -> None:
    """Diverging judges -> alpha clearly below 1."""
    scores = pd.DataFrame({"j1": [1.0, -1.0, 0.5, -0.5], "j2": [-1.0, 1.0, -0.5, 0.5]})
    assert krippendorff_alpha(scores) < 0.5


def test_alpha_no_variance_is_nan() -> None:
    """All-identical ratings leave alpha undefined -> nan, not a crash."""
    scores = pd.DataFrame({"j1": [1.0, 1.0], "j2": [1.0, 1.0]})
    assert pd.isna(krippendorff_alpha(scores))


def test_discrimination_rate_counts_sign_matches() -> None:
    """+ on human and − on AI are hits; zeros are misses."""
    scores = pd.Series([0.8, -0.5, 0.0, 0.3])
    sides = pd.Series(["human", "ai", "human", "ai"])
    # hits: 0.8/human, -0.5/ai; misses: 0.0/human, +0.3/ai -> 0.5
    assert discrimination_rate(scores, sides) == pytest.approx(0.5)