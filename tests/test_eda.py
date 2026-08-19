"""Tests for src/eda.py — split determinism/balance and prompt wiring.

llm.ask is monkeypatched in the judgement tests: they check that templates
are filled with the right pieces of text and ask for the right schema,
without any network call.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src import eda, llm
from src.eda_models import CapturesContent, CapturesPersonality, IsAI


def make_df(n_people: int = 3, n_questions: int = 10) -> pd.DataFrame:
    """Tidy frame shaped like the real data: every person answers every question."""
    rows = [
        {"person_id": f"p{p}", "question_id": q, "category": "cat",
         "question": f"q{q}", "human_text": "h", "ai_text": "a"}
        for p in range(n_people)
        for q in range(1, n_questions + 1)
    ]
    return pd.DataFrame(rows)


HELD_OUT = [7, 8, 9, 10]


def test_split_is_fixed_and_identical_per_person() -> None:
    """Every person holds out exactly the configured questions; rest is context."""
    split = eda.split_questions(make_df(), HELD_OUT)
    assert len(split) == 3
    for parts in split.values():
        assert parts["held_out"] == HELD_OUT
        assert parts["context"] == [1, 2, 3, 4, 5, 6]


def test_split_unknown_question_raises() -> None:
    """A held-out id that does not exist in the data is rejected."""
    with pytest.raises(ValueError, match="99"):
        eda.split_questions(make_df(), [7, 99])


def test_split_round_trips_through_json(tmp_path: Path) -> None:
    """save_split writes JSON that reloads to the same structure."""
    split = eda.split_questions(make_df(), HELD_OUT)
    out = tmp_path / "eda_split.json"
    eda.save_split(split, out)
    assert json.loads(out.read_text()) == split


@pytest.fixture
def captured_asks(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, type]]:
    """Replace llm.ask with a fake that records (prompt, schema) and answers."""
    calls: list[tuple[str, type]] = []

    def fake_ask(prompt: str, schema: type, model: str | None = None):
        calls.append((prompt, schema))
        # Build a minimal valid instance for whichever schema was requested.
        field = next(iter(schema.model_fields))
        return schema(**{field: True, "reason": "stub"})

    monkeypatch.setattr(llm, "ask", fake_ask)
    return calls


def test_judge_is_ai_wiring(captured_asks: list[tuple[str, type]]) -> None:
    """Q0 sends the answer text alone and asks for the IsAI schema."""
    eda.judge_is_ai("THE ANSWER")
    prompt, schema = captured_asks[0]
    assert "THE ANSWER" in prompt
    assert schema is IsAI


def test_judge_personality_wiring(captured_asks: list[tuple[str, type]]) -> None:
    """Q1 numbers the context examples and includes the AI answer."""
    eda.judge_personality(["CTX ONE", "CTX TWO"], "AI TEXT")
    prompt, schema = captured_asks[0]
    assert "Example 1:\nCTX ONE" in prompt
    assert "Example 2:\nCTX TWO" in prompt
    assert "AI TEXT" in prompt
    assert schema is CapturesPersonality


def test_judge_content_wiring(captured_asks: list[tuple[str, type]]) -> None:
    """Q2 includes question, human and AI answers, in the CapturesContent schema."""
    eda.judge_content("THE QUESTION", "HUMAN TEXT", "AI TEXT")
    prompt, schema = captured_asks[0]
    assert all(part in prompt for part in ("THE QUESTION", "HUMAN TEXT", "AI TEXT"))
    assert schema is CapturesContent
