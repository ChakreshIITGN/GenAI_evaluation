"""Tests for src/load.py — loader returns the tidy schema and fails on bad data.

Uses a small synthetic xlsx written to tmp_path, so tests run without the
confidential data file; one extra check runs on the real file if present.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.load import SHEET_NAME, TIDY_COLUMNS, load_answers

REAL_DATA = Path("data/RB_GenAI_Datatest.xlsx")


def write_xlsx(path: Path, rows: list[dict]) -> Path:
    """Write rows to an xlsx file with the source sheet name and return its path."""
    pd.DataFrame(rows).to_excel(path, sheet_name=SHEET_NAME, index=False)
    return path


def make_rows(*, blank_ai: bool = False) -> list[dict]:
    """Two questions x two people in source-file column layout."""
    rows = []
    for q_num, question in enumerate(["Where do you shop?", "Which brand?"], start=1):
        for person in ["a_human", "b_human"]:
            rows.append(
                {
                    "id": len(rows) + 1,
                    "question_category": "shopping",
                    "question": question,
                    "person_id": person,
                    "human_answers": f"human answer {q_num} {person}",
                    "ai_answers": "" if blank_ai else f"ai answer {q_num} {person}",
                }
            )
    return rows


def test_tidy_columns_and_question_ids(tmp_path: Path) -> None:
    """Loader returns exactly the tidy columns; question_id numbers questions 1..n."""
    df = load_answers(write_xlsx(tmp_path / "ok.xlsx", make_rows()))
    assert list(df.columns) == TIDY_COLUMNS
    assert len(df) == 4
    # Same question text -> same id, in order of first appearance.
    assert df.groupby("question")["question_id"].nunique().eq(1).all()
    assert sorted(df["question_id"].unique()) == [1, 2]


def test_missing_column_raises(tmp_path: Path) -> None:
    """A source file without an expected column is rejected."""
    rows = [{k: v for k, v in row.items() if k != "ai_answers"} for row in make_rows()]
    with pytest.raises(ValueError, match="ai_answers"):
        load_answers(write_xlsx(tmp_path / "missing.xlsx", rows))


def test_blank_answer_raises(tmp_path: Path) -> None:
    """Empty answer cells are rejected instead of flowing into metrics."""
    with pytest.raises(ValueError, match="ai_text"):
        load_answers(write_xlsx(tmp_path / "blank.xlsx", make_rows(blank_ai=True)))


@pytest.mark.skipif(not REAL_DATA.exists(), reason="confidential data file not present")
def test_real_file_shape() -> None:
    """The real dataset loads as 3 people x 10 questions = 30 rows."""
    df = load_answers(REAL_DATA)
    assert len(df) == 30
    assert df["person_id"].nunique() == 3
    assert df["question_id"].nunique() == 10
