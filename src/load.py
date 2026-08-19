"""Load the RB answer-pairs spreadsheet into a tidy DataFrame.

Reads data/RB_GenAI_Datatest.xlsx (sheet 'answer_pairs') and returns one row
per (person, question) with columns:
    person_id, question_id, category, question, human_text, ai_text

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import pandas as pd

# Column names as they appear in the source spreadsheet.
SOURCE_COLUMNS = {
    "question_category": "category",
    "question": "question",
    "person_id": "person_id",
    "human_answers": "human_text",
    "ai_answers": "ai_text",
}

# Columns of the tidy frame, in output order.
TIDY_COLUMNS = ["person_id", "question_id", "category", "question", "human_text", "ai_text"]

SHEET_NAME = "answer_pairs"


def load_answers(path: str | Path) -> pd.DataFrame:
    """Read the answer-pairs xlsx and return the tidy DataFrame.

    Args:
        path: location of the xlsx file (sheet 'answer_pairs').

    Returns:
        DataFrame with TIDY_COLUMNS, one row per (person, question).
        question_id is 1-based, assigned by first appearance of each distinct
        question text (the source file has no question-id column).

    Raises:
        ValueError: if expected columns are missing or any answer cell is empty.
    """
    df = pd.read_excel(path, sheet_name=SHEET_NAME)

    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {sorted(missing)}")

    df = df.rename(columns=SOURCE_COLUMNS)[list(SOURCE_COLUMNS.values())]

    # Derive a stable question_id: same question text -> same id, numbered
    # 1..n in order of first appearance.
    codes, _ = pd.factorize(df["question"])
    df["question_id"] = codes + 1

    df = df[TIDY_COLUMNS]

    # Fail loudly on blank answers — every downstream metric assumes text.
    for col in ("human_text", "ai_text"):
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if blank.any():
            raise ValueError(f"{path}: empty {col} in rows {df.index[blank].tolist()}")

    return df
