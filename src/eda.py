"""The three EDA judgements and the within-person question split (plan §1c).

Judgement functions format a template from user_prompts/eda/ and send it
through src/llm.py; the split decides, per person, which questions are shown
to the judge as context (Q1) and which are held out and scored (Q1, Q2).

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import json
from pathlib import Path

import pandas as pd

from src import llm
from src.eda_models import CapturesContent, CapturesPersonality, IsAI


def split_questions(
    df: pd.DataFrame, held_out_questions: list[int]
) -> dict[str, dict[str, list[int]]]:
    """Split each person's questions into a context set and a held-out set.

    The split is FIXED and explainable: the held-out question_ids come from
    config.yaml and are the same for every person; every other question is
    context. Within person, by question — never by person, because the
    personality judgement needs the person's own examples in context.
    Q0 (is_ai) does not use the split at all.

    Args:
        df: tidy frame from src.load.load_answers.
        held_out_questions: question_ids to hold out and judge (config.yaml).

    Returns:
        {person_id: {"context": [question_id, ...], "held_out": [...]}},
        both lists sorted.

    Raises:
        ValueError: if a held-out id does not exist in the data.
    """
    held_out = set(held_out_questions)
    all_questions = set(df["question_id"].unique())
    unknown = held_out - all_questions
    if unknown:
        raise ValueError(f"held_out_questions not in data: {sorted(unknown)}")

    split: dict[str, dict[str, list[int]]] = {}
    for person_id in sorted(df["person_id"].unique()):
        question_ids = set(df.loc[df["person_id"] == person_id, "question_id"])
        split[person_id] = {
            "context": sorted(question_ids - held_out),
            "held_out": sorted(question_ids & held_out),
        }
    return split


def save_split(split: dict, path: str | Path) -> None:
    """Write the split/fold record to JSON (output/eda_split.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))


def judge_is_ai(answer: str, model: str | None = None) -> IsAI:
    """Q0 — zero-shot: judge one answer alone, no context, no label.

    Args:
        answer: the text to judge.
        model: Ollama model tag; None uses the config default.
    """
    prompt = llm.load_prompt("eda/is_ai").format(answer=answer)
    return llm.ask(prompt, IsAI, model=model)


def judge_personality(
    context_answers: list[str], ai_answer: str, model: str | None = None
) -> CapturesPersonality:
    """Q1 — does the AI answer capture the person shown in their own answers?

    Args:
        context_answers: the person's human answers to their context questions.
        ai_answer: the AI answer to one held-out question.
        model: Ollama model tag; None uses the config default.
    """
    # Numbered blocks keep the examples visually separate for the judge.
    context = "\n\n".join(
        f"Example {i}:\n{text}" for i, text in enumerate(context_answers, start=1)
    )
    prompt = llm.load_prompt("eda/personality").format(
        context_answers=context, ai_answer=ai_answer
    )
    return llm.ask(prompt, CapturesPersonality, model=model)


def judge_content(
    question: str, human_answer: str, ai_answer: str, model: str | None = None
) -> CapturesContent:
    """Q2 — pairwise: does the AI answer carry the human answer's content?

    Args:
        question: the interview question text.
        human_answer: the person's real answer.
        ai_answer: the simulation's answer.
        model: Ollama model tag; None uses the config default.
    """
    prompt = llm.load_prompt("eda/content").format(
        question=question, human_answer=human_answer, ai_answer=ai_answer
    )
    return llm.ask(prompt, CapturesContent, model=model)
