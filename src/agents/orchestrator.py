"""Plain function-based orchestration of the judge agents — no LangGraph.

Fans the three EDA questions (Q0 human-or-AI, Q1 personality, Q2 content)
out to every `JudgeAgent` and collects the verdicts into DataFrames with the
SAME columns run_eda.py writes, so the metric functions and src/plots.py
work unchanged. Judge-bound text is lowercased exactly like run_eda.py's
normalise(). Per plan §1: if a step can be a function, it is a function —
orchestration here is three loops, not a graph.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import pandas as pd

from src.agents.judge_agent import JudgeAgent


def normalise(text: str, lowercase: bool) -> str:
    """Lowercase judge-bound text if configured — same rule as run_eda.py.

    We cannot tell transcription artefacts (caps, typos) from real style, so
    judges must not see the difference. Style features keep the raw text.

    Args:
        text: text about to be shown to a judge.
        lowercase: config eda.lowercase_judge_inputs.
    """
    return text.lower() if lowercase else text


def run_q0(df: pd.DataFrame, agents: list[JudgeAgent], lowercase: bool) -> pd.DataFrame:
    """Q0 — zero-shot human-or-AI on all answers, both sides, once per agent.

    Args:
        df: tidy frame from src.load.load_answers.
        agents: one JudgeAgent per judge model.
        lowercase: config eda.lowercase_judge_inputs.

    Returns:
        One row per (model, person, question, side): prediction, whether it
        was correct (ground truth: side == 'ai'), and the reason — the same
        columns run_eda.py's eda_q0.csv has.
    """
    rows = []
    for agent in agents:
        for row in df.itertuples():
            for side in ("human", "ai"):
                text = row.human_text if side == "human" else row.ai_text
                verdict = agent.judge_is_ai(normalise(text, lowercase))
                rows.append(
                    {
                        "model": agent.model,
                        "person_id": row.person_id,
                        "question_id": row.question_id,
                        "category": row.category,
                        "side": side,
                        "is_ai": verdict.is_ai,
                        "correct": verdict.is_ai == (side == "ai"),
                        "reason": verdict.reason,
                    }
                )
            print(f"  Q0 [{agent.model}] {row.person_id} q{row.question_id} done")
    return pd.DataFrame(rows)


def run_q1(
    df: pd.DataFrame, split: dict, agents: list[JudgeAgent], lowercase: bool
) -> pd.DataFrame:
    """Q1 — captures personality: every agent scores the same held-out set.

    Context = the person's own human answers to their context questions.

    Args:
        df: tidy frame from src.load.load_answers.
        split: {person_id: {"context": [...], "held_out": [...]}} from
            src.eda.split_questions.
        agents: one JudgeAgent per judge model.
        lowercase: config eda.lowercase_judge_inputs.

    Returns:
        One row per (model, person, held-out question) — same columns as
        run_eda.py's eda_q1.csv.
    """
    rows = []
    for agent in agents:
        for person_id, parts in split.items():
            person = df[df["person_id"] == person_id].set_index("question_id")
            context = [
                normalise(t, lowercase) for t in person.loc[parts["context"], "human_text"]
            ]
            for question_id in parts["held_out"]:
                verdict = agent.judge_personality(
                    context, normalise(person.loc[question_id, "ai_text"], lowercase)
                )
                rows.append(
                    {
                        "model": agent.model,
                        "person_id": person_id,
                        "question_id": question_id,
                        "category": person.loc[question_id, "category"],
                        "captures_personality": verdict.captures_personality,
                        "reason": verdict.reason,
                    }
                )
                print(f"  Q1 [{agent.model}] {person_id} q{question_id} done")
    return pd.DataFrame(rows)


def run_q2(
    df: pd.DataFrame, split: dict, agents: list[JudgeAgent], lowercase: bool
) -> pd.DataFrame:
    """Q2 — pairwise content check: every agent scores the same held-out pairs.

    Args:
        df: tidy frame from src.load.load_answers.
        split: fixed within-person split from src.eda.split_questions.
        agents: one JudgeAgent per judge model.
        lowercase: config eda.lowercase_judge_inputs.

    Returns:
        One row per (model, person, held-out question) — same columns as
        run_eda.py's eda_q2.csv.
    """
    rows = []
    for agent in agents:
        for person_id, parts in split.items():
            person = df[df["person_id"] == person_id].set_index("question_id")
            for question_id in parts["held_out"]:
                pair = person.loc[question_id]
                verdict = agent.judge_content(
                    pair["question"],
                    normalise(pair["human_text"], lowercase),
                    normalise(pair["ai_text"], lowercase),
                )
                rows.append(
                    {
                        "model": agent.model,
                        "person_id": person_id,
                        "question_id": question_id,
                        "category": pair["category"],
                        "captures_content": verdict.captures_content,
                        "reason": verdict.reason,
                    }
                )
                print(f"  Q2 [{agent.model}] {person_id} q{question_id} done")
    return pd.DataFrame(rows)
