"""Layer 3 — LLM judge panel: individuation and blind believability (plan §3).

Two constructs, anchored −1…+1 rubrics, structured JSON (score + reason).
Judges produce RATINGS (kept separate from Layers 1–2 measurements):

    Judge A — Individuation: persona sketch (the person's other human
        answers) + the pair for one held-out question. "Recognisably THIS
        person, or a generic segment member?" A judge shown a single pair
        cannot assess convergence to the mean — hence the sketch.
    Judge B — Believability: ONE answer, blind, no label. "How human does
        this read?" Blinding is mandatory; the discrimination rate is
        computed afterwards in code from the known labels.

Reliability: Krippendorff's alpha (interval) across the panel per construct.
All calls go through src/llm.py (cached, temp 0, one repair).

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import krippendorff

from src import llm


class IndividuationScore(BaseModel):
    """Judge A verdict: −1 generic/mean-reverting … +1 distinctly this person."""

    score: float = Field(ge=-1, le=1)
    reason: str


class BelievabilityScore(BaseModel):
    """Judge B verdict: −1 clearly AI … +1 clearly human."""

    score: float = Field(ge=-1, le=1)
    reason: str


def judge_individuation(
    persona_answers: list[str], human_answer: str, ai_answer: str, model: str
) -> IndividuationScore:
    """Judge A — is the AI answer recognisably this specific person?

    Args:
        persona_answers: the person's OTHER human answers (persona sketch).
        human_answer: the person's real answer to the judged question.
        ai_answer: the simulation's answer to the same question.
        model: judge model tag (one member of the panel).
    """
    persona = "\n\n".join(
        f"Answer {i}:\n{text}" for i, text in enumerate(persona_answers, start=1)
    )
    prompt = llm.load_prompt("judges/individuation").format(
        persona_answers=persona, human_answer=human_answer, ai_answer=ai_answer
    )
    return llm.ask(prompt, IndividuationScore, model=model)


def judge_believability(answer: str, model: str) -> BelievabilityScore:
    """Judge B — blind: how human does this single answer read?

    The caller must pass answers in randomised order with no label attached;
    this function never learns which side the text came from.
    """
    prompt = llm.load_prompt("judges/believability").format(answer=answer)
    return llm.ask(prompt, BelievabilityScore, model=model)


def krippendorff_alpha(scores: pd.DataFrame) -> float:
    """Krippendorff's alpha (interval) for one construct.

    Args:
        scores: one row per item, one column per judge, cells = scores
            (NaN allowed for missing ratings).

    Returns:
        Alpha in [-1, 1]; 1 = perfect agreement, 0 = chance level.
        Returns nan if the items have no variance at all (alpha undefined).
    """
    matrix = scores.to_numpy(dtype=float).T  # krippendorff wants judges x items
    if np.nanstd(matrix) == 0:
        return float("nan")
    return float(
        krippendorff.alpha(reliability_data=matrix, level_of_measurement="interval")
    )


def discrimination_rate(scores: pd.Series, sides: pd.Series) -> float:
    """Blind discrimination rate: does the score's SIGN match the true label?

    Human answers should score > 0, AI answers < 0. Zero scores ("cannot
    tell") count as misses. ~0.5 = the judge cannot tell at all.

    Args:
        scores: believability scores, one per judged answer.
        sides: 'human' or 'ai' per answer, aligned by position.
    """
    correct = ((scores > 0) & (sides == "human")) | ((scores < 0) & (sides == "ai"))
    return float(correct.mean())
