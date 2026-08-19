"""Pydantic response schemas for the three EDA judgements (plan §1c).

One schema per question keeps validation strict (no optional fields) and
lets src/llm.py stay generic. All fields are required; `reason` is one line.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pydantic import BaseModel


class IsAI(BaseModel):
    """Q0 — zero-shot: was this single answer written by an AI?"""

    is_ai: bool
    reason: str


class CapturesPersonality(BaseModel):
    """Q1 — given the person's own answers, does the AI capture their personality?"""

    captures_personality: bool
    reason: str


class CapturesContent(BaseModel):
    """Q2 — pairwise: does the AI answer capture the human answer's content?"""

    captures_content: bool
    reason: str
