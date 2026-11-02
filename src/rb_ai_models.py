"""Pydantic response schemas for reading the RB AI publications (RB-AI-publ).

One schema per question, all fields required, as in src/eda_models.py.
Quotes must be copied word for word; src/analyze_rb.py checks that.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from typing import Literal

from pydantic import BaseModel, Field

# How a featured person comes across on AI, in one word.
View = Literal["champion", "optimist", "balanced", "cautious", "critic"]


class Voice(BaseModel):
    """A named person quoted or featured in the article (other than the authors)."""

    name: str
    role: str  # e.g. "professor of computer science, MIT"
    view: View
    quote: str  # verbatim sentence from the article showing their view


class ArticleReading(BaseModel):
    """What one article says about AI, with verbatim evidence."""

    stance: int = Field(ge=-2, le=2)  # -2 warns ... 0 balanced ... +2 champions
    stance_reason: str  # one sentence
    evidence_quote: str  # verbatim sentence from the article backing the stance
    themes: list[str]  # 2-4 short lowercase theme phrases
    main_message: str  # the article's core claim, one sentence
    voices: list[Voice]  # may be empty


class ClusterName(BaseModel):
    """A short name and description for one cluster of related articles."""

    name: str  # 2-4 words
    description: str  # one sentence
