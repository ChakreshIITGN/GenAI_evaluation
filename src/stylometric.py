"""Deterministic style features for one text (plan §1d).

The `Stylometric` class measures seven style features on a single answer
(human or AI). Each method is a pure function of the text so it can be
wrapped as a tool later; nothing here depends on LangChain. Comparison
between humans and AI lives elsewhere — this class only measures.

The VADER sentiment model is loaded once at import, never per call.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import re
import statistics

import nltk
from lexicalrichness import LexicalRichness
from nltk.sentiment import SentimentIntensityAnalyzer

from src.lexicons import CONTRACTION_PATTERN, HEDGES, STANCE_MARKERS, count_phrases


def _load_vader() -> SentimentIntensityAnalyzer:
    """Return the VADER analyzer, downloading its lexicon on first ever use."""
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


_VADER = _load_vader()  # module-level: loaded once for the whole run

# A word is a run of letters, possibly with an internal apostrophe (can't).
_WORD_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


class Stylometric:
    """Seven deterministic style features for one text.

    Usage:
        features = Stylometric("I think it's great.").all_features()
    """

    def __init__(self, text: str):
        """Store the text and tokenise it once.

        Args:
            text: one answer (any length; empty text yields zero features).
        """
        self.text = text
        self.words: list[str] = _WORD_PATTERN.findall(text)

    def length(self) -> int:
        """Word count. Does it talk more than the person?"""
        return len(self.words)

    def lexical_diversity(self) -> float:
        """MTLD lexical diversity (lexicalrichness). Is vocabulary as varied?

        Returns 0.0 for texts of fewer than two words, where MTLD is undefined.
        """
        if len(self.words) < 2:
            return 0.0
        return float(LexicalRichness(self.text).mtld())

    def mean_word_length(self) -> float:
        """Mean letters per word (apostrophes excluded). More formal/'written'?"""
        if not self.words:
            return 0.0
        letters = sum(len(word.replace("'", "")) for word in self.words)
        return letters / len(self.words)

    def contraction_rate(self) -> float:
        """Contractions per word. Talks like chat or like a report?"""
        if not self.words:
            return 0.0
        return len(CONTRACTION_PATTERN.findall(self.text)) / len(self.words)

    def hedging_density(self) -> float:
        """Hedge words/phrases per 100 words (lexicon). Commits or hedges?"""
        if not self.words:
            return 0.0
        return count_phrases(self.text, HEDGES) / len(self.words) * 100

    def sentiment(self) -> dict[str, float]:
        """VADER compound score per sentence -> {"mean": .., "std": ..}.

        Sentences are split on ., ! and ? — a deliberate, dependency-free
        splitter; std is the population value (0.0 for a single sentence).
        Swings like a person or sits neutral?
        """
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", self.text.strip()) if s]
        if not sentences:
            return {"mean": 0.0, "std": 0.0}
        scores = [_VADER.polarity_scores(s)["compound"] for s in sentences]
        return {"mean": statistics.mean(scores), "std": statistics.pstdev(scores)}

    def stance_rate(self) -> float:
        """Stance markers ('I think/feel/prefer…') per 100 words (lexicon).

        Takes positions or describes?
        """
        if not self.words:
            return 0.0
        return count_phrases(self.text, STANCE_MARKERS) / len(self.words) * 100

    def all_features(self) -> dict[str, float]:
        """All seven features as one flat dict (sentiment as mean and std).

        Keys: length, lexical_diversity, mean_word_length, contraction_rate,
        hedging_density, sentiment_mean, sentiment_std, stance_rate.
        """
        sentiment = self.sentiment()
        return {
            "length": self.length(),
            "lexical_diversity": self.lexical_diversity(),
            "mean_word_length": self.mean_word_length(),
            "contraction_rate": self.contraction_rate(),
            "hedging_density": self.hedging_density(),
            "sentiment_mean": sentiment["mean"],
            "sentiment_std": sentiment["std"],
            "stance_rate": self.stance_rate(),
        }
