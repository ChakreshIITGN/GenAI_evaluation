"""Word lists and patterns used by the Stylometric features (plan §1d).

These lexicons are deliberately small, readable and printed in the report
appendix. They are data, not logic: Stylometric counts matches, nothing more.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import re

# Matches English contractions such as can't, it's, we're, I'll, you've.
# The apostrophe-suffix list covers the standard clitic forms.
CONTRACTION_PATTERN: re.Pattern[str] = re.compile(
    r"\b[a-zA-Z]+'(?:s|t|re|ve|ll|d|m)\b", flags=re.IGNORECASE
)

# Hedges — words/phrases that soften a claim. Curated subset of Hyland (2005),
# restricted to forms plausible in consumer-interview speech.
HEDGES: tuple[str, ...] = (
    "about",
    "almost",
    "apparently",
    "approximately",
    "around",
    "could",
    "fairly",
    "generally",
    "guess",
    "in general",
    "kind of",
    "largely",
    "likely",
    "mainly",
    "may",
    "maybe",
    "might",
    "mostly",
    "often",
    "perhaps",
    "possibly",
    "presumably",
    "probably",
    "quite",
    "rather",
    "relatively",
    "roughly",
    "seems",
    "should",
    "sometimes",
    "somewhat",
    "sort of",
    "suppose",
    "tend to",
    "typically",
    "usually",
    "would",
)

# Stance markers — first-person forms that take a position rather than
# describe. A person who talks in these commits to opinions.
STANCE_MARKERS: tuple[str, ...] = (
    "i think",
    "i feel",
    "i believe",
    "i prefer",
    "i like",
    "i love",
    "i hate",
    "i dislike",
    "i want",
    "i need",
    "i always",
    "i never",
    "in my opinion",
    "personally",
    "for me",
)


def count_phrases(text: str, phrases: tuple[str, ...]) -> int:
    """Count non-overlapping, word-bounded occurrences of each phrase in text.

    Args:
        text: raw text; matching is case-insensitive on collapsed whitespace.
        phrases: words or multi-word phrases to look for.

    Returns:
        Total number of matches across all phrases.
    """
    normalised = re.sub(r"\s+", " ", text.lower())
    total = 0
    for phrase in phrases:
        # \b keeps 'may' from matching inside 'maybe'.
        total += len(re.findall(rf"\b{re.escape(phrase)}\b", normalised))
    return total
