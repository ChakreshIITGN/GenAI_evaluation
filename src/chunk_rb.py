"""Split scraped RB articles into even, overlapping chunks with metadata.

Pure functions only; the token counter is passed in, so the chunker works
with any tokenizer (the embedding model's own one in embed_rb_ai.py, a
simple word count in the tests).
    split_sentences(text)                       -> [sentence, ...]
    chunk_sentences(sentences, count, max, overlap) -> [chunk_text, ...]
    chunk_header(article)                       -> "Title: ...\nDate: ...\n..."
    iso_date("September 9, 2025")               -> "2025-09-09"

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import re
from collections.abc import Callable
from datetime import UTC, datetime

# Sentence end: . ! or ? followed by space and a capital, digit or quote.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“‘(])")
SITE_DATE_FORMAT = "%B %d, %Y"  # how rolandberger.com prints dates


def split_sentences(text: str) -> list[str]:
    """Split article text into sentences; each line (heading, bullet) stays whole or splits further."""
    sentences: list[str] = []
    for line in text.split("\n"):
        sentences.extend(s.strip() for s in SENTENCE_END.split(line) if s.strip())
    return sentences


def _split_long(
    sentence: str, count: Callable[[str], int], max_tokens: int
) -> list[str]:
    """Cut one over-long sentence into word runs that each fit in max_tokens."""
    pieces: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        if current and count(" ".join([*current, word])) > max_tokens:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_sentences(
    sentences: list[str],
    count: Callable[[str], int],
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Pack whole sentences into chunks of at most max_tokens.

    Each new chunk starts with the last sentences of the previous one, up to
    overlap_tokens, so an idea cut at a boundary still appears whole once.
    """
    # Sentences longer than a chunk are cut first so every unit fits.
    units = [p for s in sentences for p in _split_long(s, count, max_tokens)]

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and count(" ".join([*current, unit])) > max_tokens:
            chunks.append(" ".join(current))
            # Carry trailing sentences into the next chunk, newest first.
            overlap: list[str] = []
            for prev in reversed(current):
                if count(" ".join([prev, *overlap, unit])) > max_tokens:
                    break
                if count(" ".join([prev, *overlap])) > overlap_tokens:
                    break
                overlap.insert(0, prev)
            current = overlap
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def iso_date(site_date: str) -> str:
    """Turn 'September 9, 2025' into '2025-09-09'; '' if it does not parse."""
    try:
        # Calendar date only; UTC just makes the parsed value timezone-aware.
        parsed = datetime.strptime(site_date, SITE_DATE_FORMAT).replace(tzinfo=UTC)
        return parsed.date().isoformat()
    except ValueError:
        return ""


def chunk_header(article: dict) -> str:
    """Metadata lines prepended to each chunk before embedding.

    Embedding title, date and authors with the text lets a search for
    "2025 study on AI sovereignty" match on metadata as well as content.
    """
    lines = [f"Title: {article['title']}"]
    if article.get("date"):
        lines.append(f"Date: {article['date']}")
    if article.get("authors"):
        lines.append(f"Authors: {', '.join(article['authors'])}")
    return "\n".join(lines) + "\n\n"
