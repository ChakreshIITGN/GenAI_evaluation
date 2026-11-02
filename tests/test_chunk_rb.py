"""Tests for src/chunk_rb.py, with a word count standing in for the tokenizer.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from src.chunk_rb import chunk_header, chunk_sentences, iso_date, split_sentences


def words(text: str) -> int:
    """Stand-in tokenizer: one token per word."""
    return len(text.split())


def test_split_sentences_on_punctuation_and_lines() -> None:
    text = "One two. Three four! Five?\nA heading\n- a bullet"
    assert split_sentences(text) == [
        "One two.",
        "Three four!",
        "Five?",
        "A heading",
        "- a bullet",
    ]


def test_split_sentences_keeps_decimals_and_lowercase_continuations() -> None:
    # "2.5" has no space after the dot; "e.g." is followed by lowercase.
    assert split_sentences("Up 2.5 percent. e.g. this stays.") == [
        "Up 2.5 percent. e.g. this stays."
    ]


def test_chunks_respect_max_tokens() -> None:
    sentences = [f"s{i} a b c." for i in range(20)]  # 4 words each
    for chunk in chunk_sentences(sentences, words, max_tokens=10, overlap_tokens=4):
        assert words(chunk) <= 10


def test_chunks_overlap_by_trailing_sentence() -> None:
    sentences = ["a1 a2 a3.", "b1 b2 b3.", "c1 c2 c3.", "d1 d2 d3."]
    chunks = chunk_sentences(sentences, words, max_tokens=6, overlap_tokens=3)
    assert chunks == [
        "a1 a2 a3. b1 b2 b3.",
        "b1 b2 b3. c1 c2 c3.",
        "c1 c2 c3. d1 d2 d3.",
    ]


def test_no_overlap_when_zero() -> None:
    sentences = ["a1 a2 a3.", "b1 b2 b3.", "c1 c2 c3."]
    chunks = chunk_sentences(sentences, words, max_tokens=6, overlap_tokens=0)
    assert chunks == ["a1 a2 a3. b1 b2 b3.", "c1 c2 c3."]


def test_over_long_sentence_is_cut_to_fit() -> None:
    chunks = chunk_sentences(["w " * 25], words, max_tokens=10, overlap_tokens=0)
    assert [words(c) for c in chunks] == [10, 10, 5]


def test_every_sentence_lands_in_some_chunk() -> None:
    sentences = [f"s{i} x y." for i in range(30)]
    joined = " ".join(
        chunk_sentences(sentences, words, max_tokens=12, overlap_tokens=3)
    )
    assert all(s in joined for s in sentences)


def test_iso_date() -> None:
    assert iso_date("September 9, 2025") == "2025-09-09"
    assert iso_date("not a date") == ""


def test_chunk_header_skips_missing_fields() -> None:
    full = {"title": "T", "date": "May 1, 2025", "authors": ["A", "B"]}
    assert chunk_header(full) == "Title: T\nDate: May 1, 2025\nAuthors: A, B\n\n"
    assert chunk_header({"title": "T", "date": "", "authors": []}) == "Title: T\n\n"
