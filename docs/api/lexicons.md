# `src.lexicons` — word lists behind the style features

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).

The lexicons are data, not logic: [`Stylometric`](stylometric.md) counts
matches against them, nothing more. They are deliberately small and readable,
and are reproduced in the report appendix.

## `CONTRACTION_PATTERN`

Compiled regex matching English contractions (`can't`, `it's`, `we're`,
`I'll`, `you've`, …) via the standard apostrophe-suffix forms
`'s 't 're 've 'll 'd 'm`. Case-insensitive.

## `HEDGES`

Words and phrases that soften a claim ("maybe", "probably", "kind of",
"tend to", …). Curated subset of Hyland (2005), *Metadiscourse*, restricted
to forms plausible in consumer-interview speech. Used by
`Stylometric.hedging_density()` (matches per 100 words).

## `STANCE_MARKERS`

First-person forms that take a position rather than describe ("i think",
"i prefer", "i love", "in my opinion", …). Used by
`Stylometric.stance_rate()` (matches per 100 words).

## `count_phrases(text, phrases)`

The one function: counts word-bounded, case-insensitive occurrences of each
word/phrase in the text (whitespace collapsed first, so phrases match across
line breaks).

**Arguments**

| name | type | description |
|---|---|---|
| `text` | `str` | raw text to search |
| `phrases` | `tuple[str, ...]` | words or multi-word phrases |

**Returns** — `int`, total matches across all phrases.

Word boundaries matter: `may` does **not** match inside `maybe`, and
`would` does **not** match inside `wouldn't`.

## Editing the lexicons

Add or remove entries directly in `src/lexicons.py`; every entry is plain
lowercase text. Changes take effect everywhere `Stylometric` is used —
rerun `pytest` (tests pin a few known counts) and regenerate
`results/measured_features.csv`.
