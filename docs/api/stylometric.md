# `src.stylometric` — seven deterministic style features

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).

`Stylometric` measures how a text *sounds*, not what it says. It is fully
deterministic (no LLM), so the same text always yields the same numbers.
The class only measures one text; comparing humans vs AI happens elsewhere.

```python
from src.stylometric import Stylometric

features = Stylometric("I think it's probably great.").all_features()
```

## Construction

`Stylometric(text: str)` — stores the text and tokenises it once. A word is
a run of letters with an optional internal apostrophe, so "can't" is one word.
Empty text is valid: every feature returns 0.

## Methods

| Method | Returns | Definition | Plain-English question |
|---|---|---|---|
| `length()` | `int` | word count | Does it talk more than the person? |
| `lexical_diversity()` | `float` | MTLD (via `lexicalrichness`); 0.0 under two words | Is vocabulary as varied? |
| `mean_word_length()` | `float` | letters per word (apostrophes excluded) | More formal / "written"? |
| `contraction_rate()` | `float` | contractions ÷ words | Talks like chat or like a report? |
| `hedging_density()` | `float` | hedge matches per 100 words ([lexicon](lexicons.md)) | Commits or hedges? |
| `sentiment()` | `dict` | VADER compound per sentence → `{"mean", "std"}` | Swings like a person or sits neutral? |
| `stance_rate()` | `float` | stance markers per 100 words ([lexicon](lexicons.md)) | Takes positions or describes? |
| `all_features()` | `dict` | all of the above, flat | one row of `measured_features.csv` |

`all_features()` keys, in order: `length`, `lexical_diversity`,
`mean_word_length`, `contraction_rate`, `hedging_density`, `sentiment_mean`,
`sentiment_std`, `stance_rate`.

## Implementation notes

- **VADER** (NLTK) is loaded once at module import; its lexicon is
  downloaded automatically on the very first use on a machine.
- **Sentence splitting** for `sentiment()` is a deliberate, dependency-free
  regex split on `.`, `!`, `?`. The std is the population value, so a
  single-sentence text has `std == 0.0`.
- **MTLD** is undefined for near-empty texts; under two words the method
  returns 0.0 instead of raising.

## Worked example (from the tests)

`"i can't say i won't go to the shop today"` → 10 words, 2 contractions
→ `contraction_rate() == 0.2`.
