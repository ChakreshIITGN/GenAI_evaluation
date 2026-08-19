# `src.eda` — the three EDA judgements and the question split

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).

This module answers three crude yes/no questions about the answer pairs
(plan §1c). Prompt wording lives in `user_prompts/eda/`; all model calls go
through `src.llm` (cached, deterministic).

---

## `split_questions(df, held_out_questions)`

Split each person's questions into a **context** set and a **held-out** set.

The split is fixed and explainable: the held-out `question_id`s come from
`config.yaml` (`eda.folds`, one entry per fold with its judge model) and are
the same for every person; every other question is context. The split is
within person, never by person — the personality judgement needs the person's
own examples. Q0 (`judge_is_ai`) does not use the split at all.
All judge functions accept an optional `model` argument (Ollama tag; `None`
uses the config default), and judge inputs are lowercased by `run_eda.py`
when `eda.lowercase_judge_inputs` is true.

**Arguments**

| name | type | description |
|---|---|---|
| `df` | `pd.DataFrame` | tidy frame from `src.load.load_answers` |
| `held_out_questions` | `list[int]` | question ids to hold out and judge |

**Returns** — `dict[str, dict[str, list[int]]]`:
`{person_id: {"context": [...], "held_out": [...]}}`, both lists sorted.

**Raises** — `ValueError` if a held-out id does not exist in the data.

```python
split = split_questions(df, held_out_questions=[7, 8, 9, 10])
split["c_human"]["context"]   # -> [1, 2, 3, 4, 5, 6]
```

---

## `save_split(split, path)`

Write the split to JSON (`results/eda_split.json`) so the exact evaluation
set is on record. Creates parent folders as needed.

---

## `judge_is_ai(answer)`

**Q0 — zero-shot.** Judge one answer alone: no context, no examples, no
label. The uncalibrated baseline: can the model tell human from simulation
with no help?

**Returns** — `IsAI` with `is_ai: bool` and a one-line `reason`.

---

## `judge_personality(context_answers, ai_answer)`

**Q1 — few-shot, within person.** Shows the judge the person's own human
answers (their context questions) as numbered examples, then one AI answer
to a held-out question, and asks whether the AI captures this person's
nuance and personality rather than sounding generic.

**Arguments**

| name | type | description |
|---|---|---|
| `context_answers` | `list[str]` | the person's human answers (context set) |
| `ai_answer` | `str` | the AI answer to one held-out question |

**Returns** — `CapturesPersonality` with `captures_personality: bool`, `reason`.

---

## `judge_content(question, human_answer, ai_answer)`

**Q2 — pairwise.** Question text plus both answers; asks whether the AI
answer carries the same substance regardless of style. Needs no context.

**Returns** — `CapturesContent` with `captures_content: bool`, `reason`.

---

Response schemas are defined in `src/eda_models.py`; all fields required,
`reason` is always one line.
