# user_prompts/ — all user-editable prompts in one place

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).

Every prompt sent to an LLM lives here as a plain-text file, one file per
usage, so wording can be changed without touching code. The code
(`src/eda.py` via `src/llm.py`) loads a file, fills the `{placeholders}`
with Python `str.format`, and appends the JSON output-schema instructions
automatically — **do not** describe the output format inside these files.

Prompts are grouped in one subfolder per usage stage:

```
user_prompts/
  eda/       # the three EDA sanity-check judgements (this branch)
  judges/    # Layer-3 judge prompts (added on the eval branch)
```

| File | Used by | Purpose | Placeholders |
|---|---|---|---|
| `eda/is_ai.txt` | `judge_is_ai` (EDA Q0) | Zero-shot: is this single answer human or AI? No context, no examples. | `{answer}` |
| `eda/personality.txt` | `judge_personality` (EDA Q1) | Given a person's own example answers, does the AI answer capture their personality/nuance? | `{context_answers}`, `{ai_answer}` |
| `eda/content.txt` | `judge_content` (EDA Q2) | Pairwise: does the AI answer capture the content of the human answer to the same question? | `{question}`, `{human_answer}`, `{ai_answer}` |
| `judges/individuation.txt` | `judge_individuation` (Layer 3, Judge A) | Persona sketch + pair: recognisably THIS person or a generic segment member? Score −1…+1. | `{persona_answers}`, `{human_answer}`, `{ai_answer}` |
| `judges/believability.txt` | `judge_believability` (Layer 3, Judge B) | Blind: one unlabelled answer — how human does it read? Score −1…+1. | `{answer}` |

Editing rules:
- Keep placeholders exactly as named — code fills them by keyword.
- Literal curly braces in prompt text must be doubled (`{{` / `}}`).
