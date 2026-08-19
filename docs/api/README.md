# API reference — RB GenAI evaluation

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).

User-facing documentation of the `src/` modules, one page per module.
Every public function/class in the code carries a Google-style docstring, so
if this project becomes a library these pages can be regenerated automatically
(e.g. `pdoc src` or mkdocs + mkdocstrings) — the hand-written pages here
follow the same structure a generator would produce.

| Page | Module | What it covers |
|---|---|---|
| [eda.md](eda.md) | `src/eda.py` | The three EDA judgements and the fixed question split |
| [stylometric.md](stylometric.md) | `src/stylometric.py` | The seven deterministic style features |
| [lexicons.md](lexicons.md) | `src/lexicons.py` | Word lists / patterns behind the style features |
