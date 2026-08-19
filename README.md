# RB GenAI Evaluation

Author: Dr. Chakresh Kr. Singh — prepared for Roland Berger.

Evaluates how faithfully LLM-based human simulations (built from an interview,
after Park et al., arXiv:2411.10109) mimic real consumers' open-ended answers.
Data: 3 people × 10 questions, human + AI answer per pair (confidential,
gitignored). Master spec: `eval_policy/RB_GenAI_Eval_Plan.md`.

## Install

```bash
uv sync                      # Python 3.13, deps from pyproject.toml
cp .env.example .env         # add your Ollama cloud API key
```

## Run (EDA)

```bash
uv run pytest                # fast, no network
uv run python run_eda.py     # the three EDA judgements (LLM, cached)
uv run python run_plots.py   # figures from the CSVs, no LLM
```

## What comes out (`output/`)

| File | Content |
|---|---|
| `eda_q0.csv` | zero-shot human-or-AI verdicts, all answers × 3 judges |
| `eda_q1.csv` / `eda_q2.csv` | personality / content verdicts on the held-out set × 3 judges |
| `eda_metrics.csv` | accuracies and share-of-yes, per judge / person / category |
| `eda_reasons.csv` | every judge reason, verbatim |
| `measured_features.csv` | 7 deterministic style features per answer (`src/stylometric.py`) |
| `eda_split.json`, `run_manifest.json` | the fixed split, judges, config echo, timings |
| `plots/*.png` | publication figures, Roland Berger palette (`design/tokens.md`) |

## How it works

- All LLM calls go through `src/llm.py`: Ollama cloud, temp 0, fixed seed,
  JSON enforced, validated with Pydantic (`src/eda_models.py`), one
  retry-with-repair, disk-cached under `cache/` → reruns are free and
  reproducible.
- Judge inputs are lowercased (orthography must not drive verdicts); judges
  return only booleans + a reason — every percentage is computed in code.
- Prompts live in `user_prompts/` (one file per use), docs in `docs/api/`,
  decisions in `docs/DECISIONS.md`, findings in `docs/EDA_FINDINGS.md`.
- Prior runs are kept under `archive/`.

Branches: `eda` (this baseline) → `eval` (full pipeline: NLI coverage/
fidelity, topic map, judge panel, KPIs, report, deck, app).
