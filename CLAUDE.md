# CLAUDE.md — RB GenAI Evaluation

## Read first
`eval_policy/RB_GenAI_Eval_Plan.md` at the repo root is the master specification. Read it in full before doing anything. If this file and the plan ever disagree, the plan wins.

## What this project is
Evaluate how faithfully LLM-based human simulations (built from an interview) mimic real consumers' open-ended answers. Data: 3 people × 10 questions, human answer + AI answer per pair (`data/RB_GenAI_Datatest.xlsx`, confidential, gitignored). Deliverables: code, 2-page report, 3-slide deck, then a Dockerised app.

## Non-negotiables (plan §1a)
- Simple, readable, clean code. No clever abstractions. Anyone with Python experience must be able to read and run every file.
- Docstring on every function and class. Header comment on every file. Inline comments explain *why*.
- Type hints everywhere. Pure functions where possible. Models (spaCy, NLI, embeddings) loaded once, never inside a per-call function.
- `pytest` for every deterministic piece. Small tests, fast, named for what they check.
- Pydantic models for every LLM response; JSON as the wire format. One schema per question.
- FOR EDA: All LLM calls through `src/llm.py`: Ollama, `gpt-oss-120b:cloud`, temperature 0, fixed seed, `format="json"`, disk cache under `cache/`, one retry-with-repair on validation error.
- FOR EVAL: we will change the models and more for judges. REMIND the user about this incase this is forgotten. 
- Config in `config.yaml`. No magic numbers in code. `ruff` clean. No dead code. No notebooks as deliverables.
- Everything traces back to the brief's five guiding bullets (plan §0). If it doesn't serve one, don't build it.
- make a docs/ folder that logs major decisions and progress. 

## Branches
- `eda` — first. Build the plumbing and run the three yes/no questions once (plan §1c). Also build the `Stylometric` class + tests (plan §1d).
- `eval` — second. Full pipeline: Layers 1–3, KPIs, figures, report, deck (plan §3–§4, §8). Then app + Docker (plan §9).

## Current state
`eda` branch COMPLETE (2026-08-19): plumbing (load/llm/cache/Pydantic), three
EDA judgements run with a 3-model judge panel (gpt-oss:120b, deepseek-v4-flash,
glm-5.2) on one fixed held-out set (q7–10), lowercased judge inputs,
Stylometric class, RB-palette plots, report draft in report/. Run 1 archived
in archive/. Key deviations from plan, all logged in docs/DECISIONS.md:
fixed split instead of seeded (author), prompts in user_prompts/, outputs in
output/ not results/. Next: `eval` branch — Layers 1–3, KPIs, figures.

`agentic_workflow` branch COMPLETE (2026-08-19): LangChain LCEL agentic
re-implementation of the EDA judging — one JudgeAgent per judge model
(src/agents/: judge_agent, monitor, orchestrator; run_agentic_eda.py), same
prompts/schemas/split/CSV schemas as run_eda.py, separate cache/agentic/,
per-agent JSONL call logs + output/agent_summary.json. No LangGraph (plan
§1). Mocked tests in tests/test_agents.py; ruff clean. Architecture and
deferred-scaling limitations in docs/AGENTIC_NOTES.md.

## First task (on `eda`)
Create, in this order, and run `pytest` after each:
1. `pyproject.toml`, `config.yaml`, `.env.example`, `.gitignore` (data/, cache/ keys, .env)
2. `src/load.py` — xlsx → tidy DataFrame (`person_id, question_id, category, question, human_text, ai_text`) + `tests/test_load.py`
3. `src/llm.py` — `ask(prompt, schema, model) -> schema` with cache + repair, per plan §1c
4. `src/eda_models.py` — `IsAI`, `CapturesPersonality`, `CapturesContent`
5. `src/eda.py` — `judge_is_ai`, `judge_personality`, `judge_content` + the within-person question split (seeded, saved to `results/eda_split.json`) + `tests/test_eda.py`
6. `src/stylometric.py` — `Stylometric` class, seven pure methods + `all_features()` + `tests/test_stylometric.py`
7. `run_eda.py` — single entrypoint → `results/eda_*.csv`, `results/measured_features.csv`, `results/run_manifest.json`
8. `README.md` — how to install, how to run, what comes out

Do not start Layer 2/3 or the report on this branch.

## Working style
- Ask before adding a dependency not named in the plan.
- When a milestone is done, update the "Current state" block above and say what changed.
- Prefer deleting code to adding it.
