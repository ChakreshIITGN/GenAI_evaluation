# Decision & progress log — RB GenAI evaluation

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).

Major decisions and milestones, newest last. Each entry: date, decision, why.

## 2026-08-19 — Project start (eda branch)

- **cache/ is committed, not gitignored.** CLAUDE.md's task list says to gitignore
  `cache/`, but the master plan (§7) commits the cache so every result reproduces
  without API keys, and CLAUDE.md itself says the plan wins on conflict.
- **Ollama cloud called directly with an API key** (`https://ollama.com`,
  key `OLLAMA_API_KEY` in `.env`), per author's instruction, instead of a local
  signed-in daemon. Model tag is therefore `gpt-oss:120b` (the `:cloud` suffix
  in CLAUDE.md applies only when proxying through a local daemon). Host and
  model live in `config.yaml`, so switching back is a two-line config change.
- **All prompts live in `user_prompts/`,** one plain-text file per usage with
  `{placeholders}`, documented in `user_prompts/README.md`. Code never contains
  prompt wording, so prompts can be tuned in one place without touching code.
- **Dependencies added beyond those named in the plan:** `openpyxl` (pandas
  xlsx engine), `pyyaml` (config), `python-dotenv` (.env key loading) — all
  implied by the plan's file list; flagged here for transparency.
- **Prompts stay plain text, grouped in per-stage subfolders** (`user_prompts/eda/`,
  later `user_prompts/judges/`). JSON was considered and rejected: prompts are
  multi-line prose, and JSON escaping would hurt readability and git diffs while
  adding no structure we need (model/temperature live in config.yaml, output
  schemas are appended by code). Agreed with author 2026-08-19.
- **Fixed split instead of a random one (author decision, 2026-08-19).** The
  plan (§1c) proposed a seeded random within-person split; the author judged
  that not explainable enough. Instead the held-out question_ids are listed
  explicitly in config.yaml (`held_out_questions: [7, 8, 9, 10]`, same for
  every person; all others are context). Q0 (is_ai) never used a split. Bonus:
  identical held-out questions across people make results comparable.
- **Reminder for the eval branch:** the single-model Ollama setup is EDA-only.
  Judge models (Layer 3) must come from several different model families.

## 2026-08-19 — EDA run 2: lowercase + 3 folds × 3 judge families

- **Run 1 archived** to `archive/run1_2026-08-19_rawcase_gptoss/`. Its Q1/Q2
  verdicts keyed heavily on orthography (caps/typos) — we cannot tell
  transcription artefacts from real style, so run 1 is kept for the record
  but superseded.
- **Judge inputs lowercased** (config `eda.lowercase_judge_inputs`); style
  features still measured on raw text (they measure, not judge).
- **Judge panel on ONE shared eval set (supersedes the fold idea, author
  decision):** all three judges — gpt-oss:120b, deepseek-v4-flash, glm-5.2 —
  score the SAME held-out set (q7–10) for Q1/Q2 and all 60 answers for Q0.
  Same items per judge → verdicts directly comparable head-to-head, true
  inter-judge agreement. Folds were dropped: different items per judge would
  confound judge harshness with question difficulty. qwen3.5 and kimi-k3
  probed and rejected — not on the free Ollama plan (scripts/check_models.py).
- Judges output only booleans + reasons; every percentage is computed by us
  (share of yes / accuracy), never asked from a model.
- Figures: one fixed color per judge across all plots (sorted model names →
  palette order), grouped bars, legend with model names.

## 2026-08-19 — agentic_workflow branch: one LangChain agent per judge model

- **Agentic re-implementation of the EDA judging** (`src/agents/`,
  `run_agentic_eda.py`): each judge model in `eda.judges` runs as its own
  `JudgeAgent` wrapping a LangChain LCEL chain (ChatOllama, cloud host,
  temp 0, fixed seed, format=json). Rationale: per-model tracking,
  independent scaling, judge-family diversity. Architecture and pinned
  limitations in docs/AGENTIC_NOTES.md.
- **LCEL only, no LangGraph** (plan §1): the chain is
  `ChatOllama | RunnableLambda`; orchestration is plain loops in
  `src/agents/orchestrator.py`. Dependencies added (sanctioned by plan §1):
  `langchain`, `langchain-core`, `langchain-ollama`.
- **Everything judge-visible is reused, not duplicated:** Pydantic schemas
  (`src/eda_models.py`), prompts (`user_prompts/eda/`), fence
  stripping/schema block (`src/llm.py`), split (`src/eda.py`), metric/
  reason/feature functions (`run_eda.py`). Outputs land in output/ with the
  exact run_eda.py CSV schemas, so src/plots.py works unchanged.
- **Separate cache** `cache/agentic/` (same (model, prompt hash) key as
  src/llm.py) so agentic runs never collide with the EDA cache.
- **Per-agent monitoring:** one JSONL call log per agent under
  output/agent_logs/ (timestamp, prompt hash, latency,
  success/repair/fail/cache_hit, token counts) + aggregated
  output/agent_summary.json. Scaling work (async fan-out, token budgeting,
  batching, shared context store) deliberately deferred — see
  AGENTIC_NOTES.md limitations.

## Progress

- [x] main pushed (spec, CLAUDE.md, scaffolding); `eda` branch created
- [x] Part 1: pyproject, config.yaml, .env.example, user_prompts/, docs/
- [x] Part 2: src/load.py + tests (4 passing)
- [x] Part 3: src/llm.py + tests (mocked, no network)
- [x] Part 4: src/eda_models.py
- [x] Part 5: src/eda.py + split + tests
- [x] Part 6: src/stylometric.py + lexicons + tests + docs/api/ pages
      (Google-style docstrings throughout so pdoc/mkdocstrings can generate
      docs automatically if this becomes a package)
- [x] Part 7: run_eda.py (single EDA run, 84 cached LLM calls)
- [x] Part 7b: run_plots.py + src/plots.py + tests — publication figures in the
      Roland Berger palette (design/tokens.md; colors extracted from
      rolandberger.com CSS, colorblind-validated, no red). All outputs now
      land in output/ and figures in output/plots/ (author instruction;
      config paths.results/paths.plots).
- [ ] Part 8: README, CLAUDE.md current-state update, cleanup, push eda
- [x] agentic_workflow branch: src/agents/ (JudgeAgent, monitor,
      orchestrator), run_agentic_eda.py, tests/test_agents.py (mocked, no
      network), docs/AGENTIC_NOTES.md
