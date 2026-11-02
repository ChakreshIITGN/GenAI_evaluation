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

## 2026-08-19 — Post-run simplifications (author decisions)

- **Q1/Q2 reported as yes-COUNTS per person, not percentages.** At 4 held-out
  questions per person a percentage reads like a graded "how much it
  captures", which a yes/no verdict is not. No category split (12 items are
  too few to slice twice). Volatility caveat printed on the figures. Q0
  stays a rate — 60 items make accuracy meaningful.
- **Style dumbbell removed.** Same means as the EDA per-feature bar panels;
  one style figure is enough (raw units win). The paired-difference /
  Cohen's d table (style_differences.csv) remains the eval-grade metric.
- **Topic layer fixed after a degenerate first fit** (2 topics, one of them
  "and, the, to" holding ~95% of sentences): English stopwords in the
  c-TF-IDF vectorizer (naming only), UMAP n_neighbors 15 -> 10 (defaults
  over-smooth ~200 short sentences into one cluster). Added the sentence
  cluster map (2D UMAP of all sentence embeddings, human vs AI) — the
  plan's "2D topic overview", built in matplotlib.
- **KPI 2 from the first run is void** (artifact of the degenerate topics);
  it must be recomputed after the refit before anything cites it.
- **KPI 1 / RAG matrix pooled over categories, split by PERSON.** Per-category
  groups hold only 3–15 pairs — too small for a robust verdict. Reported in
  deliverables as an explicit aggregation choice due to small data.
- **First eval outcome preserved** in
  `archive/run3_2026-08-19_evalrun1_categoryRAG_degenerate_topics/` (author
  instruction): the report will show how the degenerate topic model and the
  too-thin category split were detected and corrected, not hide them.
- **Honest-try rule for topics:** BERTopic upgraded (mpnet embeddings — same
  model as Layer 1a — stopword naming, n_neighbors 10) and guarded by
  `is_degenerate()` (config: min_topics 3, max_topic_share 0.6). If the fit
  is still degenerate, KPI 2 is reported as NOT COMPUTABLE — never as
  "all topics covered".

## 2026-08-19 — Topic layer: reported as not computable (author decision)

- The corrected BERTopic (mpnet, stopwords, n_neighbors 10) is STILL
  degenerate on this data. Interpretation: the AI's invention is diffuse and
  in-domain — false specifics inside the same personal-care themes, not new
  themes — so claim-level NLI catches it but theme-level clustering cannot.
- A similarity-graph / community-detection alternative was considered and
  rejected: at near-uniform pairwise similarity the communities are an
  artifact of the chosen threshold.
- Decision: report KPI 2 as NOT COMPUTABLE at this n, with the diagnosis
  above. "A complete picture is better than an incomplete well done."

## 2026-08-19 — Verdict wording: "believable" dropped (author decision)

- At believability alpha 0.27 the panel does not support the positive claim
  "believable". The data supports the negative claim only: simulations are
  NOT RELIABLY DISTINGUISHABLE in blind reading (0.55 to 0.62 vs 0.50
  chance), while all three judges scored AI answers somewhat less human on
  average without being able to act on it per item.
- Noted explicitly: detectability (form, one text, no ground truth) and
  fidelity (content vs the person's real answer) measure different things;
  fluent-but-false is not a contradiction, it is the core risk finding.
- Fresh empty-cache rerun reproduced every number identically; the earlier
  apparent "changes" were traced to the fold-design run judging different
  questions per judge than the final shared-set panel.

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

## 2026-11-02 — RB-AI-publ branch: scrape the RB AI publications listing

- **Decision**: `scrape_rb_ai.py` + `src/scrape_rb.py` collect every article
  tile on rolandberger.com's AI publications page (title, url, label, date,
  authors, summary, body text) into `data/rb_ai_publications.json`.
- **Why**: author request. robots.txt allows crawling (only `/search/` is
  disallowed); one request per article with a 1 s pause (config `scrape`).
- **Output in data/ (gitignored)**: article text is Roland Berger copyright
  and this repo is public; only the code is committed.
- **One listing request is enough**: "Load More" only reveals tiles already
  in the page HTML (Angular `limit`), so all 56 unique articles are there.
  Menu links to non-AI articles are excluded by keeping tile links only.
- **New dependencies** (author approved): `requests`, `beautifulsoup4`.
- **Authors**: magazine pieces have no author line, so the writer is taken
  from the opening byline ("by X Artworks by Y" -> X). 7 of 56 articles name
  no author anywhere and keep `authors: []`.
- **Known limit**: some bodies end with a sign-up / subscribe line.

## 2026-11-02 — RB-AI-publ branch: chunk + embed the scraped articles

- **Decision**: `embed_rb_ai.py` + `src/chunk_rb.py` write
  `data/rb_ai_chunks.jsonl` (one chunk per line: metadata, text, vector)
  and `data/rb_ai_chunks_manifest.json` (model, dimension, settings).
- **Model**: Ollama Cloud serves no embedding models (all embed calls
  return 401; its catalogue is chat-only). Author chose sentence-transformers
  over pulling a local Ollama model; we reuse `all-mpnet-base-v2`, already
  the eval's embedding model and cached locally.
- **Chunking** (author choice): ~300 tokens, 50-token overlap, whole
  sentences only. Tokens are counted with the model's own tokenizer, and the
  budget shrinks so header + chunk fits the model's 384-token window; no
  chunk is ever silently truncated.
- **Metadata in the vector**: each chunk is embedded as
  "Title / Date / Authors" header + text, so searches can match on metadata.
  `text` in the JSONL is the chunk alone; `date_iso` allows date filtering.

## 2026-11-02 — RB-AI-publ branch: clean-up, analysis and interactive map

- **Environment**: iCloud "Desktop & Documents" had evicted 40,777 of 46,530
  `.venv` files, so every import downloaded from iCloud (24-min embed run,
  hanging pytest). `.venv` is now a symlink to `.venv.nosync` (iCloud skips
  `*.nosync`), rebuilt with `uv sync` + the en_core_web_sm wheel. Imports 5 s,
  full suite 8 s.
- **Key name**: code read `OLLAMA_KIMIK3_KEY1`; `.env` holds `OLLAMA_API_KEY`.
  Renamed in src/llm.py, src/agents/judge_agent.py, tests, .env.example.
- **Duplicates removed**: six Finder " 2" copies (five identical, one older
  draft of src/cluster.py). No duplicate articles (0 shared URLs/titles, no
  shared body text); site boilerplate lines are dropped at scrape time
  (config `scrape.boilerplate_min_articles`).
- **Analysis** (`analyze_rb_ai.py`, `src/analyze_rb.py`): k-means on mean
  chunk vectors, k in 6-8 by silhouette (k=8, 0.185; full-range best k=4
  only splits eras); kNN network (3 neighbours, cosine >= 0.65). One
  gpt-oss:120b reading per article (stance -2..+2, verbatim evidence quote,
  themes, voices) via src/llm.py; quotes verified word for word (53/56
  evidence, 85/101 voices). LLM cache under data/ because it stores prompts.
- **src/llm.py**: `ask(..., cache_dir=None)` and list-aware example replies;
  flat schemas render identically, so existing EDA cache keys still match.
- **Map page**: published as a private claude.ai artifact, built from
  data/rb_ai_analysis.json; not in the repo (it embeds article quotes).

## 2026-11-02 — RB-AI-publ: verification, offices, call-to-action cleanup

- **Stance checked by three models** (author asked for verification):
  gpt-oss:120b (primary) + deepseek-v4.1-flash + glm-5.2
  (`analysis.check_models`). Krippendorff alpha (ordinal) 0.82; same
  direction 46/56; within one point 54/56. Headline counts hold for every
  model (2025-26: 25-27 of 28 positive; 2024: 5-7 of 10 cautious).
- **deepseek-v4-flash retired** on Ollama Cloud (2026-09-25). It is still in
  `eda.judges`: cached EDA results reproduce, fresh EDA/eval runs will fail
  on it until replaced.
- **Offices and titles** come from each author's profile page (57/57 read);
  they are today's office, not necessarily the office at time of writing.
- **Calls to action removed** sentence by sentence
  (`scrape.call_to_action_patterns`); 18 lines cleaned, real uses of words
  like "subscription" kept.
- **Reproducibility**: `docs/RB_AI_REPRODUCE.md` (scripts, score
  definitions, run steps, verification); `provenance` block in the analysis
  JSON (git commit, input hashes, settings); human audit sheet
  `data/rb_ai_stance_audit.csv`.

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
