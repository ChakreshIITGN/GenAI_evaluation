# RB AI publications: how the analysis works and how to reproduce it

Author: Dr. Chakresh Kr. Singh
Branch: `RB-AI-publ`

This note explains every script behind the "RB AI Publications Map", what
each score means, which parts are plain computation and which are an LLM's
reading, and how to rerun everything and check the results yourself.

## 1. What is computed, and by what

| Result | How it is produced | Deterministic? |
|---|---|---|
| Article text, dates, authors | Scraped from rolandberger.com | Yes, for a given snapshot of the site |
| Author title and office | Scraped from each author's profile page | Yes, for a given snapshot |
| Topic clusters | Embeddings + k-means (fixed seed) | Yes |
| Article network | Cosine similarity between article vectors | Yes |
| Stance on AI (-2 to +2) | LLM reading of the full article | No: an LLM judgement, checked by two other LLMs and a human audit sheet |
| Evidence quote, themes, core message, quoted voices | LLM reading of the full article | No: every quote is then checked word for word against the article |
| Cluster names | LLM, from the titles and themes in each cluster | No: labels only, they change no number |
| Counts per year, author, office | Plain counting over the above | Yes |

In short: the structure (who, when, which topic group, which articles are
similar) is computed; the interpretation (stance, quotes, themes) is an LLM's
reading, and is verified as described in section 4.

## 2. The scripts

Run them in this order from the repo root. Each reads `config.yaml`; no
setting is hard-coded.

| Step | Command | Reads | Writes (all under `data/`, gitignored) |
|---|---|---|---|
| 1. Scrape | `uv run python scrape_rb_ai.py` | rolandberger.com | `rb_ai_publications.json` |
| 2. Chunk + embed | `uv run python embed_rb_ai.py` | step 1 | `rb_ai_chunks.jsonl`, `rb_ai_chunks_manifest.json` |
| 3. Analyse | `uv run python analyze_rb_ai.py` | steps 1 and 2 | `rb_ai_analysis.json`, `rb_ai_stance_audit.csv`, `cache_rb_ai/` |

Code behind each step:

| File | What it does |
|---|---|
| `src/scrape_rb.py` | Parses the listing page, article pages and profile pages; removes site boilerplate and calls to action |
| `src/chunk_rb.py` | Splits articles into sentence-aligned chunks; builds the metadata header |
| `src/analyze_rb.py` | Article vectors, clustering, network, quote check, per-year / author / office counts, model agreement, LLM calls |
| `src/rb_ai_models.py` | The fixed answer format (Pydantic schema) every LLM reply must match |
| `src/llm.py` | The one LLM client: temperature 0, fixed seed, JSON output, disk cache, one repair retry |
| `user_prompts/rb_ai/*.txt` | The exact prompts sent to the LLMs |

Tests: `uv run pytest tests/test_scrape_rb.py tests/test_chunk_rb.py tests/test_analyze_rb.py`
cover every deterministic function with small hand-made inputs (no network,
no LLM).

### Setup

1. `uv sync` (installs pinned dependencies from `uv.lock`).
2. Put an Ollama Cloud key in `.env` as `OLLAMA_API_KEY=...` (see `.env.example`).
3. The embedding model (`sentence-transformers/all-mpnet-base-v2`) downloads on
   first use (about 420 MB).

Runtime: scraping about 2 minutes (1 s pause between requests, by design);
embedding under 30 s; analysis a few minutes on first run, seconds when cached.

## 3. Every score, defined

### Cleaning (step 1)
- **Boilerplate**: a line appearing word for word in 3 or more articles
  (`scrape.boilerplate_min_articles`) is a site template and is removed.
- **Calls to action**: a sentence matching any of `scrape.call_to_action_patterns`
  ("Register now...", "download the full study", "contact one of our experts")
  is removed. Matching is per sentence, because the site often appends these
  to a real paragraph.

### Chunks and embeddings (step 2)
- Articles are split into chunks of at most 300 tokens (`embed.chunk_tokens`),
  whole sentences only, with up to 50 tokens of overlap (`embed.overlap_tokens`).
  Tokens are counted with the embedding model's own tokenizer, so no chunk is
  ever truncated by the model.
- Each chunk is embedded as "Title / Date / Authors" + text. Vectors are
  normalised to length 1.

### Topics and network (step 3)
- **Article vector**: the mean of its chunk vectors, renormalised to length 1.
- **Cosine similarity**: dot product of two article vectors; 1 = same content
  direction, 0 = unrelated.
- **Clusters**: k-means for k = 6 to 8 (`analysis.k_min`, `k_max`, seed 42);
  the k with the highest **silhouette** is kept. Silhouette runs from -1 to 1;
  values near 0.2, as here, mean the topics overlap. The range starts at 6
  because the overall best (k = 4) only separates 2018 magazine pieces from
  2025-26 studies, which is a date split, not a topic split.
- **Network links**: each article is linked to its 3 most similar articles
  (`analysis.neighbours`) when similarity is at least 0.65 (`analysis.min_similarity`).

### Stance on AI (step 3)
The prompt (`user_prompts/rb_ai/read_article.txt`) asks for one integer:

| Score | Meaning given to the model | Shown on the page as |
|---|---|---|
| +2 | AI is essential and urgent; firms must adopt it now | Champions AI |
| +1 | AI is an opportunity worth pursuing, with some caveats | Positive |
| 0 | Weighs opportunities and risks evenly, or is descriptive | Balanced |
| -1 | Sees value but stresses risks, limits or hype | Cautious |
| -2 | AI is mainly a threat or a danger to guard against | Warns against AI |

The page reports **counts** ("26 of 28 positive"), grouping +1 and +2 as
positive and -1 and -2 as cautious. Means are kept in the data file only.
The headline stance is the reading of the primary model (`llm.model`).

### Quote check
Every quote the LLM returns (evidence sentence and each expert quote) is
searched for in the article text, ignoring case, quote-mark style, dash
style and spacing (`quote_in_text`). `quote_verified: false` marks a
paraphrase. The page labels unverified evidence and hides unverified expert
quotes.

### Agreement between models
The same prompt goes to the check models (`analysis.check_models`, from other
model families). For the 56 articles the analysis reports:
- **Krippendorff's alpha (ordinal)**: agreement beyond chance; 1 = perfect,
  0 = chance level. Above 0.667 is usually read as acceptable, above 0.8 as good.
- **Same direction**: articles where all models put it in the same group
  (cautious, balanced or positive).
- **Within one point**: articles where the highest and lowest score differ by at most 1.
- **Exact**: articles where all models give the same score.

Each article also carries `stance_by_model`, so any disagreement can be inspected.

### Authors and offices
- **Author counts**: articles where the person is named as author.
- **Office counts**: distinct articles with at least one author from that
  office; two co-authors from the same office count once.
- Title and office come from the author's profile page **as it is today**,
  not at the time of writing. Think:Act magazine writers have no profile and
  are shown as "Think:Act writer".

## 4. How to verify the results

1. **Rerun and compare hashes.** `rb_ai_analysis.json` → `provenance` records
   the git commit, scrape time, SHA-256 of the input files and every setting
   used. Same inputs + same cache give identical results (only the
   `generated_at` timestamp differs).
2. **Replay the LLM answers.** Every LLM reply is cached in `data/cache_rb_ai/`
   with its full prompt. A rerun with the cache in place makes no LLM calls and
   reproduces every stance and quote exactly. Delete the folder to ask the
   models again.
3. **Check the quotes.** Run the analysis and read the last lines: they report
   how many evidence quotes were found word for word. Any `quote_verified:
   false` row in the JSON is a paraphrase.
4. **Check the stance yourself.** Open `data/rb_ai_stance_audit.csv`: one row
   per article with each model's score, the evidence sentence and the link.
   Fill in `human_stance` (-2 to +2) for a sample (for example 15 articles) and
   compare with the model columns.
5. **Run the tests**: `uv run pytest -q`.

## 5. Known limits

- **The website changes.** A new scrape may add or drop articles; the
  `scraped_at` time and file hashes pin the snapshot that was analysed.
- **LLMs change.** Temperature 0 and a fixed seed make answers stable but not
  guaranteed identical across provider updates; the cache is the exact
  record. Models are also retired: `deepseek-v4-flash` was retired on
  2026-09-25 and replaced by `deepseek-v4.1-flash` in `analysis.check_models`.
- **Stance is a reading, not a measurement.** Agreement between three models
  makes it more robust, not objective; the audit sheet is there for a human check.
- **Format confound.** The cautious years (2018, 2024) are almost all Think:Act
  magazine journalism; the positive years (2025-26) are consulting studies. Part
  of the "shift" is a shift in what Roland Berger publishes.
- **Sparse years.** There are almost no AI articles from 2019 to 2023.
- **Copyright.** Article text, quotes and the page itself stay out of this
  public repo (`data/` is gitignored); only code and this note are committed.
