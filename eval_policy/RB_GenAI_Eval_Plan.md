# RB GenAI Assignment — Evaluation Plan (Master Document, v2)

**Purpose:** a complete, self-explanatory specification of the analysis, deliverables and engineering approach, written to be handed to Claude Code without further explanation. Every choice states its reason. Deliberate omissions are stated too.

**Deadline discipline:** 24h hard, ~8h target. Fixed priority: (1) analysis + results → (2) 2-page report → (3) 3-slide deck → (4) app + Docker, only after 1–3.

---

## 0. The task

A Beauty & Wellbeing company built LLM-based *human simulations* of individual consumers (young business professionals, frequent travellers) from a first open-ended interview, following Park et al. (arXiv:2411.10109 — first posted Nov 2024 as "Generative Agent Simulations of 1,000 People", later retitled "LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals"; same study, cite with a footnote). We are given a subset of a second interview: 3 people × 10 open-ended questions, each with the real human answer and the simulation's answer (`RB_GenAI_Datatest.xlsx`; columns: Question Category, Question Text, Person ID, Human Answer, AI Answer). We must evaluate how faithfully the simulations mimic the real humans and deliver code, a 2-page report, and a 3-slide CEO deck.

**Framing:** the evaluation is a *risk control*. Its job is to tell the business where simulations can be trusted for decisions and where they cannot. Ground truth = the human answers. No independent human raters (deliberate; see limitations). All judgement is deterministic, or LLM-based with reliability checks.

**Anchor — the brief's five guiding bullets.** Every section must trace to one of these. Nothing else gets built.

| Brief bullet | Where it is answered |
|---|---|
| Define evaluation criteria (what, how, why) | Three KPIs, each with method + one-line rationale. Report §1. |
| Measure alignment with real human behaviour in context | Results per category and per person; RAG matrix by category. Report §2. |
| Strengths & weaknesses of *this approach*; improvements | Deterministic-first = scalable/auditable; limitations; data requests; upgrade paths. Report §4. |
| Where the simulation does not fully mimic humans | Blind spots, phantoms, distorted-valence themes; contradiction rate by category; style deviations; individuation. Report §3 — most space. |
| Reflect before beginning | Hour-1 manual read + written hypotheses; paper-informed expectations; client wants *segment* fidelity, paper measured *individual* fidelity. Report opening. |

---

## 1. Design principles

- **Deterministic and cheap first; LLM judgement only where deterministic methods cannot answer.** Every layer runs identically on 30 pairs or 50,000.
- **Approach over answer.** n = 30. Report direction and effect size, not p-values. State what 30 pairs cannot show.
- **Transparency and logging are first-class.** Every model call cached, seeded, versioned, cost-logged. Every metric has a stated definition and limitation.
- **No manual labelling by the author** (confirmation bias; no independent rater). Named limitation.
- **Simple tooling.** LangChain LCEL chains. No LangGraph, no multi-agent choreography. If a step can be a function, it is a function.
- **Deliverables carry findings, not methods.** Method lives in appendix/README; slides carry conclusions with action titles.
- **Score decisions, not just simulations.** The end product is a decision map (use here / don't use there / why), with metrics as evidence.

## 1a. Coding standard — HARD REQUIREMENT, applies to every file

- **Simple, understandable, clean, effective.** No clever abstractions, no frameworks for their own sake. Anyone with Python experience must be able to read and run every file.
- Every function and class has a docstring (purpose, args, returns). Every file has a header comment saying what it does and how it fits.
- Inline comments explain *why*, not *what*.
- Type hints on all public functions. Pure functions where possible (input → output, no hidden state). Models loaded once, never inside a per-call function.
- `pytest` for every deterministic piece: stylometric features, segmenter, KPI maths, loaders. Tests are small, named for what they check, and run in seconds.
- Pydantic models for every structured LLM response; JSON as the default wire format.
- Config in `config.yaml`; no magic numbers in code.
- One entrypoint per branch (`run_eda.py`, `run_all.py`). No notebooks as deliverables (scratch notebooks may exist under `notebooks/` but are not the pipeline).
- `ruff` clean. No dead code.

## 1b. Branch structure

Two git branches, merged only when each is green.

- **`eda`** — naive first pass. Purpose: set up the plumbing (data loading, Ollama client, Pydantic parsing, caching, logging) and produce very basic yes/no metrics as a sanity baseline. Runs once.
- **`eval`** — the full pipeline (Layers 1–4, KPIs, deliverables). Reuses `eda`'s plumbing.

Shared from day one on both: `src/load.py`, `src/llm.py` (Ollama client + cache + Pydantic parsing), `src/log.py`, `config.yaml`, tests.

## 1c. EDA branch — specification

**Goal.** A crude, honest baseline: can an LLM tell human from simulation with no help at all; then, given the person's own examples, does it think the simulation captures the person; and does it capture the content? Three yes/no questions with a one-line reason each. Nothing more.

**Model.** Ollama, `gpt-oss-120b:cloud` first; others may be added later behind the same client. Temperature 0, fixed seed, JSON output enforced (`format="json"`), responses cached to `cache/eda/`. **Run once**; the EDA is a snapshot, not an iteration loop.

**Data split — context vs held-out (not train/test in the ML sense).**
- Split *within each person, by question*: e.g. 6 questions per person go into a **context set** shown to the judge as labelled examples; the remaining 4 per person are **held out** and scored. Never split by person — the personality question needs the person's own examples in context.
- The split is fixed by seed and recorded in `results/eda_split.json`.
- Note in report: because labelled examples are shown, this is few-shot-calibrated judgement, not zero-shot; and no unlabelled independent test exists — that is exactly why the `eval` branch adds blind judging.

**Three questions, in this order.**

| Q | Scope | Input to the LLM | Output |
|---|---|---|---|
| Q0 Human or AI? — zero-shot | all 60 answers | one text alone; no context, no examples, no label | `is_ai: bool`, `reason: str` |
| Q1 Captures nuance/personality? — within-person split | held-out AI answers (4 per person) | this person's *other* human answers as context + the held-out AI answer | `captures_personality: bool`, `reason: str` |
| Q2 Captures content? — pairwise | held-out pairs | question text + human answer + AI answer | `captures_content: bool`, `reason: str` |

Q0 runs first and needs no split; it is the uncalibrated baseline. Q1 is the only step that uses the within-person split. Q2 needs no context.

**Pydantic models — one per question (`src/eda_models.py`).** Separate schemas keep validation strict (no optional fields) and let the LLM helper be generic.
```python
class IsAI(BaseModel):                 # Q0
    is_ai: bool
    reason: str                        # one line

class CapturesPersonality(BaseModel):  # Q1
    captures_personality: bool
    reason: str

class CapturesContent(BaseModel):      # Q2
    captures_content: bool
    reason: str
```
LLM helper (`src/llm.py`): `ask(prompt, schema, model) -> schema` — puts `schema.model_json_schema()` in the prompt, calls Ollama with `format="json"`, temp 0, fixed seed, parses with `schema.model_validate_json(raw)`, one retry-with-repair on `ValidationError`, disk cache keyed on (model, prompt hash). Three functions in `src/eda.py`: `judge_is_ai(text)`, `judge_personality(context_answers, ai_answer)`, `judge_content(question, human, ai)`. Results: one CSV per question with `person_id, question_id, side/pair, answer, reason, model`.

**Metrics reported (`results/eda_metrics.csv`).**
- Q0 accuracy = share of the 60 answers correctly labelled (zero-shot). 50% = can't tell. Also report accuracy on human vs AI separately (bias toward one label is common).
- Q1 share "yes" (12 checks), Q2 share "yes" (12 pairs), per person and per category.
- Reasons kept verbatim in `results/eda_reasons.csv` for reading — the reasons are the useful part of an EDA.

**Files.** `run_eda.py` (single entrypoint), `src/eda.py` (the three prompts + loop), `src/llm.py` (Ollama call, cache, Pydantic parse with one retry-with-repair), `tests/test_eda.py` (split is deterministic and person-balanced; Pydantic parsing; cache hit).

**What the EDA is not.** Not a KPI, not the report headline. It is the sanity baseline the `eval` branch improves on, and the place where the plumbing gets built and tested.

## 1d. `Stylometric` class — specification (built first on the `eval` branch, used on both sets)

One small class, `src/stylometric.py`, computed on every answer (human and AI) before anything else.

```python
class Stylometric:
    """Deterministic style features for one text. Each method is a pure function
    (text -> number) so it can be wrapped as a LangChain @tool later."""

    def __init__(self, text: str): ...

    def length(self) -> int: ...                 # word count
    def lexical_diversity(self) -> float: ...    # MTLD via lexicalrichness
    def mean_word_length(self) -> float: ...
    def contraction_rate(self) -> float: ...     # contractions / words
    def hedging_density(self) -> float: ...      # hedges per 100 words (lexicon)
    def sentiment(self) -> dict: ...             # {"mean": .., "std": ..} VADER per sentence
    def stance_rate(self) -> float: ...          # stance markers per 100 words (lexicon)

    def all_features(self) -> dict: ...          # one flat dict, feature -> value
```
- Lexicons live in `src/lexicons.py` and are printed in the appendix.
- `all_features()` is applied row-wise to produce `results/measured_features.csv` with columns `person_id, question_id, side (human|ai), <7 features>`.
- Comparison and the dumbbell figure live in `src/style_compare.py`, not in the class — the class only measures.
- `tests/test_stylometric.py`: known inputs → known outputs for each method (e.g. a text with two contractions in ten words → 0.2).
- Later (Layer 3+), methods can be exposed as tools; LangSmith tracing is enabled via env var in `src/llm.py`. Nothing in the class depends on LangChain.

## 2. What the reference paper tells us (and how it shapes the design)

- Architecture (SM §3): full Interview-1 transcript placed in context + "expert reflections" (psychologist / behavioural economist / political scientist / demographer notes generated once and stored) + chain-of-thought answering. GPT-4o. No fine-tuning.
- Evaluation: closed-form items only (GSS categorical, Big Five Likert, economic games). Accuracy/correlation **normalised by each participant's two-week test–retest consistency** (interview agents ≈ 0.83–0.86 of the human ceiling; demographics-only ≈ 0.74). No LLM judges — none needed for closed-form outcomes.
- Ablations: removing 80% of the transcript or summarising it to bullets barely hurts → predictive power is *content*, not linguistic style.
- Mechanisms named: **direct retrieval** (answer found in transcript) and **inference** (deduced from world knowledge). Reflection layer risks **flattening** (answering as the summary of the person).
- Stated limitation: individual accuracy ≠ correct *population* structure.

Consequences for us: (a) our data is open text — the paper gives no method for scoring it, which is why our layers exist; (b) we cannot normalise against test–retest → #1 data request; (c) style layer is about believability and insight texture, not prediction; (d) our BERTopic topic map targets exactly the population-structure gap the paper admits; (e) our three layers map to retrieval (entailment), inference/flattening (individuation), and flattening (style); (f) style deviations may partly be prompt artefacts (CoT + expert reflections) → recommend the client log the generation prompt.

---

## 3. Analysis pipeline

Input: one row per (Person, Question) with `human_text`, `ai_text`, `category`.

### Layer 1 — Deterministic content and style baseline

**1a. Semantic similarity (baseline only).** `sentence-transformers/all-mpnet-base-v2`, cosine per pair. Reported, but explicitly *topic-dominated and stance-blind*; not a KPI.

**1b. Seven stylometric features (deterministic) — implemented as the `Stylometric` class (§1d), computed first, on both sets.**

| Feature | Definition | Tool | Plain-English question |
|---|---|---|---|
| Length | word count | stdlib | Does it talk more than the person? |
| Lexical diversity | MTLD | `lexicalrichness` | Is vocabulary as varied? |
| Mean word length | chars/word | stdlib | More formal/"written"? |
| Contraction rate | contractions/words | regex list | Talks like chat or like a report? |
| Hedging density | hedges per 100 words | lexicon (Hyland 2005) | Commits or hedges? |
| Sentiment mean + spread | VADER compound per sentence → mean, std | `nltk` | Swings like a person or sits neutral? |
| Stance-marker rate | "I think/feel/prefer/love/hate…" per 100 words | lexicon | Takes positions or describes? |

Comparison: paired difference (AI − human) per pair → mean and Cohen's d per feature. Bootstrap CI in appendix only. **One figure:** seven-row dumbbell chart, human dot vs AI dot. Lexicons shipped in `src/lexicons.py`, printed in appendix.
Not used in primary run (listed as upgrade path): HF sentiment/formality/zero-shot hedging models. Reason: keep the layer fully deterministic and dependency-light.

### Layer 2 — Content: micro (per pair) and macro (per topic)

Two views, two levels, no overlap in vocabulary. **2b** is *micro*: one person, one question — did the simulation say what this person said? **2c** is *macro*: pool everything — do humans and simulations talk about the same subjects in the same proportions?

**2a. Segmentation (feeds 2b only).**
- Unit = **sentence**. spaCy `en_core_web_sm` sentence splitter, nothing else. This is what SummaC-style NLI consistency work uses; no clause splitting, no coreference, no LLM decomposition.
- Report sentences-per-answer for humans vs AI (needed to read fidelity).
- If, after first results, sentences look too coarse, the only upgrade considered is LLM atomic-claim decomposition on 10 pairs as a check. Not before.

**2b. Coverage and fidelity (micro).**

The NLI model is a function that takes a pair `(premise, hypothesis)` and returns one of three labels: *entailment* (supported), *neutral* (not mentioned), *contradiction* (opposite). The premise is the reference text; the hypothesis is the sentence being checked. Model: `cross-encoder/nli-deberta-v3-base` (SNLI+MNLI, Apache-2.0), loaded once, local, deterministic.

Two numbers, same model, two directions:

| Number | Question | premise (reference) | hypothesis (checked, one at a time) | Formula |
|---|---|---|---|---|
| **Coverage** | How much of what the person said did the simulation keep? | the whole **AI** answer | each **human** unit | # human units labelled *entailment* ÷ # human units |
| **Fidelity** | How much of what the simulation said is actually true to the person? | the whole **human** answer | each **AI** unit | # AI units labelled *entailment* ÷ # AI units |

Why both: they are not symmetric. A long, padded AI answer scores 100% coverage and low fidelity; a short, safe one scores 100% fidelity and low coverage. Coverage alone rewards saying more; fidelity alone rewards saying less. Report both; harmonic mean (F1) if one headline is needed. Low coverage = thin; low fidelity = padding/invention.

Contradiction count = how many of the fidelity-direction calls returned *contradiction*. No extra step — it is one of the three labels every call already returns. Shown as a flag beside the two percentages because "opposite" is more dangerous than "not mentioned".

Reference implementation (the whole of it):
```python
nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")   # load once
LABELS = ["contradiction", "entailment", "neutral"]         # model output order

def check(reference: str, sentences: list[str]) -> list[str]:
    """Label each sentence against the reference text."""
    scores = nli.predict([(reference, s) for s in sentences])
    return [LABELS[row.argmax()] for row in scores]

human_labels = check(ai_answer, human_units)   # coverage direction
ai_labels    = check(human_answer, ai_units)   # fidelity direction
coverage = human_labels.count("entailment") / len(human_units)
fidelity = ai_labels.count("entailment")    / len(ai_units)
contradictions = ai_labels.count("contradiction")
```
- Vocabulary for 2b: *kept / missing / contradicted* (coverage side), *backed / invented / contradicted* (fidelity side). Never "blind spot" or "phantom" here.
- Report per pair, then averaged per person and per category.
- Standard lineage to cite: SummaC (Laban et al. 2022), TRUE (Honovich et al. 2022), AlignScore (Zha et al. 2023); same units→entailment recipe as RAGAS/FActScore with a local NLI checker instead of an LLM.
- Caveat: SNLI/MNLI are short clean pairs; our premise is a whole conversational answer (fits in 512 tokens). Domain shift acknowledged.

**2c. Topic map with BERTopic (macro) — the key figure.**
- Split all 30 human answers and all 30 AI answers into sentences; tag each `human` or `ai`. Fit **one** BERTopic model on the pooled sentences with default settings (let the library handle embedding, reduction and clustering; do not hand-tune UMAP/HDBSCAN at n≈400 sentences — set `min_topic_size` small, e.g. 5, and fix the seed).
- Use `topics_per_class` with `human`/`ai` as the class → per topic: count and share of human vs AI sentences.
- Classify each topic:
  - **Covered** — both sides present at similar share
  - **Blind spot** — human-heavy, AI near-absent
  - **Phantom** — AI-heavy, human near-absent
  - Optional: colour sentences by VADER polarity → **covered but wrong-way** where the two sides diverge in sentiment.
- Figure: BERTopic's per-class bar chart (topic on the y-axis, human vs AI share side by side), plus the built-in 2D topic overview if it renders cleanly. Name topics from BERTopic's top words; optionally tidy names with one LLM call.
- Vocabulary for 2c: *covered / blind spot / phantom*. Only here.

**How Layer 2 feeds the KPIs (exact mapping).**
| KPI | Comes from |
|---|---|
| KPI 1 Content Fidelity | 2b: mean Coverage and mean Fidelity (reported as two numbers, or their harmonic mean as one), with the contradiction count shown beside them, by category |
| KPI 2 Segment Coverage | 2c: share of topics classified *covered*, plus the counts of *blind spots* and *phantoms*; the RAG matrix by category uses these together with 2b's per-category coverage |
| KPI 3 Believability | Layer 1b style features + Layer 3 judges (not Layer 2) |

### Layer 3 — LLM judge panel (3–5 models, different families)

Two constructs, anchored rubric, structured JSON (score + reason), temp 0, cached. Judges are *ratings*; kept in a separate results table from Layers 1–2 *measurements*.

**Judge A — Individuation (persona vs mean).** Input: the person's *other 9 human answers* as a persona sketch + human and AI answer to the held-out question. Question: "Is the AI answer recognisably *this* person, or a generic member of the segment?" Score −1 (generic/mean-reverting) … +1 (distinctly this person). Reason required. Rationale: a judge shown a single pair cannot assess convergence to the mean.

**Judge B — Believability (blind).** Input: one answer, no label, randomised human/AI, randomised order. Question: "How human does this read?" −1 (clearly AI) … +1 (clearly human). Reason required. Also compute **blind discrimination rate** (sign matches true label; ~50% = indistinguishable). Blinding mandatory.

**Panel & reliability.** Judges are called through the same Ollama client as the EDA (`gpt-oss-120b:cloud` first; add other families behind the same interface). 3–5 judges from different model families (generator unknown → diversify against self-preference; say so). **Krippendorff's α** (interval; `krippendorff` pkg) per construct; per-judge means; high-disagreement items listed in appendix as "contested". No arbitration.

Note: the reference paper uses no LLM judges because its outcomes are closed-form; ours are open text, so a panel is required. Panel + α is standard eval practice, not the differentiator.

### Layer 4 — Optional differentiator (if ≥1h remains): individuation control (3×3)

For each question, score every simulation against every human (3×3 matrix; any similarity metric — cosine or claim-F1). Diagonal = matched. Report **individuation margin** = diagonal − mean off-diagonal, and top-1 identification rate vs 33% chance. Answers: does the simulation carry *this person's* signal or only the archetype? Underpowered at n=3 (report bootstrap CI, note humans are same-segment by construction) but decisive at scale — frame as demonstrated on subset, powerful at full data. Also possible: within-interview self-consistency of each simulation across its 10 answers (proxy for the paper's test–retest idea).

---

## 4. KPIs (three) and business translation

**KPI 1 — Content Fidelity: "Does it say what our customers said?"** From 2b: Coverage and Fidelity (0–100%), by category, with the count of contradicted units shown beside them. Coverage = completeness; fidelity = accuracy; contradiction = danger.

**KPI 2 — Segment Coverage: "Would our researchers reach the same conclusions?"** From 2c (BERTopic): share of topics *covered*; counts of **blind spots** and **phantoms**. Combined with 2b per-category coverage into a **green / amber / red matrix by question category** — this is the recommendation.

**KPI 3 — Believability: "Could a human tell?"** Blind discrimination rate + individuation score, with α as reliability tag. Style dumbbell as supporting evidence.

Optional composite Fidelity Score (0–100, equal weights, sensitivity under two alternative weightings) as headline; the three stay visible — composite hides diagnosis.

**RAG thresholds:** stated in `config.yaml`, marked provisional at n=30.

**Cost/time (deck, bottom-up, assumptions flagged).** Traditional 30-respondent qual wave ≈ €25–50k, 6–8 weeks. Simulation: interview is one-time sunk cost; incremental question ≈ €0.01–0.10 per simulated response, minutes. Framing: **cost per incremental research question** ~€30k / 6 weeks → ~€100 / 2 hours — *inside the green zone only*. Log this evaluation's actual token spend; extrapolate to 1,000 × 50; expected low hundreds of euros → "continuous validation is essentially free."

---

## 5. Deliberately NOT done

- HF transformer stylometrics, style classifiers, fingerprint retention (upgrade path only).
- LangGraph / multi-agent orchestration.
- Manual gold labels.
- Arbitration on judge disagreement (reported, not resolved).
- p-value narrative (effect sizes; CIs in appendix).
- Anything not traceable to the five brief bullets.

## 6. Limitations to state up front (Report §4)

1. n = 30 pairs; no power; results directional. Bootstrap CIs in appendix show width.
2. Human answers as ground truth, but humans aren't self-consistent; no test–retest data (paper's normalisation). **#1 data request:** re-interview subset for a human ceiling.
3. No independent human raters; author labelling avoided. All judgement LLM-based, α reported.
4. Possible judge self-preference; generator unknown; mitigated by family diversity.
5. Three people cannot represent the segment.
6. Generation/evaluation leakage (paper's "direct retrieval") cannot be checked without Interview 1. **#2 data request:** provide Interview 1 for a retrieval-vs-simulation check.
7. NLI/embedding models trained on non-conversational text; domain shift.
8. Style deviations may be prompt artefacts (CoT + expert reflections per the paper); recommend logging generation prompts so style is tuned, not judged.
Each limitation paired with the specific fix.

---

## 7. Repository and engineering standards

```
rb-genai-eval/
  README.md            # architecture diagram, how to run, results summary, limitations, cost
  pyproject.toml       # pinned deps
  .env.example         # never commit real keys
  config.yaml          # models, seeds, thresholds, weights, paths
  data/                # RB_GenAI_Datatest.xlsx (CONFIDENTIAL — see data note)
  cache/               # all LLM/NLI outputs as JSON, committed → reproducible w/o keys
  src/
    load.py  llm.py  log.py  lexicons.py
    eda.py  stylometric.py  style_compare.py
    segment.py  entail.py  cluster.py  judges.py  kpis.py  plots.py
  tests/               # pytest: test_load, test_eda, test_stylometric, test_segment, test_kpis
  run_eda.py           # EDA branch entrypoint → results/eda_*.csv
  run_all.py           # eval branch entrypoint → results/, figures/
  results/             # measured_features.csv, ratings.csv, kpis.csv, run_manifest.json
  figures/
  app/                 # app.py (Streamlit), Dockerfile
  report/  deck/  design/tokens.md
```
- Temp 0, fixed seeds, pinned versions. Disk cache keyed on (model, prompt hash, input hash).
- `run_manifest.json`: model IDs, versions, seeds, tokens, € per layer, wall time.
- Pydantic structured outputs; retry-with-repair; no regex on free text.
- Type hints, docstrings, `ruff`, tests for deterministic pieces (segmenter, style, KPI maths).
- Data note: deliver as private zip by email with the file included; `.gitignore` it for any public repo and say so.

## 8. Deliverables

**Report (≤2 pages), answer-first:** verdict → framework diagram (layers → 3 KPIs → RAG matrix) → results (KPI table, theme figure, style dumbbell, α) → limitations with fixes → recommendation. Appendix separate: definitions, lexicons, prompts, CIs, contested items, manifest.

**Deck (3 core slides + title + appendix), action titles:**
S1 — the question and the verdict (composite, one-line answer, RAG preview).
S2 — the method in one visual (deterministic first, LLM judgement second, reliability check; why each).
S3 — where to use it and what it saves (RAG matrix; blind spots/phantoms; cost per incremental question with assumptions; rollout).
Design: RB tokens (§10). No decoration.

**Code:** as §7.

## 9. App (only after 1–3)

Streamlit reading `results/` + `figures/`: KPIs, RAG matrix, theme chart, style chart, judge scores + reasons per item, manifest/cost; filters by person/category; optional "re-run judges" gated on API key, default cached mode. Dockerfile (`python:3.11-slim`, non-root, lockfile, `EXPOSE 8501`); `docker run -p 8501:8501 rb-genai-eval` must work with no keys. Working > pretty. Self-hosting = one README paragraph, not implemented.

## 10. Design tokens (PLACEHOLDER)

Derive from RB materials the author supplies; do not guess. Record primary/accent/neutral hex, heading/body typefaces (or open substitutes), ordered chart palette, slide conventions (action-title position, footer, source line, logo). Save to `design/tokens.md`; use in `plots.py`, deck, app.

## 11. Timeline (8h target)

| Time | Task |
|---|---|
| 0:00–1:00 | Read all 30 pairs by hand; write 3–5 hypotheses (retrieval / inference / flattening; where contradiction likely). |
| (eda branch, before eval) | Build plumbing (load, Ollama client, cache, Pydantic, log), `Stylometric` class + tests, run the three-question EDA once, read the reasons. |
| 1:00–2:30 | Layer 1: load, embeddings, seven style features, dumbbell. |
| 2:30–4:30 | Layer 2: sentence split, NLI coverage/fidelity, BERTopic topic map + per-class figure. |
| 4:30–5:30 | Layer 3: judge chains, panel, α. |
| 5:30–6:15 | KPIs, RAG matrix, cost model, manifest. |
| 6:15–7:15 | Deck. |
| 7:15–8:00 | Report, README, cleanup, zip. |
| after | App + Docker → then Layer 4 (3×3) if time. |

## 12. Definition of done

- `pytest` passes; every function has a docstring; files are commented; `ruff` clean.
- `python run_eda.py` and `python run_all.py` reproduce every number/figure from cache without API keys.
- Report ≤2 pages, deck 3 core slides, both answer-first, both RB style, both traceable to the five brief bullets.
- README covers architecture, run, results, limitations, cost.
- Every metric has a definition and a limitation.
- Docker app runs cached with one command.
