# Author decision log — how this evaluation was steered

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).

This document records the methodological and engineering decisions I made
while directing this project, in chronological order, each with its
rationale and consequence. It is written for the evaluator: the technical
decision log lives in `docs/DECISIONS.md`; this file explains the *approach*.
All work was AI-assisted (Claude Code) under my direction; every decision
below was mine, and all execution (tests, pipeline runs) was performed by me
in a separate terminal for consistency.

## 1. Plan first, approve, then build — in independently runnable parts
No code was written before a full plan was reviewed and approved against the
master specification (`eval_policy/RB_GenAI_Eval_Plan.md`). The build was
then split into small parts, each ending in something I could run and verify
(`pytest` after every part) before the next part started.
*Why:* traceability and control — at no point does the pipeline contain
unverified machinery. *Consequence:* the git history on `eda`/`eval` reads
as a sequence of verified increments.

## 2. All prompts externalised to `user_prompts/`
Every LLM prompt lives in its own plain-text file, grouped by usage stage,
never in code. Plain text was chosen over JSON deliberately: prompts are
multi-line prose, and JSON escaping would hurt readability and git diffs
while adding no structure (models/parameters live in `config.yaml`; output
schemas are appended by code).
*Why:* prompts are the most-revised artefact — one place to change, clean
diffs, no code edits. *Consequence:* prompt evolution is fully auditable.

## 3. The EDA before the evaluation: three crude questions, run once
Before any sophisticated metric, I ran a deliberately naive baseline on the
`eda` branch — three yes/no questions with one-line reasons: Q0 can a model
tell human from AI zero-shot (no labels, no context — raw text only; we
grade it afterwards against the known truth, including per-side accuracy to
expose a judge that defaults to one answer); Q1 given the person's own
answers, does the AI capture their personality; Q2 does it capture the
content. The EDA is a snapshot, not an iteration loop, and its most
valuable output is the judges' verbatim reasons (`eda_reasons.csv`) — read,
not just counted. Reading them is what exposed the orthography confound
(decision 6) and set the hypotheses the eval branch tests.
*Why:* cheap honesty first — know what a naive judge sees before building
the machinery. *Consequence:* every eval-branch design choice traces to an
observed EDA failure, not to speculation.

## 4. Style is measured deterministically, never judged
Seven stylometric features (length, lexical diversity, word length,
contractions, hedging, sentiment, stance) are computed by pure, tested
functions with published lexicons — no LLM involvement, identical output on
every run. LLM judgement is reserved for questions deterministic code
cannot answer.
*Why:* the plan's core principle — deterministic and cheap first;
measurements must be beyond dispute before ratings enter the picture.

## 5. A fixed, explainable evaluation split — not a random one
The plan proposed a seeded random within-person split. I replaced it with a
fixed one: questions 7–10 are held out and judged for every person; 1–6 are
context. *Why:* a seeded shuffle is reproducible but not explainable — I can
defend "these four questions, same for everyone" to a stakeholder in one
sentence, and identical held-out questions make people directly comparable.
*Consequence:* the split lives in one config line (`eda.held_out_questions`)
that anyone can read and change.

## 6. The orthography confound: lowercase everything the judges see
Run 1's judges based most personality/authenticity verdicts on capitalisation
and typos. We cannot distinguish transcription artefacts from real style, so
I ruled that evidence inadmissible: all judge inputs are lowercased
(`eda.lowercase_judge_inputs`). Deterministic style features still measure
the raw text — style is *measured* there, not *judged*. Run 1 was archived,
not deleted (`archive/`), so the before/after is inspectable.
*Why:* a verdict that flips with the caps-lock key is not a verdict about
the person. *Consequence:* remaining differences are content and voice.

## 7. A judge panel of three model families, all scoring the SAME items
Three judges — gpt-oss:120b (OpenAI), deepseek-v4-flash (DeepSeek), glm-5.2
(Z.ai) — each score every item: all 60 answers zero-shot, and the identical
held-out set for personality/content. I considered splitting questions
across models (three folds) and rejected it: different items per judge
confounds judge harshness with question difficulty. Same items → true
head-to-head and measurable inter-judge agreement (Krippendorff's alpha).
Model choice was constrained to the free Ollama cloud tier and verified
empirically with a probe script (`scripts/check_models.py`); kimi-k3 and
qwen3.5 were rejected as unavailable, with structured-output reliability as
the selection criterion for the rest.
*Why:* generator model unknown → family diversity guards against
self-preference; empirical probing beats assumed availability.

## 8. LLMs output the smallest checkable thing; all arithmetic is ours
Judges return only a boolean (or a bounded −1…+1 score) plus a one-line
reason, validated against a strict Pydantic schema with one
retry-with-repair. Every percentage, accuracy, and KPI is computed in
tested Python code — no model is ever asked for a statistic.
*Why:* verifiability. A model's "83%" is unauditable; a counted share of
validated booleans is.

## 9. Public repository, confidential data quarantined
The repo stays public so evaluators can inspect everything: code, prompts,
tests, docs, decision history. The confidential interview data AND all
artefacts that embed it (the LLM cache, which stores full prompts; result
CSVs quoting answers) are gitignored and delivered separately in a private
zip. This trade-off was made consciously when the near-leak was caught
before the first push.
*Why:* transparency to evaluators without breaching data confidentiality.

## 10. Every model call cached, seeded, and resumable
Temperature 0, fixed seed, disk cache keyed on (model, prompt). A crashed
run resumes for free; a finished run reproduces without API keys.
*Consequence:* the entire evaluation re-runs deterministically from cache.

## 11. Publication-grade figures in the client's own visual language
Colors were extracted from rolandberger.com's stylesheet (blues, greens,
black — no red, per my instruction), validated for colorblind safety, and
each judge model keeps one fixed color across every figure with its name in
the legend. Plot code is deliberately simple; every mark carries its value.
*Why:* deliverables should read as the client's, and a figure that needs a
decoder ring fails its job.

## 12. Parallel workstreams under one method
While the eval pipeline was built part-by-part, a parallel branch
(`agentic_workflow`) implements the same evaluation as LangChain agents —
one agent per judge model with per-agent call logs, latency and failure
tracking, for deployment-time monitoring and scaling. Memory and
context-window optimisation are deliberately deferred and pinned as
limitations (`docs/AGENTIC_NOTES.md`) until scaling is actually needed.
The 2-page report (APA two-column) was likewise drafted in parallel with
placeholders, to be filled only with verified pipeline numbers.
*Why:* the deadline is fixed; independent work belongs in parallel, but
numbers enter deliverables only after verification.

## 13. Attribution and accountability
Every file and every commit carries the author line and the declaration
"Prepared for Roland Berger". AI assistance is disclosed in the commit
trailers. Progress and state are continuously recorded (`docs/DECISIONS.md`,
`CLAUDE.md` current-state block) so the project survives interruption
without losing the thread.
