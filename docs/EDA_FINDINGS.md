# EDA findings — first read of the results (2026-08-19)

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).

Model: gpt-oss:120b (Ollama cloud), temp 0, seed 42. 60 Q0 + 12 Q1 + 12 Q2
judgements, all cached. Style features fully deterministic (src/stylometric.py).

## Headline

**The simulations are believable but not faithful.** Read alone, AI answers
pass as human most of the time (Q0). Compared against what the person
actually said, they fail on both personality (Q1: 17% yes) and content
(Q2: 8% yes). Convincing-but-wrong is the risk profile the eval branch must
quantify.

## Q0 — zero-shot human-or-AI: 58% overall, strongly asymmetric

- Human answers: 80% correctly called human. AI answers: only 37% correctly
  called AI — 19 of 30 AI answers fooled the judge.
- The judge leans toward "human" (43 of 60 verdicts). So the 58% overall is
  not "can't tell": it can spot humans, but the simulations read human too.

## Q1 — captures personality: 2 of 12 yes

- Every "no" reason says the same thing: the AI is polished, capitalized,
  marketing-toned; the real people write informal, lowercase, with typos,
  brief and pragmatic. Classic **flattening** (the paper's term).
- Caveat for eval branch: the judge keys heavily on orthography (lowercase,
  typos), which may be a transcription artefact rather than personality.
  Consider case-normalising or instructing judges to ignore formatting.
- The only 2 yes verdicts are for g_human — that simulation sometimes mirrors
  the person's informal style. Individuation quality varies by person.

## Q2 — captures content: 1 of 12 yes

- The dominant failure is **addition, not omission**: the AI injects
  preferences never stated (sustainability, reviews, dermatologists,
  eco-packaging) and omits the person's specifics (price sensitivity,
  glass bottle, oud smell).
- Two outright contradictions: g_human q8 (AI picks a different lotion
  option than the person) and g_human q7 (AI invents ingredients where the
  person explicitly had none). Contradictions are the dangerous class.
- Caveat: the binary yes/no is harsh — several "no" reasons concede the core
  points matched. Layer 2's coverage/fidelity split will measure this
  properly instead of collapsing it to one bit.
- "Sustainability" recurs across people as an injected theme → likely a
  **phantom topic** for BERTopic to confirm at Layer 2c.

## Style features (deterministic) — the flattening, measured

| Feature (mean) | Human | AI | Reading |
|---|---|---|---|
| length (words) | 31.5 | 55.0 | AI talks ~75% more |
| lexical diversity (MTLD) | 56.3 | 95.3 | far more varied vocabulary |
| mean word length | 4.27 | 4.74 | more "written", formal |
| contraction rate | 0.017 | 0.010 | less chatty |
| hedging /100w | 5.57 | 2.35 | humans hedge 2.4× more |
| sentiment mean | 0.28 | 0.68 | **AI is strongly positive; humans near neutral** |
| sentiment std | 0.12 | 0.11 | similar swing |
| stance /100w | 1.70 | 1.42 | slightly fewer positions |

The sentiment gap (0.28 → 0.68) is the sharpest single signal: the
simulations sell, the humans shrug. Distorted valence, exactly what the
plan's "covered but wrong-way" check targets.

## Hypotheses carried into the eval branch

1. Fidelity (AI units backed by the human answer) will be low; coverage
   moderate — padding/invention is the failure mode, not thinness.
2. BERTopic will show phantom topics (sustainability, ingredient talk) that
   humans barely mention.
3. Style deviations may be generation-prompt artefacts (CoT + expert
   reflections per Park et al.) → keep the "log generation prompts" data
   request.
4. Judge sensitivity to orthography must be controlled before trusting
   personality scores.
