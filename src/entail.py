"""NLI-based coverage and fidelity — the heart of Layer 2 (plan §2b).

One local NLI cross-encoder labels each sentence (hypothesis) against a
whole reference answer (premise) as contradiction / entailment / neutral.
Run in both directions:

    Coverage  — premise = AI answer,    hypotheses = human sentences.
                "How much of what the person said did the simulation keep?"
    Fidelity  — premise = human answer, hypotheses = AI sentences.
                "How much of what the simulation said is true to the person?"

They are not symmetric: a padded AI answer scores high coverage / low
fidelity; a short safe one the reverse. Contradictions are counted on the
fidelity direction — "opposite" is more dangerous than "not mentioned".
Lineage: SummaC (Laban et al. 2022), TRUE (Honovich et al. 2022),
AlignScore (Zha et al. 2023). The model is loaded once, on first use.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import yaml
from sentence_transformers import CrossEncoder

_CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())
_NLI_MODEL_NAME: str = _CONFIG["eval"]["nli_model"]

# Output order of cross-encoder/nli-deberta-v3-base — fixed by the model.
LABELS: list[str] = ["contradiction", "entailment", "neutral"]

# Loaded lazily on first use, then cached (module import stays light).
_NLI: CrossEncoder | None = None


def _nli() -> CrossEncoder:
    """Return the shared NLI cross-encoder, loading it on first call only."""
    global _NLI
    if _NLI is None:
        _NLI = CrossEncoder(_NLI_MODEL_NAME)
    return _NLI


def check(reference: str, sentences: list[str]) -> list[str]:
    """Label each sentence against the reference text.

    Args:
        reference: the premise — one whole answer.
        sentences: hypotheses — sentences checked one at a time.

    Returns:
        One label per sentence: 'contradiction' | 'entailment' | 'neutral'.
    """
    if not sentences:
        return []
    scores = _nli().predict([(reference, s) for s in sentences])
    return [LABELS[row.argmax()] for row in scores]


def coverage_fidelity(
    human_answer: str, ai_answer: str, human_units: list[str], ai_units: list[str]
) -> dict[str, float | int]:
    """Compute both directions for one (human, AI) answer pair.

    Args:
        human_answer: the person's whole answer (fidelity premise).
        ai_answer: the simulation's whole answer (coverage premise).
        human_units: the human answer split into sentences.
        ai_units: the AI answer split into sentences.

    Returns:
        dict with coverage (share of human units entailed by the AI answer),
        fidelity (share of AI units entailed by the human answer),
        contradictions (count, fidelity direction), and both unit counts.
        Empty unit lists yield 0.0 for that direction.
    """
    human_labels = check(ai_answer, human_units)  # coverage direction
    ai_labels = check(human_answer, ai_units)     # fidelity direction
    return {
        "coverage": (
            human_labels.count("entailment") / len(human_units) if human_units else 0.0
        ),
        "fidelity": ai_labels.count("entailment") / len(ai_units) if ai_units else 0.0,
        "contradictions": ai_labels.count("contradiction"),
        "n_human_units": len(human_units),
        "n_ai_units": len(ai_units),
    }
