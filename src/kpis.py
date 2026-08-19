"""KPI computation and the RAG matrix (plan §4).

KPI 1 — Content Fidelity: mean coverage and fidelity from Layer 2b, with the
contradiction count beside them, overall and by category.
KPI 2 — Segment Coverage: share of covered topics, blind spot and phantom
counts from Layer 2c; combined with per-category coverage into the
green/amber/red matrix — the actual recommendation.
KPI 3 — Believability: assembled in run_all.py from the Layer 3 judge
helpers (discrimination rate, individuation mean, Krippendorff's alpha).

Pure functions over DataFrames; thresholds come from config (provisional
at n = 30, and stated so wherever they appear).

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import pandas as pd


def harmonic_mean(a: float, b: float) -> float:
    """Harmonic mean of two non-negative rates; 0.0 when both are 0."""
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def rag_status(value: float, green: float, amber: float) -> str:
    """Map a 0–1 score to 'green' / 'amber' / 'red' (thresholds inclusive)."""
    if value >= green:
        return "green"
    if value >= amber:
        return "amber"
    return "red"


def content_fidelity_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """KPI 1 — coverage/fidelity summary, one row overall + one per category.

    Args:
        pairs: per-pair Layer 2b results with columns
            category, coverage, fidelity, contradictions.

    Returns:
        Rows with scope, coverage, fidelity, f1 (harmonic mean of the two
        means), contradictions (summed), n_pairs.
    """

    def summarise(scope: str, part: pd.DataFrame) -> dict:
        coverage = part["coverage"].mean()
        fidelity = part["fidelity"].mean()
        return {
            "scope": scope,
            "coverage": coverage,
            "fidelity": fidelity,
            "f1": harmonic_mean(coverage, fidelity),
            "contradictions": int(part["contradictions"].sum()),
            "n_pairs": len(part),
        }

    rows = [summarise("overall", pairs)]
    rows += [summarise(cat, part) for cat, part in pairs.groupby("category")]
    return pd.DataFrame(rows)


def segment_coverage_summary(topics: pd.DataFrame) -> dict[str, float | int]:
    """KPI 2 — topic-level summary from the Layer 2c table.

    Args:
        topics: fit_topics output (one row per topic with classification).

    Returns:
        share_covered plus the raw counts per classification.
    """
    counts = topics["classification"].value_counts()
    n_topics = len(topics)
    return {
        "n_topics": n_topics,
        "covered": int(counts.get("covered", 0)),
        "blind_spots": int(counts.get("blind_spot", 0)),
        "phantoms": int(counts.get("phantom", 0)),
        "share_covered": counts.get("covered", 0) / n_topics if n_topics else 0.0,
    }


def rag_matrix(pairs: pd.DataFrame, green: float, amber: float) -> pd.DataFrame:
    """The recommendation: green/amber/red per question category.

    Status is driven by the harmonic mean of category coverage and fidelity
    (both must be decent — one cannot buy the other), with the contradiction
    count carried beside it as the danger flag.
    """
    table = content_fidelity_table(pairs)
    table = table[table["scope"] != "overall"].rename(columns={"scope": "category"})
    table["status"] = [rag_status(f1, green, amber) for f1 in table["f1"]]
    return table.reset_index(drop=True)
