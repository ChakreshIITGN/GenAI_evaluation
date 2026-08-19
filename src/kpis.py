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


def content_fidelity_table(pairs: pd.DataFrame, by: str = "person_id") -> pd.DataFrame:
    """KPI 1 — coverage/fidelity summary, one row overall + one per group.

    Grouping defaults to PERSON (author decision): per-category groups hold
    only 3–15 pairs in this data — too small for a robust verdict — so
    results are pooled over categories and split by person instead. The
    aggregation choice is reported in the deliverables.

    Args:
        pairs: per-pair Layer 2b results with columns
            coverage, fidelity, contradictions, and the `by` column.
        by: grouping column ('person_id' default; 'category' possible at
            larger n).

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
    rows += [summarise(str(group), part) for group, part in pairs.groupby(by)]
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


def rag_matrix(
    pairs: pd.DataFrame, green: float, amber: float, by: str = "person_id"
) -> pd.DataFrame:
    """The recommendation: green/amber/red per group (person by default).

    Status is driven by the harmonic mean of group coverage and fidelity
    (both must be decent — one cannot buy the other), with the contradiction
    count carried beside it as the danger flag. Grouping follows
    content_fidelity_table: pooled over categories, split by person.
    """
    table = content_fidelity_table(pairs, by=by)
    table = table[table["scope"] != "overall"].rename(columns={"scope": by})
    table["status"] = [rag_status(f1, green, amber) for f1 in table["f1"]]
    return table.reset_index(drop=True)
