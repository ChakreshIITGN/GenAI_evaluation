"""Style comparison — Layer 1b analysis (plan §1b).

Takes the deterministic per-answer features from measured_features.csv
(produced by src/stylometric.py, which only measures) and compares the two
sides: paired difference (AI − human) per feature with Cohen's d.
The visual story lives in the EDA per-feature bar panels (author decision:
one style figure, raw units; the z-scored dumbbell was removed as a
duplicate representation).

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import pandas as pd

# Identity columns in measured_features.csv; everything else is a feature.
_ID_COLUMNS = ("person_id", "question_id", "side")


def feature_names(features: pd.DataFrame) -> list[str]:
    """The feature columns of a measured_features frame, in file order."""
    return [c for c in features.columns if c not in _ID_COLUMNS]


def cohens_d(diffs: pd.Series) -> float:
    """Paired Cohen's d: mean(diff) / sd(diff), sample sd (ddof=1).

    Returns 0.0 when the differences have no spread (sd == 0) — by
    convention here, a constant zero difference is 'no effect'.
    """
    sd = diffs.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return 0.0
    return float(diffs.mean() / sd)


def paired_differences(features: pd.DataFrame) -> pd.DataFrame:
    """Paired (AI − human) comparison per feature.

    Args:
        features: measured_features frame (person_id, question_id, side, 7+ features).

    Returns:
        One row per feature: human_mean, ai_mean, mean_diff (AI − human),
        cohens_d, n_pairs. Effect sizes, not p-values (plan §1: n=30).
    """
    names = feature_names(features)
    wide = features.pivot_table(
        index=["person_id", "question_id"], columns="side", values=names
    )
    rows = []
    for name in names:
        diffs = wide[(name, "ai")] - wide[(name, "human")]
        rows.append(
            {
                "feature": name,
                "human_mean": wide[(name, "human")].mean(),
                "ai_mean": wide[(name, "ai")].mean(),
                "mean_diff": diffs.mean(),
                "cohens_d": cohens_d(diffs),
                "n_pairs": len(diffs),
            }
        )
    return pd.DataFrame(rows)
