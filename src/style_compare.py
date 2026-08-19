"""Style comparison — Layer 1b analysis and the dumbbell figure (plan §1b).

Takes the deterministic per-answer features from measured_features.csv
(produced by src/stylometric.py, which only measures) and compares the two
sides: paired difference (AI − human) per feature with Cohen's d, and the
seven-row dumbbell chart (human dot vs AI dot).

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.plots import RB_BLACK, RB_BLUE, RB_GRAY, RB_GREEN, RB_GRID, STYLE, _save

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


def plot_dumbbell(features: pd.DataFrame, path: str | Path) -> None:
    """The style dumbbell (plan §1b): one row per feature, human vs AI dot.

    Features live on different units, so each row is drawn in z-scores of
    that feature's pooled distribution (all 60 answers): 0 = pooled mean.
    Raw means are annotated beside the dots so no information is hidden.
    """
    names = feature_names(features)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(names) + 1.2))
        for row_pos, name in enumerate(names):
            pooled_mean = features[name].mean()
            pooled_sd = features[name].std(ddof=1)
            side_means = features.groupby("side")[name].mean()
            # Degenerate feature (no spread): both dots sit at 0 by definition.
            z = {
                side: 0.0 if pooled_sd == 0 else (side_means[side] - pooled_mean) / pooled_sd
                for side in ("human", "ai")
            }
            ax.plot([z["human"], z["ai"]], [row_pos, row_pos],
                    color=RB_GRAY, linewidth=1.5, zorder=1)
            ax.scatter(z["human"], row_pos, s=70, color=RB_BLUE, zorder=2,
                       label="human" if row_pos == 0 else None)
            ax.scatter(z["ai"], row_pos, s=70, color=RB_GREEN, zorder=2,
                       label="AI" if row_pos == 0 else None)
            ax.annotate(f"{side_means['human']:.2f}", (z["human"], row_pos),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color=RB_BLUE)
            ax.annotate(f"{side_means['ai']:.2f}", (z["ai"], row_pos),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color=RB_GREEN)
        ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
        ax.invert_yaxis()
        ax.axvline(0, color=RB_GRAY, linestyle="--", linewidth=1)
        ax.grid(axis="x", color=RB_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("z-score of the pooled distribution (labels: raw means)")
        ax.set_title("Style dumbbell: where the simulations drift from the humans")
        ax.legend(loc="lower right", fontsize=8, labelcolor=RB_BLACK)
        _save(fig, path)
