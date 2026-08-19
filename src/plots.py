"""Publication-grade figures for the EDA outputs, in the Roland Berger palette.

Three plain matplotlib functions, one per figure; each takes a DataFrame and
a save path and returns nothing. Colors come from design/tokens.md (blues,
greens, black — no red); the categorical order is fixed and colorblind-safe.
Every bar/dot carries its value as a visible label, so no reading depends on
color alone. run_plots.py is the entrypoint.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # headless backend: files only, no display window

# Palette — see design/tokens.md for provenance and validation.
RB_BLUE = "#156C9C"   # human side / first series
RB_GREEN = "#72B656"  # AI side / second series
RB_TURQUOISE = "#00AAC9"   # third series
RB_DARK_GREEN = "#2B8D28"  # fourth series
RB_BLACK = "#000000"
RB_GRAY = "#8D9399"       # reference lines
RB_GRID = "#DEE0E3"       # grid lines

# Fixed categorical order for multi-model figures — never cycled.
SERIES = [RB_BLUE, RB_GREEN, RB_TURQUOISE, RB_DARK_GREEN]


def model_colors(models: list[str]) -> dict[str, str]:
    """Fixed color per judge model, identical in every figure.

    Sorting makes the assignment independent of row order in the CSVs, so
    a judge keeps its color across Q0, Q1 and Q2 figures.
    """
    return dict(zip(sorted(models), SERIES))


def _grouped_percent_bars(
    ax: plt.Axes, group_labels: list[str], series: dict[str, list[float]]
) -> None:
    """Horizontal grouped 0–100% bars: one group per label, one bar per model.

    `series` maps model name -> one value per group label. Colors come from
    model_colors(); every bar carries its value; a legend names the models.
    """
    colors = model_colors(list(series))
    n_series = len(series)
    group_height = 0.8 / n_series
    for i, (model, values) in enumerate(sorted(series.items())):
        positions = [j + (i - (n_series - 1) / 2) * group_height
                     for j in range(len(group_labels))]
        ax.barh(positions, values, height=group_height * 0.85,
                color=colors[model], label=model)
        for pos, value in zip(positions, values):
            ax.text(value + 1.5, pos, f"{value:.0f}%", va="center",
                    color=RB_BLACK, fontsize=8)
    ax.set_yticks(range(len(group_labels)), group_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 25))
    ax.grid(axis="x", color=RB_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)

STYLE: dict = {
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": RB_BLACK,
    "axes.labelcolor": RB_BLACK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": RB_BLACK,
    "ytick.color": RB_BLACK,
    "legend.frameon": False,
}


def _save(fig: plt.Figure, path: str | Path) -> None:
    """Write the figure as PNG (300 dpi, tight) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_q0_accuracy(metrics: pd.DataFrame, path: str | Path) -> None:
    """Q0 figure: zero-shot human-or-AI accuracy per judge model, per side.

    One bar group per scope, one color per model (legend gives the model
    names). The dashed gray line marks 50% = indistinguishable (coin flip).
    """
    q0 = metrics[metrics["metric"] == "q0_accuracy"]
    order = [("overall", "All answers"), ("side:human", "Human answers"),
             ("side:ai", "AI answers")]
    series = {
        model: [part.set_index("scope").loc[scope, "value"] * 100 for scope, _ in order]
        for model, part in q0.groupby("model")
    }

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 0.9 * len(order) + 1.2))
        _grouped_percent_bars(ax, [label for _, label in order], series)
        ax.axvline(50, color=RB_GRAY, linestyle="--", linewidth=1)
        ax.set_xlabel("correctly labelled (%) — dashed line: coin flip")
        ax.set_title("Can the judges tell human from AI? (zero-shot)")
        _save(fig, path)


def plot_share_yes(metrics: pd.DataFrame, flag: str, title: str, path: str | Path) -> None:
    """Q1/Q2 figure: share of 'yes' per scope, one bar per judge model.

    Every judge scored the same held-out items, so the grouped bars are a
    direct head-to-head: scope groups (overall, each person, each category)
    on the y-axis, one consistently colored bar per judge, legend with the
    model names. The pooled row is excluded — the judges ARE the story.

    Args:
        metrics: tidy frame from output/eda_metrics.csv.
        flag: 'captures_personality' or 'captures_content'.
        title: figure title (action-style, states the question).
        path: output PNG path.
    """
    rows = metrics[(metrics["metric"] == f"{flag}_share_yes") & (metrics["model"] != "pooled")]
    # Scope order as written by compute_metrics: overall, persons, categories.
    scopes = list(dict.fromkeys(rows["scope"]))
    labels = [scope.split(":", 1)[-1] for scope in scopes]
    series = {
        model: [part.set_index("scope").loc[scope, "value"] * 100 for scope in scopes]
        for model, part in rows.groupby("model")
    }

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 0.75 * len(labels) + 1.2))
        _grouped_percent_bars(ax, labels, series)
        ax.set_xlabel("judged 'yes' (%)")
        ax.set_title(title)
        _save(fig, path)


def plot_style_features(features: pd.DataFrame, path: str | Path) -> None:
    """Style figure: mean of each stylometric feature, human vs AI, per panel.

    Small multiples (one panel per feature, own scale) because the features
    live on very different units — a shared axis would hide everything.
    """
    feature_names = [c for c in features.columns if c not in ("person_id", "question_id", "side")]
    means = features.groupby("side")[feature_names].mean()

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(10, 4.6))
        for ax, name in zip(axes.flat, feature_names):
            values = [means.loc["human", name], means.loc["ai", name]]
            ax.bar([0, 1], values, color=[RB_BLUE, RB_GREEN], width=0.6)
            for x, value in zip([0, 1], values):
                ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
            ax.set_title(name.replace("_", " "), fontsize=9, fontweight="normal")
            ax.set_xticks([0, 1], ["human", "AI"])
            # Always include 0; give labels headroom even for tiny values.
            ax.set_ylim(min(0.0, min(values) * 1.3), max(0.001, max(values) * 1.25))
            ax.grid(axis="y", color=RB_GRID, linewidth=0.6)
            ax.set_axisbelow(True)
        fig.suptitle("How the answers sound: style features, human vs AI mean (deterministic, no LLM)",
                     x=0.02, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(fig, path)
