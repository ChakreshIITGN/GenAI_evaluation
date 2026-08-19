"""Topic map with BERTopic — Layer 2c, the macro content view (plan §2c).

One BERTopic model is fitted on ALL sentences pooled (human + AI, tagged by
side). Per topic we count each side's sentences and classify:

    covered     — both sides present in comparable share
    blind spot  — human-heavy, AI near-absent (the simulation misses it)
    phantom     — AI-heavy, human near-absent (the simulation invents it)

This targets the population-structure gap the reference paper admits:
individual accuracy does not guarantee the right distribution of themes.
Vocabulary "covered / blind spot / phantom" belongs to THIS layer only.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from bertopic import BERTopic
from umap import UMAP

from src.plots import RB_BLUE, RB_GREEN, RB_GRID, STYLE, _save

_CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())
_BT_CFG: dict = _CONFIG["eval"]["bertopic"]


def classify_topic(n_human: int, n_ai: int, dominance: float) -> str:
    """Classify one topic from its per-side sentence counts (pure function).

    Args:
        n_human: human sentences in the topic.
        n_ai: AI sentences in the topic.
        dominance: fraction of the topic one side must hold to make it a
            blind spot (human side) or phantom (AI side).

    Returns:
        'blind_spot' | 'phantom' | 'covered'.
    """
    total = n_human + n_ai
    if total == 0:
        return "covered"  # degenerate, cannot happen for a fitted topic
    if n_human / total >= dominance:
        return "blind_spot"
    if n_ai / total >= dominance:
        return "phantom"
    return "covered"


def fit_topics(sentences: list[str], sides: list[str]) -> pd.DataFrame:
    """Fit ONE seeded BERTopic on the pooled sentences and summarise per topic.

    Args:
        sentences: all sentences, human and AI pooled.
        sides: 'human' or 'ai' per sentence, aligned by position.

    Returns:
        One row per topic (outliers, topic -1, excluded): topic_id, name
        (top words), n_human, n_ai, human_share, classification.
    """
    # BERTopic's own UMAP defaults, plus a random_state — seeding only,
    # no tuning (plan §2c: let the library handle everything else).
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
        random_state=_BT_CFG["seed"],
    )
    model = BERTopic(min_topic_size=_BT_CFG["min_topic_size"], umap_model=umap_model)
    topics, _ = model.fit_transform(sentences)

    frame = pd.DataFrame({"topic": topics, "side": sides})
    rows = []
    for topic_id, part in frame[frame["topic"] != -1].groupby("topic"):
        n_human = int((part["side"] == "human").sum())
        n_ai = int((part["side"] == "ai").sum())
        top_words = [word for word, _ in model.get_topic(topic_id)][:4]
        rows.append(
            {
                "topic_id": topic_id,
                "name": ", ".join(top_words),
                "n_human": n_human,
                "n_ai": n_ai,
                "human_share": n_human / (n_human + n_ai),
                "classification": classify_topic(n_human, n_ai, _BT_CFG["dominance"]),
            }
        )
    return pd.DataFrame(rows)


def plot_topic_map(topic_frame: pd.DataFrame, path: str | Path) -> None:
    """The key figure: per topic, human vs AI sentence counts side by side.

    Topics sorted by human share, so blind spots gather at the top and
    phantoms at the bottom; the classification is written beside each topic.
    """
    frame = topic_frame.sort_values("human_share", ascending=False).reset_index(drop=True)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(frame) + 1.4))
        positions = range(len(frame))
        bar = 0.38
        ax.barh([p - bar / 2 for p in positions], frame["n_human"], height=bar,
                color=RB_BLUE, label="human")
        ax.barh([p + bar / 2 for p in positions], frame["n_ai"], height=bar,
                color=RB_GREEN, label="AI")
        # Classification tag at the row end — identity never by color alone.
        row_totals = frame["n_human"].combine(frame["n_ai"], max)
        for pos, (total, tag) in enumerate(zip(row_totals, frame["classification"])):
            ax.text(total + 0.4, pos, tag.replace("_", " "), va="center", fontsize=8)
        ax.set_yticks(positions, frame["name"])
        ax.invert_yaxis()
        ax.set_xlabel("sentences in topic")
        ax.grid(axis="x", color=RB_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title("Do humans and simulations talk about the same things?")
        ax.legend(loc="lower right", fontsize=8)
        _save(fig, path)
