"""One-off: run ONLY the Layer 2c topic model and inspect the result.

Pools all sentences, fits the (fixed) seeded BERTopic, prints the topic
table and the degeneracy verdict, and writes output/topics.csv plus the
topic-map and sentence-map figures. No LLM calls; local models only.

Run:  uv run python scripts/run_topics.py

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import sys
from pathlib import Path

# Running "python scripts/run_topics.py" puts scripts/ on sys.path, not the
# repo root — add the root so "src" imports like everywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_all import CONFIG, pool_sentences  # noqa: E402
from src.cluster import fit_topics, is_degenerate, plot_sentence_map, plot_topic_map  # noqa: E402
from src.load import load_answers  # noqa: E402


def main() -> None:
    """Fit the topic model once and print everything a human needs to judge it."""
    results_dir = Path(CONFIG["paths"]["results"])
    plots_dir = Path(CONFIG["paths"]["plots"])

    df = load_answers(CONFIG["paths"]["data"])
    sentences, sides = pool_sentences(df)
    print(f"Fitting BERTopic on {len(sentences)} sentences "
          f"({sides.count('human')} human / {sides.count('ai')} ai)…")

    topics = fit_topics(sentences, sides)
    topics.to_csv(results_dir / "topics.csv", index=False)
    plot_topic_map(topics, plots_dir / "topic_map.png")
    plot_sentence_map(sentences, sides, plots_dir / "sentence_map.png")

    print(topics.to_string(index=False))
    n_assigned = int((topics["n_human"] + topics["n_ai"]).sum())
    print(f"\n{len(topics)} topics, {n_assigned}/{len(sentences)} sentences assigned "
          f"(rest are outliers)")
    if is_degenerate(topics):
        print("VERDICT: DEGENERATE — no usable topic structure; KPI 2 not computable.")
    else:
        counts = topics["classification"].value_counts().to_dict()
        print(f"VERDICT: usable — {counts}")


if __name__ == "__main__":
    main()
