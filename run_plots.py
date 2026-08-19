"""run_plots.py — build all EDA figures from the CSVs in output/.

Separate from run_eda.py so figures can be regenerated any time without a
single LLM call. Reads output/eda_metrics.csv and output/measured_features.csv,
writes PNGs to output/plots/ (config paths.plots).

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import pandas as pd
import yaml

from src.plots import plot_q0_accuracy, plot_style_features, plot_yes_counts

CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())


def main() -> None:
    """Read the EDA result CSVs and write every figure to output/plots/."""
    results_dir = Path(CONFIG["paths"]["results"])
    plots_dir = Path(CONFIG["paths"]["plots"])

    metrics = pd.read_csv(results_dir / "eda_metrics.csv")
    features = pd.read_csv(results_dir / "measured_features.csv")

    plot_q0_accuracy(metrics, plots_dir / "eda_q0_accuracy.png")
    plot_yes_counts(
        metrics, "captures_personality",
        "Does the AI capture the person? (yes-verdicts per person)",
        plots_dir / "eda_q1_personality.png",
    )
    plot_yes_counts(
        metrics, "captures_content",
        "Does the AI capture the content? (yes-verdicts per person)",
        plots_dir / "eda_q2_content.png",
    )
    plot_style_features(features, plots_dir / "eda_style_features.png")
    print(f"4 figures written to {plots_dir}/")


if __name__ == "__main__":
    main()
