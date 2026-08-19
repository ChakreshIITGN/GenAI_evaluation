"""Tests for src/plots.py — every figure function writes a non-empty PNG.

Synthetic frames mirror the real CSV shapes; no LLM output is needed.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import pandas as pd

from src.plots import plot_q0_accuracy, plot_share_yes, plot_style_features
from src.stylometric import Stylometric


def make_metrics() -> pd.DataFrame:
    """Tidy metrics frame shaped like output/eda_metrics.csv (two judges)."""
    rows = []
    for model in ("model-a", "model-b"):
        rows += [
            {"metric": "q0_accuracy", "scope": "overall", "model": model, "value": 0.55, "n": 60},
            {"metric": "q0_accuracy", "scope": "side:human", "model": model, "value": 0.40, "n": 30},
            {"metric": "q0_accuracy", "scope": "side:ai", "model": model, "value": 0.70, "n": 30},
        ]
    for flag in ("captures_personality", "captures_content"):
        rows.append({"metric": f"{flag}_share_yes", "scope": "overall", "model": "pooled",
                     "value": 0.5, "n": 24})
        for model in ("model-a", "model-b"):
            rows += [
                {"metric": f"{flag}_share_yes", "scope": "overall", "model": model, "value": 0.5, "n": 12},
                {"metric": f"{flag}_share_yes", "scope": "person_id:p1", "model": model, "value": 0.75, "n": 4},
                {"metric": f"{flag}_share_yes", "scope": "category:shopping", "model": model, "value": 1.0, "n": 6},
            ]
    return pd.DataFrame(rows)


def make_features() -> pd.DataFrame:
    """Feature frame shaped like output/measured_features.csv (two rows)."""
    rows = [
        {"person_id": "p1", "question_id": 1, "side": side}
        | Stylometric(text).all_features()
        for side, text in (("human", "I love it, it's great."), ("ai", "It is acceptable."))
    ]
    return pd.DataFrame(rows)


def assert_png(path: Path) -> None:
    """The file exists and starts with the PNG magic bytes."""
    assert path.exists()
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_q0_accuracy_writes_png(tmp_path: Path) -> None:
    """Q0 accuracy figure is written as a valid PNG."""
    out = tmp_path / "q0.png"
    plot_q0_accuracy(make_metrics(), out)
    assert_png(out)


def test_share_yes_writes_png(tmp_path: Path) -> None:
    """Share-yes figure works for both Q1 and Q2 metrics."""
    for flag in ("captures_personality", "captures_content"):
        out = tmp_path / f"{flag}.png"
        plot_share_yes(make_metrics(), flag, "title", out)
        assert_png(out)


def test_style_features_writes_png(tmp_path: Path) -> None:
    """Style small-multiples figure is written as a valid PNG."""
    out = tmp_path / "style.png"
    plot_style_features(make_features(), out)
    assert_png(out)
