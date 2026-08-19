"""Tests for src/cluster.py — topic classification logic and the figure.

fit_topics itself is NOT tested here (it downloads and runs the embedding
model); the deterministic pieces around it are.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import pandas as pd

from src.cluster import classify_topic, plot_topic_map


def test_blind_spot_when_human_dominates() -> None:
    """9 human vs 1 AI at dominance 0.8 -> the simulation misses this topic."""
    assert classify_topic(9, 1, dominance=0.8) == "blind_spot"


def test_phantom_when_ai_dominates() -> None:
    """1 human vs 9 AI -> the simulation invents this topic."""
    assert classify_topic(1, 9, dominance=0.8) == "phantom"


def test_covered_when_balanced() -> None:
    """Both sides present in comparable share -> covered."""
    assert classify_topic(5, 5, dominance=0.8) == "covered"
    assert classify_topic(6, 4, dominance=0.8) == "covered"


def test_dominance_boundary_is_inclusive() -> None:
    """Exactly at the threshold (8 of 10 = 0.8) counts as dominated."""
    assert classify_topic(8, 2, dominance=0.8) == "blind_spot"
    assert classify_topic(2, 8, dominance=0.8) == "phantom"


def test_topic_map_writes_png(tmp_path: Path) -> None:
    """The per-topic figure lands on disk as a valid PNG."""
    frame = pd.DataFrame(
        [
            {"topic_id": 0, "name": "price, cheap", "n_human": 9, "n_ai": 1,
             "human_share": 0.9, "classification": "blind_spot"},
            {"topic_id": 1, "name": "smell, scent", "n_human": 5, "n_ai": 5,
             "human_share": 0.5, "classification": "covered"},
            {"topic_id": 2, "name": "sustainability, eco", "n_human": 1, "n_ai": 9,
             "human_share": 0.1, "classification": "phantom"},
        ]
    )
    out = tmp_path / "topics.png"
    plot_topic_map(frame, out)
    assert out.exists()
    assert out.read_bytes()[:4] == b"\x89PNG"
