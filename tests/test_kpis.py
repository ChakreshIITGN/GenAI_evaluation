"""Tests for src/kpis.py — KPI maths and RAG classification on known frames.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import pandas as pd
import pytest

from src.kpis import (
    content_fidelity_table,
    harmonic_mean,
    rag_matrix,
    rag_status,
    segment_coverage_summary,
)


def make_pairs() -> pd.DataFrame:
    """Four pairs across two categories with hand-checkable values."""
    return pd.DataFrame(
        [
            {"category": "shopping", "coverage": 0.8, "fidelity": 0.6, "contradictions": 0},
            {"category": "shopping", "coverage": 0.6, "fidelity": 1.0, "contradictions": 1},
            {"category": "brand", "coverage": 0.2, "fidelity": 0.4, "contradictions": 2},
            {"category": "brand", "coverage": 0.4, "fidelity": 0.2, "contradictions": 0},
        ]
    )


def test_harmonic_mean_known_values() -> None:
    """2*0.8*0.6/(0.8+0.6) ≈ 0.686; both-zero stays 0 instead of dividing."""
    assert harmonic_mean(0.8, 0.6) == pytest.approx(0.6857, abs=1e-3)
    assert harmonic_mean(0.0, 0.0) == 0.0


def test_rag_status_boundaries_inclusive() -> None:
    """Thresholds are inclusive: exactly green -> green, exactly amber -> amber."""
    assert rag_status(0.70, green=0.70, amber=0.40) == "green"
    assert rag_status(0.40, green=0.70, amber=0.40) == "amber"
    assert rag_status(0.39, green=0.70, amber=0.40) == "red"


def test_content_fidelity_table_means_and_sums() -> None:
    """Overall row: coverage 0.5, fidelity 0.55, contradictions summed to 3."""
    table = content_fidelity_table(make_pairs()).set_index("scope")
    assert table.loc["overall", "coverage"] == pytest.approx(0.5)
    assert table.loc["overall", "fidelity"] == pytest.approx(0.55)
    assert table.loc["overall", "contradictions"] == 3
    assert table.loc["shopping", "coverage"] == pytest.approx(0.7)
    assert table.loc["brand", "n_pairs"] == 2


def test_rag_matrix_statuses() -> None:
    """shopping f1 ≈ 0.75 -> green; brand f1 = 0.3 -> red (0.70/0.40 thresholds)."""
    matrix = rag_matrix(make_pairs(), green=0.70, amber=0.40).set_index("category")
    assert matrix.loc["shopping", "status"] == "green"
    assert matrix.loc["brand", "status"] == "red"
    assert "overall" not in matrix.index


def test_segment_coverage_summary_counts() -> None:
    """Counts and share follow the classification column."""
    topics = pd.DataFrame(
        {"classification": ["covered", "covered", "blind_spot", "phantom"]}
    )
    summary = segment_coverage_summary(topics)
    assert summary["covered"] == 2
    assert summary["blind_spots"] == 1
    assert summary["phantoms"] == 1
    assert summary["share_covered"] == pytest.approx(0.5)