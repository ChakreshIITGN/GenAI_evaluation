"""Tests for src/style_compare.py and src/embed.py — maths and figure output.

The embedding model is mocked; style maths run on small hand-built frames
with hand-computed expectations.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import numpy as np
import pandas as pd
import pytest

from src import embed
from src.style_compare import cohens_d, paired_differences


def make_features() -> pd.DataFrame:
    """Two pairs, two features, hand-checkable values."""
    return pd.DataFrame(
        [
            {"person_id": "p1", "question_id": 1, "side": "human", "length": 10, "hedges": 4.0},
            {"person_id": "p1", "question_id": 1, "side": "ai", "length": 20, "hedges": 1.0},
            {"person_id": "p2", "question_id": 1, "side": "human", "length": 30, "hedges": 6.0},
            {"person_id": "p2", "question_id": 1, "side": "ai", "length": 36, "hedges": 2.0},
        ]
    )


def test_paired_differences_means() -> None:
    """AI − human diffs: length +10/+6 -> mean 8; hedges −3/−4 -> mean −3.5."""
    result = paired_differences(make_features()).set_index("feature")
    assert result.loc["length", "mean_diff"] == pytest.approx(8.0)
    assert result.loc["hedges", "mean_diff"] == pytest.approx(-3.5)
    assert result.loc["length", "human_mean"] == pytest.approx(20.0)
    assert result.loc["length", "n_pairs"] == 2


def test_cohens_d_known_value() -> None:
    """diffs [2, 0, 4, 2]: mean 2, sd 1.633 -> d ≈ 1.2247."""
    assert cohens_d(pd.Series([2.0, 0.0, 4.0, 2.0])) == pytest.approx(1.2247, abs=1e-3)


def test_cohens_d_zero_spread_is_zero() -> None:
    """Constant differences (sd = 0) return 0.0, never a division error."""
    assert cohens_d(pd.Series([1.0, 1.0, 1.0])) == 0.0


class FakeEmbedder:
    """Stands in for SentenceTransformer with fixed unit vectors."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        return np.array([self.vectors[t] for t in texts])


def test_pair_similarity_cosine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical vectors -> 1.0; orthogonal -> 0.0."""
    fake = FakeEmbedder({"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]})
    monkeypatch.setattr(embed, "_MODEL", fake)
    sims = embed.pair_similarity(["a", "a"], ["b", "c"])
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == pytest.approx(0.0)


def test_pair_similarity_length_mismatch_raises() -> None:
    """Misaligned lists are rejected before any encoding happens."""
    with pytest.raises(ValueError, match="same length"):
        embed.pair_similarity(["a"], [])
