"""Tests for the deterministic parts of src/analyze_rb.py (no LLM, no network).

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import numpy as np

from src.analyze_rb import (
    article_vectors,
    author_profiles,
    cluster_articles,
    knn_edges,
    office_profiles,
    quote_in_text,
    stance_agreement,
    stance_by_year,
)


def test_article_vectors_average_and_renormalise() -> None:
    chunks = [
        {"url": "a", "embedding": [1.0, 0.0]},
        {"url": "a", "embedding": [0.0, 1.0]},
        {"url": "b", "embedding": [0.0, 1.0]},
    ]
    urls, vectors = article_vectors(chunks)
    assert urls == ["a", "b"]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[0], [2**-0.5, 2**-0.5])


def test_cluster_articles_finds_two_obvious_groups() -> None:
    rng = np.random.default_rng(0)
    group_a = np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.01, (5, 3))
    group_b = np.array([0.0, 1.0, 0.0]) + rng.normal(0, 0.01, (5, 3))
    vectors = np.vstack([group_a, group_b])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    labels, k, scores = cluster_articles(vectors, k_min=2, k_max=4, seed=0)
    assert k == 2 and set(scores) == {2, 3, 4}
    assert len(set(labels[:5])) == 1 and labels[0] != labels[5]


def test_knn_edges_are_undirected_unique_and_thresholded() -> None:
    vectors = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    edges = knn_edges(vectors, k=1, min_sim=0.5)
    assert [(i, j) for i, j, _ in edges] == [(0, 1)]  # 2 is too far from both


def test_quote_in_text_ignores_case_quotes_and_spacing() -> None:
    text = "He said: “AI isn’t magic –  it’s maths.” Then left."
    assert quote_in_text("\"AI isn't magic - it's maths.\"", text)
    assert not quote_in_text("AI is magic.", text)
    assert not quote_in_text("", text)


def test_stance_by_year_means_in_year_order() -> None:
    articles = [
        {"year": 2026, "stance": 2},
        {"year": 2018, "stance": -1},
        {"year": 2026, "stance": 1},
    ]
    assert stance_by_year(articles) == [
        {"year": 2018, "n": 1, "mean_stance": -1.0},
        {"year": 2026, "n": 2, "mean_stance": 1.5},
    ]


ARTICLES = [
    {"id": "x", "authors": ["Ann", "Bo"], "year": 2025, "stance": 2, "cluster": 0},
    {"id": "y", "authors": ["Bo"], "year": 2026, "stance": 0, "cluster": 1},
    {"id": "z", "authors": ["Cy"], "year": 2026, "stance": -1, "cluster": 1},
]
PEOPLE = {
    "Ann": {"title": "Partner", "office": "Munich Office", "region": "Central Europe"},
    "Bo": {"title": "Principal", "office": "Munich Office", "region": "Central Europe"},
}


def test_author_profiles_rank_by_count_then_stance() -> None:
    profiles = author_profiles(ARTICLES, PEOPLE)
    assert [p["name"] for p in profiles] == ["Bo", "Ann", "Cy"]
    bo = profiles[0]
    assert (bo["n"], bo["n_positive"], bo["years"], bo["mean_stance"]) == (
        2,
        1,
        [2025, 2026],
        1.0,
    )
    assert (bo["title"], bo["office"]) == ("Principal", "Munich Office")


def test_author_without_profile_is_a_magazine_writer() -> None:
    cy = author_profiles(ARTICLES, PEOPLE)[2]
    assert (cy["title"], cy["office"]) == ("Think:Act writer", "")


def test_office_counts_each_article_once() -> None:
    # Ann and Bo (same office) co-wrote "x": it counts once for Munich.
    assert office_profiles(ARTICLES, PEOPLE) == [
        {
            "office": "Munich Office",
            "region": "Central Europe",
            "n": 2,
            "n_positive": 1,
            "authors": ["Ann", "Bo"],
            "articles": ["x", "y"],
        }
    ]


def test_stance_agreement_counts_direction_and_closeness() -> None:
    result = stance_agreement(
        {"m1": [2, 1, -1, 0], "m2": [1, 1, -1, 1], "m3": [2, 1, 1, 0]}
    )
    assert result["n"] == 4
    assert result["exact"] == 1  # only article 2 (all 1)
    assert result["same_direction"] == 2  # articles 1 and 2
    assert result["within_one"] == 3  # article 3 spans -1..1


def test_stance_agreement_is_perfect_for_identical_readers() -> None:
    result = stance_agreement({"m1": [2, 0, -1], "m2": [2, 0, -1]})
    assert result["alpha_ordinal"] == 1.0
