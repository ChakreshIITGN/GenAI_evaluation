"""Analyse the RB AI publications: clusters, network, stance, themes, voices.

Deterministic parts (pure functions, tested):
    article_vectors(chunks)          -> urls, one unit vector per article
    cluster_articles(vectors, ...)   -> labels, chosen k, silhouette per k
    knn_edges(vectors, k, min_sim)   -> [(i, j, similarity), ...]
    quote_in_text(quote, text)       -> is the quote really in the article?
    stance_by_year(articles)         -> [{year, n, mean_stance}, ...]
    author_profiles(articles, people) -> [{name, n, title, office, ...}, ...]
    office_profiles(articles, people) -> [{office, region, n, authors, ...}, ...]
    stance_agreement(by_model)       -> how far independent LLMs agree on stance
LLM parts (via src/llm.py, cached under a gitignored folder):
    read_article(article, cache_dir, model) -> ArticleReading
    name_cluster(lines, cache_dir)   -> ClusterName

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import re
from collections import defaultdict
from pathlib import Path

import krippendorff
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.llm import ask, load_prompt
from src.rb_ai_models import ArticleReading, ClusterName

# Typographic variants the model may normalise when copying a quote.
_QUOTE_CHARS = str.maketrans(
    {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
)


def article_vectors(chunks: list[dict]) -> tuple[list[str], np.ndarray]:
    """Average each article's chunk vectors into one unit vector.

    Returns the article urls in first-seen order and a (n_articles, dim) matrix.
    """
    by_url: dict[str, list[list[float]]] = defaultdict(list)
    for chunk in chunks:
        by_url[chunk["url"]].append(chunk["embedding"])
    urls = list(by_url)
    vectors = np.array([np.mean(by_url[u], axis=0) for u in urls])
    # Re-normalise: the mean of unit vectors is shorter than 1.
    return urls, vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def cluster_articles(
    vectors: np.ndarray, k_min: int, k_max: int, seed: int
) -> tuple[np.ndarray, int, dict[int, float]]:
    """K-means for each k in [k_min, k_max]; keep the k with the best silhouette.

    Vectors are unit length, so Euclidean k-means groups by cosine similarity.
    """
    labels_by_k: dict[int, np.ndarray] = {}
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels_by_k[k] = model.fit_predict(vectors)
        scores[k] = float(silhouette_score(vectors, labels_by_k[k], metric="cosine"))
    best_k = max(scores, key=scores.get)
    return labels_by_k[best_k], best_k, scores


def knn_edges(
    vectors: np.ndarray, k: int, min_sim: float
) -> list[tuple[int, int, float]]:
    """Link each article to its k most similar others, if at least min_sim.

    Undirected and de-duplicated: each pair appears once with i < j.
    """
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1.0)  # never link an article to itself
    edges: dict[tuple[int, int], float] = {}
    for i in range(len(vectors)):
        for j in np.argsort(-sims[i])[:k]:
            if sims[i, j] >= min_sim:
                edges[(min(i, j), max(i, j))] = float(sims[i, j])
    return [(i, j, s) for (i, j), s in sorted(edges.items())]


def _normalise(text: str) -> str:
    """Lowercase, unify quote marks and dashes, collapse whitespace."""
    return re.sub(r"\s+", " ", text.translate(_QUOTE_CHARS).lower()).strip()


def quote_in_text(quote: str, text: str) -> bool:
    """True if the quote appears in the text (ignoring case, quote style, spacing)."""
    needle = _normalise(quote).strip("\"' ")
    return bool(needle) and needle in _normalise(text)


def stance_by_year(articles: list[dict]) -> list[dict]:
    """Number of articles and mean stance per publication year, oldest first."""
    by_year: dict[int, list[int]] = defaultdict(list)
    for article in articles:
        by_year[article["year"]].append(article["stance"])
    return [
        {"year": y, "n": len(s), "mean_stance": round(float(np.mean(s)), 2)}
        for y, s in sorted(by_year.items())
    ]


def _direction(stance: int) -> int:
    """-1 cautious, 0 balanced, +1 positive: the plain-language groups."""
    return int(stance > 0) - int(stance < 0)


def author_profiles(articles: list[dict], people: dict[str, dict]) -> list[dict]:
    """Per author: articles written, years, stance, and title/office from profile.

    Authors without a profile page (Think:Act magazine writers) get
    "Think:Act writer" as title and an empty office.
    Sorted by number of articles, then by mean stance (most positive first).
    """
    by_author: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        for name in article["authors"]:
            by_author[name].append(article)
    profiles = []
    for name, written in by_author.items():
        person = people.get(name, {})
        profiles.append(
            {
                "name": name,
                "title": person.get("title") or "Think:Act writer",
                "office": person.get("office", ""),
                "region": person.get("region", ""),
                "n": len(written),
                "n_positive": sum(_direction(a["stance"]) > 0 for a in written),
                "years": sorted({a["year"] for a in written}),
                "mean_stance": round(float(np.mean([a["stance"] for a in written])), 2),
                "clusters": sorted({a["cluster"] for a in written}),
                "articles": [a["id"] for a in written],
            }
        )
    return sorted(profiles, key=lambda p: (-p["n"], -p["mean_stance"], p["name"]))


def office_profiles(articles: list[dict], people: dict[str, dict]) -> list[dict]:
    """Per office: distinct articles with at least one author from it, and authors.

    An article co-written by two people from one office counts once for it.
    """
    by_office: dict[str, dict[str, dict]] = defaultdict(dict)
    authors_of: dict[str, set[str]] = defaultdict(set)
    region_of: dict[str, str] = {}
    for article in articles:
        for name in article["authors"]:
            office = people.get(name, {}).get("office", "")
            if not office:
                continue
            by_office[office][article["id"]] = article
            authors_of[office].add(name)
            region_of[office] = people[name].get("region", "")
    rows = [
        {
            "office": office,
            "region": region_of[office],
            "n": len(arts),
            "n_positive": sum(_direction(a["stance"]) > 0 for a in arts.values()),
            "authors": sorted(authors_of[office]),
            "articles": sorted(arts),
        }
        for office, arts in by_office.items()
    ]
    return sorted(rows, key=lambda r: (-r["n"], r["office"]))


def stance_agreement(by_model: dict[str, list[int]]) -> dict:
    """How far independent LLM readers agree on each article's stance.

    Args:
        by_model: model name -> stance per article, all lists aligned.

    Returns:
        n articles; Krippendorff's alpha (ordinal, 1 = perfect, 0 = chance);
        counts of articles where every model gives the same direction
        (cautious / balanced / positive), scores within 1 point, and the
        exact same score.
    """
    matrix = np.array(list(by_model.values()))  # models x articles
    directions = np.sign(matrix)  # same groups as _direction, for a matrix
    return {
        "models": list(by_model),
        "n": int(matrix.shape[1]),
        "alpha_ordinal": round(
            float(
                krippendorff.alpha(
                    reliability_data=matrix.astype(float),
                    level_of_measurement="ordinal",
                )
            ),
            3,
        ),
        "same_direction": int((directions == directions[0]).all(axis=0).sum()),
        "within_one": int((matrix.max(axis=0) - matrix.min(axis=0) <= 1).sum()),
        "exact": int((matrix == matrix[0]).all(axis=0).sum()),
    }


def read_article(
    article: dict, cache_dir: Path, model: str | None = None
) -> ArticleReading:
    """Ask an LLM (default: config llm.model) for stance, evidence, themes, voices."""
    prompt = load_prompt("rb_ai/read_article").format(
        title=article["title"],
        date=article["date"],
        label=article["label"],
        authors=", ".join(article["authors"]) or "not named",
        content=article["content"],
    )
    return ask(prompt, ArticleReading, model=model, cache_dir=cache_dir)


def name_cluster(lines: list[str], cache_dir: Path) -> ClusterName:
    """Ask the LLM to name a cluster from its members' titles and themes."""
    prompt = load_prompt("rb_ai/name_cluster").format(articles="\n".join(lines))
    return ask(prompt, ClusterName, cache_dir=cache_dir)
