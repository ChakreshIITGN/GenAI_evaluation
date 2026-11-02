"""analyze_rb_ai.py — clusters, network, stance and voices of the RB AI articles.

Reads data/rb_ai_publications.json and data/rb_ai_chunks.jsonl, then:
  1. one vector per article (mean of its chunk vectors);
  2. k-means topic clusters (k chosen by silhouette in config range);
  3. a similarity network (each article linked to its nearest neighbours);
  4. one LLM reading per article: stance on AI (-2..+2) with a verbatim
     evidence quote, themes, main message and featured voices;
  5. one LLM name per cluster.
  6. the same stance question to the check models (config
     analysis.check_models) and their agreement (Krippendorff's alpha);
  7. author and office rollups from the scraped profile pages.
Every quote is checked against the article text (`*_verified` flags).
Writes data/rb_ai_analysis.json (feeds the interactive page) and
data/rb_ai_stance_audit.csv (for a human spot check); both gitignored
because they hold article quotes. See docs/RB_AI_REPRODUCE.md.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import csv
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.analyze_rb import (
    article_vectors,
    author_profiles,
    cluster_articles,
    knn_edges,
    name_cluster,
    office_profiles,
    quote_in_text,
    read_article,
    stance_agreement,
    stance_by_year,
)
from src.chunk_rb import iso_date

CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())


def main() -> None:
    """Run the analysis and write the page's data file."""
    cfg = CONFIG["analysis"]
    cache_dir = Path(cfg["cache_dir"])
    scrape_file = Path(CONFIG["scrape"]["output"])
    scrape_data = json.loads(scrape_file.read_text())
    scraped, people = scrape_data["articles"], scrape_data["people"]
    by_url = {a["url"]: a for a in scraped}
    chunks_text = Path(CONFIG["embed"]["output"]).read_text()
    chunks = [json.loads(line) for line in chunks_text.splitlines()]

    urls, vectors = article_vectors(chunks)
    labels, k, silhouettes = cluster_articles(
        vectors, cfg["k_min"], cfg["k_max"], cfg["seed"]
    )
    edges = knn_edges(vectors, cfg["neighbours"], cfg["min_similarity"])
    print(f"{len(urls)} articles, {k} clusters, {len(edges)} links.")

    print("Reading articles with the LLM (cached after the first run)...")
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        readings = list(pool.map(lambda u: read_article(by_url[u], cache_dir), urls))

    # Independent second opinions on stance, from other model families.
    primary = CONFIG["llm"]["model"]
    stances = {primary: [r.stance for r in readings]}
    for model in cfg["check_models"]:
        print(f"Checking stance with {model}...")
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
            stances[model] = [
                r.stance
                for r in pool.map(
                    lambda u, m=model: read_article(by_url[u], cache_dir, m), urls
                )
            ]
    agreement = stance_agreement(stances)

    articles: list[dict] = []
    for i, (url, label, reading) in enumerate(zip(urls, labels, readings)):
        source = by_url[url]
        date_iso = iso_date(source["date"])
        articles.append(
            {
                "id": url.rsplit("/", 1)[-1].removesuffix(".html"),
                "title": source["title"],
                "url": url,
                "date": date_iso,
                "year": int(date_iso[:4]),
                "label": source["label"].title(),
                "authors": source["authors"],
                "cluster": int(label),
                "stance": reading.stance,
                "stance_by_model": {m: s[i] for m, s in stances.items()},
                "stance_reason": reading.stance_reason,
                "evidence_quote": reading.evidence_quote,
                "quote_verified": quote_in_text(
                    reading.evidence_quote, source["content"]
                ),
                "themes": [t.lower() for t in reading.themes],
                "main_message": reading.main_message,
                "voices": [
                    {
                        **v.model_dump(),
                        "quote_verified": quote_in_text(v.quote, source["content"]),
                    }
                    for v in reading.voices
                    # Authors are counted separately, not as featured voices.
                    if v.name not in source["authors"]
                ],
            }
        )

    clusters: list[dict] = []
    for c in range(k):
        members = [a for a in articles if a["cluster"] == c]
        named = name_cluster(
            [f"- {a['title']} ({', '.join(a['themes'])})" for a in members], cache_dir
        )
        clusters.append(
            {
                "id": c,
                "name": named.name,
                "description": named.description,
                "size": len(members),
                "mean_stance": round(
                    sum(a["stance"] for a in members) / len(members), 2
                ),
                "top_themes": [
                    t
                    for t, _ in Counter(
                        t for a in members for t in a["themes"]
                    ).most_common(5)
                ],
                "by_year": dict(sorted(Counter(a["year"] for a in members).items())),
            }
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": CONFIG["scrape"]["listing_url"],
        "method": {
            "embedding_model": CONFIG["embed"]["model"],
            "llm": CONFIG["llm"]["model"],
            "k": k,
            "silhouette_by_k": {str(kk): round(s, 3) for kk, s in silhouettes.items()},
            "neighbours": cfg["neighbours"],
            "min_similarity": cfg["min_similarity"],
        },
        "clusters": clusters,
        "articles": articles,
        "edges": [
            {
                "source": articles[i]["id"],
                "target": articles[j]["id"],
                "sim": round(s, 3),
            }
            for i, j, s in edges
        ],
        "stance_by_year": stance_by_year(articles),
        "stance_agreement": agreement,
        "authors": author_profiles(articles, people),
        "offices": office_profiles(articles, people),
        "provenance": provenance(scrape_file),
    }
    out = Path(cfg["output"])
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    write_audit_sheet(articles, Path(cfg["audit_sheet"]))
    print(
        f"Stance agreement across {len(stances)} models: alpha "
        f"{agreement['alpha_ordinal']}, same direction "
        f"{agreement['same_direction']}/{agreement['n']}."
    )
    unverified = sum(not a["quote_verified"] for a in articles)
    print(
        f"Wrote {out}. Evidence quotes not found verbatim: {unverified}/{len(articles)}."
    )


def _sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file, so a rerun can prove it used the same inputs."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(scrape_file: Path) -> dict:
    """What produced this result: input hashes, settings and code version."""
    # check=False: outside a git checkout the commit is simply "unknown".
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "git_commit": commit or "unknown",
        "scraped_at": json.loads(scrape_file.read_text())["scraped_at"],
        "sha256": {
            "publications": _sha256(scrape_file),
            "chunks": _sha256(CONFIG["embed"]["output"]),
        },
        "settings": {
            "scrape": CONFIG["scrape"],
            "embed": CONFIG["embed"],
            "analysis": CONFIG["analysis"],
            "llm": {k: CONFIG["llm"][k] for k in ("model", "temperature", "seed")},
        },
    }


def write_audit_sheet(articles: list[dict], path: Path) -> None:
    """One row per article for a human to check the LLM stance by hand.

    The last two columns are left blank for the reviewer to fill in.
    """
    models = list(articles[0]["stance_by_model"])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "title",
                "date",
                "label",
                "url",
                *[f"stance_{m}" for m in models],
                "evidence_quote",
                "quote_verified",
                "human_stance",
                "human_note",
            ]
        )
        for a in articles:
            writer.writerow(
                [
                    a["id"],
                    a["title"],
                    a["date"],
                    a["label"],
                    a["url"],
                    *[a["stance_by_model"][m] for m in models],
                    a["evidence_quote"],
                    a["quote_verified"],
                    "",
                    "",
                ]
            )


if __name__ == "__main__":
    main()
