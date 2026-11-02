"""scrape_rb_ai.py — scrape every article on the RB AI publications listing.

Reads the listing page once, then visits each article (pausing between
requests, per config scrape.delay_s) and writes one JSON file:
    {source, scraped_at, count, failed,
     articles: [{title, url, label, date, authors, author_urls, summary,
                 content, error}, ...],
     people: {author name: {url, title, office, region, expertise}}}

A failed article keeps its listing title and url, with the reason in `error`,
so one bad page never stops the run. Output lands under data/ (gitignored):
the article text is Roland Berger copyright and stays out of the public repo.
Lines repeated word for word across several articles (sign-up footers) and
call-to-action sentences ("register now", "download the full study") are
removed after all pages are read.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
import yaml

from src.scrape_rb import (
    drop_boilerplate,
    drop_calls_to_action,
    fetch,
    parse_article,
    parse_listing,
    parse_person,
)

CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())


def main() -> None:
    """Scrape the listing and every article, then write the JSON file."""
    cfg = CONFIG["scrape"]
    session = requests.Session()
    session.headers["User-Agent"] = cfg["user_agent"]

    tiles = parse_listing(
        fetch(session, cfg["listing_url"], cfg["timeout_s"]), cfg["article_prefix"]
    )
    print(f"Found {len(tiles)} articles on the listing page.")

    articles: list[dict] = []
    for i, tile in enumerate(tiles, start=1):
        record = {
            **tile,
            "date": "",
            "authors": [],
            "author_urls": {},
            "summary": "",
            "content": "",
            "error": "",
        }
        try:
            parsed = parse_article(fetch(session, tile["url"], cfg["timeout_s"]))
            # Keep the listing title if the page has none.
            record.update({k: v for k, v in parsed.items() if v or k != "title"})
        except requests.RequestException as exc:
            record["error"] = str(exc)
        status = "FAILED" if record["error"] else f"{len(record['content'])} chars"
        print(f"[{i}/{len(tiles)}] {record['title']} — {status}")
        articles.append(record)
        time.sleep(cfg["delay_s"])

    # Site templates (sign-up lines etc.) repeat across articles; drop them.
    cleaned = drop_boilerplate(
        [a["content"] for a in articles], cfg["boilerplate_min_articles"]
    )
    for article, content in zip(articles, cleaned):
        article["content"] = drop_calls_to_action(
            content, cfg["call_to_action_patterns"]
        )

    # One visit per author profile (title, office), shared across articles.
    profile_urls = {
        name: url for a in articles for name, url in a["author_urls"].items()
    }
    people: dict[str, dict] = {}
    for name, url in sorted(profile_urls.items()):
        try:
            people[name] = {
                "url": url,
                **parse_person(fetch(session, url, cfg["timeout_s"])),
            }
        except requests.RequestException as exc:
            # Profiles of people who left the firm are often removed.
            people[name] = {"url": url, "error": str(exc)}
        time.sleep(cfg["delay_s"])
    print(f"Read {len(people)} author profiles.")

    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": cfg["listing_url"],
        "scraped_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "count": len(articles),
        "failed": sum(1 for a in articles if a["error"]),
        "articles": articles,
        "people": people,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {out} ({payload['count']} articles, {payload['failed']} failed).")


if __name__ == "__main__":
    main()
