"""Scrape the Roland Berger AI publications listing into plain records.

Two pure parsers (HTML string in, dicts out) plus one small fetch helper:
    parse_listing(html, article_prefix) -> [{title, url, label}, ...]
    parse_article(html)                 -> {title, date, authors, author_urls,
                                            summary, content}
    parse_person(html)                  -> {title, office, region, expertise}
    drop_boilerplate(contents, n)       -> contents without site templates
    drop_calls_to_action(content, pats) -> content without sign-up sentences
    fetch(session, url, timeout_s)      -> HTML string

The listing page holds every article tile in its HTML; the "Load More"
button only reveals tiles already on the page, so one request is enough.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import re
from collections import Counter

import requests
from bs4 import BeautifulSoup, Tag

from src.chunk_rb import SENTENCE_END

# Article body blocks we keep, in reading order. Everything else on the page
# (menus, related tiles, footer) is left out.
BODY_BLOCKS = (
    "h2.c-text-subheadline, h3.c-text-subheadline, p.c-text-paragraph, "
    "ul.c-text-list > li, ol.c-text-list > li, blockquote"
)
TITLE_SUFFIX = " | Roland Berger"

# Magazine pieces carry no author line; their body opens with a byline such
# as "by Steffan Heuer Artworks by Carsten Gueth". Keep only the writer.
BYLINE = re.compile(
    r"^(?:article\s+)?by\s+(.+?)"
    r"(?:\s+(?:illustrations?|artworks?|photos?|this piece)\b.*)?$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Collapse whitespace and remove the space links leave before punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?)])", r"\1", text)


def parse_listing(html: str, article_prefix: str) -> list[dict]:
    """Return the article tiles on the listing page, de-duplicated, in page order.

    Only tile links count: plain links under `article_prefix` also appear in
    the site menu and are not part of the AI listing.
    """
    soup = BeautifulSoup(html, "html.parser")
    tiles: list[dict] = []
    seen: set[str] = set()
    for link in soup.select("a.c-tile-content, a.c-tile-content-new"):
        url = link.get("href", "")
        if not url.startswith(article_prefix) or url in seen:
            continue
        seen.add(url)
        headline = link.select_one(".headline")
        label = link.select_one(".c-label")
        tiles.append(
            {
                "title": clean_text(headline.get_text(" ")) if headline else "",
                "url": url,
                "label": clean_text(label.get_text(" ")) if label else "",
            }
        )
    return tiles


def _meta(soup: BeautifulSoup, prop: str) -> str:
    """Return the content of a <meta property=...> tag, or '' if absent."""
    tag = soup.find("meta", attrs={"property": prop})
    return clean_text(tag.get("content", "")) if tag else ""


def byline_author(first_line: str) -> str:
    """Return the writer named in a magazine byline, or '' if it is not one."""
    match = BYLINE.match(first_line)
    return match.group(1).strip() if match else ""


def _block_text(block: Tag) -> str:
    """Render one body block as a line of text; list items get a '- ' prefix."""
    text = clean_text(block.get_text(" "))
    if block.name == "li":
        return f"- {text}"
    if block.name == "blockquote":
        return f"> {text}"
    return text


def parse_article(html: str) -> dict:
    """Extract title, date, authors, summary and body text from an article page."""
    soup = BeautifulSoup(html, "html.parser")

    title = _meta(soup, "og:title")
    if not title and soup.title:
        title = clean_text(soup.title.get_text()).removesuffix(TITLE_SUFFIX)

    date_tag = soup.select_one(".c-text-timestamp .date")
    author_tag = soup.select_one("p.c-text-autor")

    # The site nests some lists inside <p>; move each list just after its
    # paragraph so the list text is not counted twice.
    for nested in soup.select("p.c-text-paragraph > ul, p.c-text-paragraph > ol"):
        parent = nested.parent
        parent.insert_after(nested.extract())

    lines: list[str] = []
    for section in soup.select("section.m-text"):
        for block in section.select(BODY_BLOCKS):
            line = _block_text(block)
            if line and line not in ("-", ">"):
                lines.append(line)

    author_links = author_tag.find_all("a") if author_tag else []
    authors = [clean_text(a.get_text(" ")) for a in author_links]
    # Profile pages give each author's title and office (see parse_person).
    author_urls = {
        clean_text(a.get_text(" ")): a.get("href", "")
        for a in author_links
        if a.get("href")
    }
    if not authors and lines and byline_author(lines[0]):
        authors = [byline_author(lines[0])]

    return {
        "title": title,
        "date": clean_text(date_tag.get_text(" ")) if date_tag else "",
        "authors": authors,
        "author_urls": author_urls,
        "summary": _meta(soup, "og:description"),
        "content": "\n".join(lines),
    }


def parse_person(html: str) -> dict:
    """Read job title, office, region and expertise from an author's profile page.

    Missing fields come back as '' (or [] for expertise): profiles of people
    who have left the firm are often removed or reduced.
    """
    soup = BeautifulSoup(html, "html.parser")
    person = soup.select_one(".m-stage-person .person")
    if person is None:
        return {"title": "", "office": "", "region": "", "expertise": []}
    title = person.select_one(".job-title")
    office = person.select_one(".address a")
    region = person.select_one(".address p")
    return {
        "title": clean_text(title.get_text(" ")) if title else "",
        "office": clean_text(office.get_text(" ")) if office else "",
        "region": clean_text(region.get_text(" ")).lstrip(", ") if region else "",
        "expertise": [
            clean_text(e.get_text(" "))
            for e in soup.select(".m-person-expertise a, .m-person-expertise li")
        ],
    }


def drop_boilerplate(contents: list[str], min_articles: int) -> list[str]:
    """Remove lines that appear word for word in at least `min_articles` articles.

    Such lines are site templates (newsletter sign-ups, "register to access
    the full study", shared bylines), not article content; left in, they
    would pull unrelated articles together in the embeddings.
    """
    line_counts = Counter(
        line for content in contents for line in set(content.split("\n"))
    )
    return [
        "\n".join(
            line for line in content.split("\n") if line_counts[line] < min_articles
        )
        for content in contents
    ]


def drop_calls_to_action(content: str, patterns: list[str]) -> str:
    """Remove sentences matching any call-to-action pattern; drop emptied lines.

    Works sentence by sentence because the site often appends "download the
    full study" or "register now" to the last real paragraph of an article.
    """
    rules = [re.compile(p, re.IGNORECASE) for p in patterns]
    kept_lines = []
    for line in content.split("\n"):
        sentences = SENTENCE_END.split(line)
        kept = [s for s in sentences if not any(r.search(s.strip()) for r in rules)]
        if " ".join(kept).strip():
            kept_lines.append(" ".join(kept).strip())
    return "\n".join(kept_lines)


def fetch(session: requests.Session, url: str, timeout_s: float) -> str:
    """GET a page and return its HTML; raises on HTTP errors."""
    response = session.get(url, timeout=timeout_s)
    response.raise_for_status()
    return response.text
