"""Tests for src/scrape_rb.py: the pure HTML parsers, on small inline pages.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from src.scrape_rb import (
    byline_author,
    clean_text,
    drop_boilerplate,
    drop_calls_to_action,
    parse_article,
    parse_listing,
    parse_person,
)

PREFIX = "https://www.rolandberger.com/en/Insights/Publications/"

LISTING = f"""
<nav><a href="{PREFIX}Menu-link.html">Menu link</a></nav>
<a href="{PREFIX}First.html" class="c-tile-content is-large">
  <div class="c-label">Study</div><div class="headline"><span>First  article</span></div>
</a>
<a href="{PREFIX}Second.html" class="c-tile-content-new style-regular">
  <div class="headline"><span>Second article</span></div>
</a>
<a href="{PREFIX}First.html" class="c-tile-content">
  <div class="headline"><span>First article</span></div>
</a>
<a href="https://example.com/other.html" class="c-tile-content">
  <div class="headline"><span>Elsewhere</span></div>
</a>
"""

ARTICLE = """
<html><head>
<title>Fallback title | Roland Berger</title>
<meta property="og:title" content="Real title" />
<meta property="og:description" content="Short summary." />
</head><body>
<div class="c-text-timestamp is-top"><span class="date">September 9, 2025</span></div>
<p class="c-text-autor">By <a href="#">Ann Lee</a> and <a href="#">Bo Chen</a></p>
<section class="m-text">
  <p class="c-text-paragraph">Intro with a <a href="#"> link </a>, then more.</p>
  <h3 class="c-text-subheadline is-black">Key insights</h3>
  <p class="c-text-paragraph">Lead-in:<ol class="c-text-list"><li>One</li><li>Two</li></ol></p>
  <figure class="c-text-blockquote"><blockquote>"A quote."</blockquote></figure>
</section>
<p class="c-text-paragraph">Related tile text outside the article.</p>
</body></html>
"""


def test_clean_text_collapses_space_and_fixes_punctuation() -> None:
    assert clean_text("  a \n b , c . ") == "a b, c."


def test_listing_keeps_only_tiles_under_prefix() -> None:
    urls = [t["url"] for t in parse_listing(LISTING, PREFIX)]
    assert urls == [f"{PREFIX}First.html", f"{PREFIX}Second.html"]


def test_listing_reads_title_and_label() -> None:
    first, second = parse_listing(LISTING, PREFIX)
    assert first["title"] == "First article" and first["label"] == "Study"
    assert second["label"] == ""


def test_article_metadata() -> None:
    article = parse_article(ARTICLE)
    assert article["title"] == "Real title"
    assert article["date"] == "September 9, 2025"
    assert article["authors"] == ["Ann Lee", "Bo Chen"]
    assert article["summary"] == "Short summary."


def test_article_body_in_order_without_duplicates() -> None:
    assert parse_article(ARTICLE)["content"].split("\n") == [
        "Intro with a link, then more.",
        "Key insights",
        "Lead-in:",
        "- One",
        "- Two",
        '> "A quote."',
    ]


def test_article_title_falls_back_to_page_title() -> None:
    html = ARTICLE.replace('<meta property="og:title" content="Real title" />', "")
    assert parse_article(html)["title"] == "Fallback title"


def test_byline_author_keeps_only_the_writer() -> None:
    assert (
        byline_author("by Steffan Heuer Artworks by Carsten Gueth") == "Steffan Heuer"
    )
    assert (
        byline_author("Article by Steffan Heuer Photos by Tony Luong")
        == "Steffan Heuer"
    )
    assert byline_author("by Henrik Bork") == "Henrik Bork"
    assert byline_author("HR is changing. Digitalization is here.") == ""


def test_article_without_author_line_uses_byline() -> None:
    byline = '<p class="c-text-paragraph">by Cy Diaz Illustrations by Kim</p>'
    html = ARTICLE.replace(
        '<p class="c-text-autor">By <a href="#">Ann Lee</a> and <a href="#">Bo Chen</a></p>',
        "",
    ).replace('<section class="m-text">', f'<section class="m-text">{byline}')
    assert parse_article(html)["authors"] == ["Cy Diaz"]


def test_drop_boilerplate_removes_lines_shared_by_many_articles() -> None:
    footer = "Sign up now to access the full study."
    contents = [f"Unique {i}.\n{footer}" for i in range(3)] + ["Only here."]
    cleaned = drop_boilerplate(contents, min_articles=3)
    assert cleaned == ["Unique 0.", "Unique 1.", "Unique 2.", "Only here."]


def test_drop_boilerplate_keeps_lines_below_threshold() -> None:
    contents = ["Shared line.\nA.", "Shared line.\nB."]
    assert drop_boilerplate(contents, min_articles=3) == contents


PERSON = """
<section class="m-stage-person"><div class="person">
  <div class="name"><h1>Ann Lee</h1></div>
  <div class="details"><div class="job-title">Senior Partner</div>
    <div class="address"><a href="#">Berlin Office</a><p>,  Central Europe</p></div></div>
</div></section>
<section class="m-person-expertise"><div class="headline"><a href="#">Digital</a></div></section>
"""


def test_article_keeps_author_profile_links() -> None:
    article = parse_article(
        ARTICLE.replace('<a href="#">Ann Lee</a>', '<a href="/p/ann">Ann Lee</a>')
    )
    assert article["author_urls"] == {"Ann Lee": "/p/ann", "Bo Chen": "#"}


def test_parse_person_reads_title_office_region_expertise() -> None:
    assert parse_person(PERSON) == {
        "title": "Senior Partner",
        "office": "Berlin Office",
        "region": "Central Europe",
        "expertise": ["Digital"],
    }


def test_parse_person_without_profile_block_returns_blanks() -> None:
    assert parse_person("<html></html>") == {
        "title": "",
        "office": "",
        "region": "",
        "expertise": [],
    }


CTA = [
    "^(register|sign up|subscribe|stay current|stay up to date|curious about)\\b",
    "download (a copy|the full|our)",
    "contact one of our experts",
]


def test_calls_to_action_removed_sentence_by_sentence() -> None:
    content = (
        "Banks face a shift. For more, download a copy of the report.\n"
        "Register now to access the full study.\n"
        "Embedded subscription-management tools offer the biggest gains."
    )
    assert drop_calls_to_action(content, CTA) == (
        "Banks face a shift.\n"
        "Embedded subscription-management tools offer the biggest gains."
    )


def test_register_only_matches_at_sentence_start() -> None:
    content = "Users must register their devices. Sign up today."
    assert drop_calls_to_action(content, CTA) == "Users must register their devices."
