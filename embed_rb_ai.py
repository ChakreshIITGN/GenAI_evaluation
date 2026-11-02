"""embed_rb_ai.py — chunk and embed the scraped RB AI articles.

Reads data/rb_ai_publications.json (from scrape_rb_ai.py), splits each
article into even ~300-token chunks with a small overlap, and embeds each
chunk with its metadata header (title, date, authors) prepended. Writes:
    data/rb_ai_chunks.jsonl          one chunk per line:
        {chunk_id, url, title, date, date_iso, authors, label,
         chunk_index, n_chunks, text, embedding}
    data/rb_ai_chunks_manifest.json  model, dimension, settings, counts

Vectors are L2-normalised, so a dot product is the cosine similarity.
Chunk lengths are counted with the embedding model's own tokenizer and
leave room for the header, so no chunk is truncated by the model.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer

from src.chunk_rb import chunk_header, chunk_sentences, iso_date, split_sentences

CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())


def main() -> None:
    """Chunk every article, embed all chunks in one batch run, write the files."""
    cfg = CONFIG["embed"]
    articles = json.loads(Path(CONFIG["scrape"]["output"]).read_text())["articles"]
    model = SentenceTransformer(cfg["model"])

    def count(text: str) -> int:
        """Tokens the embedding model sees for this text (no special tokens)."""
        return len(model.tokenizer.encode(text, add_special_tokens=False))

    # The model adds [CLS]/[SEP]-style tokens around every input.
    special_tokens = model.tokenizer.num_special_tokens_to_add()

    chunks: list[dict] = []
    for article in articles:
        if article["error"] or not article["content"]:
            continue
        header = chunk_header(article)
        # Header + chunk must fit in the model's window, or the end is cut off.
        room = model.max_seq_length - special_tokens - count(header)
        texts = chunk_sentences(
            split_sentences(article["content"]),
            count,
            min(cfg["chunk_tokens"], room),
            cfg["overlap_tokens"],
        )
        slug = article["url"].rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")
        for i, text in enumerate(texts):
            chunks.append(
                {
                    "chunk_id": f"{slug}#{i}",
                    "url": article["url"],
                    "title": article["title"],
                    "date": article["date"],
                    "date_iso": iso_date(article["date"]),
                    "authors": article["authors"],
                    "label": article["label"],
                    "chunk_index": i,
                    "n_chunks": len(texts),
                    "text": text,
                    "_embed_input": header + text,
                }
            )

    print(f"{len(chunks)} chunks from {len(articles)} articles; embedding...")
    vectors = model.encode(
        [c.pop("_embed_input") for c in chunks],
        batch_size=cfg["batch_size"],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = [round(float(x), 6) for x in vector]
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    manifest = {
        "model": cfg["model"],
        "dimension": int(vectors.shape[1]),
        "normalized": True,
        "embedded_text": "metadata header (title, date, authors) + chunk text",
        "chunk_tokens": cfg["chunk_tokens"],
        "overlap_tokens": cfg["overlap_tokens"],
        "articles": len(articles),
        "chunks": len(chunks),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    Path(cfg["manifest"]).write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out} and {cfg['manifest']} ({len(chunks)} chunks).")


if __name__ == "__main__":
    main()
