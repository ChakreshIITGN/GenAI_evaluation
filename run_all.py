"""run_all.py — single entrypoint for the eval branch: Layers 1–3 + KPIs.

Order (plan §3–§4):
    1. Load pairs; Layer 1a cosine similarity (baseline only, not a KPI).
    2. Layer 1b style features (deterministic) + paired diffs + dumbbell.
    3. Layer 2a/2b: sentence split, NLI coverage/fidelity/contradictions
       per pair (local model, deterministic).
    4. Layer 2c: one seeded BERTopic over all pooled sentences -> topic
       table + topic-map figure.
    5. Layer 3: judge panel (config eval.judges) — Judge A individuation
       (persona sketch = the person's other human answers) on all pairs,
       Judge B believability BLIND on all answers in seeded random order.
       Judge inputs lowercased, same rule as the EDA.
    6. KPIs 1–3 + RAG matrix + Krippendorff's alpha; manifest.

Everything lands in output/ (CSVs) and output/plots/ (figures). LLM replies
cache under cache/ — re-running is free; deterministic layers are seeded.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from run_eda import measure_features, normalise
from src.cluster import fit_topics, is_degenerate, plot_sentence_map, plot_topic_map
from src.embed import pair_similarity
from src.entail import coverage_fidelity
from src.judges import (
    discrimination_rate,
    judge_believability,
    judge_individuation,
    krippendorff_alpha,
)
from src.kpis import content_fidelity_table, rag_matrix, segment_coverage_summary
from src.load import load_answers
from src.segment import split_sentences
from src.style_compare import paired_differences

CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())


def run_content_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Layers 1a + 2a/2b: similarity, units, coverage/fidelity per pair."""
    similarities = pair_similarity(df["human_text"].tolist(), df["ai_text"].tolist())
    rows = []
    for row, similarity in zip(df.itertuples(), similarities):
        result = coverage_fidelity(
            row.human_text, row.ai_text,
            split_sentences(row.human_text), split_sentences(row.ai_text),
        )
        rows.append(
            {
                "person_id": row.person_id,
                "question_id": row.question_id,
                "category": row.category,
                "similarity": similarity,
            }
            | result
        )
        print(f"  2b {row.person_id} q{row.question_id} done")
    return pd.DataFrame(rows)


def pool_sentences(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """All sentences of both sides, pooled and tagged, for Layer 2c."""
    sentences: list[str] = []
    sides: list[str] = []
    for row in df.itertuples():
        for side, text in (("human", row.human_text), ("ai", row.ai_text)):
            units = split_sentences(text)
            sentences += units
            sides += [side] * len(units)
    return sentences, sides


def run_individuation(df: pd.DataFrame, judges: list[str]) -> pd.DataFrame:
    """Layer 3 Judge A on every pair: persona = the person's OTHER answers."""
    rows = []
    for model in judges:
        for row in df.itertuples():
            others = df[
                (df["person_id"] == row.person_id) & (df["question_id"] != row.question_id)
            ]["human_text"]
            verdict = judge_individuation(
                [normalise(t) for t in others],
                normalise(row.human_text), normalise(row.ai_text), model=model,
            )
            rows.append(
                {
                    "model": model,
                    "person_id": row.person_id,
                    "question_id": row.question_id,
                    "category": row.category,
                    "score": verdict.score,
                    "reason": verdict.reason,
                }
            )
            print(f"  3A [{model}] {row.person_id} q{row.question_id} done")
    return pd.DataFrame(rows)


def run_believability(df: pd.DataFrame, judges: list[str], seed: int) -> pd.DataFrame:
    """Layer 3 Judge B, blind: every answer, seeded random order, no labels."""
    items = [
        {"person_id": row.person_id, "question_id": row.question_id, "side": side,
         "text": row.human_text if side == "human" else row.ai_text}
        for row in df.itertuples()
        for side in ("human", "ai")
    ]
    random.Random(seed).shuffle(items)  # blind protocol: no positional pattern
    rows = []
    for model in judges:
        for item in items:
            verdict = judge_believability(normalise(item["text"]), model=model)
            rows.append(
                {"model": model, "person_id": item["person_id"],
                 "question_id": item["question_id"], "side": item["side"],
                 "score": verdict.score, "reason": verdict.reason}
            )
            print(f"  3B [{model}] {item['person_id']} q{item['question_id']} "
                  f"({item['side']}) done")
    return pd.DataFrame(rows)


def judge_summaries(individuation: pd.DataFrame, believability: pd.DataFrame) -> pd.DataFrame:
    """KPI 3 rows: per-judge and pooled discrimination/means + panel alphas."""
    rows = []
    for model, part in believability.groupby("model"):
        rows.append({"metric": "blind_discrimination_rate", "model": model,
                     "value": discrimination_rate(part["score"], part["side"]),
                     "n": len(part)})
    rows.append({"metric": "blind_discrimination_rate", "model": "pooled",
                 "value": discrimination_rate(believability["score"], believability["side"]),
                 "n": len(believability)})
    for model, part in individuation.groupby("model"):
        rows.append({"metric": "individuation_mean", "model": model,
                     "value": part["score"].mean(), "n": len(part)})
    rows.append({"metric": "individuation_mean", "model": "pooled",
                 "value": individuation["score"].mean(), "n": len(individuation)})
    # Panel reliability per construct: items x judges matrices.
    for name, frame, item_cols in (
        ("individuation", individuation, ["person_id", "question_id"]),
        ("believability", believability, ["person_id", "question_id", "side"]),
    ):
        matrix = frame.pivot_table(index=item_cols, columns="model", values="score")
        rows.append({"metric": f"krippendorff_alpha_{name}", "model": "panel",
                     "value": krippendorff_alpha(matrix), "n": len(matrix)})
    return pd.DataFrame(rows)


def main() -> None:
    """Run Layers 1–3 and the KPIs once; write everything under output/."""
    start = time.time()
    judges: list[str] = CONFIG["eval"]["judges"]
    thresholds: dict = CONFIG["eval"]["rag_thresholds"]
    results_dir = Path(CONFIG["paths"]["results"])
    plots_dir = Path(CONFIG["paths"]["plots"])
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_answers(CONFIG["paths"]["data"])

    print("Layer 1b — style features and paired differences…")
    features = measure_features(df)
    features.to_csv(results_dir / "measured_features.csv", index=False)
    paired_differences(features).to_csv(results_dir / "style_differences.csv", index=False)

    print("Layers 1a + 2a/2b — similarity and NLI coverage/fidelity…")
    pairs = run_content_layer(df)
    pairs.to_csv(results_dir / "pairs_content.csv", index=False)

    print("Layer 2c — topic map and sentence cluster map…")
    sentences, sides = pool_sentences(df)
    print(f"  2c fitting BERTopic on {len(sentences)} sentences…")
    topics = fit_topics(sentences, sides)
    topics.to_csv(results_dir / "topics.csv", index=False)
    plot_topic_map(topics, plots_dir / "topic_map.png")
    plot_sentence_map(sentences, sides, plots_dir / "sentence_map.png")

    print(f"Layer 3 — judge panel ({', '.join(judges)})…")
    individuation = run_individuation(df, judges)
    individuation.to_csv(results_dir / "ratings_individuation.csv", index=False)
    believability = run_believability(df, judges, CONFIG["llm"]["seed"])
    believability.to_csv(results_dir / "ratings_believability.csv", index=False)

    print("KPIs and RAG matrix…")
    # Pooled over categories, split by person (author decision: 3–15 pairs
    # per category is too small for a robust verdict; reported as such).
    content_fidelity_table(pairs).to_csv(results_dir / "kpi1_content_fidelity.csv", index=False)
    rag = rag_matrix(pairs, thresholds["green"], thresholds["amber"])
    rag.to_csv(results_dir / "rag_matrix.csv", index=False)
    kpi3 = judge_summaries(individuation, believability)
    kpi3.to_csv(results_dir / "kpi3_believability.csv", index=False)
    if is_degenerate(topics):
        # Honest reporting: a collapsed clustering must never read as
        # "all topics covered" — KPI 2 is simply not computable here.
        kpi2 = {"computable": False,
                "note": "topic model degenerate at this n; KPI 2 not computable"}
        print("WARNING: topic model is degenerate — KPI 2 reported as not computable.")
    else:
        kpi2 = {"computable": True} | segment_coverage_summary(topics)

    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": CONFIG,
        "kpi2_segment_coverage": kpi2,
        "llm_calls": {"individuation": len(individuation), "believability": len(believability)},
        "wall_time_s": round(time.time() - start, 1),
    }
    (results_dir / "run_all_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done in {manifest['wall_time_s']}s — outputs in {results_dir}/, "
          f"figures in {plots_dir}/")
    print(f"KPI 2: {kpi2}")
    print(rag.to_string(index=False))


if __name__ == "__main__":
    main()
