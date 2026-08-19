"""run_eda.py — single entrypoint for the EDA branch (plan §1c, extended).

Design (author decisions, see docs/DECISIONS.md):
  - Judge inputs are LOWERCASED (config eda.lowercase_judge_inputs) so
    orthography cannot drive verdicts; style features use the raw text.
  - A panel of judge models (config eda.judges) each answers EVERY question:
    Q0 on all answers, Q1/Q2 on the SAME held-out set — verdicts are
    directly comparable across judges.
  - One fixed within-person split (config eda.held_out_questions), identical
    for every person and judge; the rest of a person's questions are the
    Q1 context. Q0 uses no split.

Outputs (output/): eda_split.json, eda_q0.csv, eda_q1.csv, eda_q2.csv,
eda_metrics.csv, eda_reasons.csv, measured_features.csv, run_manifest.json.
All LLM replies are cached under cache/ — re-running is free and identical.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.eda import judge_content, judge_is_ai, judge_personality, save_split, split_questions
from src.load import load_answers
from src.stylometric import Stylometric

CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())


def normalise(text: str) -> str:
    """Lowercase text bound for a judge, if configured.

    We cannot tell transcription artefacts (caps, typos) from real style, so
    judges must not see the difference. Style features keep the raw text.
    """
    return text.lower() if CONFIG["eda"]["lowercase_judge_inputs"] else text


def run_q0(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Q0 — zero-shot human-or-AI on all answers, both sides, once per model.

    Returns one row per (model, person, question, side) with the prediction,
    whether it was correct (ground truth: side == 'ai'), and the reason.
    """
    rows = []
    for model in models:
        for row in df.itertuples():
            for side in ("human", "ai"):
                text = row.human_text if side == "human" else row.ai_text
                verdict = judge_is_ai(normalise(text), model=model)
                rows.append(
                    {
                        "model": model,
                        "person_id": row.person_id,
                        "question_id": row.question_id,
                        "category": row.category,
                        "side": side,
                        "is_ai": verdict.is_ai,
                        "correct": verdict.is_ai == (side == "ai"),
                        "reason": verdict.reason,
                    }
                )
            print(f"  Q0 [{model}] {row.person_id} q{row.question_id} done")
    return pd.DataFrame(rows)


def run_q1(df: pd.DataFrame, split: dict, judges: list[str]) -> pd.DataFrame:
    """Q1 — captures personality: every judge scores the same held-out set.

    Context = the person's own human answers to their context questions.
    """
    rows = []
    for model in judges:
        for person_id, parts in split.items():
            person = df[df["person_id"] == person_id].set_index("question_id")
            context = [normalise(t) for t in person.loc[parts["context"], "human_text"]]
            for question_id in parts["held_out"]:
                verdict = judge_personality(
                    context, normalise(person.loc[question_id, "ai_text"]), model=model
                )
                rows.append(
                    {
                        "model": model,
                        "person_id": person_id,
                        "question_id": question_id,
                        "category": person.loc[question_id, "category"],
                        "captures_personality": verdict.captures_personality,
                        "reason": verdict.reason,
                    }
                )
                print(f"  Q1 [{model}] {person_id} q{question_id} done")
    return pd.DataFrame(rows)


def run_q2(df: pd.DataFrame, split: dict, judges: list[str]) -> pd.DataFrame:
    """Q2 — pairwise content check: every judge scores the same held-out pairs."""
    rows = []
    for model in judges:
        for person_id, parts in split.items():
            person = df[df["person_id"] == person_id].set_index("question_id")
            for question_id in parts["held_out"]:
                pair = person.loc[question_id]
                verdict = judge_content(
                    pair["question"], normalise(pair["human_text"]),
                    normalise(pair["ai_text"]), model=model,
                )
                rows.append(
                    {
                        "model": model,
                        "person_id": person_id,
                        "question_id": question_id,
                        "category": pair["category"],
                        "captures_content": verdict.captures_content,
                        "reason": verdict.reason,
                    }
                )
                print(f"  Q2 [{model}] {person_id} q{question_id} done")
    return pd.DataFrame(rows)


def compute_metrics(q0: pd.DataFrame, q1: pd.DataFrame, q2: pd.DataFrame) -> pd.DataFrame:
    """Tidy metrics, one block per judge model on identical items.

    Q0: accuracy overall + per side, per model (a genuine rate over 60 items).
    Q1/Q2: yes-COUNTS per person and per judge (value = count, n = items).
    All judges score the same items, so rows are directly comparable.
    """
    rows = []
    for model, part in q0.groupby("model"):
        rows.append({"metric": "q0_accuracy", "scope": "overall", "model": model,
                     "value": part["correct"].mean(), "n": len(part)})
        # Per-side accuracy shows judge bias toward one answer.
        for side, side_part in part.groupby("side"):
            rows.append({"metric": "q0_accuracy", "scope": f"side:{side}", "model": model,
                         "value": side_part["correct"].mean(), "n": len(side_part)})
    # Q1/Q2 are reported as raw yes-COUNTS, not percentages (author decision):
    # at 4 held-out questions per person a share reads like a graded "how
    # much it captures", which it is not. No category split — 12 items are
    # too few to slice twice. Volatile measure; stated on the figures.
    for frame, flag in ((q1, "captures_personality"), (q2, "captures_content")):
        rows.append({"metric": f"{flag}_yes_count", "scope": "overall", "model": "pooled",
                     "value": int(frame[flag].sum()), "n": len(frame)})
        for model, part in frame.groupby("model"):
            rows.append({"metric": f"{flag}_yes_count", "scope": "overall", "model": model,
                         "value": int(part[flag].sum()), "n": len(part)})
            for person, person_part in part.groupby("person_id"):
                rows.append({"metric": f"{flag}_yes_count", "scope": f"person_id:{person}",
                             "model": model, "value": int(person_part[flag].sum()),
                             "n": len(person_part)})
    return pd.DataFrame(rows)


def collect_reasons(q0: pd.DataFrame, q1: pd.DataFrame, q2: pd.DataFrame) -> pd.DataFrame:
    """Verbatim judge reasons across all three questions, one readable file."""
    frames = []
    for name, frame, flag in (
        ("q0", q0, "is_ai"), ("q1", q1, "captures_personality"), ("q2", q2, "captures_content")
    ):
        part = frame.copy()
        part["question"] = name
        part["verdict"] = part[flag]
        columns = ["question", "model", "person_id", "question_id", "verdict", "reason"]
        if "side" in part.columns:
            columns.insert(4, "side")
        frames.append(part[columns])
    return pd.concat(frames, ignore_index=True)


def measure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stylometric.all_features() on every RAW answer, both sides.

    Deliberately not lowercased: style is measured, not judged, and none of
    the features depend on case.
    """
    rows = []
    for row in df.itertuples():
        for side in ("human", "ai"):
            text = row.human_text if side == "human" else row.ai_text
            rows.append(
                {"person_id": row.person_id, "question_id": row.question_id, "side": side}
                | Stylometric(text).all_features()
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the whole EDA once and write every output under output/."""
    start = time.time()
    judges: list[str] = CONFIG["eda"]["judges"]
    results_dir = Path(CONFIG["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_answers(CONFIG["paths"]["data"])
    split = split_questions(df, CONFIG["eda"]["held_out_questions"])
    save_split({"judges": judges, "split": split}, results_dir / "eda_split.json")
    print(f"Loaded {len(df)} pairs; split written; judges: {', '.join(judges)}")

    print(f"Q0 — human or AI, zero-shot, {len(judges)} judges × 60 answers…")
    q0 = run_q0(df, judges)
    print("Q1 — captures personality, same held-out set per judge…")
    q1 = run_q1(df, split, judges)
    print("Q2 — captures content, same held-out set per judge…")
    q2 = run_q2(df, split, judges)

    q0.to_csv(results_dir / "eda_q0.csv", index=False)
    q1.to_csv(results_dir / "eda_q1.csv", index=False)
    q2.to_csv(results_dir / "eda_q2.csv", index=False)
    compute_metrics(q0, q1, q2).to_csv(results_dir / "eda_metrics.csv", index=False)
    collect_reasons(q0, q1, q2).to_csv(results_dir / "eda_reasons.csv", index=False)
    measure_features(df).to_csv(results_dir / "measured_features.csv", index=False)

    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": CONFIG,
        "llm_calls": {"q0": len(q0), "q1": len(q1), "q2": len(q2)},
        "tokens": "not captured on the EDA branch",
        "wall_time_s": round(time.time() - start, 1),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done in {manifest['wall_time_s']}s — outputs in {results_dir}/")


if __name__ == "__main__":
    main()
