"""run_agentic_eda.py — entrypoint for the agentic re-implementation of the EDA.

Same judgements, same fixed split, same output files as run_eda.py
(output/eda_*.csv, measured_features.csv, run_manifest.json — src/plots.py
works unchanged), but every judge model runs as its own LangChain-based
`JudgeAgent` with per-call logging. Extra outputs: one JSONL call log per
agent under output/agent_logs/ and an aggregated per-agent monitoring
summary in output/agent_summary.json, also printed at the end.

Metric/reason/feature computation is imported from run_eda.py — one
definition, two pipelines, no drift.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from run_eda import collect_reasons, compute_metrics, measure_features
from src.agents.judge_agent import JudgeAgent
from src.agents.monitor import summary
from src.agents.orchestrator import run_q0, run_q1, run_q2
from src.eda import save_split, split_questions
from src.load import load_answers

CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())


def print_summary(stats: dict[str, dict]) -> None:
    """Print the per-agent monitoring summary as a small readable table.

    Args:
        stats: output of src.agents.monitor.summary().
    """
    print("\nPer-agent monitoring summary:")
    for model, row in stats.items():
        latency = row["mean_latency_s"]
        print(
            f"  {model}: {row['calls']} calls "
            f"({row['llm_calls']} LLM, {row['cache_hits']} cached), "
            f"mean latency {latency if latency is not None else 'n/a'}s, "
            f"repairs {row['repairs']}, failure rate {row['failure_rate']:.0%}"
        )


def main() -> None:
    """Run the agentic EDA once and write every output under output/."""
    start = time.time()
    lowercase: bool = CONFIG["eda"]["lowercase_judge_inputs"]
    results_dir = Path(CONFIG["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # One agent per judge model, built once for the whole run — the chat
    # client lives inside the agent and is never rebuilt per call.
    agents = [JudgeAgent(model) for model in CONFIG["eda"]["judges"]]

    df = load_answers(CONFIG["paths"]["data"])
    split = split_questions(df, CONFIG["eda"]["held_out_questions"])
    save_split(
        {"judges": [a.model for a in agents], "split": split},
        results_dir / "eda_split.json",
    )
    print(f"Loaded {len(df)} pairs; split written; agents: "
          f"{', '.join(a.model for a in agents)}")

    print(f"Q0 — human or AI, zero-shot, {len(agents)} agents × 60 answers…")
    q0 = run_q0(df, agents, lowercase)
    print("Q1 — captures personality, same held-out set per agent…")
    q1 = run_q1(df, split, agents, lowercase)
    print("Q2 — captures content, same held-out set per agent…")
    q2 = run_q2(df, split, agents, lowercase)

    q0.to_csv(results_dir / "eda_q0.csv", index=False)
    q1.to_csv(results_dir / "eda_q1.csv", index=False)
    q2.to_csv(results_dir / "eda_q2.csv", index=False)
    compute_metrics(q0, q1, q2).to_csv(results_dir / "eda_metrics.csv", index=False)
    collect_reasons(q0, q1, q2).to_csv(results_dir / "eda_reasons.csv", index=False)
    measure_features(df).to_csv(results_dir / "measured_features.csv", index=False)

    stats = summary()
    Path(CONFIG["agents"]["summary_file"]).write_text(json.dumps(stats, indent=2))

    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline": "agentic (LangChain LCEL, one JudgeAgent per model)",
        "config": CONFIG,
        "llm_calls": {"q0": len(q0), "q1": len(q1), "q2": len(q2)},
        "agent_summary": stats,
        "wall_time_s": round(time.time() - start, 1),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print_summary(stats)
    print(f"\nDone in {manifest['wall_time_s']}s — outputs in {results_dir}/")


if __name__ == "__main__":
    main()
