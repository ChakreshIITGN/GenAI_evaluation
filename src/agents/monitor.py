"""Per-agent call monitoring: one JSONL log per judge model + a summary.

Every `JudgeAgent` owns an `AgentMonitor` that appends one JSON line per LLM
call (timestamp, model, prompt hash, latency, status, token counts) to
output/agent_logs/<model>.jsonl. `summary()` aggregates all logs into
per-agent statistics (calls, mean latency, failure rate) — this is what lets
each agent be tracked and compared independently.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import json
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

import yaml

_CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())
LOG_DIR = Path(_CONFIG["agents"]["log_dir"])

# The four ways a call can end. String constants, not an Enum, so the JSONL
# stays greppable and readable without importing anything.
STATUS_CACHE_HIT = "cache_hit"  # served from disk, no network
STATUS_SUCCESS = "success"      # first reply validated
STATUS_REPAIR = "repair"        # first reply invalid, repair reply validated
STATUS_FAIL = "fail"            # both replies invalid, error propagated


def sanitize_model(model: str) -> str:
    """Turn a model tag into a safe file stem (e.g. 'gpt-oss:120b' -> 'gpt-oss-120b').

    Args:
        model: Ollama model tag, may contain ':' or '/'.

    Returns:
        The tag with every non-filename character replaced by '-'.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", model)


class AgentMonitor:
    """Append-only JSONL call log for ONE agent (one judge model).

    Attributes:
        model: the judge model this monitor belongs to.
        path: the JSONL file all records are appended to.
    """

    def __init__(self, model: str, log_dir: Path | None = None) -> None:
        """Bind the monitor to its model and log file.

        Args:
            model: Ollama model tag of the agent being monitored.
            log_dir: where the JSONL lives; None uses config agents.log_dir.
        """
        self.model = model
        # LOG_DIR is looked up here (not as a default argument) so tests can
        # monkeypatch the module constant before constructing monitors.
        directory = Path(log_dir) if log_dir is not None else LOG_DIR
        self.path = directory / f"{sanitize_model(model)}.jsonl"

    def record(
        self,
        prompt_hash: str,
        latency_s: float,
        status: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Append one call record to the agent's JSONL log.

        Args:
            prompt_hash: cache key of the prompt (joins log to cache entry).
            latency_s: wall time of the whole call, repair round included.
            status: one of the STATUS_* constants above.
            input_tokens: prompt tokens if the response metadata reported them.
            output_tokens: completion tokens if reported.
        """
        entry = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": self.model,
            "prompt_hash": prompt_hash,
            "latency_s": round(latency_s, 3),
            "status": status,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One JSON object per line: append-safe, greppable, no read-modify-write.
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")


def summary(log_dir: Path | None = None) -> dict[str, dict]:
    """Aggregate every agent log in `log_dir` into per-agent statistics.

    Args:
        log_dir: directory of JSONL logs; None uses config agents.log_dir.

    Returns:
        {model: {calls, llm_calls, cache_hits, repairs, failures,
                 failure_rate, mean_latency_s, input_tokens, output_tokens}}.
        mean_latency_s covers real LLM calls only (cache hits are ~0 and
        would flatter the number); token totals are None when the backend
        never reported usage.
    """
    directory = Path(log_dir) if log_dir is not None else LOG_DIR
    result: dict[str, dict] = {}
    for path in sorted(directory.glob("*.jsonl")) if directory.exists() else []:
        entries = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not entries:
            continue
        model = entries[0]["model"]
        llm_entries = [e for e in entries if e["status"] != STATUS_CACHE_HIT]
        failures = sum(e["status"] == STATUS_FAIL for e in entries)
        tokens_in = [e["input_tokens"] for e in entries if e["input_tokens"] is not None]
        tokens_out = [e["output_tokens"] for e in entries if e["output_tokens"] is not None]
        result[model] = {
            "calls": len(entries),
            "llm_calls": len(llm_entries),
            "cache_hits": sum(e["status"] == STATUS_CACHE_HIT for e in entries),
            "repairs": sum(e["status"] == STATUS_REPAIR for e in entries),
            "failures": failures,
            "failure_rate": failures / len(entries),
            "mean_latency_s": (
                round(statistics.mean(e["latency_s"] for e in llm_entries), 3)
                if llm_entries
                else None
            ),
            "input_tokens": sum(tokens_in) if tokens_in else None,
            "output_tokens": sum(tokens_out) if tokens_out else None,
        }
    return result
