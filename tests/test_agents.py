"""Tests for src/agents/ — JudgeAgent parsing/repair/cache, monitor, orchestrator.

The LCEL chain is replaced by a scripted fake throughout: no test here
touches Ollama, LangChain networking, or needs a real API key.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.agents import judge_agent, monitor, orchestrator
from src.agents.judge_agent import JudgeAgent
from src.agents.monitor import AgentMonitor, sanitize_model, summary
from src.eda import split_questions
from src.eda_models import CapturesContent, CapturesPersonality, IsAI

VALID = '{"is_ai": true, "reason": "fine"}'
INVALID = '{"is_ai": "not-a-bool", "missing": 1}'


class FakeChain:
    """Stands in for the LCEL chain: pops scripted replies, records prompts."""

    def __init__(self) -> None:
        """Start with no scripted replies and no recorded calls."""
        self.replies: list[str] = []
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> dict:
        """Return the next scripted reply in the chain's output shape."""
        self.prompts.append(prompt)
        return {"text": self.replies.pop(0), "input_tokens": 7, "output_tokens": 3}


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JudgeAgent:
    """A JudgeAgent with tmp cache/log dirs and the chain replaced by a fake."""
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")  # no real key needed
    monkeypatch.setattr(judge_agent, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(monitor, "LOG_DIR", tmp_path / "logs")
    built = JudgeAgent("test-model")
    built._chain = FakeChain()
    return built


def read_log(agent_under_test: JudgeAgent) -> list[dict]:
    """All JSONL records the agent's monitor has written so far."""
    return [
        json.loads(line)
        for line in agent_under_test.monitor.path.read_text().splitlines()
    ]


# --- JudgeAgent -------------------------------------------------------------


def test_valid_reply_parses(agent: JudgeAgent) -> None:
    """A valid JSON reply comes back as a validated IsAI instance."""
    agent._chain.replies = [VALID]
    result = agent.judge_is_ai("some answer text")
    assert result == IsAI(is_ai=True, reason="fine")
    # The prompt template was filled with the answer being judged.
    assert "some answer text" in agent._chain.prompts[0]


def test_fenced_json_is_unwrapped(agent: JudgeAgent) -> None:
    """A reply wrapped in ```json fences still parses without a repair round."""
    agent._chain.replies = [f"```json\n{VALID}\n```"]
    result = agent.judge_is_ai("answer")
    assert result.is_ai is True
    assert len(agent._chain.prompts) == 1


def test_repair_retry_recovers(agent: JudgeAgent) -> None:
    """An invalid first reply triggers exactly one repair call, logged as such."""
    agent._chain.replies = [INVALID, VALID]
    result = agent.judge_is_ai("answer")
    assert result.reason == "fine"
    assert len(agent._chain.prompts) == 2
    assert read_log(agent)[-1]["status"] == "repair"


def test_second_failure_raises_and_logs(agent: JudgeAgent) -> None:
    """Two invalid replies propagate the ValidationError and log a failure."""
    agent._chain.replies = [INVALID, INVALID]
    with pytest.raises(ValidationError):
        agent.judge_is_ai("answer")
    assert read_log(agent)[-1]["status"] == "fail"


def test_cache_hit_skips_network(agent: JudgeAgent) -> None:
    """The second identical ask is served from disk and logged as a cache hit."""
    agent._chain.replies = [VALID]
    agent.judge_is_ai("answer")
    result = agent.judge_is_ai("answer")  # would raise if it popped an empty list
    assert result.is_ai is True
    assert len(agent._chain.prompts) == 1
    assert [r["status"] for r in read_log(agent)] == ["success", "cache_hit"]


def test_personality_prompt_wiring(agent: JudgeAgent) -> None:
    """Q1 puts every context answer and the AI answer into the prompt."""
    agent._chain.replies = ['{"captures_personality": false, "reason": "flat"}']
    result = agent.judge_personality(["ctx one", "ctx two"], "the ai answer")
    assert result == CapturesPersonality(captures_personality=False, reason="flat")
    prompt = agent._chain.prompts[0]
    assert "ctx one" in prompt and "ctx two" in prompt and "the ai answer" in prompt


def test_content_prompt_wiring(agent: JudgeAgent) -> None:
    """Q2 puts question, human and AI answers into the prompt."""
    agent._chain.replies = ['{"captures_content": true, "reason": "same points"}']
    result = agent.judge_content("the question?", "human said", "ai said")
    assert result == CapturesContent(captures_content=True, reason="same points")
    prompt = agent._chain.prompts[0]
    assert "the question?" in prompt and "human said" in prompt and "ai said" in prompt


# --- Monitor ----------------------------------------------------------------


def test_sanitize_model_makes_safe_stems() -> None:
    """Colons and slashes in model tags become dashes for filenames."""
    assert sanitize_model("gpt-oss:120b") == "gpt-oss-120b"
    assert sanitize_model("org/model:v1") == "org-model-v1"


def test_monitor_writes_jsonl(tmp_path: Path) -> None:
    """record() appends one parseable JSON line per call."""
    mon = AgentMonitor("m:1", log_dir=tmp_path)
    mon.record("abc", 1.25, "success", input_tokens=10, output_tokens=5)
    mon.record("def", 0.75, "fail")
    lines = [json.loads(line) for line in mon.path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["model"] == "m:1" and lines[0]["latency_s"] == 1.25
    assert lines[1]["status"] == "fail" and lines[1]["input_tokens"] is None


def test_summary_aggregates_per_agent(tmp_path: Path) -> None:
    """summary() reports calls, mean latency (LLM calls only) and failure rate."""
    mon = AgentMonitor("m:1", log_dir=tmp_path)
    mon.record("a", 1.0, "success", 10, 5)
    mon.record("a", 0.0, "cache_hit")
    mon.record("b", 3.0, "fail", 10, 5)
    stats = summary(log_dir=tmp_path)["m:1"]
    assert stats["calls"] == 3 and stats["llm_calls"] == 2 and stats["cache_hits"] == 1
    assert stats["mean_latency_s"] == 2.0  # (1.0 + 3.0) / 2, cache hit excluded
    assert stats["failure_rate"] == pytest.approx(1 / 3)
    assert stats["input_tokens"] == 20 and stats["output_tokens"] == 10


def test_summary_empty_dir_is_empty(tmp_path: Path) -> None:
    """No logs -> empty summary, not an error."""
    assert summary(log_dir=tmp_path / "nowhere") == {}


# --- Orchestrator -----------------------------------------------------------


class FakeAgent:
    """Duck-typed JudgeAgent that returns fixed verdicts and records inputs."""

    def __init__(self, model: str) -> None:
        """Remember the model tag and start with an empty input record."""
        self.model = model
        self.seen: list[str] = []

    def judge_is_ai(self, answer: str) -> IsAI:
        """Always says 'AI'; records the text it was shown."""
        self.seen.append(answer)
        return IsAI(is_ai=True, reason="r")

    def judge_personality(self, context_answers: list[str], ai_answer: str) -> CapturesPersonality:
        """Always says yes; records everything it was shown."""
        self.seen.extend(context_answers + [ai_answer])
        return CapturesPersonality(captures_personality=True, reason="r")

    def judge_content(self, question: str, human_answer: str, ai_answer: str) -> CapturesContent:
        """Always says yes; records the two answers it was shown."""
        self.seen.extend([human_answer, ai_answer])
        return CapturesContent(captures_content=True, reason="r")


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Two people x two questions, mixed case to test lowercasing."""
    rows = [
        {"person_id": p, "question_id": q, "category": f"cat{q}",
         "question": f"Question {q}?", "human_text": f"HUMAN {p} q{q}",
         "ai_text": f"AI {p} q{q}"}
        for p in ("P1", "P2") for q in (1, 2)
    ]
    return pd.DataFrame(rows)


def test_run_q0_columns_and_counts(tiny_df: pd.DataFrame) -> None:
    """Q0 frame matches run_eda's eda_q0.csv schema: one row per model/pair/side."""
    agents = [FakeAgent("m1"), FakeAgent("m2")]
    frame = orchestrator.run_q0(tiny_df, agents, lowercase=True)
    assert list(frame.columns) == [
        "model", "person_id", "question_id", "category", "side",
        "is_ai", "correct", "reason",
    ]
    assert len(frame) == 2 * 4 * 2  # 2 agents x 4 pairs x 2 sides
    # FakeAgent always says AI: correct on the ai side, wrong on the human side.
    assert frame.loc[frame["side"] == "ai", "correct"].all()
    assert not frame.loc[frame["side"] == "human", "correct"].any()


def test_run_q1_q2_columns_and_counts(tiny_df: pd.DataFrame) -> None:
    """Q1/Q2 frames match run_eda's schemas: one row per model/person/held-out q."""
    split = split_questions(tiny_df, held_out_questions=[2])
    agents = [FakeAgent("m1")]
    q1 = orchestrator.run_q1(tiny_df, split, agents, lowercase=True)
    q2 = orchestrator.run_q2(tiny_df, split, agents, lowercase=True)
    assert list(q1.columns) == [
        "model", "person_id", "question_id", "category",
        "captures_personality", "reason",
    ]
    assert list(q2.columns) == [
        "model", "person_id", "question_id", "category",
        "captures_content", "reason",
    ]
    assert len(q1) == len(q2) == 2  # 1 agent x 2 people x 1 held-out question


def test_judge_inputs_are_lowercased(tiny_df: pd.DataFrame) -> None:
    """With lowercase=True every text an agent sees is lowercase (run_eda rule)."""
    split = split_questions(tiny_df, held_out_questions=[2])
    agent_under_test = FakeAgent("m1")
    orchestrator.run_q0(tiny_df, [agent_under_test], lowercase=True)
    orchestrator.run_q1(tiny_df, split, [agent_under_test], lowercase=True)
    orchestrator.run_q2(tiny_df, split, [agent_under_test], lowercase=True)
    assert agent_under_test.seen  # sanity: the agent was actually called
    assert all(text == text.lower() for text in agent_under_test.seen)
