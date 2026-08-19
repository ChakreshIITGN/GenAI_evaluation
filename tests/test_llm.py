"""Tests for src/llm.py — caching, parsing, and retry-with-repair.

The raw network call (llm._call) is monkeypatched throughout: no test here
touches Ollama or needs an API key.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from src import llm


class Verdict(BaseModel):
    """Minimal schema standing in for the real EDA models."""

    ok: bool
    reason: str


VALID = '{"ok": true, "reason": "fine"}'
INVALID = '{"ok": "not-a-bool-at-all", "missing": 1}'


@pytest.fixture
def fake_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the cache at tmp_path and replace _call with a scripted fake.

    Returns a dict; set its 'replies' list to script successive raw replies.
    'calls' counts how often the fake network call ran.
    """
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    state = {"replies": [], "calls": 0}

    def scripted_call(_prompt: str, _model: str) -> str:
        state["calls"] += 1
        return state["replies"].pop(0)

    monkeypatch.setattr(llm, "_call", scripted_call)
    return state


def test_valid_reply_parses(fake_llm: dict) -> None:
    """A valid JSON reply comes back as a validated schema instance."""
    fake_llm["replies"] = [VALID]
    result = llm.ask("prompt", Verdict)
    assert result == Verdict(ok=True, reason="fine")
    assert fake_llm["calls"] == 1


def test_cache_hit_skips_network(fake_llm: dict) -> None:
    """The second identical ask is served from disk, not the model."""
    fake_llm["replies"] = [VALID]
    llm.ask("prompt", Verdict)
    result = llm.ask("prompt", Verdict)  # would raise if it popped an empty list
    assert result.ok is True
    assert fake_llm["calls"] == 1


def test_repair_retry_recovers(fake_llm: dict) -> None:
    """An invalid first reply triggers exactly one repair call."""
    fake_llm["replies"] = [INVALID, VALID]
    result = llm.ask("prompt", Verdict)
    assert result.reason == "fine"
    assert fake_llm["calls"] == 2


def test_second_failure_raises(fake_llm: dict) -> None:
    """Two invalid replies propagate the ValidationError — no silent retry loop."""
    fake_llm["replies"] = [INVALID, INVALID]
    with pytest.raises(ValidationError):
        llm.ask("prompt", Verdict)
    assert fake_llm["calls"] == 2


def test_fenced_json_is_unwrapped(fake_llm: dict) -> None:
    """A reply wrapped in ```json fences (glm-5.2 habit) still parses."""
    fake_llm["replies"] = [f"```json\n{VALID}\n```"]
    result = llm.ask("prompt", Verdict)
    assert result.ok is True
    assert fake_llm["calls"] == 1  # no repair round needed


def test_load_prompt_reads_templates() -> None:
    """Prompt templates load from user_prompts/ with placeholders intact."""
    template = llm.load_prompt("eda/is_ai")
    assert "{answer}" in template
