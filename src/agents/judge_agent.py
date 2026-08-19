"""One LangChain-based judge agent per model (agentic_workflow branch).

`JudgeAgent` wraps a single judge model (config.yaml eda.judges) in a
LangChain LCEL chain: ChatOllama against the Ollama cloud host, temperature 0,
fixed seed, JSON output. It mirrors src/llm.py exactly — schema appended to
the prompt, markdown fences stripped, replies validated into the SAME
Pydantic models from src/eda_models.py, one retry-with-repair, disk cache
keyed on (model, prompt hash) under cache/agentic/ — and adds per-call
monitoring through src/agents/monitor.py. Prompts load from user_prompts/;
no prompt wording lives in code.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import TypeVar

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from src.agents.monitor import (
    STATUS_CACHE_HIT,
    STATUS_FAIL,
    STATUS_REPAIR,
    STATUS_SUCCESS,
    AgentMonitor,
)
from src.eda_models import CapturesContent, CapturesPersonality, IsAI

# Reused from the non-agentic gateway on purpose: same prompt loading, fence
# stripping and schema instructions mean the two pipelines cannot drift apart
# and their cached prompts stay byte-identical.
from src.llm import _schema_block, _strip_fences, load_prompt

load_dotenv()  # makes OLLAMA_KIMIK3_KEY1 available before any agent is built

_CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())
LLM_CFG: dict = _CONFIG["llm"]
CACHE_DIR = Path(_CONFIG["agents"]["cache_dir"])

S = TypeVar("S", bound=BaseModel)


def _extract_reply(message: AIMessage) -> dict:
    """Final LCEL step: pull text and token usage out of the chat message.

    Args:
        message: the AIMessage returned by ChatOllama.

    Returns:
        {"text": str, "input_tokens": int | None, "output_tokens": int | None}.
        Token counts are None when the backend reports no usage metadata.
    """
    usage = message.usage_metadata or {}
    return {
        "text": str(message.content),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def _sum_tokens(first: int | None, second: int | None) -> int | None:
    """Add two optional token counts; None + None stays None (unknown)."""
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)


class JudgeAgent:
    """One judge model behind one LCEL chain, cached and monitored.

    One instance per judge in config.yaml eda.judges. The chat model is
    constructed ONCE here, never per call. Each instance owns its cache
    entries (keyed on its model tag) and its own JSONL call log, so judges
    can be tracked, compared and scaled independently.

    Attributes:
        model: Ollama model tag this agent judges with.
        monitor: per-agent call logger (output/agent_logs/<model>.jsonl).
    """

    def __init__(
        self,
        model: str,
        cache_dir: Path | None = None,
        monitor: AgentMonitor | None = None,
    ) -> None:
        """Build the agent and its LCEL chain.

        Args:
            model: Ollama model tag (e.g. 'gpt-oss:120b').
            cache_dir: disk cache location; None uses config agents.cache_dir.
            monitor: call logger; None builds one for this model.
        """
        self.model = model
        # CACHE_DIR is looked up here (not as a default argument) so tests
        # can monkeypatch the module constant before constructing agents.
        self._cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        self.monitor = monitor if monitor is not None else AgentMonitor(model)
        # os.environ[...] (not .get) fails fast with a clear KeyError when the
        # key is missing — better than a confusing 401 mid-run.
        chat = ChatOllama(
            model=model,
            base_url=LLM_CFG["host"],
            temperature=LLM_CFG["temperature"],
            seed=LLM_CFG["seed"],
            format="json",
            client_kwargs={
                "headers": {"Authorization": f"Bearer {os.environ['OLLAMA_KIMIK3_KEY1']}"}
            },
        )
        # The LCEL chain: chat model piped into reply extraction. Everything
        # around it (cache, repair, logging) is plain functions, per plan §1.
        self._chain = chat | RunnableLambda(_extract_reply)

    def _cache_path(self, prompt: str) -> Path:
        """Deterministic cache file for (model, prompt) — mirrors src/llm.py."""
        key = hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()[:24]
        return self._cache_dir / f"{key}.json"

    def _ask(self, prompt: str, schema: type[S]) -> S:
        """Send `prompt` through the chain and return a validated `schema`.

        Appends the JSON Schema instructions, serves cache hits from disk,
        retries once with a repair prompt on validation failure, caches only
        validated replies, and logs every call to the agent's monitor.

        Args:
            prompt: fully formatted prompt text (no output-format wording).
            schema: Pydantic model class the reply must satisfy.

        Returns:
            A validated `schema` instance.

        Raises:
            pydantic.ValidationError: if the reply is still invalid after
                one repair attempt (logged as a failure first).
        """
        full_prompt = prompt + _schema_block(schema)
        prompt_hash = hashlib.sha256(full_prompt.encode()).hexdigest()[:24]
        cache_file = self._cache_path(full_prompt)
        start = time.perf_counter()

        if cache_file.exists():
            raw = json.loads(cache_file.read_text())["raw"]
            parsed = schema.model_validate_json(_strip_fences(raw))
            self.monitor.record(prompt_hash, time.perf_counter() - start, STATUS_CACHE_HIT)
            return parsed

        reply = self._chain.invoke(full_prompt)
        raw = reply["text"]
        tokens_in: int | None = reply["input_tokens"]
        tokens_out: int | None = reply["output_tokens"]
        status = STATUS_SUCCESS
        try:
            parsed = schema.model_validate_json(_strip_fences(raw))
        except ValidationError as err:
            # One repair attempt: show the model its invalid reply and the error.
            repair = (
                full_prompt
                + f"\n\nYour previous reply was invalid.\nReply: {raw}\nError: {err}\n"
                "Return only a corrected JSON object matching the schema."
            )
            reply = self._chain.invoke(repair)
            raw = reply["text"]
            tokens_in = _sum_tokens(tokens_in, reply["input_tokens"])
            tokens_out = _sum_tokens(tokens_out, reply["output_tokens"])
            try:
                parsed = schema.model_validate_json(_strip_fences(raw))
                status = STATUS_REPAIR
            except ValidationError:
                # Log the failure before propagating so the monitor sees it.
                self.monitor.record(
                    prompt_hash, time.perf_counter() - start, STATUS_FAIL,
                    tokens_in, tokens_out,
                )
                raise

        self.monitor.record(
            prompt_hash, time.perf_counter() - start, status, tokens_in, tokens_out
        )
        # Only validated replies are cached, so cache hits never need repair.
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"model": self.model, "prompt": full_prompt, "raw": raw})
        )
        return parsed

    def judge_is_ai(self, answer: str) -> IsAI:
        """Q0 — zero-shot: judge one answer alone, no context, no label.

        Args:
            answer: the text to judge.
        """
        prompt = load_prompt("eda/is_ai").format(answer=answer)
        return self._ask(prompt, IsAI)

    def judge_personality(self, context_answers: list[str], ai_answer: str) -> CapturesPersonality:
        """Q1 — does the AI answer capture the person shown in their own answers?

        Args:
            context_answers: the person's human answers to their context questions.
            ai_answer: the AI answer to one held-out question.
        """
        # Numbered blocks keep the examples visually separate for the judge.
        context = "\n\n".join(
            f"Example {i}:\n{text}" for i, text in enumerate(context_answers, start=1)
        )
        prompt = load_prompt("eda/personality").format(
            context_answers=context, ai_answer=ai_answer
        )
        return self._ask(prompt, CapturesPersonality)

    def judge_content(self, question: str, human_answer: str, ai_answer: str) -> CapturesContent:
        """Q2 — pairwise: does the AI answer carry the human answer's content?

        Args:
            question: the interview question text.
            human_answer: the person's real answer.
            ai_answer: the simulation's answer.
        """
        prompt = load_prompt("eda/content").format(
            question=question, human_answer=human_answer, ai_answer=ai_answer
        )
        return self._ask(prompt, CapturesContent)
