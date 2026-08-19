"""Single gateway for all LLM calls in the EDA branch.

Provides:
    load_prompt(name)              -> prompt template text from user_prompts/
    ask(prompt, schema[, model])   -> validated Pydantic instance

Calls Ollama cloud (host, model, temperature, seed from config.yaml; API key
from OLLAMA_KIMIK3_KEY1 in .env) with format="json", appends the schema's JSON
Schema to the prompt, validates the reply, retries once with a repair prompt
on validation failure, and caches every raw reply as JSON on disk keyed on
(model, prompt hash) so re-runs are free and reproducible without keys.

NOTE (eval branch): judge models/families will change; only this client is
meant to be reused, not the single-model setup.

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import TypeVar

import yaml
from dotenv import load_dotenv
from ollama import Client
from pydantic import BaseModel, ValidationError

load_dotenv()  # makes OLLAMA_API_KEY available before the client is built

_CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())
LLM_CFG: dict = _CONFIG["llm"]
PROMPTS_DIR = Path(_CONFIG["paths"]["prompts"])
CACHE_DIR = Path(LLM_CFG["cache_dir"])

# One client for the whole run — built once at import, never per call.
_client = Client(
    host=LLM_CFG["host"],
    headers={"Authorization": f"Bearer {os.environ.get('OLLAMA_KIMIK3_KEY1', '')}"},
)

S = TypeVar("S", bound=BaseModel)


def load_prompt(name: str) -> str:
    """Return the prompt template for `name`, e.g. 'eda/is_ai'.

    Templates live in user_prompts/<name>.txt and contain {placeholders}
    filled by the caller with str.format. See user_prompts/README.md.
    """
    return (PROMPTS_DIR / f"{name}.txt").read_text()


def _cache_path(model: str, prompt: str) -> Path:
    """Deterministic cache file for one (model, prompt) pair."""
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()[:24]
    return CACHE_DIR / f"{key}.json"


def _call(prompt: str, model: str) -> str:
    """Raw chat call to Ollama: one user message, JSON output enforced."""
    response = _client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": LLM_CFG["temperature"], "seed": LLM_CFG["seed"]},
    )
    return response.message.content or ""


def _strip_fences(raw: str) -> str:
    """Remove a markdown code fence around a JSON reply, if present.

    Some models (e.g. glm-5.2) wrap their JSON in ```json … ``` despite
    format="json". Validation still guards the content — this only unwraps.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -len("```")]
    return text.strip()


def _schema_block(schema: type[BaseModel]) -> str:
    """Instruction appended to every prompt describing the required JSON."""
    return (
        "\n\nRespond with a single JSON object that matches this JSON Schema "
        "exactly (no extra keys, no surrounding text):\n"
        f"{json.dumps(schema.model_json_schema())}"
    )


def ask(prompt: str, schema: type[S], model: str | None = None) -> S:
    """Send `prompt` and return the reply parsed into `schema`.

    Args:
        prompt: fully formatted prompt text (schema instructions are appended
            here — prompt templates must not describe the output format).
        schema: Pydantic model class the reply must satisfy.
        model: Ollama model tag; defaults to config.yaml llm.model.

    Returns:
        A validated `schema` instance.

    Raises:
        pydantic.ValidationError: if the reply is still invalid after one
            repair attempt.
    """
    model = model or LLM_CFG["model"]
    full_prompt = prompt + _schema_block(schema)

    cache_file = _cache_path(model, full_prompt)
    if cache_file.exists():
        raw = json.loads(cache_file.read_text())["raw"]
        return schema.model_validate_json(_strip_fences(raw))

    raw = _call(full_prompt, model)
    try:
        parsed = schema.model_validate_json(_strip_fences(raw))
    except ValidationError as err:
        # One repair attempt: show the model its invalid reply and the error.
        repair = (
            full_prompt
            + f"\n\nYour previous reply was invalid.\nReply: {raw}\nError: {err}\n"
            "Return only a corrected JSON object matching the schema."
        )
        raw = _call(repair, model)
        parsed = schema.model_validate_json(_strip_fences(raw))  # second failure propagates

    # Only validated replies are cached, so cache hits never need repair.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"model": model, "prompt": full_prompt, "raw": raw}))
    return parsed
