"""One-off probe: which Ollama cloud models work on this account's plan?

Sends each candidate model one tiny structured-output request through the
same client the EDA uses (src/llm.ask). Prints OK + latency, or the error —
quota/tier errors show up here instead of mid-run. Safe to re-run: replies
cache like everything else.

Run:  uv run python scripts/check_models.py

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import sys
import time
from pathlib import Path

from pydantic import BaseModel

# Running "python scripts/check_models.py" puts scripts/ on sys.path, not the
# repo root — add the root so "src" imports like everywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import ask  # noqa: E402  (needs the path fix above first)

# kimi-k3 and qwen3.5 dropped: not available on the free plan (probed).
CANDIDATES: list[str] = [
    "gpt-oss:120b",       # re-check: ran the EDA earlier — a fail now = quota, not tier
    "deepseek-v4-flash",  # confirmed working
    "glm-5.2",
    "mistral-large-3",
    "nemotron-3-super",
    "minimax-m3",
    "gemma4:31b",
]


class Probe(BaseModel):
    """Trivial schema: proves the model can follow a JSON schema at all."""

    answer: int
    reason: str


def main() -> None:
    """Probe every candidate and print a one-line verdict per model."""
    prompt = "What is 2 + 3? Answer with the number and a one-line reason."
    for model in CANDIDATES:
        start = time.time()
        try:
            result = ask(prompt, Probe, model=model)
            status = f"OK    answer={result.answer}  ({time.time() - start:.1f}s)"
        except Exception as err:  # any failure: auth, quota, unknown model
            status = f"FAIL  {type(err).__name__}: {str(err)[:120]}"
        print(f"{model:<22} {status}")


if __name__ == "__main__":
    main()
