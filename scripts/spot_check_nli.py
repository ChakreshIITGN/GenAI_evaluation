"""One-off: show the raw NLI labels for one pair, sentence by sentence.

Human verification of Layer 2b: pick a (person, question), print every
human sentence with its coverage-direction label and every AI sentence with
its fidelity-direction label, so the entailment calls can be eyeballed
against the actual texts. No LLM calls; local NLI model only.

Run:  uv run python scripts/spot_check_nli.py c_human 8

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""

import sys
from pathlib import Path

# Running "python scripts/spot_check_nli.py" puts scripts/ on sys.path, not
# the repo root — add the root so "src" imports like everywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.entail import check  # noqa: E402
from src.load import load_answers  # noqa: E402
from src.segment import split_sentences  # noqa: E402

CONFIG: dict = yaml.safe_load(Path("config.yaml").read_text())


def main() -> None:
    """Print per-sentence NLI labels for the pair given on the command line."""
    person_id = sys.argv[1] if len(sys.argv) > 1 else "c_human"
    question_id = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    df = load_answers(CONFIG["paths"]["data"])
    pair = df[(df["person_id"] == person_id) & (df["question_id"] == question_id)].iloc[0]

    human_units = split_sentences(pair["human_text"])
    ai_units = split_sentences(pair["ai_text"])

    print(f"=== {person_id} q{question_id} — {pair['category']}\n")
    print(f"HUMAN answer:\n{pair['human_text']}\n")
    print(f"AI answer:\n{pair['ai_text']}\n")

    print("COVERAGE direction — is each HUMAN sentence entailed by the AI answer?")
    for unit, label in zip(human_units, check(pair["ai_text"], human_units)):
        print(f"  [{label:^13}] {unit}")

    print("\nFIDELITY direction — is each AI sentence entailed by the HUMAN answer?")
    for unit, label in zip(ai_units, check(pair["human_text"], ai_units)):
        print(f"  [{label:^13}] {unit}")


if __name__ == "__main__":
    main()
