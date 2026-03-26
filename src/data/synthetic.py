"""Rule-based synthetic SFT data generation for backoff training.

Generates two types of examples from GSM8K gold solutions:
- Clean (60%): steps with <continue> at boundaries, </think> + answer
- Backoff (40%): inject wrong step, <backoff> <depth_1> [directive] <continue>,
  then correct continuation
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from src.data.gsm8k import load_gsm8k

# ── Directives for backoff examples ──

ARITHMETIC_DIRECTIVES = [
    "The calculation was wrong, redo carefully.",
    "Check the arithmetic again.",
    "That math is incorrect, recalculate.",
    "Wrong result, try the computation again.",
    "The numbers don't add up, redo this step.",
]

COPY_DIRECTIVES = [
    "Used the wrong number from a previous step, check again.",
    "That references an incorrect value, look back at the earlier steps.",
    "Wrong number carried forward, recheck.",
    "The value from the prior step was copied incorrectly.",
    "Go back and use the correct number.",
]

GENERIC_DIRECTIVES = [
    "That step has an error, redo it.",
    "Something went wrong, try again.",
    "This reasoning is off, reconsider.",
    "Let me rethink this step.",
    "That doesn't look right, redo.",
]


def parse_steps(answer_text: str) -> list[str]:
    """Parse a GSM8K gold solution into individual reasoning steps.

    Each step is typically one sentence with an embedded <<expr=result>>result.
    Steps are split on newlines (primary) and then on sentence-ending periods
    if a line contains multiple sentences.
    """
    # Remove the #### answer line
    lines = answer_text.strip().split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("####"):
            continue
        # Remove <<expr=result>> annotations for cleaner text
        # but keep the result number
        cleaned = re.sub(r"<<[^>]*>>", "", line)
        cleaned = cleaned.strip()
        if cleaned:
            steps.append(cleaned)
    return steps


def _find_numbers(text: str) -> list[tuple[int, int, str]]:
    """Find all numbers in text. Returns [(start, end, number_str), ...]."""
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"-?\d+\.?\d*", text)]


def _perturb_number(num_str: str, rng: random.Random) -> str:
    """Create a wrong version of a number (arithmetic error)."""
    try:
        val = float(num_str)
    except ValueError:
        return num_str

    if val == 0:
        return str(rng.randint(1, 10))

    # Pick a perturbation strategy
    strategy = rng.choice(["off_by_one", "double_half", "digit_swap", "random_near"])

    if strategy == "off_by_one":
        delta = rng.choice([-1, 1, -2, 2])
        result = val + delta
    elif strategy == "double_half":
        result = val * 2 if rng.random() < 0.5 else val / 2
    elif strategy == "digit_swap" and len(num_str) >= 2 and "." not in num_str:
        digits = list(num_str.lstrip("-"))
        if len(digits) >= 2:
            i, j = rng.sample(range(len(digits)), 2)
            digits[i], digits[j] = digits[j], digits[i]
            result_str = "".join(digits)
            if num_str.startswith("-"):
                result_str = "-" + result_str
            return result_str
        result = val + rng.choice([-3, 3, -5, 5])
    else:  # random_near
        magnitude = max(1, abs(val) * 0.3)
        result = val + rng.uniform(-magnitude, magnitude)

    # Format to match original style (int vs float)
    if "." in num_str:
        return f"{result:.{len(num_str.split('.')[1])}f}"
    return str(int(result))


def make_wrong_step(step: str, rng: random.Random) -> tuple[str, str]:
    """Create a wrong version of a step and pick a directive.

    Returns (wrong_step, directive).
    """
    numbers = _find_numbers(step)
    if not numbers:
        # No numbers to perturb — use generic error
        return step + " (this seems off)", rng.choice(GENERIC_DIRECTIVES)

    # Pick error type
    error_type = rng.choice(["arithmetic", "copy"])

    if error_type == "arithmetic" and len(numbers) >= 1:
        # Perturb the last number in the step (typically the result)
        idx = len(numbers) - 1
        start, end, num_str = numbers[idx]
        wrong_num = _perturb_number(num_str, rng)
        # Avoid no-op perturbation
        if wrong_num == num_str:
            wrong_num = str(int(float(num_str)) + rng.choice([3, -3, 7, -7]))
        wrong_step = step[:start] + wrong_num + step[end:]
        directive = rng.choice(ARITHMETIC_DIRECTIVES)
    else:
        # Copy error: replace a number with one from elsewhere in the step
        if len(numbers) >= 2:
            target_idx = rng.randint(0, len(numbers) - 1)
            source_idx = rng.choice(
                [i for i in range(len(numbers)) if i != target_idx]
            )
            start, end, _ = numbers[target_idx]
            _, _, source_num = numbers[source_idx]
            wrong_step = step[:start] + source_num + step[end:]
            directive = rng.choice(COPY_DIRECTIVES)
        else:
            # Only one number — fall back to arithmetic error
            start, end, num_str = numbers[0]
            wrong_num = _perturb_number(num_str, rng)
            if wrong_num == num_str:
                wrong_num = str(int(float(num_str)) + 5)
            wrong_step = step[:start] + wrong_num + step[end:]
            directive = rng.choice(ARITHMETIC_DIRECTIVES)

    return wrong_step, directive


def build_clean_example(
    question: str, steps: list[str], answer: str
) -> str:
    """Build a clean SFT example (no backoff).

    Format:
        <think>
        [step 1] <continue>
        [step 2] <continue>
        ...
        [final step] </think>
        #### [answer]
    """
    parts = ["<think>"]
    for i, step in enumerate(steps):
        if i < len(steps) - 1:
            parts.append(f"{step} <continue>")
        else:
            parts.append(f"{step} </think>")
    parts.append(f"#### {answer}")
    return "\n".join(parts)


def build_backoff_example(
    question: str,
    steps: list[str],
    answer: str,
    rng: random.Random,
) -> str | None:
    """Build a backoff SFT example with an injected error.

    Picks a step to corrupt, injects the error, then adds:
        <backoff> <depth_1> [directive] <continue>
    followed by the correct continuation.

    Returns None if there aren't enough steps for a meaningful backoff.
    """
    if len(steps) < 2:
        return None

    # Pick which step to make wrong (not the first or last)
    if len(steps) == 2:
        error_idx = 0
    else:
        error_idx = rng.randint(0, len(steps) - 2)

    wrong_step, directive = make_wrong_step(steps[error_idx], rng)

    parts = ["<think>"]
    for i in range(error_idx):
        parts.append(f"{steps[i]} <continue>")

    # Wrong step followed by backoff
    parts.append(f"{wrong_step} <backoff> <depth_1> {directive} <continue>")

    # Correct continuation from the error point
    for i in range(error_idx, len(steps)):
        if i < len(steps) - 1:
            parts.append(f"{steps[i]} <continue>")
        else:
            parts.append(f"{steps[i]} </think>")

    parts.append(f"#### {answer}")
    return "\n".join(parts)


def format_chat(question: str, assistant_content: str) -> list[dict]:
    """Format as Qwen3.5 chat messages."""
    return [
        {
            "role": "user",
            "content": (
                "Solve the following math problem. "
                "Give the final answer after ####.\n\n" + question
            ),
        },
        {"role": "assistant", "content": assistant_content},
    ]


def generate_sft_dataset(
    split: str = "train",
    subset_size: int | None = None,
    backoff_ratio: float = 0.4,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic SFT dataset from GSM8K.

    Returns a list of dicts with keys: question, answer, messages, has_backoff.
    """
    rng = random.Random(seed)
    data = load_gsm8k(split, subset_size)
    examples = []

    for ex in data:
        steps = parse_steps(ex["full_answer"])
        answer = ex["answer_number"]
        question = ex["question"]

        if not steps or answer is None:
            continue

        use_backoff = rng.random() < backoff_ratio

        if use_backoff:
            content = build_backoff_example(question, steps, answer, rng)
            if content is None:
                # Not enough steps — fall back to clean
                content = build_clean_example(question, steps, answer)
                use_backoff = False
        else:
            content = build_clean_example(question, steps, answer)

        examples.append({
            "question": question,
            "answer": answer,
            "messages": format_chat(question, content),
            "has_backoff": use_backoff,
        })

    rng.shuffle(examples)
    return examples


def save_sft_dataset(examples: list[dict], path: str | Path) -> None:
    """Save SFT dataset as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
