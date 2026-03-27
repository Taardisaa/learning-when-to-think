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

def _make_arithmetic_directive(
    wrong_num: str, correct_num: str, rng: random.Random
) -> str:
    """Generate a specific directive referencing the actual arithmetic error."""
    templates = [
        f"{wrong_num} is wrong, it should be {correct_num}",
        f"that gives {wrong_num} but the correct result is {correct_num}",
        f"got {wrong_num}, recalculate — the answer is {correct_num}",
        f"error: {wrong_num} instead of {correct_num}, redo",
        f"the result should be {correct_num} not {wrong_num}",
    ]
    return rng.choice(templates)


def _make_copy_directive(
    wrong_num: str, correct_num: str, rng: random.Random
) -> str:
    """Generate a directive for a wrong-number-carried-forward error."""
    templates = [
        f"used {wrong_num} but should be {correct_num} from earlier",
        f"wrong value: {wrong_num}, the correct one is {correct_num}",
        f"carried {wrong_num} forward incorrectly, it's {correct_num}",
        f"should use {correct_num} not {wrong_num}",
        f"the value {wrong_num} is from the wrong step, use {correct_num}",
    ]
    return rng.choice(templates)


def _make_step_directive(wrong_step: str, correct_step: str, rng: random.Random) -> str:
    """Generate a directive that quotes part of the correct step as guidance.

    Used when we can't identify specific wrong/correct numbers (e.g., no
    numbers in the step). Gives the model a concrete target to aim for.
    """
    # Take first ~60 chars of correct step as a hint
    hint = correct_step[:60].strip()
    templates = [
        f"redo: {hint}",
        f"should be: {hint}",
        f"correct version: {hint}",
    ]
    return rng.choice(templates)


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
    """Create a wrong version of a step and a specific directive.

    Returns (wrong_step, directive) where directive references the actual
    wrong and correct values.
    """
    numbers = _find_numbers(step)
    if not numbers:
        # No numbers — swap a word to make it wrong, use step-level directive
        words = step.split()
        if len(words) > 3:
            i = rng.randint(1, len(words) - 2)
            wrong_step = " ".join(words[:i] + ["WRONG"] + words[i+1:])
        else:
            wrong_step = step + " (wrong)"
        return wrong_step, _make_step_directive(wrong_step, step, rng)

    error_type = rng.choice(["arithmetic", "copy"])

    if error_type == "arithmetic":
        idx = len(numbers) - 1
        start, end, num_str = numbers[idx]
        wrong_num = _perturb_number(num_str, rng)
        if wrong_num == num_str:
            wrong_num = str(int(float(num_str)) + rng.choice([3, -3, 7, -7]))
        wrong_step = step[:start] + wrong_num + step[end:]
        directive = _make_arithmetic_directive(wrong_num, num_str, rng)
    else:
        if len(numbers) >= 2:
            # Find pairs with different values for copy error
            target_idx = rng.randint(0, len(numbers) - 1)
            candidates = [
                i for i in range(len(numbers))
                if i != target_idx and numbers[i][2] != numbers[target_idx][2]
            ]
            if candidates:
                source_idx = rng.choice(candidates)
                start, end, correct_num = numbers[target_idx]
                _, _, wrong_num = numbers[source_idx]
                wrong_step = step[:start] + wrong_num + step[end:]
                directive = _make_copy_directive(wrong_num, correct_num, rng)
            else:
                # All numbers are the same — fall back to arithmetic error
                start, end, num_str = numbers[-1]
                wrong_num = _perturb_number(num_str, rng)
                if wrong_num == num_str:
                    wrong_num = str(int(float(num_str)) + rng.choice([3, -3, 7, -7]))
                wrong_step = step[:start] + wrong_num + step[end:]
                directive = _make_arithmetic_directive(wrong_num, num_str, rng)
        else:
            start, end, num_str = numbers[0]
            wrong_num = _perturb_number(num_str, rng)
            if wrong_num == num_str:
                wrong_num = str(int(float(num_str)) + 5)
            wrong_step = step[:start] + wrong_num + step[end:]
            directive = _make_arithmetic_directive(wrong_num, num_str, rng)

    # Final safety: if the wrong step is identical to the original,
    # force a visible arithmetic perturbation
    if wrong_step.strip() == step.strip():
        idx = len(numbers) - 1
        start, end, num_str = numbers[idx]
        forced_wrong = str(int(float(num_str)) + rng.choice([5, -5, 10, -10]))
        wrong_step = step[:start] + forced_wrong + step[end:]
        directive = _make_arithmetic_directive(forced_wrong, num_str, rng)

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
    """Build a backoff SFT example with injected error(s).

    Injects 1-3 wrong steps (depending on available steps), then backs off
    with the appropriate depth token to rewind past all wrong steps.

    Depth distribution:
      - <depth_1>: rewind 1 boundary (1 wrong step) — most common
      - <depth_2>: rewind 2 boundaries (2 wrong steps)
      - <depth_3>: rewind 3 boundaries (3 wrong steps)

    Returns None if there aren't enough steps for a meaningful backoff.
    """
    if len(steps) < 2:
        return None

    # Decide how many wrong steps to inject (1, 2, or 3)
    # Weighted toward 1: [60% depth_1, 25% depth_2, 15% depth_3]
    max_wrong = min(3, len(steps) - 1)  # leave at least 1 correct step at end
    if max_wrong >= 3 and rng.random() < 0.15:
        num_wrong = 3
    elif max_wrong >= 2 and rng.random() < 0.35:
        num_wrong = 2
    else:
        num_wrong = 1

    depth_token = f"<depth_{num_wrong}>"

    # Pick where the wrong steps start (not the last step)
    latest_start = len(steps) - 1 - num_wrong
    if latest_start < 0:
        latest_start = 0
        num_wrong = len(steps) - 1
        depth_token = f"<depth_{num_wrong}>"
    error_start = rng.randint(0, max(0, latest_start))

    # Generate wrong versions of the affected steps
    # Keep the directive from the last wrong step (most specific)
    wrong_steps = []
    directive = ""
    for i in range(num_wrong):
        wrong, d = make_wrong_step(steps[error_start + i], rng)
        wrong_steps.append(wrong)
        directive = d  # use the last one — most relevant to the backoff point

    # Build the example
    parts = ["<think>"]

    # Correct steps before the error
    for i in range(error_start):
        parts.append(f"{steps[i]} <continue>")

    # Wrong steps with <continue> between them
    for i, wrong in enumerate(wrong_steps):
        if i < len(wrong_steps) - 1:
            parts.append(f"{wrong} <continue>")
        else:
            # Last wrong step: followed by backoff
            parts.append(f"{wrong} <backoff> {depth_token} {directive} <continue>")

    # Correct continuation from the error point
    for i in range(error_start, len(steps)):
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
