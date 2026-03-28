"""MATH-500 dataset (Lightman et al., 2023) loader and answer grading.

Uses the exact MATH-500 split from HuggingFaceH4/MATH-500 (same 500 problems
used by S²R and SCoRe for evaluation). Pre-extracted gold answers in the
`answer` field.

For training data generation, loads the full MATH train set from
EleutherAI/hendrycks_math (7 subjects, ~7.5k problems).

Usage:
    from src.data.math import load_math500, load_math_train, grade_math_answer

    # Eval on MATH-500
    problems = load_math500()

    # Training set for rollout generation
    problems = load_math_train(subset_size=1000)
"""

import re

from datasets import load_dataset


SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def normalize_math_answer(answer: str) -> str:
    """Normalize a MATH answer string for comparison."""
    s = answer.strip()
    s = re.sub(r"\\text(?:bf|it|rm|sf)?\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\!", "")
    s = s.replace(",", "")
    s = s.rstrip(".")
    s = s.replace(" ", "")
    return s


def grade_math_answer(predicted: str | None, gold: str | None) -> bool:
    """Compare predicted vs gold MATH answer.

    Uses normalized string comparison, then numeric, then fraction parsing.
    """
    if predicted is None or gold is None:
        return False

    p = normalize_math_answer(predicted)
    g = normalize_math_answer(gold)

    if p == g:
        return True

    # Numeric comparison
    try:
        pf = float(p)
        gf = float(g)
        return abs(pf - gf) < 1e-6
    except (ValueError, OverflowError):
        pass

    # Fraction: \frac{a}{b} -> a/b
    def parse_frac(s):
        m = re.match(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
        if m:
            return int(m.group(1)) / int(m.group(2))
        m = re.match(r"(-?\d+)/(-?\d+)", s)
        if m:
            return int(m.group(1)) / int(m.group(2))
        return None

    pv = parse_frac(p)
    gv = parse_frac(g)
    if pv is not None and gv is not None:
        return abs(pv - gv) < 1e-6

    return False


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract answer from \boxed{...} in model output, handling nested braces."""
    # Look after </think> if present
    think_end = text.find("</think>")
    if think_end >= 0:
        text = text[think_end:]

    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[start : i - 1].strip()


def load_math500() -> list[dict]:
    """Load the exact MATH-500 eval split (Lightman et al., 2023).

    Returns list of dicts with keys:
        question, answer, answer_number, level, subject, unique_id, full_solution
    """
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    for ex in ds:
        problems.append({
            "question": ex["problem"],
            "answer": ex["answer"],
            "answer_number": ex["answer"],  # compat with GSM8K interface
            "level": ex["level"],
            "subject": ex["subject"],
            "unique_id": ex["unique_id"],
            "full_solution": ex["solution"],
        })
    return problems


def load_math_train(subset_size: int | None = None, seed: int = 42) -> list[dict]:
    """Load MATH training set for rollout generation.

    Loads all 7 subjects from EleutherAI/hendrycks_math.

    Returns list of dicts with keys:
        question, answer, answer_number, level, type, full_solution
    """
    problems = []
    for subj in SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
        for ex in ds:
            # Extract gold from \boxed{} in solution
            gold = _extract_gold_from_solution(ex["solution"])
            problems.append({
                "question": ex["problem"],
                "answer": gold,
                "answer_number": gold,
                "level": ex["level"],
                "type": ex["type"],
                "full_solution": ex["solution"],
            })

    if subset_size is not None:
        import random
        rng = random.Random(seed)
        problems = rng.sample(problems, min(subset_size, len(problems)))

    return problems


def _extract_gold_from_solution(solution: str) -> str | None:
    r"""Extract gold answer from \boxed{} in a MATH solution string."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return solution[start : i - 1].strip()
