import re

from datasets import load_dataset


def load_gsm8k(split: str = "test", subset_size: int | None = None) -> list[dict]:
    """Load GSM8K and return a list of dicts with question, answer_number, full_answer."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if subset_size is not None:
        ds = ds.select(range(min(subset_size, len(ds))))

    examples = []
    for ex in ds:
        examples.append({
            "question": ex["question"],
            "answer_number": extract_answer_number(ex["answer"]),
            "full_answer": ex["answer"],
        })
    return examples


def extract_answer_number(text: str) -> str | None:
    """Extract the gold answer from GSM8K's #### format.

    Returns the number as a cleaned string, or None if not found.
    """
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def extract_predicted_number(text: str) -> str | None:
    r"""Extract predicted answer from model output.

    Looks after </think> (if present), only in the last 300 chars.
    Tries formats in order: \boxed{N}, #### N, then last number.
    """
    think_end = text.find("</think>")
    if think_end >= 0:
        text = text[think_end + 8:]

    # Only look at tail to avoid matching thinking-trace numbers
    if len(text) > 300:
        text = text[-300:]

    # Strict: \boxed{N} format (last match)
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        raw = matches[-1]
        # Strip LaTeX currency/formatting: \$, \, \text{...}, commas
        raw = raw.replace("\\$", "").replace("$", "").replace(",", "")
        raw = re.sub(r"\\text\{[^}]*\}", "", raw)
        raw = re.sub(r"\\[a-zA-Z]+", "", raw)  # strip \dfrac, \frac, etc.
        raw = re.sub(r"[{}]", "", raw)  # leftover braces
        raw = raw.strip()
        # Extract the first number from whatever remains
        num = re.search(r"-?\d+\.?\d*", raw)
        if num:
            return num.group()
        return raw if raw else None

    # Legacy: #### N format (last match)
    matches = re.findall(r"####\s*\$?(-?[\d,.]+)", text)
    if matches:
        return matches[-1].replace(",", "").replace("$", "").strip()

    # Flexible: last number in text
    matches = re.findall(r"(-?[\d,.]+)", text)
    for ans in reversed(matches):
        ans = ans.replace(",", "").strip().strip(".")
        if ans:
            return ans

    return None


def grade_answer(predicted: str | None, gold: str | None, tol: float = 1e-3) -> bool:
    """Compare predicted vs gold answer with float tolerance."""
    if predicted is None or gold is None:
        return False

    # Try exact string match first
    if predicted == gold:
        return True

    # Try numeric comparison with tolerance
    try:
        p = float(predicted)
        g = float(gold)
        return abs(p - g) <= tol
    except (ValueError, OverflowError):
        return False
