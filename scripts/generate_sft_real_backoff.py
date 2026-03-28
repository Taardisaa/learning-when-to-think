"""Generate SFT data using real wrong rollouts (S²R-style).

Instead of synthetically perturbing numbers in correct rollouts, this script:
1. Generates K rollouts per problem (some correct, some wrong).
2. Stitches real wrong rollouts + backoff token + correct rollout.
3. Harder problems (lower pass rate) get more wrong attempts.

The model learns from its own actual failure patterns, not synthetic errors.

Usage:
    # MATH dataset (default) — full pipeline
    python scripts/generate_sft_real_backoff.py --model Qwen/Qwen3-1.7B --dataset math

    # GSM8K dataset
    python scripts/generate_sft_real_backoff.py --model Qwen/Qwen3-1.7B --dataset gsm8k

    # Use pre-generated grouped rollouts
    python scripts/generate_sft_real_backoff.py --rollouts data/rollouts_grouped_math_Qwen3-1.7B.jsonl

    # Dry-run on small subset
    python scripts/generate_sft_real_backoff.py --model Qwen/Qwen3-1.7B --subset 200
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams

from src.data.gsm8k import load_gsm8k, extract_predicted_number, grade_answer
from src.data.math import load_math_train, extract_boxed_answer, grade_math_answer
from src.data.synthetic import format_chat, save_sft_dataset, make_wrong_step
from src.prompt import build_prompt

# Import rollout-based builders from existing script
from scripts.generate_sft_from_rollouts import (
    build_clean_from_rollout,
    build_backoff_from_rollout,
    parse_rollout_steps,
    has_numbers,
)


# ── Directive generation for real backoff ──


def _extract_first_step(text: str) -> str:
    """Extract the first meaningful reasoning sentence from a rollout."""
    text = re.sub(r"</?think>", "", text).strip()
    # Split into sentences/paragraphs, skip filler
    for line in re.split(r"\n+", text):
        line = line.strip()
        if len(line) > 20 and not line.lower().startswith(("okay", "let me", "hmm", "so,", "alright")):
            return line[:120]
    # Fallback: just take first substantial line
    for line in re.split(r"\n+", text):
        line = line.strip()
        if len(line) > 20:
            return line[:120]
    return ""


def _make_real_directive(
    wrong_predicted: str | None,
    correct_rollout: str,
    gold_answer: str,
    attempt_num: int,
    rng: random.Random,
) -> str:
    """Build a semantically meaningful directive from real rollout data.

    Uses the wrong prediction, the gold answer, and a hint from the
    correct rollout's opening to guide the retry.
    """
    hint = _extract_first_step(correct_rollout)

    # Build the directive from concrete information
    parts = []

    # Clean up LaTeX artifacts in predicted answer
    if wrong_predicted:
        wrong_predicted = re.sub(r"[\\!,]", "", wrong_predicted).strip()

    # Part 1: what went wrong (reference the wrong answer)
    if wrong_predicted and wrong_predicted != gold_answer:
        wrong_refs = [
            f"That gives {wrong_predicted}, which is wrong.",
            f"I got {wrong_predicted}, but that can't be right.",
            f"The result {wrong_predicted} doesn't match — I made an error.",
        ]
        parts.append(rng.choice(wrong_refs))
    else:
        parts.append("That reasoning has an error.")

    # Part 2: what to do differently (hint from correct rollout)
    if hint:
        hint_refs = [
            f"Instead: {hint}",
            f"Let me restart: {hint}",
            f"Try this: {hint}",
        ]
        parts.append(rng.choice(hint_refs))

    return " ".join(parts)


# ── Multi-sample rollout generation ──


def _gsm8k_extract_and_grade(text: str, gold: str) -> tuple[str | None, bool]:
    """Extract and grade for GSM8K (numeric answers)."""
    pred = extract_predicted_number(text)
    return pred, grade_answer(pred, gold)


def _math_extract_and_grade(text: str, gold: str) -> tuple[str | None, bool]:
    """Extract and grade for MATH (LaTeX boxed answers)."""
    pred = extract_boxed_answer(text)
    return pred, grade_math_answer(pred, gold)


def generate_grouped_rollouts(
    model_name: str,
    dataset: list[dict],
    num_samples: int = 8,
    max_tokens: int = 8192,
    tp: int = 2,
    gpu_mem: float = 0.90,
    extract_and_grade=_gsm8k_extract_and_grade,
) -> list[dict]:
    """Generate K rollouts per problem and group them with pass rates."""
    print(f"\nGenerating {num_samples} rollouts/problem with {model_name}...")
    print(f"  Problems: {len(dataset)}")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        max_model_len=16384,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
    )
    tokenizer = llm.get_tokenizer()

    prompts = []
    for ex in dataset:
        prompts.append(build_prompt(tokenizer, ex["question"]))

    params = SamplingParams(
        n=num_samples,
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.3,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/len(dataset):.2f}s/problem)")

    grouped = []
    total_correct = 0
    total_rollouts = 0

    for ex, output in zip(dataset, outputs):
        rollouts = []
        for completion in output.outputs:
            text = completion.text
            predicted, is_correct = extract_and_grade(text, ex["answer_number"])
            rollouts.append({
                "text": text,
                "predicted": predicted,
                "correct": is_correct,
                "num_tokens": len(completion.token_ids),
            })

        num_correct = sum(1 for r in rollouts if r["correct"])
        total_correct += num_correct
        total_rollouts += len(rollouts)

        grouped.append({
            "question": ex["question"],
            "answer_number": ex["answer_number"],
            "rollouts": rollouts,
            "pass_rate": num_correct / len(rollouts),
        })

    print(f"  Overall pass rate: {total_correct}/{total_rollouts} "
          f"({100*total_correct/total_rollouts:.1f}%)")

    # Difficulty distribution
    n_trivial = sum(1 for g in grouped if g["pass_rate"] == 1.0)
    n_easy = sum(1 for g in grouped if 0.75 <= g["pass_rate"] < 1.0)
    n_medium = sum(1 for g in grouped if 0.25 <= g["pass_rate"] < 0.75)
    n_hard = sum(1 for g in grouped if 0 < g["pass_rate"] < 0.25)
    n_impossible = sum(1 for g in grouped if g["pass_rate"] == 0.0)
    print(f"  Difficulty: trivial={n_trivial} easy={n_easy} medium={n_medium} "
          f"hard={n_hard} impossible={n_impossible}")

    # Free GPU
    del llm
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    return grouped


# ── Stitching real wrong + correct rollouts ──


def _strip_rollout(text: str) -> str:
    """Strip <think>, </think>, and \\boxed{...} from a rollout, leaving raw reasoning."""
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"\\boxed\{[^}]*\}", "", text)
    # Also strip any backoff tokens that might have leaked in
    text = re.sub(r"<backoff_[123]>", "", text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (chars / 3.5 for English/math)."""
    return len(text) // 4 + 1


def _truncate_rollout(text: str, max_tokens: int) -> str:
    """Truncate a rollout at the last sentence boundary before max_tokens."""
    if _estimate_tokens(text) <= max_tokens:
        return text

    # Approximate char limit
    char_limit = max_tokens * 4
    truncated = text[:char_limit]

    # Find last sentence boundary
    for pattern in [r"\.\s+", r"\n", r"\.\s*$"]:
        match = None
        for m in re.finditer(pattern, truncated):
            match = m
        if match and match.end() > len(truncated) // 2:
            return truncated[:match.end()].strip()

    return truncated.strip()


def build_backoff_from_real_rollouts(
    question: str,
    wrong_rollout_records: list[dict],
    correct_rollout: str,
    gold_answer: str,
    num_wrong_attempts: int,
    rng: random.Random,
    token_budget: int = 8192,
) -> str | None:
    """Stitch real wrong rollout(s) + backoff + correct rollout.

    Args:
        question: the math problem text
        wrong_rollout_records: list of dicts with "text" and "predicted" keys
        correct_rollout: a correct rollout text
        gold_answer: the ground truth answer string
        num_wrong_attempts: how many wrong attempts to include (1 or 2)
        rng: random state
        token_budget: max approximate tokens for the full sequence

    Returns:
        Stitched text or None if can't build a valid example.
    """
    if len(wrong_rollout_records) < num_wrong_attempts:
        num_wrong_attempts = len(wrong_rollout_records)
    if num_wrong_attempts == 0:
        return None

    # Select distinct wrong rollouts
    selected_wrong = rng.sample(wrong_rollout_records, num_wrong_attempts)

    # Strip wrong rollouts (remove tags/boxed), keep predictions
    wrong_predictions = [r["predicted"] for r in selected_wrong]
    stripped_wrong = [_strip_rollout(r["text"]) for r in selected_wrong]

    # Filter out very short wrong rollouts
    stripped_wrong = [w for w in stripped_wrong if len(w) >= 50]
    if not stripped_wrong:
        return None
    num_wrong_attempts = len(stripped_wrong)

    # Prepare correct rollout — keep </think> and \boxed{} tail
    correct_text = correct_rollout.strip()
    if correct_text.startswith("<think>"):
        correct_text = correct_text[len("<think>"):].strip()
    # correct_text still has </think>\boxed{...} at the end

    # Budget: estimate tokens and truncate if needed
    overhead = 50  # <think>, backoff tokens, directives
    correct_tokens = _estimate_tokens(correct_text)
    available_for_wrong = token_budget - correct_tokens - overhead

    if available_for_wrong < 100:
        # Correct rollout alone is too long
        return None

    per_wrong_budget = available_for_wrong // num_wrong_attempts

    # Truncate wrong rollouts if needed
    for i in range(len(stripped_wrong)):
        stripped_wrong[i] = _truncate_rollout(stripped_wrong[i], per_wrong_budget)

    # If after truncation + budget check we'd still overflow, drop to 1 wrong
    total_wrong_tokens = sum(_estimate_tokens(w) for w in stripped_wrong)
    if total_wrong_tokens + correct_tokens + overhead > token_budget and len(stripped_wrong) > 1:
        stripped_wrong = [stripped_wrong[0]]
        stripped_wrong[0] = _truncate_rollout(stripped_wrong[0], available_for_wrong)
        num_wrong_attempts = 1

    # Assemble
    parts = ["<think>"]

    for i, wrong_text in enumerate(stripped_wrong):
        parts.append(wrong_text)
        directive = _make_real_directive(
            wrong_predicted=wrong_predictions[i],
            correct_rollout=correct_rollout,
            gold_answer=gold_answer,
            attempt_num=i,
            rng=rng,
        )
        parts.append(f"<backoff_3> {directive}")

    parts.append(correct_text)

    return "\n\n".join(parts)


# ── Dataset construction ──


def build_sft_dataset(
    grouped_rollouts: list[dict],
    backoff_ratio: float = 0.75,
    synthetic_ratio: float = 0.25,
    max_wrong_attempts: int = 2,
    seed: int = 42,
) -> list[dict]:
    """Build SFT dataset from grouped rollouts.

    Mix:
        - backoff_ratio * (1 - synthetic_ratio): real wrong rollout backoff
        - backoff_ratio * synthetic_ratio: synthetic perturbation backoff
        - (1 - backoff_ratio): clean examples

    Args:
        grouped_rollouts: list of per-problem dicts with rollouts + pass_rate
        backoff_ratio: fraction of examples that should have backoff
        synthetic_ratio: of the backoff examples, fraction using synthetic perturbation
        max_wrong_attempts: max wrong attempts per trajectory (capped at 2)
        seed: random seed
    """
    rng = random.Random(seed)
    examples = []
    stats = {"clean": 0, "real_backoff": 0, "synthetic_backoff": 0,
             "skipped_impossible": 0, "skipped_no_wrong": 0,
             "real_fallback_to_synthetic": 0, "real_fallback_to_clean": 0,
             "wrong_attempts_1": 0, "wrong_attempts_2": 0}

    for group in grouped_rollouts:
        question = group["question"]
        answer = group["answer_number"]
        pass_rate = group["pass_rate"]
        rollouts = group["rollouts"]

        correct_rollouts = [r for r in rollouts if r["correct"]]
        wrong_rollouts = [r for r in rollouts if not r["correct"]]

        # Skip impossible problems (no correct rollout)
        if not correct_rollouts:
            stats["skipped_impossible"] += 1
            continue

        # Pick a correct rollout for this example
        correct_r = rng.choice(correct_rollouts)

        # Decide example type
        has_wrong = len(wrong_rollouts) > 0

        if not has_wrong:
            # No wrong rollouts — nothing to teach, skip
            stats["skipped_no_wrong"] += 1
            continue

        # Real wrong rollout backoff
        num_wrong = _decide_num_wrong(pass_rate, max_wrong_attempts, rng)
        content = build_backoff_from_real_rollouts(
            question,
            wrong_rollouts,
            correct_r["text"],
            answer,
            num_wrong,
            rng,
        )
        if content is None:
            stats["real_fallback_to_clean"] += 1
            continue

        backoff_type = "real"
        stats["real_backoff"] += 1
        if num_wrong == 1:
            stats["wrong_attempts_1"] += 1
        else:
            stats["wrong_attempts_2"] += 1

        examples.append({
            "question": question,
            "answer": answer,
            "messages": format_chat(question, content),
            "has_backoff": backoff_type is not None,
            "backoff_type": backoff_type,
        })

    rng.shuffle(examples)
    return examples, stats


def _decide_num_wrong(
    pass_rate: float, max_wrong: int, rng: random.Random
) -> int:
    """Decide number of wrong attempts based on difficulty."""
    max_wrong = min(max_wrong, 2)  # cap at b_max=2

    if pass_rate >= 0.75:
        return 1
    elif pass_rate >= 0.25:
        return rng.choice([1, 2]) if max_wrong >= 2 else 1
    else:
        return max_wrong


# ── Main ──


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT data using real wrong rollouts (S²R-style)"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                        help="Model for rollout generation")
    parser.add_argument("--dataset", default="math", choices=["gsm8k", "math"],
                        help="Training dataset (default: math)")
    parser.add_argument("--rollouts", default=None,
                        help="Pre-generated grouped rollouts JSONL")
    parser.add_argument("--output", default=None,
                        help="Output SFT JSONL (default: auto)")
    parser.add_argument("--rollouts-output", default=None,
                        help="Save grouped rollouts JSONL")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit number of train problems")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="K rollouts per problem")
    parser.add_argument("--backoff-ratio", type=float, default=0.75)
    parser.add_argument("--synthetic-ratio", type=float, default=0.25,
                        help="Fraction of backoff examples using synthetic perturbation")
    parser.add_argument("--max-wrong-attempts", type=int, default=2)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Select dataset and grading function
    if args.dataset == "math":
        extract_and_grade = _math_extract_and_grade
        dataset_tag = "math"
    else:
        extract_and_grade = _gsm8k_extract_and_grade
        dataset_tag = "gsm8k"

    if args.output is None:
        args.output = f"data/sft_real_backoff_{dataset_tag}_train.jsonl"

    if args.rollouts:
        print(f"Loading grouped rollouts from {args.rollouts}...")
        with open(args.rollouts) as f:
            grouped = [json.loads(line) for line in f]
        print(f"  Total problems: {len(grouped)}")
    else:
        if args.dataset == "math":
            data = load_math_train(subset_size=args.subset)
        else:
            data = load_gsm8k("train", subset_size=args.subset)
        print(f"Dataset: {args.dataset}, {len(data)} problems")

        grouped = generate_grouped_rollouts(
            args.model, data,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            tp=args.tp,
            gpu_mem=args.gpu_mem,
            extract_and_grade=extract_and_grade,
        )

        # Save grouped rollouts for reuse
        if args.rollouts_output:
            rollouts_path = args.rollouts_output
        else:
            short_name = args.model.split("/")[-1]
            rollouts_path = f"data/rollouts_grouped_{dataset_tag}_{short_name}.jsonl"
        Path(rollouts_path).parent.mkdir(parents=True, exist_ok=True)
        with open(rollouts_path, "w") as f:
            for g in grouped:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"  Grouped rollouts saved to {rollouts_path}")

    # Build SFT dataset
    print(f"\nBuilding SFT dataset...")
    print(f"  backoff_ratio={args.backoff_ratio}, "
          f"synthetic_ratio={args.synthetic_ratio}, "
          f"max_wrong_attempts={args.max_wrong_attempts}")

    examples, stats = build_sft_dataset(
        grouped,
        backoff_ratio=args.backoff_ratio,
        synthetic_ratio=args.synthetic_ratio,
        max_wrong_attempts=args.max_wrong_attempts,
        seed=args.seed,
    )

    # Print stats
    total = len(examples)
    print(f"\n  Total examples: {total}")
    print(f"  Clean:             {stats['clean']:5d} ({100*stats['clean']/max(total,1):.1f}%)")
    print(f"  Real backoff:      {stats['real_backoff']:5d} ({100*stats['real_backoff']/max(total,1):.1f}%)")
    print(f"  Synthetic backoff: {stats['synthetic_backoff']:5d} ({100*stats['synthetic_backoff']/max(total,1):.1f}%)")
    if stats["real_backoff"] > 0:
        print(f"    1 wrong attempt: {stats['wrong_attempts_1']}")
        print(f"    2 wrong attempts:{stats['wrong_attempts_2']}")
    print(f"  Skipped (no correct): {stats['skipped_impossible']}")
    print(f"  No wrong rollouts:    {sum(1 for g in grouped if g['pass_rate'] == 1.0)}")
    print(f"  Fallback synthetic:   {stats['real_fallback_to_synthetic']}")
    print(f"  Fallback clean:       {stats['real_fallback_to_clean']}")

    # Token length stats
    lengths = [len(ex["messages"][1]["content"]) // 4 for ex in examples]
    arr = np.array(lengths)
    print(f"\n  Token estimate: mean={arr.mean():.0f}, median={np.median(arr):.0f}, "
          f"p95={np.percentile(arr, 95):.0f}, max={arr.max()}")

    # Save
    save_sft_dataset(examples, args.output)
    print(f"\nSaved to {args.output}")

    # Show sample examples
    print(f"\n{'='*60}")
    print("SAMPLE EXAMPLES")
    print(f"{'='*60}")

    for btype in ["real", "synthetic", None]:
        matches = [ex for ex in examples if ex.get("backoff_type") == btype]
        if matches:
            ex = matches[0]
            label = btype or "clean"
            print(f"\n--- {label} (backoff={ex['has_backoff']}) ---")
            content = ex["messages"][1]["content"]
            print(content[:800])
            if len(content) > 800:
                print(f"  ... ({len(content)} chars total)")
            print()


if __name__ == "__main__":
    main()
