"""Generate SFT data from model rollouts instead of gold solutions.

Step 1: Generate rollouts on GSM8K train set using the base model (vLLM).
Step 2: Filter to correct rollouts only.
Step 3: Inject perturbations (arithmetic/copy errors) + backoff tokens.
Step 4: Save as SFT JSONL for Phase 1 training.

This preserves the model's natural reasoning style — SFT only teaches
the backoff token format, not a new reasoning policy.

Usage:
    # Full pipeline: generate rollouts + build SFT data
    python scripts/generate_sft_from_rollouts.py --model Qwen/Qwen3-1.7B

    # Use pre-generated rollouts (skip generation)
    python scripts/generate_sft_from_rollouts.py --rollouts data/rollouts_Qwen3-1.7B.jsonl

    # Use rollouts from a stronger model (distillation)
    python scripts/generate_sft_from_rollouts.py --model Qwen/Qwen3-4B-Thinking-2507
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

from src.data.gsm8k import load_gsm8k, extract_predicted_number
from src.data.synthetic import (
    make_wrong_step,
    format_chat,
    save_sft_dataset,
)
from src.prompt import build_prompt


# ── Step parsing for model rollouts ──


def parse_rollout_steps(text: str) -> list[str]:
    """Parse a model's CoT rollout into reasoning steps.

    Unlike gold solutions (which are terse 1-line steps), model rollouts
    are verbose multi-paragraph text. We split on double-newlines and
    sentence boundaries to get chunks that can be individually perturbed.

    Returns list of non-empty step strings.
    """
    # Remove <think> / </think> tags
    text = re.sub(r"</?think>", "", text)

    # Remove the final answer (everything after \boxed or ####)
    for pattern in [r"\\boxed\{[^}]*\}", r"####.*$"]:
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    text = text.strip()
    if not text:
        return []

    # Split on double-newlines first (paragraph boundaries)
    paragraphs = re.split(r"\n\n+", text)

    steps = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # For long paragraphs, further split on sentence-ending periods
        # followed by a space and capital letter (heuristic sentence boundary)
        if len(para) > 200:
            sentences = re.split(r"(?<=\.)\s+(?=[A-Z])", para)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    steps.append(sent)
        else:
            steps.append(para)

    return steps


def has_numbers(text: str) -> bool:
    """Check if text contains any numbers (needed for perturbation)."""
    return bool(re.search(r"\d+", text))


# ── Rollout generation ──


def generate_rollouts(
    model_name: str,
    dataset: list[dict],
    max_tokens: int = 8192,
    tp: int = 2,
    gpu_mem: float = 0.90,
) -> list[dict]:
    """Generate CoT rollouts for all problems using vLLM."""
    print(f"\nGenerating rollouts with {model_name}...")
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
        max_tokens=max_tokens,
        temperature=0.6,  # Qwen3 thinking temperature
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.3,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/len(dataset):.2f}s/problem)")

    rollouts = []
    for ex, output in zip(dataset, outputs):
        text = output.outputs[0].text
        predicted = extract_predicted_number(text)
        is_correct = predicted is not None and predicted == ex["answer_number"]

        rollouts.append({
            "question": ex["question"],
            "answer_number": ex["answer_number"],
            "rollout_text": text,
            "predicted": predicted,
            "correct": is_correct,
            "num_tokens": len(output.outputs[0].token_ids),
        })

    correct = sum(1 for r in rollouts if r["correct"])
    print(f"  Correct: {correct}/{len(rollouts)} ({100*correct/len(rollouts):.1f}%)")

    # Free GPU
    del llm
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    return rollouts


# ── SFT data construction from rollouts ──


def build_clean_from_rollout(question: str, rollout_text: str) -> str:
    """Build a clean SFT example directly from the model's rollout.

    Preserves the model's natural <think>...</think> and answer format.
    """
    # The rollout already has the model's natural format
    # Just ensure it has <think> wrapper
    text = rollout_text.strip()
    if not text.startswith("<think>"):
        text = "<think>\n" + text

    # Ensure </think> is present and answer follows
    if "</think>" not in text:
        text += "\n</think>"

    return text


def build_backoff_from_rollout(
    question: str,
    rollout_text: str,
    answer: str,
    rng: random.Random,
) -> str | None:
    """Build a backoff SFT example by perturbing a correct rollout.

    Injects 1-3 wrong steps into the model's natural reasoning,
    adds <backoff_N> + directive, then appends the original correct
    continuation.

    Returns None if the rollout can't be meaningfully perturbed.
    """
    steps = parse_rollout_steps(rollout_text)

    # Need at least 2 steps with numbers for perturbation
    numeric_steps = [(i, s) for i, s in enumerate(steps) if has_numbers(s)]
    if len(numeric_steps) < 2:
        return None
    if len(steps) < 2:
        return None

    # Decide how many wrong steps to inject (1, 2, or 3)
    max_wrong = min(3, len(numeric_steps) - 1)
    if max_wrong >= 3 and rng.random() < 0.15:
        num_wrong = 3
    elif max_wrong >= 2 and rng.random() < 0.35:
        num_wrong = 2
    else:
        num_wrong = 1

    backoff_token = f"<backoff_{num_wrong}>"

    # Pick where to inject the error (must be a numeric step, not the last)
    eligible = [(i, s) for i, s in numeric_steps if i < len(steps) - 1]
    if not eligible:
        return None

    # Pick the start of the error region
    latest_eligible = len(eligible) - num_wrong
    if latest_eligible < 0:
        num_wrong = len(eligible)
        latest_eligible = 0
        backoff_token = f"<backoff_{num_wrong}>"

    err_idx = rng.randint(0, max(0, latest_eligible))
    error_start = eligible[err_idx][0]  # index into steps[]

    # Generate wrong versions
    wrong_steps = []
    directive = ""
    for k in range(num_wrong):
        step_idx = error_start + k
        if step_idx >= len(steps):
            break
        wrong, d = make_wrong_step(steps[step_idx], rng)
        wrong_steps.append(wrong)
        directive = d  # use the last directive

    if not wrong_steps:
        return None

    # Assemble the backoff example
    parts = ["<think>"]

    # Steps before the error (from original rollout)
    for i in range(error_start):
        parts.append(steps[i])

    # Wrong steps
    for k, wrong in enumerate(wrong_steps):
        if k < len(wrong_steps) - 1:
            parts.append(wrong)
        else:
            parts.append(f"{wrong}\n{backoff_token} {directive}")

    # Correct continuation from error point (from original rollout)
    for i in range(error_start, len(steps)):
        parts.append(steps[i])

    parts.append("</think>")

    # Extract the original answer format from the rollout
    boxed_match = re.search(r"\\boxed\{[^}]+\}", rollout_text)
    if boxed_match:
        parts.append(boxed_match.group())
    else:
        parts.append(f"\\boxed{{{answer}}}")

    return "\n".join(parts)


def build_sft_dataset(
    rollouts: list[dict],
    backoff_ratio: float = 0.75,
    seed: int = 42,
) -> list[dict]:
    """Build SFT dataset from correct rollouts.

    Args:
        rollouts: list of rollout dicts (must have correct=True filtered)
        backoff_ratio: fraction of examples with backoff injection
        seed: random seed
    """
    rng = random.Random(seed)
    examples = []
    skipped = 0

    for r in rollouts:
        question = r["question"]
        answer = r["answer_number"]
        rollout_text = r["rollout_text"]

        use_backoff = rng.random() < backoff_ratio

        if use_backoff:
            content = build_backoff_from_rollout(
                question, rollout_text, answer, rng
            )
            if content is None:
                # Can't perturb — fall back to clean
                content = build_clean_from_rollout(question, rollout_text)
                use_backoff = False
                skipped += 1
        else:
            content = build_clean_from_rollout(question, rollout_text)

        examples.append({
            "question": question,
            "answer": answer,
            "messages": format_chat(question, content),
            "has_backoff": use_backoff,
        })

    rng.shuffle(examples)

    if skipped:
        print(f"  Skipped {skipped} perturbations (fell back to clean)")

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT data from model rollouts"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                        help="Model for rollout generation")
    parser.add_argument("--rollouts", default=None,
                        help="Pre-generated rollouts JSONL (skip generation)")
    parser.add_argument("--output", default="data/sft_rollout_train.jsonl")
    parser.add_argument("--rollouts-output", default=None,
                        help="Save rollouts to JSONL (default: data/rollouts_<model>.jsonl)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit number of GSM8K train problems")
    parser.add_argument("--backoff-ratio", type=float, default=0.75)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rollouts:
        # Load pre-generated rollouts
        print(f"Loading rollouts from {args.rollouts}...")
        with open(args.rollouts) as f:
            rollouts = [json.loads(line) for line in f]
        print(f"  Total: {len(rollouts)}")
    else:
        # Generate rollouts
        data = load_gsm8k("train", subset_size=args.subset)
        rollouts = generate_rollouts(
            args.model, data,
            max_tokens=args.max_tokens,
            tp=args.tp,
            gpu_mem=args.gpu_mem,
        )

        # Save rollouts for reuse
        if args.rollouts_output:
            rollouts_path = args.rollouts_output
        else:
            short_name = args.model.split("/")[-1]
            rollouts_path = f"data/rollouts_{short_name}.jsonl"
        Path(rollouts_path).parent.mkdir(parents=True, exist_ok=True)
        with open(rollouts_path, "w") as f:
            for r in rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Rollouts saved to {rollouts_path}")

    # Filter to correct rollouts only
    correct_rollouts = [r for r in rollouts if r["correct"]]
    print(f"\nCorrect rollouts: {len(correct_rollouts)}/{len(rollouts)} "
          f"({100*len(correct_rollouts)/len(rollouts):.1f}%)")

    if not correct_rollouts:
        print("ERROR: No correct rollouts found!")
        return

    # Token stats on correct rollouts
    token_counts = [r["num_tokens"] for r in correct_rollouts]
    arr = np.array(token_counts)
    print(f"  Token stats: mean={arr.mean():.0f}, median={np.median(arr):.0f}, "
          f"p95={np.percentile(arr, 95):.0f}, max={arr.max()}")

    # Build SFT dataset
    print(f"\nBuilding SFT dataset (backoff_ratio={args.backoff_ratio})...")
    examples = build_sft_dataset(
        correct_rollouts,
        backoff_ratio=args.backoff_ratio,
        seed=args.seed,
    )

    backoff_count = sum(1 for ex in examples if ex["has_backoff"])
    print(f"  Total examples: {len(examples)}")
    print(f"  Clean:   {len(examples) - backoff_count}")
    print(f"  Backoff: {backoff_count} ({100*backoff_count/len(examples):.1f}%)")

    # Save
    save_sft_dataset(examples, args.output)
    print(f"\nSaved to {args.output}")

    # Show a few examples
    print(f"\n{'='*60}")
    print("SAMPLE EXAMPLES")
    print(f"{'='*60}")
    for ex in examples[:3]:
        print(f"\n--- backoff={ex['has_backoff']} ---")
        content = ex["messages"][1]["content"]
        # Show first 500 chars
        print(content[:500])
        if len(content) > 500:
            print(f"  ... ({len(content)} chars total)")
        print()


if __name__ == "__main__":
    main()
