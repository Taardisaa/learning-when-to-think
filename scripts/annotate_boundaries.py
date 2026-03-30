"""Annotate semantic boundaries in CoT reasoning traces using Qwen3-32B.

Uses vLLM in non-thinking mode to identify where reasoning steps transition
in model-generated rollouts. Outputs character offsets for each boundary.

Usage:
    # Default: annotate 2000 rollouts from existing grouped rollouts
    python scripts/annotate_boundaries.py \
        --rollouts data/rollouts_grouped_math_Qwen3-1.7B.jsonl \
        --model Qwen/Qwen3-32B \
        --num-rollouts 2000

    # Smaller test run
    python scripts/annotate_boundaries.py \
        --rollouts data/rollouts_grouped_math_Qwen3-1.7B.jsonl \
        --model Qwen/Qwen3-32B \
        --num-rollouts 50
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm import LLM, SamplingParams


BOUNDARY_TAG = "<|seg|>"

ANNOTATION_PROMPT = """\
You are annotating semantic boundaries in a mathematical reasoning trace.
A semantic boundary is a natural transition point where the reasoning shifts to a distinct step.

Your task: reproduce the reasoning trace below EXACTLY, but insert the marker {boundary_tag} at every semantic boundary point.

Boundary types (insert {boundary_tag} here):
1. After a sentence-ending period where the next sentence starts a NEW reasoning step
2. Before a logical connective that introduces a new direction ("So now...", "Therefore...", "Wait...", "Let me...", "Next...", "However...")
3. Between problem setup → equation formulation → solving → verification → answer extraction
4. After a completed sub-calculation before its result is used downstream
5. After a paragraph break where the topic shifts

NOT a boundary (do NOT insert {boundary_tag}):
- Mid-sentence or mid-formula (e.g., inside "x^2 + 3x = 0")
- Between consecutive lines of the same algebraic simplification
- Inside a single logical step that hasn't concluded yet
- Between a connective and its clause (e.g., between "Therefore" and "x = 5")

Example:
Input: "First, x = 2 + 3 = 5. Now I need to find y. Since y = x^2, y = 25."
Output: "First, x = 2 + 3 = 5. {boundary_tag}Now I need to find y. {boundary_tag}Since y = x^2, y = 25."

Now reproduce the following trace with {boundary_tag} markers inserted at boundary points. Output ONLY the annotated text, nothing else.

{rollout_text}"""


def _build_annotation_prompt(tokenizer, rollout_text: str) -> str:
    """Build chat-formatted prompt for boundary annotation."""
    content = ANNOTATION_PROMPT.replace("{rollout_text}", rollout_text)
    content = content.replace("{boundary_tag}", BOUNDARY_TAG)
    messages = [{"role": "user", "content": content}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return prompt


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> wrapper, keep inner reasoning."""
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"\\boxed\{[^}]*\}", "", text)
    return text.strip()


def _parse_tagged_response(response: str) -> list[int]:
    """Extract boundary char offsets from LLM response containing <|seg|> tags.

    Strips the tags and computes the char offset in the original (untagged)
    text where each tag appeared.
    """
    tag = BOUNDARY_TAG
    offsets = []
    # Walk through the response, tracking position in the clean text
    clean_pos = 0
    i = 0
    while i < len(response):
        if response[i:i + len(tag)] == tag:
            offsets.append(clean_pos)
            i += len(tag)
        else:
            clean_pos += 1
            i += 1

    return offsets


def _select_rollouts(
    grouped_path: str, num_rollouts: int, seed: int,
    min_chars: int = 100, max_chars: int = 16000,
) -> list[dict]:
    """Load and sample rollouts from grouped JSONL.

    Returns flat list of {question, answer, text, correct, problem_idx, rollout_idx}.
    Samples across all problems, mixing correct and incorrect rollouts.
    """
    rng = random.Random(seed)

    with open(grouped_path) as f:
        groups = [json.loads(line) for line in f]

    # Flatten all rollouts with metadata
    all_rollouts = []
    for i, g in enumerate(groups):
        for j, r in enumerate(g["rollouts"]):
            all_rollouts.append({
                "question": g["question"],
                "answer": g["answer_number"],
                "text": _strip_think_tags(r["text"]),
                "correct": r["correct"],
                "problem_idx": i,
                "rollout_idx": j,
            })

    # Filter by length
    before = len(all_rollouts)
    all_rollouts = [r for r in all_rollouts
                    if min_chars <= len(r["text"]) <= max_chars]
    skipped = before - len(all_rollouts)
    if skipped:
        print(f"  Filtered {skipped} rollouts outside [{min_chars}, {max_chars}] chars")

    # Sample
    if num_rollouts < len(all_rollouts):
        all_rollouts = rng.sample(all_rollouts, num_rollouts)

    return all_rollouts


def annotate_boundaries(
    model_name: str,
    rollouts: list[dict],
    tp: int = 2,
    gpu_mem: float = 0.90,
) -> list[dict]:
    """Run Qwen3-32B to annotate boundaries on rollout texts."""
    max_model_len = 32768

    print(f"\nLoading {model_name} for boundary annotation...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts, skip those that won't fit (prompt + output must fit in context)
    # Output ≈ same length as input text (it's a copy with tags inserted)
    prompts = []
    valid_rollouts = []
    output_token_limits = []
    skipped = 0
    for r in rollouts:
        prompt = _build_annotation_prompt(tokenizer, r["text"])
        prompt_tokens = len(tokenizer.encode(prompt))
        # Output needs roughly as many tokens as the rollout text itself + margin for tags
        text_tokens = len(tokenizer.encode(r["text"]))
        out_tokens = int(text_tokens * 1.2) + 100  # 20% margin for tags
        if prompt_tokens + out_tokens > max_model_len:
            skipped += 1
            continue
        prompts.append(prompt)
        valid_rollouts.append(r)
        output_token_limits.append(out_tokens)

    if skipped:
        print(f"  Skipped {skipped} rollouts exceeding context limit")

    # Use the max output limit across all rollouts (vLLM uses shared params)
    max_out = max(output_token_limits) if output_token_limits else 4096
    params = SamplingParams(
        max_tokens=max_out,
        temperature=0.0,  # deterministic for annotation
    )

    print(f"  Annotating {len(prompts)} rollouts (max_output_tokens={max_out})...")
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/len(prompts):.2f}s/rollout)")
    rollouts = valid_rollouts

    # Parse responses — extract boundary positions from tagged text
    results = []
    parse_failures = 0
    for r, output in zip(rollouts, outputs):
        response = output.outputs[0].text
        offsets = _parse_tagged_response(response)

        if not offsets:
            parse_failures += 1

        # Filter offsets that are out of range
        text_len = len(r["text"])
        offsets = [o for o in offsets if 0 < o < text_len]

        results.append({
            "question": r["question"],
            "answer": r["answer"],
            "text": r["text"],
            "correct": r["correct"],
            "boundary_char_offsets": offsets,
            "num_boundaries": len(offsets),
            "problem_idx": r["problem_idx"],
            "rollout_idx": r["rollout_idx"],
            "raw_response": response,
        })

    print(f"  Parse failures: {parse_failures}/{len(rollouts)}")
    boundary_counts = [r["num_boundaries"] for r in results]
    if boundary_counts:
        import numpy as np
        arr = np.array(boundary_counts)
        print(f"  Boundaries per rollout: mean={arr.mean():.1f}, "
              f"median={np.median(arr):.0f}, min={arr.min()}, max={arr.max()}")

    # Free GPU
    del llm
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    return results


def _validate_against_heuristic(results: list[dict], tolerance: int = 10):
    """Compare LLM annotations against heuristic boundary detection.

    Reports agreement rate on obvious boundaries (punctuation + connectives).
    """
    from src.boundary import BoundaryTracker

    agreements = 0
    heuristic_total = 0
    llm_total = 0

    for r in results[:200]:  # Check first 200
        text = r["text"]
        llm_offsets = set(r["boundary_char_offsets"])
        llm_total += len(llm_offsets)

        # Run heuristic on the text character by character (approximate)
        tracker = BoundaryTracker(k_min=15, k_max=80)
        heuristic_offsets = set()
        char_idx = 0
        # Simple word-level simulation
        for word in re.split(r"(\s+)", text):
            if word.strip():
                is_b = tracker.step(word)
                if is_b:
                    heuristic_offsets.add(char_idx)
            char_idx += len(word)

        heuristic_total += len(heuristic_offsets)

        # Check agreement with tolerance (chars within ±tolerance)
        for h_off in heuristic_offsets:
            for l_off in llm_offsets:
                if abs(h_off - l_off) <= tolerance:
                    agreements += 1
                    break

    if heuristic_total > 0:
        print(f"\n  Heuristic validation (first 200 rollouts):")
        print(f"    Heuristic boundaries: {heuristic_total}")
        print(f"    LLM boundaries: {llm_total}")
        print(f"    Agreements (±{tolerance} chars): {agreements}/{heuristic_total} "
              f"({100*agreements/heuristic_total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate semantic boundaries in CoT reasoning traces"
    )
    parser.add_argument("--rollouts", required=True,
                        help="Path to grouped rollouts JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-32B",
                        help="Annotation model (default: Qwen/Qwen3-32B)")
    parser.add_argument("--output", default="data/boundary_annotations.jsonl",
                        help="Output path for annotated data")
    parser.add_argument("--num-rollouts", type=int, default=2000,
                        help="Number of rollouts to annotate")
    parser.add_argument("--tp", type=int, default=2,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--min-chars", type=int, default=100,
                        help="Skip rollouts shorter than this (chars)")
    parser.add_argument("--max-chars", type=int, default=16000,
                        help="Skip rollouts longer than this (chars, ~4000 tokens)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true",
                        help="Run heuristic validation after annotation")
    args = parser.parse_args()

    # Select rollouts
    print(f"Loading rollouts from {args.rollouts}...")
    rollouts = _select_rollouts(
        args.rollouts, args.num_rollouts, args.seed,
        min_chars=args.min_chars, max_chars=args.max_chars,
    )
    print(f"  Selected {len(rollouts)} rollouts")
    correct_pct = 100 * sum(1 for r in rollouts if r["correct"]) / len(rollouts)
    print(f"  Correct: {correct_pct:.1f}%, Incorrect: {100-correct_pct:.1f}%")

    # Annotate
    results = annotate_boundaries(
        args.model, rollouts,
        tp=args.tp, gpu_mem=args.gpu_mem,
    )

    # Validate against heuristic
    if args.validate:
        _validate_against_heuristic(results)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(results)} annotated rollouts to {args.output}")

    # Show sample
    print(f"\n{'='*60}")
    print("SAMPLE ANNOTATION")
    print(f"{'='*60}")
    sample = results[0]
    text = sample["text"]
    offsets = sample["boundary_char_offsets"]
    print(f"Question: {sample['question'][:100]}...")
    print(f"Boundaries: {len(offsets)}")
    print(f"\nText with boundaries marked [>>]:")
    marked = text
    for off in reversed(offsets):
        marked = marked[:off] + " [>>] " + marked[off:]
    print(marked[:1000])
    if len(marked) > 1000:
        print(f"  ... ({len(text)} chars total)")


if __name__ == "__main__":
    main()
