"""Evaluate a model on MATH-500 (Lightman et al., 2023).

Usage:
    # Base model
    python -m scripts.eval_math500 --model Qwen/Qwen3-1.7B --n 500

    # With LoRA checkpoint
    python -m scripts.eval_math500 --model Qwen/Qwen3-1.7B --checkpoint checkpoints/phase1/final --n 500

    # Quick test
    python -m scripts.eval_math500 --model Qwen/Qwen3-1.7B --n 50
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from vllm import LLM, SamplingParams

from src.data.math import load_math500, extract_boxed_answer, grade_math_answer
from src.prompt import build_prompt


def main():
    parser = argparse.ArgumentParser(description="Evaluate on MATH-500")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", default=None,
                        help="LoRA checkpoint to merge (optional)")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of problems (default: all 500)")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--output", default=None,
                        help="Output JSON (default: auto-generated)")
    args = parser.parse_args()

    # Load problems
    problems = load_math500()
    if args.n < 500:
        problems = problems[: args.n]
    print(f"MATH-500: {len(problems)} problems")

    # Load model
    lora_request = None
    if args.checkpoint:
        from vllm.lora.request import LoRARequest
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=args.tp,
            max_model_len=16384,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_mem,
            enable_lora=True,
            max_lora_rank=64,
        )
        lora_request = LoRARequest("adapter", 1, args.checkpoint)
        print(f"Model: {args.model} + LoRA from {args.checkpoint}")
    else:
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=args.tp,
            max_model_len=16384,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_mem,
        )
        print(f"Model: {args.model} (base)")

    tokenizer = llm.get_tokenizer()

    # Build prompts
    prompts = [build_prompt(tokenizer, p["question"]) for p in problems]

    if args.temperature == 0:
        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            top_k=20,
            repetition_penalty=args.repetition_penalty,
        )

    # Generate
    print(f"Generating (temp={args.temperature}, max_tokens={args.max_tokens})...")
    t0 = time.time()
    if lora_request:
        outputs = llm.generate(prompts, params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / len(problems):.2f}s/problem)")

    # Grade
    results = []
    correct = 0
    for p, o in zip(problems, outputs):
        text = o.outputs[0].text
        predicted = extract_boxed_answer(text)
        is_correct = grade_math_answer(predicted, p["answer"])
        if is_correct:
            correct += 1
        results.append({
            "question": p["question"],
            "gold": p["answer"],
            "predicted": predicted,
            "correct": is_correct,
            "level": p["level"],
            "subject": p["subject"],
            "num_tokens": len(o.outputs[0].token_ids),
            "output": text,
        })

    total = len(results)
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"{'='*60}")

    # By level
    print("\nBy level:")
    for lvl in sorted(set(r["level"] for r in results)):
        subset = [r for r in results if r["level"] == lvl]
        c = sum(1 for r in subset if r["correct"])
        print(f"  Level {lvl}: {c}/{len(subset)} ({100 * c / len(subset):.1f}%)")

    # By subject
    print("\nBy subject:")
    for subj in sorted(set(r["subject"] for r in results)):
        subset = [r for r in results if r["subject"] == subj]
        c = sum(1 for r in subset if r["correct"])
        print(f"  {subj:30s} {c}/{len(subset)} ({100 * c / len(subset):.1f}%)")

    # Token stats
    token_counts = [r["num_tokens"] for r in results]
    import numpy as np
    arr = np.array(token_counts)
    print(f"\nTokens: mean={arr.mean():.0f}, median={np.median(arr):.0f}, "
          f"p95={np.percentile(arr, 95):.0f}")

    # Save
    if args.output:
        out_path = args.output
    else:
        short = args.model.split("/")[-1]
        tag = "_lora" if args.checkpoint else "_base"
        Path("eval_results").mkdir(exist_ok=True)
        out_path = f"eval_results/math500_{short}{tag}_n{total}.json"

    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "temperature": args.temperature,
        "time_seconds": elapsed,
    }

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
