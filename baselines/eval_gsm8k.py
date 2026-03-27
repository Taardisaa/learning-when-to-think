"""
Baseline evaluation on GSM8K (forward-only, no backoff).

Uses vLLM for fast batched inference.

Usage:
    # Single model
    python eval_gsm8k.py --model Qwen/Qwen3.5-2B

    # Multiple models sequentially
    python eval_gsm8k.py --model Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B

Output goes to baselines/<model_short_name>/gsm8k_results.json automatically.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams


def extract_gsm8k_answer(text: str, method: str = "strict") -> str | None:
    r"""Extract the final numerical answer from model output.

    Only looks at text AFTER </think>, and only the last 300 chars
    (following verl convention) to avoid matching thinking-trace numbers.

    Methods:
        strict:   \boxed{N} or #### N format (last match)
        flexible: last number in the text
    """
    think_end = text.find("</think>")
    if think_end >= 0:
        text = text[think_end + 8:]

    # Only look at tail — answer should be near the end
    if len(text) > 300:
        text = text[-300:]

    if method == "strict":
        # \boxed{N} format (preferred for Qwen3)
        matches = re.findall(r"\\boxed\{([^}]+)\}", text)
        if matches:
            return matches[-1].replace(",", "").replace("$", "").strip()
        # Legacy #### N format
        matches = re.findall(r"####\s*\$?(-?[\d,.]+)", text)
        if not matches:
            return None
        return matches[-1].replace(",", "").replace("$", "").strip()
    else:  # flexible
        matches = re.findall(r"(-?[\d,.]+)", text)
        for ans in reversed(matches):
            ans = ans.replace(",", "").strip().strip(".")
            if ans:
                return ans
        return None


def extract_gold_answer(answer_text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_text.strip()


def evaluate_model(
    model_name: str,
    dataset,
    max_new_tokens: int = 8192,
    tensor_parallel: int = 2,
):
    """Evaluate a single model on GSM8K."""
    short_name = model_name.split("/")[-1]
    output_dir = Path(__file__).parent / short_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gsm8k_results.json"

    print(f"\n{'='*60}")
    print(f"Model:          {model_name}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"TP:             {tensor_parallel}")
    print(f"Samples:        {len(dataset)}")
    print(f"Output:         {output_path}")
    print(f"{'='*60}")

    questions = [ex["question"] for ex in dataset]
    gold_answers = [extract_gold_answer(ex["answer"]) for ex in dataset]

    # Build chat-formatted prompts
    conversations = []
    for q in questions:
        conversations.append([
            {"role": "user", "content": (
                f"Solve the following math problem. "
                f"Please reason step by step, and put your final answer "
                f"within \\boxed{{}}.\n\n{q}"
            )}
        ])

    # Load model via vLLM
    print("Loading model via vLLM...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel,
        max_model_len=16384,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,  # greedy
        repetition_penalty=1.3,
    )

    # Apply chat template
    tokenizer = llm.get_tokenizer()
    prompts = []
    for conv in conversations:
        try:
            prompts.append(tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            ))
        except TypeError:
            # Fallback if enable_thinking not supported
            prompts.append(tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True,
            ))

    # Batched inference (vLLM handles batching internally)
    print(f"Running inference on {len(prompts)} problems...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - t0
    print(f"Inference done in {total_time:.1f}s ({total_time/len(dataset):.2f}s/problem)")

    # Score results
    results = []
    correct = 0
    token_counts = []

    for i, (output, gold) in enumerate(zip(outputs, gold_answers)):
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)

        predicted = extract_gsm8k_answer(generated_text)
        is_correct = predicted is not None and predicted == gold

        if is_correct:
            correct += 1
        token_counts.append(num_tokens)

        # Check if thinking was completed
        has_think_end = "</think>" in generated_text

        results.append({
            "index": i,
            "question": questions[i],
            "gold_answer": gold,
            "predicted_answer": predicted,
            "correct": is_correct,
            "num_tokens": num_tokens,
            "thinking_completed": has_think_end,
            "generated_text": generated_text,
        })

    # Summary stats
    total = len(dataset)
    token_arr = np.array(token_counts)
    thinking_completed = sum(1 for r in results if r["thinking_completed"])

    summary = {
        "model": model_name,
        "dataset": "GSM8K",
        "split": "test",
        "num_samples": total,
        "max_new_tokens": max_new_tokens,
        "accuracy": round(correct / total * 100, 2),
        "correct": correct,
        "total": total,
        "inference_time_sec": round(total_time, 1),
        "sec_per_problem": round(total_time / total, 3),
        "thinking_completed": thinking_completed,
        "thinking_completed_pct": round(thinking_completed / total * 100, 1),
        "answer_extracted": sum(1 for r in results if r["predicted_answer"] is not None),
        "token_stats": {
            "mean": round(float(token_arr.mean()), 1),
            "median": round(float(np.median(token_arr)), 1),
            "std": round(float(token_arr.std()), 1),
            "min": int(token_arr.min()),
            "max": int(token_arr.max()),
            "p90": round(float(np.percentile(token_arr, 90)), 1),
            "p95": round(float(np.percentile(token_arr, 95)), 1),
            "p99": round(float(np.percentile(token_arr, 99)), 1),
            "total": int(token_arr.sum()),
        },
        "hit_max_tokens_count": int((token_arr >= max_new_tokens).sum()),
        "hit_max_tokens_pct": round(float((token_arr >= max_new_tokens).mean() * 100), 1),
    }

    print(f"\n--- {short_name} ---")
    print(f"Accuracy:       {summary['accuracy']}% ({correct}/{total})")
    print(f"Thinking done:  {thinking_completed}/{total} ({summary['thinking_completed_pct']}%)")
    print(f"Answer found:   {summary['answer_extracted']}/{total}")
    print(f"Inference:      {summary['inference_time_sec']}s ({summary['sec_per_problem']}s/problem)")
    print(f"Avg tokens:     {summary['token_stats']['mean']}")
    print(f"Median tokens:  {summary['token_stats']['median']}")
    print(f"P95 tokens:     {summary['token_stats']['p95']}")
    print(f"Hit max limit:  {summary['hit_max_tokens_count']} ({summary['hit_max_tokens_pct']}%)")

    # Save
    output_data = {"summary": summary, "results": results}
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"Saved to {output_path}")

    # Free GPU memory before loading next model
    del llm
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on GSM8K")
    parser.add_argument("--model", nargs="+", required=True, help="Model name(s)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max new tokens")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    args = parser.parse_args()

    # Load dataset once
    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Total samples: {len(dataset)}")

    # Run each model
    summaries = []
    for model_name in args.model:
        summary = evaluate_model(
            model_name=model_name,
            dataset=dataset,
            max_new_tokens=args.max_tokens,
            tensor_parallel=args.tp,
        )
        summaries.append(summary)

    # Print comparison table
    if len(summaries) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print(f"{'Model':<25} {'Acc%':>6} {'AvgTok':>7} {'P95Tok':>7} {'HitMax%':>8} {'s/prob':>7}")
        print("-" * 80)
        for s in summaries:
            name = s["model"].split("/")[-1]
            print(f"{name:<25} {s['accuracy']:>6.1f} {s['token_stats']['mean']:>7.0f} "
                  f"{s['token_stats']['p95']:>7.0f} {s['hit_max_tokens_pct']:>7.1f}% "
                  f"{s['sec_per_problem']:>7.2f}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
