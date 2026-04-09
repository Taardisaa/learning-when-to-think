"""Evaluate a model on MATH-500 using the pivot eval harness.

Usage:
    # From config (recommended)
    python -m scripts.eval_pivot --config configs/eval_qwen_math_7b.yaml

    # Override config values via CLI
    python -m scripts.eval_pivot --config configs/eval_qwen_math_7b.yaml --n 10

    # Without config (all CLI args)
    python -m scripts.eval_pivot --model Qwen/Qwen2.5-Math-7B-Instruct --n 500
"""

import argparse
import json
import time
from pathlib import Path

import yaml
from vllm import LLM, SamplingParams

from src.data.math import extract_boxed_answer, grade_math_answer, load_math500
from src.pivot.eval_harness import evaluate, metrics_to_dict, print_eval_report
from src.pivot.eval_types import RolloutResult


SYSTEM_PROMPT = (
    "Solve the following math problem. "
    r"Please reason step by step, and put your final answer within \boxed{}."
)

DEFAULTS = {
    "model": "Qwen/Qwen2.5-Math-7B-Instruct",
    "checkpoint": None,
    "temperature": 0.0,
    "max_tokens": 2048,
    "top_p": 0.95,
    "top_k": -1,
    "tp": 1,
    "gpu_mem": 0.90,
    "max_model_len": 4096,
    "n": 500,
    "output": None,
}


def load_config(config_path: str) -> dict:
    """Load eval config from YAML, falling back to defaults for missing keys."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    merged = {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
    return merged


def build_eval_prompt(tokenizer, question: str) -> str:
    """Build a chat-templated prompt for evaluation."""
    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Evaluate on MATH-500 (pivot)")
    parser.add_argument("--config", default=None,
                        help="YAML config file (e.g. configs/eval_qwen_math_7b.yaml)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--gpu-mem", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Build config: defaults <- yaml <- cli overrides
    if args.config:
        cfg = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        cfg = dict(DEFAULTS)

    # CLI overrides (only non-None values)
    cli_map = {
        "model": args.model, "checkpoint": args.checkpoint, "n": args.n,
        "tp": args.tp, "gpu_mem": args.gpu_mem, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k, "output": args.output,
    }
    for k, v in cli_map.items():
        if v is not None:
            cfg[k] = v

    # Load problems
    problems = load_math500()
    if cfg["n"] < len(problems):
        problems = problems[: cfg["n"]]
    print(f"MATH-500: {len(problems)} problems")

    # Load model
    lora_request = None
    llm_kwargs = dict(
        model=cfg["model"],
        trust_remote_code=True,
        tensor_parallel_size=cfg["tp"],
        max_model_len=cfg["max_model_len"],
        dtype="bfloat16",
        gpu_memory_utilization=cfg["gpu_mem"],
    )
    if cfg["checkpoint"]:
        from vllm.lora.request import LoRARequest
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        llm = LLM(**llm_kwargs)
        lora_request = LoRARequest("adapter", 1, cfg["checkpoint"])
        print(f"Model: {cfg['model']} + LoRA from {cfg['checkpoint']}")
    else:
        llm = LLM(**llm_kwargs)
        print(f"Model: {cfg['model']} (base)")

    tokenizer = llm.get_tokenizer()

    # Build prompts
    prompts = [build_eval_prompt(tokenizer, p["question"]) for p in problems]

    # Sampling params
    sampling_kwargs = dict(max_tokens=cfg["max_tokens"])
    if cfg["temperature"] == 0:
        sampling_kwargs["temperature"] = 0
    else:
        sampling_kwargs["temperature"] = cfg["temperature"]
        sampling_kwargs["top_p"] = cfg["top_p"]
        if cfg["top_k"] > 0:
            sampling_kwargs["top_k"] = cfg["top_k"]
    params = SamplingParams(**sampling_kwargs)

    # Generate
    print(f"Generating (temp={cfg['temperature']}, max_tokens={cfg['max_tokens']})...")
    t0 = time.time()
    if lora_request:
        outputs = llm.generate(prompts, params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / len(problems):.2f}s/problem)")

    # Build RolloutResults
    results = []
    for p, o in zip(problems, outputs):
        text = o.outputs[0].text
        predicted = extract_boxed_answer(text)
        is_correct = grade_math_answer(predicted, p["answer"])
        results.append(RolloutResult(
            prompt_id=p["unique_id"],
            question=p["question"],
            gold_answer=p["answer"],
            level=p["level"],
            subject=p["subject"],
            generated_text=text,
            predicted_answer=predicted,
            correct=is_correct,
            num_tokens=len(o.outputs[0].token_ids),
            actions=[],  # baseline: no actions
        ))

    # Evaluate and report
    metrics = evaluate(results)
    print_eval_report(metrics)

    # Save
    if cfg["output"]:
        out_path = cfg["output"]
    else:
        short = cfg["model"].split("/")[-1]
        tag = "_lora" if cfg["checkpoint"] else "_base"
        Path("eval_results").mkdir(exist_ok=True)
        out_path = f"eval_results/pivot_{short}{tag}_n{len(results)}.json"

    output_data = {
        "summary": {
            "model": cfg["model"],
            "checkpoint": cfg["checkpoint"],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "time_seconds": elapsed,
            **metrics_to_dict(metrics),
        },
        "results": [
            {
                "prompt_id": r.prompt_id,
                "question": r.question,
                "gold": r.gold_answer,
                "predicted": r.predicted_answer,
                "correct": r.correct,
                "level": r.level,
                "subject": r.subject,
                "num_tokens": r.num_tokens,
                "output": r.generated_text,
            }
            for r in results
        ],
    }

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
