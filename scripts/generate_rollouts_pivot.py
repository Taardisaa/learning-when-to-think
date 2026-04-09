"""Generate grouped rollouts from Qwen2.5-Math-7B-Instruct on MATH train.

Produces K rollouts per problem, graded and grouped with pass rates.
Used as input for SFT warmup dataset construction (generate_sft_3action.py).

Usage:
    # From config (recommended)
    python -m scripts.generate_rollouts_pivot --config configs/generate_rollouts_qwen25.yaml

    # CLI overrides
    python -m scripts.generate_rollouts_pivot --config configs/generate_rollouts_qwen25.yaml --subset 100

    # Without config
    python -m scripts.generate_rollouts_pivot --model Qwen/Qwen2.5-Math-7B-Instruct --subset 1000
"""

import argparse
import json
import time
from pathlib import Path

import yaml
from vllm import LLM, SamplingParams

from src.data.math import extract_boxed_answer, grade_math_answer, load_math_train


SYSTEM_PROMPT = (
    "Solve the following math problem. "
    r"Please reason step by step, and put your final answer within \boxed{}."
)

DEFAULTS = {
    "model": "Qwen/Qwen2.5-Math-7B-Instruct",
    "subset": 1000,
    "num_samples": 8,
    "max_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.95,
    "tp": 1,
    "gpu_mem": 0.90,
    "max_model_len": 4096,
    "output": None,
    "seed": 42,
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}


def build_prompt(tokenizer, question: str) -> str:
    """Build a chat-templated prompt for Qwen2.5-Math."""
    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def generate_grouped_rollouts(cfg: dict) -> list[dict]:
    """Generate K rollouts per problem, grade them, and group with pass rates."""
    # Load data
    data = load_math_train(subset_size=cfg["subset"], seed=cfg["seed"])
    print(f"Loaded {len(data)} MATH train problems")

    # Load model
    llm = LLM(
        model=cfg["model"],
        trust_remote_code=True,
        tensor_parallel_size=cfg["tp"],
        max_model_len=cfg["max_model_len"],
        dtype="bfloat16",
        gpu_memory_utilization=cfg["gpu_mem"],
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts
    prompts = [build_prompt(tokenizer, ex["question"]) for ex in data]

    params = SamplingParams(
        n=cfg["num_samples"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
    )

    # Generate
    print(f"Generating {cfg['num_samples']} rollouts/problem "
          f"(temp={cfg['temperature']}, max_tokens={cfg['max_tokens']})...")
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / len(data):.2f}s/problem)")

    # Grade and group
    grouped = []
    total_correct = 0
    total_rollouts = 0

    for ex, output in zip(data, outputs):
        rollouts = []
        for completion in output.outputs:
            text = completion.text
            predicted = extract_boxed_answer(text)
            is_correct = grade_math_answer(predicted, ex["answer_number"])
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
            "answer": ex["answer_number"],
            "rollouts": rollouts,
            "pass_rate": num_correct / len(rollouts),
        })

    print(f"Overall pass rate: {total_correct}/{total_rollouts} "
          f"({100 * total_correct / total_rollouts:.1f}%)")

    # Difficulty distribution
    n_trivial = sum(1 for g in grouped if g["pass_rate"] == 1.0)
    n_mixed = sum(1 for g in grouped if 0 < g["pass_rate"] < 1.0)
    n_impossible = sum(1 for g in grouped if g["pass_rate"] == 0.0)
    print(f"Difficulty: trivial(all correct)={n_trivial}  "
          f"mixed={n_mixed}  impossible(all wrong)={n_impossible}")

    # Free GPU
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Generate grouped rollouts for SFT warmup data"
    )
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--model", default=None)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--gpu-mem", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Build config: defaults <- yaml <- cli
    cfg = load_config(args.config) if args.config else dict(DEFAULTS)
    if args.config:
        print(f"Loaded config from {args.config}")

    cli_map = {
        "model": args.model, "subset": args.subset, "num_samples": args.num_samples,
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "tp": args.tp, "gpu_mem": args.gpu_mem, "output": args.output,
        "seed": args.seed,
    }
    for k, v in cli_map.items():
        if v is not None:
            cfg[k] = v

    # Generate
    grouped = generate_grouped_rollouts(cfg)

    # Save
    if cfg["output"]:
        out_path = cfg["output"]
    else:
        short = cfg["model"].split("/")[-1]
        Path("data").mkdir(exist_ok=True)
        out_path = f"data/rollouts_grouped_math_{short}.jsonl"

    with open(out_path, "w") as f:
        for g in grouped:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"Saved {len(grouped)} grouped rollouts to {out_path}")


if __name__ == "__main__":
    main()
