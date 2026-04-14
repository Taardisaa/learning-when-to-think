"""HF-based eval with optional action-injection strategy (hard_mask / soft_bias).

Uses transformers+PEFT instead of vLLM because vLLM 0.18 dropped custom
logits_processors in SamplingParams. Slower than eval_pivot.py but supports
hard_mask / soft_bias at inference time.

Usage:
    # no-injection (same policy shape as eval_pivot but via HF)
    PYTHONPATH=. venv/bin/python -m scripts.eval_pivot_hf \
        --model checkpoints/sft_pivot/merged \
        --checkpoint checkpoints/grpo_pivot/final/checkpoint-30 \
        --n 100 --action-strategy no_injection

    # hard_mask (forces action at \\n\\n boundaries)
    PYTHONPATH=. venv/bin/python -m scripts.eval_pivot_hf \
        --model checkpoints/sft_pivot/merged \
        --checkpoint checkpoints/grpo_pivot/final/checkpoint-30 \
        --n 100 --action-strategy hard_mask
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.math import extract_boxed_answer, grade_math_answer, load_math500
from src.pivot.action_strategies import get_strategy
from src.pivot.eval_harness import evaluate, metrics_to_dict, print_eval_report
from src.pivot.eval_types import RolloutResult
from src.pivot.tokens import ACTION_TOKENS


SYSTEM_PROMPT = (
    "Solve the following math problem. "
    r"Please reason step by step, and put your final answer within \boxed{}."
)


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description="HF-based eval with action strategies")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--checkpoint", default=None, help="LoRA checkpoint path")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--action-strategy", default="no_injection",
                        choices=["no_injection", "hard_mask", "soft_bias"])
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Problems per HF generate call")
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Load tokenizer + model + adapter
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    if args.checkpoint:
        print(f"Attaching LoRA: {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    # Build logits processor
    action_ids = [tokenizer.convert_tokens_to_ids(t) for t in ACTION_TOKENS]
    action_id_to_name = {tid: t.strip("<>") for tid, t in zip(action_ids, ACTION_TOKENS)}
    strategy = get_strategy(args.action_strategy)
    processor = strategy.logits_processor(action_ids)
    print(f"Strategy: {strategy.name}")

    # Problems
    problems = load_math500()[: args.n]
    prompts = [build_prompt(tokenizer, p["question"]) for p in problems]

    # Batched generation
    results = []
    t0 = time.time()
    for batch_start in range(0, len(problems), args.batch_size):
        batch_probs = problems[batch_start : batch_start + args.batch_size]
        batch_prompts = prompts[batch_start : batch_start + args.batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=args.max_tokens,
            do_sample=(args.temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
        )
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
        if processor is not None:
            gen_kwargs["logits_processor"] = [processor]

        with torch.no_grad():
            out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, **gen_kwargs)
        prompt_len = enc.input_ids.shape[1]
        gen_ids_batch = out[:, prompt_len:]

        for p, gen_ids in zip(batch_probs, gen_ids_batch):
            ids = gen_ids.tolist()
            # Strip trailing pad tokens
            while ids and ids[-1] == tokenizer.pad_token_id:
                ids.pop()
            text = tokenizer.decode(ids, skip_special_tokens=False)
            predicted = extract_boxed_answer(text)
            correct = grade_math_answer(predicted, p["answer"])
            actions = [action_id_to_name[tid] for tid in ids if tid in action_id_to_name]
            results.append(RolloutResult(
                prompt_id=p["unique_id"], question=p["question"], gold_answer=p["answer"],
                level=p["level"], subject=p["subject"],
                generated_text=text, predicted_answer=predicted, correct=correct,
                num_tokens=len(ids), actions=actions,
            ))

        elapsed = time.time() - t0
        done = len(results)
        print(f"  [{done}/{len(problems)}] {elapsed:.0f}s ({elapsed/done:.1f}s/problem)")

    # Report
    metrics = evaluate(results)
    print_eval_report(metrics)

    # Save
    if args.output is None:
        short = args.model.split("/")[-1]
        tag = f"_{args.action_strategy}"
        if args.checkpoint:
            ckpt_tag = Path(args.checkpoint).name
            tag = f"_{ckpt_tag}_{args.action_strategy}"
        Path("eval_results").mkdir(exist_ok=True)
        args.output = f"eval_results/hf_{short}{tag}_n{len(results)}.json"
    with open(args.output, "w") as f:
        json.dump({
            "summary": {
                "model": args.model, "checkpoint": args.checkpoint,
                "action_strategy": args.action_strategy,
                "temperature": args.temperature, "max_tokens": args.max_tokens,
                **metrics_to_dict(metrics),
            },
            "results": [{
                "prompt_id": r.prompt_id, "level": r.level, "subject": r.subject,
                "gold": r.gold_answer, "predicted": r.predicted_answer,
                "correct": r.correct, "num_tokens": r.num_tokens,
                "actions": r.actions, "output": r.generated_text,
            } for r in results],
        }, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
