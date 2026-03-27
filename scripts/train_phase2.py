"""Phase 2: GRPO with progress reward.

Loads Phase 1 checkpoint (merge LoRA → base), applies fresh LoRA for Phase 2,
trains with GRPO + dense per-segment progress signal on GSM8K.

Usage:
    python -m scripts.train_phase2
    python -m scripts.train_phase2 --phase1 checkpoints/phase1/final --steps 500
"""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BackoffConfig
from src.data.gsm8k import load_gsm8k
from src.tokens import (
    setup_tokenizer_and_model,
    enable_new_token_grad,
    NEW_SPECIAL_TOKENS,
    TERMINATE_TOKEN,
)
from src.train.grpo import grpo_step


def load_phase1_as_base(base_model_name: str, phase1_path: str, device: str = "cuda"):
    """Load Phase 1 checkpoint and merge LoRA into base weights.

    Returns (merged_model, tokenizer, token_ids).
    The merged model has the special token embeddings baked in.
    """
    # Load tokenizer from Phase 1 (has special tokens)
    tokenizer = AutoTokenizer.from_pretrained(phase1_path)

    # Load base model and resize embeddings to match
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, device_map=device
    )
    base_model.resize_token_embeddings(len(tokenizer))

    # Load and merge Phase 1 LoRA
    model = PeftModel.from_pretrained(base_model, phase1_path)
    model = model.merge_and_unload()
    # Clean up stale peft_config left by merge_and_unload
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")

    # Build token_ids from tokenizer
    all_special = NEW_SPECIAL_TOKENS + [TERMINATE_TOKEN]
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in all_special}

    return model, tokenizer, token_ids


def main():
    parser = argparse.ArgumentParser(description="Phase 2: GRPO + exploration")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--phase1", default="checkpoints/phase1/final")
    parser.add_argument("--output", default="checkpoints/phase2")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Questions per GRPO step")
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-problems", type=int, default=500,
                        help="GSM8K train problems to use")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 1 checkpoint: {args.phase1}")
    print(f"Output:             {args.output}")
    print(f"Steps:              {args.steps}")
    print(f"Batch size:         {args.batch_size} questions x {args.num_rollouts} rollouts")
    print(f"LR:                 {args.lr}")

    # ── Device placement ──
    train_device = ref_device = "cuda"

    # ── Load Phase 1 merged model as reference ──
    print("Loading Phase 1 → merged base (reference model)...")
    ref_model, tokenizer, token_ids = load_phase1_as_base(
        args.base_model, args.phase1, device=ref_device
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    print(f"Reference model loaded. Token IDs: {token_ids}")

    # ── Create trainable model: same merged base + fresh LoRA ──
    print("Creating trainable model (fresh LoRA on merged base)...")
    train_model, _, _ = load_phase1_as_base(
        args.base_model, args.phase1, device=train_device
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    train_model = get_peft_model(train_model, lora_config)
    hooks = enable_new_token_grad(train_model, token_ids)
    train_model.train()

    trainable = sum(p.numel() for p in train_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in train_model.parameters())
    print(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.1f}M")

    # ── Config ──
    config = BackoffConfig(
        num_rollouts=args.num_rollouts,
        lr=args.lr,
        alpha=0.5,
        num_probes=1,
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in train_model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=0.01,
    )

    # ── Load data ──
    print(f"\nLoading GSM8K train set (max {args.max_problems} problems)...")
    data = load_gsm8k("train", subset_size=args.max_problems)
    questions = [d["question"] for d in data]
    answers = [d["answer_number"] for d in data]
    print(f"Loaded {len(data)} problems")

    # ── Training loop ──
    print(f"\nStarting GRPO training ({args.steps} steps)...")
    log_file = output_dir / "train_log.jsonl"

    for step in range(1, args.steps + 1):
        # Sample a batch of questions
        indices = random.sample(range(len(questions)), min(args.batch_size, len(questions)))
        batch_q = [questions[i] for i in indices]
        batch_a = [answers[i] for i in indices]

        stats = grpo_step(
            model=train_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            token_ids=token_ids,
            questions=batch_q,
            gold_answers=batch_a,
            config=config,
            optimizer=optimizer,
            step=step,
        )

        # Log
        log_entry = {
            "step": step,
            "loss": round(stats.loss, 4),
            "avg_reward": round(stats.avg_reward, 4),
            "accuracy": round(stats.accuracy, 4),
            "backoff_rate": round(stats.backoff_rate, 4),
            "avg_progress": round(stats.avg_progress, 4),
            "avg_rho": round(stats.avg_rho, 4),
            "num_traj": stats.num_trajectories,
        }

        if step % 10 == 0 or step <= 5:
            print(f"Step {step:4d} | loss={stats.loss:.3f} reward={stats.avg_reward:.3f} "
                  f"acc={stats.accuracy:.2f} backoff={stats.backoff_rate:.2f} "
                  f"progress={stats.avg_progress:.3f} rho={stats.avg_rho:.3f}")

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Save checkpoint
        if step % args.save_every == 0:
            ckpt_path = output_dir / f"step_{step}"
            train_model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            print(f"  Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = output_dir / "final"
    train_model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nPhase 2 complete. Final checkpoint: {final_path}")

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
