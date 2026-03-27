"""Phase 1: SFT warm-up — teach the model the backoff token format.

Trains LoRA on Qwen3 with synthetic SFT data containing
<backoff_1/2/3> and </think> tokens. New token embeddings are
selectively unfrozen via gradient masking.

Usage:
    python -m scripts.train_phase1
    python -m scripts.train_phase1 --model Qwen/Qwen3-4B-Thinking-2507 --epochs 3
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.config import BackoffConfig
from src.tokens import setup_tokenizer_and_model, enable_new_token_grad


def load_sft_data(path: str, tokenizer) -> Dataset:
    """Load JSONL SFT data and apply chat template."""
    with open(path) as f:
        raw = [json.loads(line) for line in f]

    texts = []
    for ex in raw:
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser(description="Phase 1: SFT warm-up")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--data", default="data/sft_train.jsonl")
    parser.add_argument("--output", default="checkpoints/phase1/final")
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint dir to resume training from")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    print(f"Model:      {args.model}")
    print(f"Data:       {args.data}")
    print(f"Output:     {args.output}")
    print(f"Epochs:     {args.epochs}")
    print(f"LR:         {args.lr}")
    print(f"Batch:      {args.batch_size} x {args.grad_accum} grad_accum")
    print(f"Max seq:    {args.max_seq_len}")
    print(f"LoRA:       r={args.lora_r}, alpha={args.lora_alpha}")

    # ── Load model + tokenizer ──
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    )

    # Add special tokens + initialize embeddings
    token_ids = setup_tokenizer_and_model(tokenizer, model)
    print(f"Token IDs: {token_ids}")

    # ── Apply LoRA ──
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Selectively unfreeze new token embeddings
    hooks = enable_new_token_grad(model, token_ids)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.2f}%)")

    # ── Load data ──
    print(f"\nLoading SFT data from {args.data}...")
    dataset = load_sft_data(args.data, tokenizer)
    print(f"Examples: {len(dataset)}")

    # ── Training config ──
    output_dir = args.output
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_seq_len,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # ── Train ──
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # ── Save ──
    print(f"\nSaving to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Clean up hooks
    for h in hooks:
        h.remove()

    print("Phase 1 SFT complete.")


if __name__ == "__main__":
    main()
