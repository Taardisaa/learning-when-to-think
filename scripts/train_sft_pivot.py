"""SFT warmup: teach Qwen2.5-Math-7B-Instruct the 3-action token format.

Trains LoRA + new token embeddings on SFT data containing
<continue>, <refine>, <terminate> action tokens.

Usage:
    python -m scripts.train_sft_pivot --config configs/sft_3action.yaml
    python -m scripts.train_sft_pivot --config configs/sft_3action.yaml --gpus 1
    python -m scripts.train_sft_pivot --config configs/sft_3action.yaml --epochs 1 --no-sanity-check
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.pivot.tokens import ACTION_TOKENS, enable_new_token_grad, setup_tokenizer_and_model


DEFAULTS = {
    "model": "Qwen/Qwen2.5-Math-7B-Instruct",
    "data": "data/sft_3action_math_train.jsonl",
    "output_dir": "checkpoints/sft_pivot/final",
    "resume_from": None,
    "epochs": 3,
    "lr": 2e-5,
    "batch_size": 1,
    "grad_accum": 8,
    "max_seq_len": 2048,
    "warmup_ratio": 0.03,
    "lr_scheduler": "cosine",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    "bf16": True,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "seed": 42,
    "gpus": None,
    "sanity_check_n": 3,
    "sanity_check_max_tokens": 1024,
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}


def load_sft_data(path: str, tokenizer) -> Dataset:
    """Load JSONL SFT data and apply chat template to produce training text."""
    with open(path) as f:
        raw = [json.loads(line) for line in f]

    texts = []
    for ex in raw:
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})


def run_sanity_check(model, tokenizer, data_path: str, n: int, max_tokens: int):
    """Generate on a few training problems to verify action tokens appear."""
    with open(data_path) as f:
        examples = [json.loads(line) for line in f]

    clean = [ex for ex in examples if not ex["has_refine"]]
    refine = [ex for ex in examples if ex["has_refine"]]
    samples = []
    if clean:
        samples.append(clean[0])
    if refine:
        samples.append(refine[0])
    for ex in examples:
        if len(samples) >= n:
            break
        if ex not in samples:
            samples.append(ex)

    model.eval()
    print(f"\n{'='*60}")
    print(f"SANITY CHECK: generating on {len(samples)} problems")
    print(f"{'='*60}")

    for i, ex in enumerate(samples):
        messages = [{"role": "user", "content": ex["messages"][0]["content"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=False
        )

        counts = {tok: generated.count(tok) for tok in ACTION_TOKENS}
        has_any = any(v > 0 for v in counts.values())

        print(f"\n--- Sample {i+1} (has_refine={ex['has_refine']}) ---")
        print(f"Q: {ex['question'][:100]}...")
        print(f"Action tokens: {counts}")
        print(f"Output (first 500 chars): {generated[:500]}")
        if not has_any:
            print("WARNING: No action tokens found in output!")
        print()


def main():
    parser = argparse.ArgumentParser(description="SFT warmup for 3-action tokens")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-sanity-check", action="store_true")
    args = parser.parse_args()

    # Build config: defaults <- yaml <- cli
    cfg = load_config(args.config) if args.config else dict(DEFAULTS)
    if args.config:
        print(f"Loaded config from {args.config}")

    cli_map = {
        "model": args.model, "data": args.data, "output_dir": args.output_dir,
        "resume_from": args.resume_from, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "max_seq_len": args.max_seq_len, "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha, "gpus": args.gpus, "seed": args.seed,
    }
    for k, v in cli_map.items():
        if v is not None:
            cfg[k] = v

    # GPU selection
    if cfg["gpus"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
        print(f"Using GPUs: {cfg['gpus']}")

    # Print config summary
    print(f"\n{'='*50}")
    print(f"SFT Warmup Training — 3-Action Tokens")
    print(f"{'='*50}")
    print(f"Model:      {cfg['model']}")
    print(f"Data:       {cfg['data']}")
    print(f"Output:     {cfg['output_dir']}")
    print(f"Epochs:     {cfg['epochs']}")
    print(f"LR:         {cfg['lr']}")
    print(f"Batch:      {cfg['batch_size']} x {cfg['grad_accum']} grad_accum "
          f"(eff={cfg['batch_size'] * cfg['grad_accum']})")
    print(f"Max seq:    {cfg['max_seq_len']}")
    print(f"LoRA:       r={cfg['lora_r']}, alpha={cfg['lora_alpha']}, "
          f"targets={cfg['lora_targets']}")
    print(f"Scheduler:  {cfg['lr_scheduler']}, warmup={cfg['warmup_ratio']}")

    # Load model + tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16
    )

    # Add action tokens + initialize embeddings
    token_ids = setup_tokenizer_and_model(tokenizer, model)
    print(f"Action token IDs: {token_ids}")

    # Apply LoRA
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_targets"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Selectively unfreeze new token embeddings (AFTER get_peft_model)
    hooks = enable_new_token_grad(model, token_ids)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.2f}%)")

    # Load data
    print(f"\nLoading SFT data from {cfg['data']}...")
    dataset = load_sft_data(cfg["data"], tokenizer)
    print(f"Examples: {len(dataset)}")

    with open(cfg["data"]) as f:
        raw = [json.loads(line) for line in f]
    n_refine = sum(1 for ex in raw if ex["has_refine"])
    print(f"  Clean: {len(raw) - n_refine}, Refine: {n_refine} "
          f"({100*n_refine/len(raw):.1f}% refine)")

    steps_per_epoch = len(dataset) // (cfg["batch_size"] * cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    print(f"  Steps/epoch: {steps_per_epoch}, total: {total_steps}")

    # Training config
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        max_length=cfg["max_seq_len"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler"],
        seed=cfg["seed"],
        report_to="tensorboard",
        logging_dir=f"{cfg['output_dir']}/tb",
        remove_unused_columns=False,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=cfg["resume_from"])
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Save
    output_dir = cfg["output_dir"]
    print(f"Saving to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    config_save_path = Path(output_dir) / "train_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Config saved to {config_save_path}")

    # Sanity check
    if not args.no_sanity_check:
        run_sanity_check(
            model, tokenizer, cfg["data"],
            n=cfg["sanity_check_n"],
            max_tokens=cfg["sanity_check_max_tokens"],
        )

    # Clean up hooks
    for h in hooks:
        h.remove()

    print("\nSFT warmup complete.")


if __name__ == "__main__":
    main()
