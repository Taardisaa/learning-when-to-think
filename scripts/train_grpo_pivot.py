"""GRPO training with ALP reward and pluggable action-injection strategies.

Usage:
    # Hard-mask strategy (prototype default)
    python -m scripts.train_grpo_pivot --config configs/grpo_3action.yaml --gpus 1

    # Smoke test: few steps with no injection
    python -m scripts.train_grpo_pivot --config configs/grpo_3action.yaml \
        --gpus 1 --action-strategy no_injection --max-steps 5
"""

import argparse
import os
import time
from pathlib import Path

import yaml


SYSTEM_PROMPT = (
    "Solve the following math problem. "
    r"Please reason step by step, and put your final answer within \boxed{}."
)

DEFAULTS = {
    "model": "checkpoints/sft_pivot/merged",
    "output_dir": "checkpoints/grpo_pivot/final",
    "data_subset": 200,
    "num_generations": 4,
    "max_completion_length": 1024,
    "temperature": 0.6,
    "top_p": 0.95,
    "action_strategy": "hard_mask",
    "beta": 0.01,
    "epsilon": 0.2,
    "loss_type": "dapo",
    "alp_beta": 0.05,
    "alp_l_max": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    "num_train_epochs": 1,
    "learning_rate": 1e-5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_steps": -1,
    "bf16": True,
    "logging_steps": 1,
    "save_strategy": "steps",
    "save_steps": 50,
    "seed": 42,
    "gpus": None,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}


def main():
    parser = argparse.ArgumentParser(description="GRPO training with ALP + action strategies")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-subset", type=int, default=None)
    parser.add_argument("--action-strategy", default=None,
                        choices=["no_injection", "hard_mask", "soft_bias"])
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--alp-beta", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else dict(DEFAULTS)
    if args.config:
        print(f"Loaded config from {args.config}")

    cli_overrides = {
        "model": args.model, "output_dir": args.output_dir,
        "data_subset": args.data_subset, "action_strategy": args.action_strategy,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature, "beta": args.beta,
        "alp_beta": args.alp_beta, "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs, "max_steps": args.max_steps,
        "logging_steps": args.logging_steps, "gpus": args.gpus, "seed": args.seed,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v

    if cfg["gpus"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
        print(f"Using GPUs: {cfg['gpus']}")

    # Lazy imports (after CUDA_VISIBLE_DEVICES)
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from src.data.math import load_math_train
    from src.pivot.action_strategies import get_strategy
    from src.pivot.reward_wrapper import make_alp_reward_func
    from src.pivot.rollout import make_rollout_func

    print(f"\n{'='*50}")
    print("GRPO Training — 3-Action Pivot")
    print(f"{'='*50}")
    print(f"Model:          {cfg['model']}")
    print(f"Output:         {cfg['output_dir']}")
    print(f"Action strat:   {cfg['action_strategy']}")
    print(f"Num gens (K):   {cfg['num_generations']}")
    print(f"ALP beta:       {cfg['alp_beta']}")
    print(f"KL beta:        {cfg['beta']}")
    print(f"LR:             {cfg['learning_rate']}")

    # Build training dataset
    print(f"\nLoading MATH train ({cfg['data_subset']} problems)...")
    problems = load_math_train(subset_size=cfg["data_subset"], seed=cfg["seed"])

    def make_record(p):
        user_msg = f"{SYSTEM_PROMPT}\n\n{p['question']}"
        return {
            "prompt": [{"role": "user", "content": user_msg}],
            "gold_answer": p["answer"],
        }

    dataset = Dataset.from_list([make_record(p) for p in problems])
    print(f"Dataset size: {len(dataset)}")

    # Strategy
    strategy = get_strategy(cfg["action_strategy"])
    print(f"Strategy:       {strategy.name}")

    # Reward and rollout
    reward_func = make_alp_reward_func(
        beta=cfg["alp_beta"],
        l_max=cfg["alp_l_max"],
        num_generations=cfg["num_generations"],
    )
    rollout_func = make_rollout_func(strategy)

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_generations=cfg["num_generations"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        beta=cfg["beta"],
        epsilon=cfg["epsilon"],
        loss_type=cfg["loss_type"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_steps=cfg["max_steps"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"],
        seed=cfg["seed"],
        report_to="none",
        remove_unused_columns=False,
        use_vllm=False,  # HF generate; vLLM has version mismatch
    )

    # Silence the rollout_func experimental warning
    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_targets"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("\nInitializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=[reward_func],
        args=grpo_config,
        train_dataset=dataset,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )

    print(f"\nStarting training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Save
    output_dir = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)

    cfg_save_path = Path(output_dir) / "train_config.yaml"
    with open(cfg_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Saved model + config to {output_dir}")


if __name__ == "__main__":
    main()
