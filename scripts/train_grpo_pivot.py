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
    "save_steps": 2,
    "save_total_limit": 5,
    "seed": 42,
    "gpus": None,
    "resume_from_checkpoint": None,
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
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint dir to resume training from")
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
        "resume_from_checkpoint": args.resume_from,
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
    def make_record(question, answer):
        user_msg = f"{SYSTEM_PROMPT}\n\n{question}"
        return {
            "prompt": [{"role": "user", "content": user_msg}],
            "gold_answer": answer,
        }

    # Option: filter by pass rate from pre-generated rollouts to focus on mid-difficulty
    rollouts_file = cfg.get("rollouts_file")
    pass_rate_range = cfg.get("pass_rate_range", [0.0, 1.0])
    if rollouts_file:
        import json
        print(f"\nLoading rollouts from {rollouts_file}...")
        with open(rollouts_file) as f:
            groups = [json.loads(line) for line in f]
        lo, hi = pass_rate_range
        filtered = [g for g in groups if lo <= g["pass_rate"] <= hi]
        print(f"  Filtered {len(filtered)}/{len(groups)} problems with {lo} <= pass_rate <= {hi}")
        # Optionally subset
        if cfg["data_subset"] and cfg["data_subset"] < len(filtered):
            import random
            random.Random(cfg["seed"]).shuffle(filtered)
            filtered = filtered[: cfg["data_subset"]]
        records = [make_record(g["question"], g["answer"]) for g in filtered]
    else:
        print(f"\nLoading MATH train ({cfg['data_subset']} problems)...")
        problems = load_math_train(subset_size=cfg["data_subset"], seed=cfg["seed"])
        records = [make_record(p["question"], p["answer"]) for p in problems]

    dataset = Dataset.from_list(records)
    print(f"Dataset size: {len(dataset)}")

    # Strategy
    strategy = get_strategy(cfg["action_strategy"])
    print(f"Strategy:       {strategy.name}")

    # Safety: if strategy needs action tokens, verify the model/tokenizer has them
    if cfg["action_strategy"] != "no_injection":
        from transformers import AutoTokenizer
        from src.pivot.tokens import ACTION_TOKENS
        _tok = AutoTokenizer.from_pretrained(cfg["model"])
        missing = [t for t in ACTION_TOKENS if _tok.convert_tokens_to_ids(t) == _tok.unk_token_id or _tok.convert_tokens_to_ids(t) is None]
        if missing:
            raise ValueError(
                f"Strategy '{cfg['action_strategy']}' requires action tokens in vocab, "
                f"but model '{cfg['model']}' is missing: {missing}. "
                f"Use action_strategy=no_injection for base models without SFT warmup."
            )

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
        report_to="tensorboard",
        logging_dir=f"{cfg['output_dir']}/tb",
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

    # Circuit breaker: abort training if clipped_ratio indicates model collapse
    from transformers import TrainerCallback

    class ClippedRatioCircuitBreaker(TrainerCallback):
        """Stops training if completions/clipped_ratio reaches threshold for N steps.

        This indicates the model stopped terminating completions (usually means
        it collapsed into non-stopping generation). Better to bail early so the
        latest good checkpoint is preserved.
        """

        def __init__(self, threshold: float = 0.99, patience: int = 1):
            self.threshold = threshold
            self.patience = patience
            self.bad_streak = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            cr = logs.get("completions/clipped_ratio")
            if cr is None:
                return
            if cr >= self.threshold:
                self.bad_streak += 1
                print(f"[CircuitBreaker] clipped_ratio={cr:.2f} >= {self.threshold} "
                      f"(streak {self.bad_streak}/{self.patience})")
                if self.bad_streak >= self.patience:
                    print("[CircuitBreaker] Triggered — stopping training. "
                          "Resume from last healthy checkpoint.")
                    control.should_training_stop = True
            else:
                self.bad_streak = 0

    print("\nInitializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=[reward_func],
        args=grpo_config,
        train_dataset=dataset,
        rollout_func=rollout_func,
        peft_config=peft_config,
        callbacks=[ClippedRatioCircuitBreaker(threshold=0.99, patience=1)],
    )

    resume = cfg.get("resume_from_checkpoint")
    if resume:
        print(f"\nResuming from checkpoint: {resume}")

    print(f"\nStarting training...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
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
