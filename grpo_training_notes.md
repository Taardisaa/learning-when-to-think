# GRPO Training Notes

Running notes on gotchas, failure modes, and fixes discovered during Stage 4 GRPO training. Key lessons to not forget.

## Gotcha 1: `unwrap_model_for_generation` is mandatory for `rollout_func`

**Symptom:** Model generates complete garbage (random multilingual tokens, never terminates) inside GRPOTrainer, even though the same model generates coherent math outside.

**Root cause:** trl's default generation path uses `unwrap_model_for_generation(model, accelerator, ...)` which:
1. Unwraps DDP / FSDP / accelerate wrappers
2. Sets model to `eval()` mode (disables dropout)
3. Re-enables `use_cache=True` during generation (gradient checkpointing disables it for training)

Without this, `model.training == True` during generation → dropout active → garbage output.

**Fix:** Use `unwrap_model_for_generation` context manager inside `rollout_func`. See `src/pivot/rollout.py`.

## Gotcha 2: Reward variance collapse (the main GRPO killer)

**Symptom:** `loss = 0.0`, `reward_std ≈ 0.001`, training appears to "work" but no gradient. Sometimes followed by sudden model collapse in next step.

**Root cause:** GRPO uses **group-relative advantages**: `A_k = (R_k - mean_group) / std_group`. If all K rollouts in a group get nearly identical reward, `std_group ≈ 0` → advantage ≈ 0 → no gradient.

This happens when:
- Problems are too easy (all K rollouts correct → all reward ~1.0)
- Problems are too hard (all K rollouts wrong → all reward 0.0)
- K is too small (K=2 often has no variance)

**Our case:** SFT model gets ~80% on MATH-500. Random 200 problems from MATH train → most are easy → all K rollouts correct → no signal.

**Fix:** Train on **pre-filtered mid-difficulty problems**. Use rollouts from `data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl` filtered by `pass_rate ∈ [0.125, 0.875]`. These are problems where the SFT model is uncertain — exactly where GRPO has signal.

Config:
```yaml
rollouts_file: data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl
pass_rate_range: [0.125, 0.875]
```

Also: increase `num_generations` (K) to 4 or 8 for more intra-group variance.

## Gotcha 3: Step 2 model collapse

**Symptom:** Step 1 has healthy metrics (reward > 0, loss ≠ 0). Step 2 suddenly all completions hit `max_completion_length`, reward=0, model generates garbage.

**Root cause:** SFT model has very low entropy (~0.06). Even a tiny gradient update (from KL penalty alone) can push the policy across a cliff. After the update, model gets stuck in repetitive / non-terminating generation modes.

**Fixes (try in order):**
1. **Lower learning rate**: `1e-5 → 2e-6 or lower`. SFT-warmed models need gentle updates.
2. **Train on mid-difficulty problems** (Gotcha 2) — ensures actual learning happens, not just drift from KL.
3. **Higher KL penalty**: `beta: 0.01 → 0.05`. Stays closer to reference model.
4. **Increase K** for variance + stability.

## Gotcha 4: DAPO vs GRPO loss

- `loss_type: dapo` (default in trl 0.29.1) has asymmetric clip (clip-higher) and dynamic sampling — better for exploration.
- `loss_type: grpo` is the vanilla PPO-style clipped surrogate.

DAPO is generally preferred for math reasoning (it's what the DAPO paper was designed for). We switched to `grpo` briefly thinking it was more conservative, but actually DAPO+low-LR is the more principled choice.

## Quick pre-flight checklist

Before starting a long GRPO run:

- [ ] SFT merged model at `checkpoints/sft_pivot/merged/` exists
- [ ] Rollouts file exists: `data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl`
- [ ] Config has `rollouts_file` set (not random MATH train)
- [ ] `pass_rate_range: [0.125, 0.875]`
- [ ] `num_generations >= 4`
- [ ] `learning_rate <= 2e-6`
- [ ] `beta >= 0.01`
- [ ] `loss_type: dapo`

Smoke test with 2-3 steps first to check:
- `reward_std > 0.1` (has gradient signal)
- `completions/clipped_ratio < 0.3` (not stuck on max_tokens)
- `entropy` stays 0.02-0.1 (not collapsing to 0)

## Debugging

- Set `ROLLOUT_DUMP_EVERY=1` env var to dump every step's rollouts to `debug_rollouts/step_XXXXXX.json` for inspection
- Tensorboard: `venv/bin/tensorboard --logdir checkpoints/grpo_pivot/final/tb`
- Key metrics to watch:
  - `rewards/alp_reward/mean` — should be in [0.3, 0.7] range (means problems are at right difficulty)
  - `rewards/alp_reward/std` — should be > 0.1 (learning signal)
  - `completions/clipped_ratio` — should be < 0.3
  - `entropy` — should decrease gradually, not collapse to 0
  - `kl` — should grow slowly, not spike
