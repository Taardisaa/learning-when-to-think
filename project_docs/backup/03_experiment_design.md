# Experiment Design

## Base Model

**Qwen/Qwen2.5-1.5B-Instruct** (pilot)
- 4-bit QLoRA + small controller/value heads
- Fits single GPU, allows multiple seeds within 3600s budget
- Paper target: scale up to 7B-8B for final results

## Learned Conditions (5 trained)

| # | Condition | Actions | Tests |
|---|-----------|---------|-------|
| 1 | **ChunkwiseTerminateOrContinueController** | continue, terminate | H1 |
| 2 | **TaskAwareProbeCommitConflictVerificationRouter** | probe -> mode commit (early_terminate / continue_linear / branch_once / late_verify) | H2, H3, H4 |
| 3 | **StepwiseFourActionMicromanager** (ablation) | continue, verify, branch, terminate at every step | H2 (comparison) |
| 4 | **ConfidenceTriggeredProbeRouter** (ablation) | same as #2 but verify on low confidence, not conflict | H3 (comparison) |
| 5 | **PerDomainSpecialistProbeRouters** (ablation) | separate math/code controllers | H4 (comparison) |

## Inference-Only Baselines (3, no training budget)

| Baseline | Purpose |
|----------|---------|
| **BudgetForcedFixedChainBaseline** | Fixed-length CoT, H1 comparison |
| **ThreeBinDifficultyBudgetRouter** | Simple upfront routing, must be beaten for novelty |
| **SelfConsistencySamplingBaseline** | Best-of-N at matched token budget |

## Training Protocol

- **RL algorithm:** PPO with clipped surrogate, GAE (lambda=0.95), KL penalty to frozen reference
- **Reward:** correctness - lambda_tok * tokens - lambda_branch * branches - lambda_verify * verifications
- **Chunk size:** 32 tokens per reasoning step
- **Probe:** 24 tokens (for probe-based conditions)
- **Controller MLP:** Linear(hidden+stats+domain, 256) -> Tanh -> Linear(256,256) -> Tanh -> policy/value heads
- **Seeds:** 5 per condition (pilot), 10 for top methods (confirmatory)
- **Max updates per seed:** capped by time budget (~120s per seed per condition)

## Key Hyperparameters

```
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
controller_lr: 1e-4
lora_lr: 5e-5
ppo_clip: 0.2
kl_coef: 0.02
entropy_coef: 0.01-0.02
gradient_clip: 1.0
reward_token_penalty: 0.002
reward_branch_penalty: 0.01
reward_verify_penalty: 0.008
```

## Evaluation

- **Regimes:** tight budget (2-4 decision steps) and moderate budget (4-8 steps)
- **Episodes:** 30 per seed per regime
- **Primary metric:** accuracy (exact-match for GSM8K, pass@1 for HumanEval)
- **Secondary metrics:** compute_per_correct_answer, reasoning_token_usage, branch_rate, verification_rate, late_verification_share, token_reduction_at_matched_accuracy
- **Statistics:** Wilcoxon signed-rank, paired bootstrap, Cohen's d, 95% CI

## Datasets

**Current pilot:** MMLU + HellaSwag (multiple-choice reasoning)
**Paper target:** GSM8K + HumanEval (math + code)

Note: The benchmark agent selected MMLU/HellaSwag for the pilot. The goal.md specifies GSM8K/HumanEval as the real targets. This mismatch needs to be resolved for the final paper.

## Risk Mitigations

- Policy collapse detection: abort if terminate rate >95% or <5% for 2 consecutive evals
- NaN guard: abort and restart seed on any NaN in logits/values/loss
- Reward bounded to [-1, 1]
- Verification limited to 1x per episode in pilot
