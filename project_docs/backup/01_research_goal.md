# Research Goal

## SMART Goal

By June 30, 2026, implement and evaluate a LoRA-trained RL controller on top of a small open LLM that dynamically selects `continue`, `verify`, `branch`, or `terminate` during inference, and demonstrate:

- On **GSM8K**: at least +2 to +4 absolute accuracy points over fixed-budget CoT at matched average reasoning-token cost, OR at least 20% lower compute-per-correct-answer than self-consistency/best-of-N under equal or better accuracy

- On **HumanEval**: at least +1 to +2 pass@1 points OR 15% lower compute-per-passing-solution than fixed-budget baselines

## Why This Is Novel (The Gap)

Most test-time compute work (2024-2026) addresses **how much** compute to spend, not **what kind**:

- Snell et al. (ICLR 2025): showed adaptive per-prompt compute allocation matters, but uses fixed strategies
- s1 (2025): made controllable reasoning-budget scaling practical, but only scales length
- Yang et al. (NeurIPS 2025): studied optimal reasoning-length distributions, still length-only

**The gap:** No one has trained a unified sequential policy that chooses among semantically different reasoning actions (continue/verify/branch/terminate) online under an explicit compute cost.

## Novelty Risks

- Reviewers may argue this collapses to "better routing" unless it beats termination-only control AND upfront difficulty routing AND verifier-guided search
- RL-for-self-correction work (ICLR 2025) is nearby but focuses on correction training, not budget-aware online action selection

## Constraints

- Single GPU (1x RTX 6000 Ada, 49GB VRAM)
- Open 7B-8B instruct model only (currently using Qwen/Qwen2.5-1.5B-Instruct for pilot)
- LoRA/QLoRA + lightweight RL (PPO)
- Training runs should finish in hours
- Benchmarks: GSM8K + HumanEval (primary), MMLU + HellaSwag (pilot/screening)
