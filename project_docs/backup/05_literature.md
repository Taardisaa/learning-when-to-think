# Key Literature

## Primary Sources (Verified)

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Snell et al., "Scaling LLM Test-Time Compute Optimally..." | ICLR 2025 Oral | Shows adaptive per-prompt compute allocation matters; mainly allocates amount, not action type |
| s1: Simple test-time scaling | 2025 preprint | Makes controllable reasoning-budget scaling practical; only scales length |
| Yang et al., "Towards Thinking-Optimal Scaling..." | NeurIPS 2025 | Optimal reasoning-length distributions; still length-allocation, not action-level |
| MRT: "Optimizing Test-Time Compute via Meta RL Finetuning" | ICML 2025 | Meta-RL for test-time compute; related but different formulation |

## Novelty Threats

| Paper | Why It's a Threat |
|-------|-------------------|
| "Training LMs to Self-Correct via RL" (ICLR 2025 Oral) | RL for self-correction, but not budget-aware online action selection |
| "S^2R: Teaching LLMs to Self-verify and Self-correct via RL" (2025 preprint) | Nearest verification-oriented RL threat; emphasizes correction training, not compute control |
| "Rewarding Progress: Scaling Automated Process Verifiers" (ICLR 2025 Spotlight) | Best modern verifier/search comparison for step-level reward shaping |

## Key Related Work by Theme

### Test-Time Scaling
- Huang et al. (2025) — per-query dynamic compute with latency awareness
- Liu et al. (2025) — prompting strategies change rank as sampling budgets grow
- Chen et al. (2025) — systems innovations for TTS latency on edge hardware

### Verification & Multi-Path
- Forest of Thought (Bi et al.) — multiple reasoning trees with self-correction
- Reflexion (Shinn et al.) — verbal feedback + episodic memory across attempts
- Singhi et al. (2025) — solution generation usually more efficient than verification
- ParaThinker (Wen et al.) — width scaling via parallel paths

### RL for Reasoning
- PPO (Schulman et al., 2017) — stable policy gradient method
- Setlur et al. (2025) — RL with negative gradients for exploratory search

## Identified Gaps (From Synthesis)

1. No unified sequential policy over inference actions (continue/verify/branch/terminate)
2. Verification studied as module, not as budgeted action
3. RL-for-reasoning papers don't tightly connect to correctness-cost tradeoffs on standard benchmarks
4. Termination decisions are underexplored
5. Cross-domain evidence for adaptive compute policies is limited
6. Efficiency metrics are fragmented across papers
