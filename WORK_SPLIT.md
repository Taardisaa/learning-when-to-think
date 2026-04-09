# Work Split — Learning When to Think

## Project Pipeline

The core idea of this project is: **LLMs waste compute by thinking the same amount on every problem.** Easy problems get over-processed, hard problems don't get enough. We want to train a model that learns *when to keep thinking and when to stop*.

### 1. Data

We use **MATH-500**, a 500-problem subset of the MATH benchmark covering competition-level math. Each problem has a ground-truth answer we can check against. This is our training and evaluation dataset.

### 2. Base Model

We start from **Qwen2.5-Math-7B-Instruct**, a 7B-parameter math-specialized LLM. Rather than fine-tuning the entire model (which would be expensive and risk catastrophic forgetting), we attach **LoRA adapters** (rank 16, alpha 32) to both the attention layers and the MLP layers. This lets us train the reasoning behavior while keeping most of the model frozen.

### 3. Action Policy (The Key Idea)

At each reasoning step, the model doesn't just generate tokens freely — it first picks one of **three actions**:

- **`continue`**: Keep reasoning along the current line of thought.
- **`refine`**: Pause and re-check recent reasoning for mistakes before continuing (a lightweight self-correction).
- **`terminate`**: Stop reasoning and output the final answer.

This turns reasoning into a **sequential decision problem**. The model must learn when a problem is easy enough to stop early, when it needs more thinking, and when it should go back and double-check.

### 4. Reward Function (ALP)

We use **Adaptive Length Penalty (ALP)** to score each rollout. The reward combines two signals:

- **Correctness**: Did the model get the right answer? (0 or 1)
- **Efficiency penalty**: How many tokens did it use? But crucially, this penalty is **scaled by how easy the problem is** for the current model. If the model already solves a problem most of the time (high group solve rate), we penalize long solutions more — it should learn to be brief on easy problems. If the problem is hard (low solve rate), we penalize length less — it's okay to think longer when the problem genuinely needs it.

### 5. Training (GRPO + DeGRPO)

We train using **GRPO (Group Relative Policy Optimization)**: for each problem, sample K rollouts, score them with ALP, and compute advantages relative to the group (no separate critic model needed).

On top of GRPO, we apply **DeGRPO weighting**: the loss function treats **control tokens** (the action decisions — continue/refine/terminate) differently from **response tokens** (the actual math reasoning). Control tokens get a higher weight in the gradient, so the model prioritizes learning *when to act* over *what to say*. Without this, the control signal gets drowned out by the much larger number of response tokens.

### 6. Baselines

To show our method actually helps, we compare against:

- **CoT single-pass**: Standard chain-of-thought — the model just reasons once with no adaptive stopping.
- **Direct-answer**: No reasoning at all — the model just outputs an answer immediately.
- **Vanilla GRPO**: Same GRPO training but without the DeGRPO weighting (control and response tokens treated equally).

### 7. Evaluation

We measure:

- **Accuracy**: How many problems does the model get right?
- **Avg tokens per problem**: How much compute does it use on average?
- **Cost per correct answer**: Tokens spent divided by correct answers — the efficiency metric.
- **Action usage by difficulty**: Do easy problems get more `terminate` actions and hard problems get more `continue`/`refine`? This is the key behavioral evidence that the model learned adaptive reasoning.

---

## Member 1 — Data Pipeline, Evaluation & Reward

- MATH-500 data loading and preprocessing (evaluation only)
- Answer extraction & grading (`\boxed{}` and `####` parsing, normalized math-string matching)
- Evaluation harness: compute accuracy, avg tokens, cost-per-correct, per-difficulty breakdowns
- **ALP reward function**: compute group solve rate SR(q), token-length penalty, final reward R_k
- **SFT warmup dataset**: generate K=8 rollouts from Qwen2.5-Math-7B-Instruct on MATH train (~1000 problems), construct ~1500–2000 training examples with `<continue>`/`<refine>`/`<terminate>` action tokens injected at semantic boundaries
- **Action token setup**: add 3 new special tokens to tokenizer, initialize embeddings

> Owns all scoring logic and training data — from raw model output to final metrics, reward signals, and SFT warmup data.

## Member 2 — Core RL Training

- LoRA setup on Qwen2.5-Math-7B-Instruct (attention + MLP projections)
- 3-action policy with special control tokens (`continue`/`refine`/`terminate`)
- GRPO advantage computation and policy update step
- **DeGRPO gradient weighting**: split tokens into control/response slices, apply weighted loss
- `allow_refine` and `degrpo` ablation switches

> Owns the RL training core — takes reward from Member 1, rollouts from Member 3.

## Member 3 — Generation Infrastructure, Experiments & Analysis

- **Rollout / generation pipeline**: K-rollout sampling loop per prompt using the base model (shared by baselines and GRPO)
- **Baseline implementations**: CoT single-pass decoding and direct-answer (no reasoning)
- Run all experiments: main method, ablations (β values, `allow_refine` toggle, `degrpo` flag)
- Vanilla GRPO baseline (degrpo=false)
- Results collection, tables, and plots (accuracy-efficiency Pareto curves, action distribution bar charts by difficulty)
- Hypothesis testing: verify H1/H2/H3 with the collected data

> Owns the generation loop and all experiment execution — can start immediately by building the inference pipeline and running baselines.

---

## Dependency Order

```
Member 1 (data/reward)  ──────────────────────>  Member 2 (RL training)
         |                                              |
         v                                              v
Member 3 (generation/baselines)  ──────────>  Member 3 (full experiments)
```

We have **13 days** (Apr 7 – Apr 20). Presentation is Apr 20.

- **Apr 7–11 (Days 1–5)**: All three work in parallel.
  - Member 1: data loader + grader + ALP reward + baseline eval ✅ → then start SFT warmup data (rollout generation + dataset construction)
  - Member 2: LoRA setup + action token scheme + GRPO update logic
  - Member 3: rollout generation loop + baselines (CoT, direct-answer)
- **Apr 12–14 (Days 6–8)**: Integration + SFT warmup.
  - Member 1 delivers SFT warmup dataset + token setup
  - Member 2 runs SFT warmup training, then wires up GRPO with Member 1's ALP reward
  - Member 3 integrates eval harness with rollout loop, collects baseline results
- **Apr 15–18 (Days 9–12)**: Experiments.
  - Member 3 runs full method + ablations
  - Everyone works on results analysis, plots, and writeup
- **Apr 19–20 (Days 13–14)**: Presentation prep.
  - Finalize slides, rehearse, present on Apr 20
