# Project Overview

## What This Project Is

**Goal:** Train a small open LLM (via LoRA + GRPO) that learns to **undo bad reasoning mid-generation** — and show this improves accuracy over standard forward-only reasoning at the same token cost.

**Problem:** Current LLMs can only append tokens. Once a model goes down a wrong reasoning path, those tokens stay in context, waste the context window, and can mislead subsequent reasoning. The model has no way to say "that was wrong, let me start this part over."

**Our approach:** Add a $\langle\texttt{backoff}\rangle$ action that hard-truncates unhelpful tokens from the KV cache, reclaims context space, and optionally injects a short directive before resuming. The model learns when to use it through RL. The full action space is $\mathcal{A} = \{\langle\texttt{continue}\rangle, \langle\texttt{backoff}\rangle, \langle\texttt{terminate}\rangle\}$, decided at semantic boundaries during generation.

**Key claim:** A model with access to $\langle\texttt{backoff}\rangle$ achieves higher accuracy than one that can only continue or stop, at matched token cost — because it can recover from mistakes instead of being stuck with them.

## What IS The Novelty

Prior work controls **quantity** of reasoning. We control **type** — the model learns qualitatively different operations, not just how long to think. Backoff gives the model an "undo" operation: recognize a dead end, delete it, reclaim context, and try again with a directive.

| Prior work | Model | Ours |
|-----------|-------|------|
| Snell et al. (ICLR 2025) | Forward-only, vary length | Forward + undo |
| s1 (2025) | Forward-only, learned stop | Forward + undo + redirect |
| Yang et al. (NeurIPS 2025) | Forward-only, optimal length | Non-monotonic (can shorten mid-trajectory) |
| DeepSeek-R1 | Forward-only, `<think>`/`</think>` | `<continue>`/`<backoff>`/`<terminate>` with KV cache truncation |

**What is NOT novel:** Adaptive compute allocation (spending more tokens on hard problems). If our contribution reduces to "the model learns to stop early on easy problems," we have no paper.

**Context reclamation is the key differentiator:** In a context-limited setting, every useless token actively hurts. Backoff via hard KV cache truncation actually reclaims context space, not just masks it. This is what differentiates us from self-correction work (which only appends corrections).

**Key ablation:** backoff with KV truncation vs. backoff with attention masking only — to prove context reclamation matters.

## Architecture

The LLM itself is the controller. Action decisions occur at **semantic boundaries** (punctuation, logical connectives, structural markers) within $[K_{\min}, K_{\max}]$ tokens. At each boundary, the model's logits over $\mathcal{A} \subset \mathcal{V}$ determine the next action. GRPO updates the LoRA weights to optimize $R(\tau) = r(\hat{a}, a^*) - \lambda_{\text{tok}} \cdot T_{\text{net}}(\tau)$.

RH: 为了知道$K_{\min}$和$K_{\max}$的最佳值，我们可以从现有的output里面去smaple一下。

See [02_method.md](02_method.md) for the full MDP, generation loop, and training objectives.

## Baselines

**Category 1: Test-time reasoning (no weight updates):**
CoT, Self-Consistency (majority vote), Tree-of-Thought, ReAct, Budget forcing

**Category 2: Finetuning-based (weight updates, no backoff):**
SFT on CoT traces, RL with continue/terminate only ($|\mathcal{A}| = 2$), DeepSeek-R1 style GRPO

Our method is Category 2 + $\langle\texttt{backoff}\rangle$. The key comparison is against Category 2 at matched token cost.

## Benchmarks

| Benchmark | Role | Why |
|-----------|------|-----|
| GSM8K | Primary | Multi-step math reasoning; backoff = undo wrong calculation, retry |
| HumanEval | Primary | Code generation; backoff = discard buggy impl, redirect |
| MMLU | Pilot | Quick eval, multiple-choice, broad coverage |
| HellaSwag | Pilot | Commonsense transfer test |

## Constraints

- 2 × A6000, 48 × 2 = 96 GB VRAM
- Small models only (Qwen3.5 0.8B / 2B / 4B / 9B)
- LoRA + GRPO (no PPO, no value network)
- Target venue: NeurIPS/ICLR/ICML

## Current Status

Baseline forward-only evaluations on GSM8K (500 samples, strict `####` extraction, last-300-char window):

| Model | Accuracy | Extracted | Avg Tokens |
|-------|----------|-----------|------------|
| Qwen3.5-0.8B | 7.4% | 15.2% | 5732 |
| Qwen3.5-2B | 31.6% | 38.2% | 4988 |
| Qwen3.5-4B | 70.0% | 78.4% | 3140 |
| Qwen3.5-9B | 88.2% | 93.8% | 3428 |

Next: complete baselines, then implement three-phase training (SFT warm-up → GRPO + exploration bonus → pure GRPO).

