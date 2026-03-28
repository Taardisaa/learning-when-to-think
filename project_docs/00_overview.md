# Project Overview

## What This Project Is

**Goal:** Train a small open LLM (via LoRA + GRPO) that learns to **undo bad reasoning mid-generation** — and show this improves accuracy over standard forward-only reasoning at the same token cost.

**Problem:** Current LLMs can only append tokens. Once a model goes down a wrong reasoning path, those tokens stay in context, waste the context window, and can mislead subsequent reasoning. The model has no way to say "that was wrong, let me start this part over."

**Our approach:** Add `<backoff_N>` actions (N = rewind depth) that hard-truncate unhelpful tokens from the KV cache, reclaim context space, and optionally inject a short directive before resuming. The model learns when to use them through RL. The action tokens are $\mathcal{A} = \{\texttt{<backoff\_1>}, \texttt{<backoff\_2>}, \texttt{<backoff\_3>}, \texttt{</think>}\}$. At semantic boundaries during generation, the model may emit a backoff token (to rewind) or `</think>` (to terminate reasoning). If neither is emitted, generation continues naturally — no explicit `<continue>` token is needed.

**Key claim:** A model with access to `<backoff_N>` achieves higher accuracy than one that can only continue or stop, at matched token cost — because it can recover from mistakes instead of being stuck with them.

## What IS The Novelty

Prior work controls **quantity** of reasoning. We control **type** — the model learns qualitatively different operations, not just how long to think. Backoff gives the model an "undo" operation: recognize a dead end, delete it, reclaim context, and try again with a directive.

| Prior work | Model | Ours |
|-----------|-------|------|
| Snell et al. (ICLR 2025) | Forward-only, vary length | Forward + undo |
| s1 (2025) | Forward-only, learned stop | Forward + undo + redirect |
| Yang et al. (NeurIPS 2025) | Forward-only, optimal length | Non-monotonic (can shorten mid-trajectory) |
| DeepSeek-R1 | Forward-only, `<think>`/`</think>` | `<backoff_1/2/3>` + `</think>` with KV cache truncation |

**What is NOT novel:** Adaptive compute allocation (spending more tokens on hard problems). If our contribution reduces to "the model learns to stop early on easy problems," we have no paper.

**Context reclamation is the key differentiator:** In a context-limited setting, every useless token actively hurts. Backoff via hard KV cache truncation actually reclaims context space, not just masks it. This is what differentiates us from self-correction work (which only appends corrections).

**Key ablation:** backoff with KV truncation vs. backoff with attention masking only — to prove context reclamation matters.

## Architecture

The LLM itself is the controller. Semantic boundaries (punctuation, logical connectives, structural markers) are detected heuristically within $[K_{\min}, K_{\max}]$ tokens. At each boundary, the model may emit a backoff or terminate token; if it emits neither, generation continues naturally. GRPO updates the LoRA weights to optimize $R(\tau) = r(\hat{a}, a^*) + \alpha \sum_j r_{\text{prg}}(\mathbf{z}_j; \mathbf{c}_{j-1})$.

RH: 为了知道$K_{\min}$和$K_{\max}$的最佳值，我们可以从现有的output里面去smaple一下。

See [02_method.md](02_method.md) for the full MDP, generation loop, and training objectives.

## Baselines

**Category 1: Test-time reasoning (no weight updates):**
CoT, Self-Consistency (majority vote), Tree-of-Thought, ReAct, Budget forcing

**Category 2: Finetuning-based (weight updates, no backoff):**
SFT on CoT traces, RL with terminate only, DeepSeek-R1 style GRPO

Our method is Category 2 + `<backoff_N>`. The key comparison is against Category 2 at matched token cost.

## Benchmarks

| Benchmark | Role | Why |
|-----------|------|-----|
| MATH (Hendrycks et al.) | Training | Competition-level problems; 1.7B model fails often enough to produce usable wrong rollouts for S²R-style SFT |
| MATH-500 (HuggingFaceH4) | Evaluation | 500-problem subset; `\boxed{}` answer extraction; standard eval benchmark |

## Constraints

- 2 × A6000, 48 × 2 = 96 GB VRAM
- Primary target model: Qwen3-1.7B (also tested on Qwen3-4B-Thinking-2507)
- LoRA + GRPO (no PPO, no value network)
- Target venue: NeurIPS/ICLR/ICML

## Current Status

Baseline forward-only evaluations on MATH-500 (`\boxed{}` extraction):

| Model | Accuracy | Notes |
|-------|----------|-------|
| Qwen3-1.7B (base, thinking) | 68.4% | Greedy, vLLM tp=2 |
| Qwen3-1.7B-Base (non-thinking) | 0% | Cannot produce `\boxed{}` answers |
| Qwen3-1.7B + Phase 1 SFT (real backoff) | 90.4% | No backoffs triggered yet — format learned but not used |

SFT data: 104 real-backoff examples from MATH train (S²R-style stitching of model's own wrong + correct rollouts).

Next: scale up SFT data (full MATH train), then Phase 2 GRPO with progress reward.

