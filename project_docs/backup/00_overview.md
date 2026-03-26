# Project Overview

**Run ID:** `rc-20260324-215102-f9199b`

**Started:** 2026-03-24 ~21:51 UTC

**Pipeline reached:** Stage 15 (RESEARCH_DECISION) -> verdict: REFINE (loop back)

**Experiment status:** FAILED (no metrics produced)

## What This Project Is

A research paper attempting to show that a **lightweight RL controller** can make LLMs reason more efficiently at inference time. Instead of using fixed chain-of-thought or best-of-N sampling, the idea is to train a small policy (via LoRA + PPO) that dynamically chooses **what kind** of reasoning to do next:

- `continue` — keep generating
- `verify` — check intermediate results
- `branch` — try an alternative path
- `terminate` — stop early, emit answer

The key claim: this learned controller achieves better accuracy-per-token than fixed strategies, especially on smaller open models.

## One-Line Paper Pitch

> Learn a discrete action policy over reasoning operations for small open LLMs, and show improved accuracy-per-token over fixed CoT and self-consistency.

## Target Venue

Conference-level (NeurIPS/ICLR/ICML style)

## What Actually Happened

1. Stages 1-9 (research scoping through experiment design) completed successfully
2. Stage 10 (code generation) initially failed due to missing `hitl` attribute, then succeeded on rerun
3. Stage 12-13 (execution + refinement) ran but the Docker sandbox had recurring bugs:
   - `__future__` in `requirements.txt` (invalid pip requirement)
   - Unterminated string literal in `data.py:332`/`437`
   - Self-healing tried 10 iterations but couldn't fix the syntax error
4. Stage 14 (analysis) correctly noted: zero experimental results
5. Stage 15 decided: REFINE (go back and fix)
6. 3 repair cycles also failed (score stayed 0.0)

**Bottom line: The research design is solid but no experiment has successfully run yet.**
