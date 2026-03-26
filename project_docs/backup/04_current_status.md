# Current Status

## Pipeline Progress

| Stage | Name | Status | Notes |
|-------|------|--------|-------|
| 1 | TOPIC_INIT | done | Goal + scope defined |
| 2 | PROBLEM_DECOMP | done | 6 sub-questions prioritized |
| 3 | LIT_SEARCH | done | 12 sources found |
| 4 | LIT_COLLECT | done | Web search returned 0 additional results |
| 5 | LIT_SCREEN (GATE) | done | Auto-approved |
| 6 | LIT_EXTRACT | done | |
| 7 | SYNTHESIS | done | 4 clusters, 6 gaps identified |
| 8 | HYPOTHESIS_GEN | done | 4 hypotheses, novelty score 1.0 |
| 9 | EXP_DESIGN (GATE) | done | Full experiment plan with conditions/baselines |
| 10 | CODE_GENERATION | done (2nd attempt) | 1st failed: `hitl` attribute error. 2nd succeeded after rerun |
| 11 | RESOURCE_PLAN | done | Schedule created |
| 12 | RUN_EXPERIMENTS | done | Docker sandbox execution |
| 13 | ITERATIVE_REFINE | done | 10 iterations, all failed with same bugs |
| 14 | RESULT_ANALYSIS | done | Correctly noted: zero results |
| 15 | RESEARCH_DECISION | done | Verdict: REFINE |

## What Failed and Why

The experiment code never ran successfully. Two recurring bugs across all 10 refinement iterations:

1. **`requirements.txt` includes `__future__`** — pip rejects this as it's a stdlib module, not a pip package. The LLM-generated requirements scanner included it from `from __future__ import annotations`.

2. **`data.py` line 332/437: unterminated string literal** — A regex pattern `re.findall(r"` was split across lines incorrectly by the code generator, creating a syntax error. The self-healing system tried to fix it 10 times but kept reproducing the same error.

## Repair Attempts After Stage 15

3 additional repair cycles ran, all failed (score stayed 0.0 -> 0.0). The same two bugs persisted.

## What's Good

- Research framing is solid and well-structured
- Hypotheses are testable with clear success/failure criteria
- Experiment design is thorough with proper controls
- The code architecture is reasonable (config.py, data.py, models.py, training.py, main.py)
- Docker sandbox correctly installs deps and downloads datasets
- The AdaptiveControllerBase with PPO training is implemented

## What Needs Fixing Before Rerun

1. Fix the `requirements.txt` generation (exclude stdlib modules)
2. Fix the `data.py` string literal syntax error
3. Consider: the pilot uses MMLU/HellaSwag but the paper targets GSM8K/HumanEval
4. Stage 14 analysis flagged: baseline set is insufficient — need termination-only and difficulty-router baselines
5. Compute accounting needs to count all branches/model calls, not just reasoning tokens
