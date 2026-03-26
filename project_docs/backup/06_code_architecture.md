# Experiment Code Architecture

## Location

Generated code lives in:
```
artifacts/rc-20260324-215102-f9199b/stage-10/agent_sandbox/
  _project_1/    # Initial generation (local sandbox)
  _project_2/    # After datasets shim fix
  _project_3/    # After scipy shim fix
  _docker_project_1/  # Docker sandbox run (deps installed via pip)
```

Final experiment versions (from stage 13 refinement):
```
artifacts/rc-20260324-215102-f9199b/stage-13/experiment_v{1-10}/
artifacts/rc-20260324-215102-f9199b/stage-13/experiment_final/
```

## File Structure

| File | Purpose | Size |
|------|---------|------|
| `config.py` | ConditionSpec, ExperimentConfig, TimeBudget, seed setup, condition registry | ~11KB |
| `data.py` | Dataset loading (MMLU/HellaSwag), prompt formatting, EpisodeBatch collation | ~11-17KB |
| `models.py` | AdaptiveControllerBase (backbone + LoRA + controller MLP + policy/value heads) | ~45KB |
| `training.py` | RolloutBuffer, PPOTrainer, ExperimentRunner, Evaluator, statistical tests | ~13KB |
| `main.py` | Entry point — runs all conditions across seeds, emits metrics | ~15KB |
| `experiment_harness.py` | ResearchClaw metric output wrapper | ~4KB |
| `setup.py` | Dataset download script (runs with network before experiment) | varies |

## Key Classes

### AdaptiveControllerBase (models.py)
- Loads Qwen2.5-1.5B-Instruct with QLoRA (4-bit NF4)
- Adds controller MLP: hidden_state + stats + domain_embed -> 256 -> 256 -> policy/value heads
- Policy head outputs action logits over {continue, terminate, verify, branch}
- Value head outputs scalar state value

### PPOTrainer (training.py)
- Separate LR for controller params vs backbone LoRA params
- Clipped surrogate PPO with GAE
- KL penalty to frozen reference policy
- Gradient clipping at 1.0

### RolloutBuffer (training.py)
- Stores (state, action, old_logprob, value, reward, done) tuples
- Used for PPO updates after collecting rollout episodes

### ExperimentRunner (training.py)
- Orchestrates training across conditions and seeds
- Manages time budget
- Handles per-condition training loops

## Execution Flow

1. `main.py` builds config, sets seeds
2. For each condition x seed:
   a. Build model (AdaptiveControllerBase with condition-specific action space)
   b. Load data (BenchmarkDataModule)
   c. Train with PPO (RolloutBuffer -> PPOTrainer.train_step)
   d. Evaluate on tight + moderate budget regimes
3. Compute cross-condition statistics (Wilcoxon, bootstrap CI)
4. Emit metrics via experiment_harness

## Docker Execution Phases

When mode=docker:
1. **Phase 0:** `pip install -r requirements.txt`
2. **Phase 1:** `python3 setup.py` (dataset download)
3. **Phase 2:** `python3 main.py` (experiment, network optionally disabled)
