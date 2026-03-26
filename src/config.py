from dataclasses import dataclass, field


@dataclass
class BackoffConfig:
    # Model
    model_name: str = "Qwen/Qwen3.5-0.8B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_dropout: float = 0.05

    # Boundary detection
    k_min: int = 15   # minimum tokens before boundary check
    k_max: int = 80   # forced boundary after this many tokens

    # Backoff constraints
    d_max: int = 3    # max rewind depth (number of boundaries)
    k_dir: int = 20   # max directive tokens after backoff
    b_max: int = 2    # max backoffs per trajectory
    t_max: int = 2048 # max total tokens in trajectory

    # Reward
    lambda_tok: float = 0.0001    # token cost coefficient
    lambda_explore: float = 0.1   # exploration bonus (Phase 2, annealed)
    anneal_steps: int = 500       # steps to anneal lambda_explore to 0

    # GRPO
    num_rollouts: int = 4         # G — group size
    clip_epsilon: float = 0.2     # PPO clipping
    kl_coef: float = 0.01        # KL penalty coefficient
    lr: float = 1e-4             # learning rate (Phase 2/3)

    # SFT (Phase 1)
    sft_epochs: int = 3
    sft_lr: float = 2e-5
    sft_max_seq_len: int = 2048
    sft_batch_size: int = 1
    sft_grad_accum: int = 4

    # Generation
    temperature: float = 0.7     # sampling temperature for rollouts
    eval_temperature: float = 0.0  # greedy for eval

    # System prompt used in chat template
    system_prompt: str = (
        "Solve the following math problem. "
        "Give the final answer after ####."
    )
