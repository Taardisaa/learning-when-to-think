"""Concrete running example of the backoff generation loop.

Shows how cache truncation works step by step:
- Forces a specific action sequence (continue → backoff → continue → terminate)
- Prints the full trace: chunk text, actions, cache positions, snapshot restore
- Shows the forward pass sequences used for segment-wise log-prob in GRPO

Usage:
    python examples/backoff_trace.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BackoffConfig
from src.tokens import setup_tokenizer_and_model, TERMINATE_TOKEN, DEPTH_TOKENS
from src.generation import BackoffGenerator


class DemoGenerator(BackoffGenerator):
    """Generator that forces specific action choices for demonstration."""

    def __init__(self, *a, forced_actions=None, **kw):
        super().__init__(*a, **kw)
        self._forced = forced_actions or []
        self._idx = 0

    def _sample_masked(self, logits, allowed_ids, temperature):
        if set(allowed_ids).issubset(set(self.action_ids)):
            if self._idx < len(self._forced):
                forced = self._forced[self._idx]
                self._idx += 1
                if forced in allowed_ids:
                    logits_flat = logits[0, -1, :]
                    mask = torch.full_like(logits_flat, float("-inf"))
                    mask[allowed_ids] = 0.0
                    masked = logits_flat + mask
                    lp = torch.nn.functional.log_softmax(masked, dim=-1)[forced].item()
                    return forced, lp
        return super()._sample_masked(logits, allowed_ids, temperature)


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    token_ids = setup_tokenizer_and_model(tokenizer, model)

    config = BackoffConfig(
        t_max=512, k_min=10, k_max=25, temperature=0.0, k_dir=8, b_max=2
    )
    gen = DemoGenerator(
        model, tokenizer, token_ids, config,
        forced_actions=[
            token_ids["<continue>"],      # after chunk 1
            token_ids["<backoff>"],        # after chunk 2 → backoff!
            token_ids["<continue>"],       # after chunk 3
            token_ids[TERMINATE_TOKEN],    # after chunk 4 → done
        ],
    )

    question = "What is 15 + 27?"
    messages = [
        {"role": "user", "content": (
            f"Solve the following math problem. "
            f"Give the final answer after ####.\n\n{question}"
        )}
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to("cuda")
    prompt_len = prompt_ids.shape[1]

    traj = gen.generate(prompt_ids)

    # ── Print the full story ──
    id2name = {v: k for k, v in token_ids.items()}

    print("=" * 80)
    print("BACKOFF GENERATION TRACE")
    print("=" * 80)
    print(f"\nPrompt ({prompt_len} tokens): {tokenizer.decode(prompt_ids[0])}")

    for i, seg in enumerate(traj.segments):
        action_name = id2name.get(seg.action, str(seg.action))
        print(f"\n{'─' * 80}")
        print(f"SEGMENT {i}")
        print(f"{'─' * 80}")

        chunk_text = tokenizer.decode(seg.chunk_ids)
        print(f"  Chunk ({len(seg.chunk_ids)} tokens, pos {seg.kv_start_pos}→{seg.kv_end_pos}):")
        print(f"    \"{chunk_text[:200]}\"")
        print(f"  Action: {action_name}")

        if seg.depth is not None:
            depth_name = id2name.get(seg.depth, str(seg.depth))
            depth_val = (
                DEPTH_TOKENS.index(depth_name) + 1
                if depth_name in DEPTH_TOKENS else "?"
            )
            print(f"  Depth:  {depth_name} (rewind {depth_val} boundary)")
            print(f"  Rewind target: position {seg.rewind_pos}")
            print()
            print(f"  ┌─ BEFORE BACKOFF: cache has {seg.kv_end_pos + 2} tokens")
            print(f"  │  (chunk={seg.kv_end_pos - seg.kv_start_pos}, "
                  f"+<backoff>, +{depth_name})")
            print(f"  │")
            print(f"  │  TRUNCATE: crop cache from "
                  f"{seg.kv_end_pos + 2} → {seg.rewind_pos}")
            print(f"  │  (deleted {seg.kv_end_pos + 2 - seg.rewind_pos} tokens)")
            print(f"  │")

            dir_text = tokenizer.decode(seg.directive_ids)
            print(f"  │  DIRECTIVE ({len(seg.directive_ids)} tokens): "
                  f"\"{dir_text}\"")
            if seg.force_continued:
                print(f"  │  (force-appended <continue>)")
            print(f"  └─ AFTER DIRECTIVE: cache has "
                  f"{seg.rewind_pos + len(seg.directive_ids)} tokens")
        else:
            print(f"  Cache after action: {seg.kv_end_pos + 1} tokens")

    print(f"\n{'─' * 80}")
    print(f"ANSWER")
    print(f"{'─' * 80}")
    if traj.answer_text:
        print(f"  Text: \"{traj.answer_text[:300]}\"")
    print(f"  Extracted number: {traj.answer_number}")

    print(f"\n{'─' * 80}")
    print(f"SUMMARY")
    print(f"{'─' * 80}")
    print(f"  Segments:           {len(traj.segments)}")
    print(f"  Backoffs:           {traj.backoff_count}")
    print(f"  Total generated:    {traj.total_generated_tokens} tokens")
    print(f"  Final KV length:    {traj.final_kv_length} tokens (T_net for reward)")
    print(f"  Terminated:         {traj.terminated}")
    print(f"  Stored log-prob:    {traj.stored_log_prob:.2f}")

    # ── Show the forward pass sequences ──
    print(f"\n{'─' * 80}")
    print(f"FORWARD PASS SEQUENCES (for segment-wise log-prob in GRPO)")
    print(f"{'─' * 80}")
    seqs = traj.build_forward_pass_sequences()
    for j, seq in enumerate(seqs):
        text = tokenizer.decode(seq)
        print(f"\n  Pass {j} ({len(seq)} tokens):")
        if len(text) > 300:
            print(f"    ...{text[-250:]}")
        else:
            print(f"    {text}")


if __name__ == "__main__":
    main()
