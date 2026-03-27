"""Concrete running example of the backoff generation loop.

Shows how cache truncation works step by step:
- Forces a specific token sequence (continue → backoff_1 → continue → terminate)
- Prints the full trace: chunk text, actions, cache positions, snapshot restore
- Shows the forward pass sequences used for segment-wise log-prob in GRPO

Usage:
    python examples/backoff_trace.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BackoffConfig
from src.tokens import setup_tokenizer_and_model, TERMINATE_TOKEN, BACKOFF_TOKENS
from src.generation import BackoffGenerator


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    token_ids = setup_tokenizer_and_model(tokenizer, model)

    config = BackoffConfig(
        t_max=512, temperature=0.0, k_dir=8,
    )
    gen = BackoffGenerator(model, tokenizer, token_ids, config)

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
    backoff_id_set = set(token_ids[t] for t in BACKOFF_TOKENS)

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
        print(f"    \"{chunk_text}\"")
        print(f"  Action: {action_name}")

        if seg.action in backoff_id_set:
            depth_val = BACKOFF_TOKENS.index(action_name) + 1 if action_name in BACKOFF_TOKENS else "?"
            print(f"  Rewind depth: {depth_val} boundary(ies)")
            print(f"  Rewind target: position {seg.rewind_pos}")
            print()
            print(f"  ┌─ BEFORE BACKOFF: cache has {seg.kv_end_pos + 1} tokens")
            print(f"  │  (chunk={seg.kv_end_pos - seg.kv_start_pos}, +{action_name})")
            print(f"  │")
            print(f"  │  TRUNCATE: crop cache from "
                  f"{seg.kv_end_pos + 1} → {seg.rewind_pos}")
            print(f"  │  (deleted {seg.kv_end_pos + 1 - seg.rewind_pos} tokens)")
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
        print(f"  Text: \"{traj.answer_text}\"")
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
        print(f"    {text}")


if __name__ == "__main__":
    main()
