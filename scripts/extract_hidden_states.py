"""Extract Qwen3-1.7B hidden states and align with boundary annotations.

Runs frozen Qwen3-1.7B forward passes on annotated rollouts, extracts
hidden states at all layers, and maps boundary char offsets to token positions.

Usage:
    # Full extraction (all layers, for layer sweep pilot)
    python scripts/extract_hidden_states.py \
        --annotations data/boundary_annotations.jsonl \
        --model Qwen/Qwen3-1.7B \
        --num-rollouts 200 \
        --all-layers

    # Targeted extraction (specific layers, for full dataset)
    python scripts/extract_hidden_states.py \
        --annotations data/boundary_annotations.jsonl \
        --model Qwen/Qwen3-1.7B \
        --layers 8 12 16 20
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def _map_char_offsets_to_tokens(
    text: str,
    char_offsets: list[int],
    tokenizer,
) -> tuple[list[int], list[int]]:
    """Map character offsets to token indices using offset_mapping.

    Returns:
        token_ids: list of token IDs
        boundary_token_indices: token indices that correspond to boundary positions
    """
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    token_ids = encoding["input_ids"][0].tolist()
    offset_mapping = encoding["offset_mapping"][0].tolist()  # [(start, end), ...]

    boundary_token_indices = []
    for char_off in char_offsets:
        # Find the token whose span contains this character offset
        for tok_idx, (start, end) in enumerate(offset_mapping):
            if start <= char_off < end:
                boundary_token_indices.append(tok_idx)
                break
        else:
            # If exact match fails, find nearest token
            min_dist = float("inf")
            best_idx = 0
            for tok_idx, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:
                    continue  # skip special tokens
                dist = min(abs(char_off - start), abs(char_off - end))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = tok_idx
            boundary_token_indices.append(best_idx)

    return token_ids, sorted(set(boundary_token_indices))


def extract_hidden_states(
    model_name: str,
    annotations: list[dict],
    layers: list[int] | None = None,
    all_layers: bool = False,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict:
    """Extract hidden states from frozen model and align with boundary labels.

    Returns dict with:
        features: {layer_idx: tensor [total_tokens, hidden_dim]}
        labels: tensor [total_tokens] (1 = boundary, 0 = not)
        rollout_boundaries: list of (start_idx, end_idx) per rollout
        metadata: list of {question, correct, num_tokens, num_boundaries}
    """
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Layers: {num_layers}, Hidden dim: {hidden_dim}")

    if all_layers:
        target_layers = list(range(num_layers + 1))  # +1 for embedding layer
    elif layers:
        target_layers = layers
    else:
        # Default: middle and late layers (most informative per literature)
        target_layers = [num_layers // 4, num_layers // 2,
                         3 * num_layers // 4, num_layers]

    print(f"  Extracting layers: {target_layers}")

    # Process rollouts
    all_features = {l: [] for l in target_layers}
    all_labels = []
    rollout_boundaries = []
    metadata = []
    global_idx = 0

    t0 = time.time()
    for i, ann in enumerate(annotations):
        text = ann["text"]
        char_offsets = ann["boundary_char_offsets"]

        # Tokenize and map boundaries
        token_ids, boundary_tok_indices = _map_char_offsets_to_tokens(
            text, char_offsets, tokenizer
        )
        num_tokens = len(token_ids)

        # Create binary labels
        labels = torch.zeros(num_tokens, dtype=torch.long)
        for idx in boundary_tok_indices:
            if idx < num_tokens:
                labels[idx] = 1

        # Forward pass
        input_ids = torch.tensor([token_ids], device=device)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # Extract hidden states at target layers
        # outputs.hidden_states is tuple of (num_layers+1) tensors, each [1, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states

        for layer_idx in target_layers:
            if layer_idx < len(hidden_states):
                hs = hidden_states[layer_idx][0].cpu().float()  # [seq_len, hidden_dim]
                all_features[layer_idx].append(hs)

        all_labels.append(labels)
        rollout_boundaries.append((global_idx, global_idx + num_tokens))
        global_idx += num_tokens

        metadata.append({
            "question": ann["question"][:100],
            "correct": ann["correct"],
            "num_tokens": num_tokens,
            "num_boundaries": len(boundary_tok_indices),
            "boundary_token_indices": boundary_tok_indices,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i+1}/{len(annotations)} "
                  f"({elapsed:.1f}s, {elapsed/(i+1):.2f}s/rollout)")

    elapsed = time.time() - t0
    print(f"  Done: {len(annotations)} rollouts in {elapsed:.1f}s")

    # Concatenate
    features = {}
    for layer_idx in target_layers:
        if all_features[layer_idx]:
            features[layer_idx] = torch.cat(all_features[layer_idx], dim=0)
    labels = torch.cat(all_labels, dim=0)

    total_tokens = len(labels)
    total_boundaries = labels.sum().item()
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total boundaries: {total_boundaries} ({100*total_boundaries/total_tokens:.1f}%)")

    # Free model
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "features": features,
        "labels": labels,
        "rollout_boundaries": rollout_boundaries,
        "metadata": metadata,
        "target_layers": target_layers,
        "hidden_dim": hidden_dim,
    }


def compute_displacement(features: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Compute layer-wise displacement: d[l] = h[l+1] - h[l].

    Returns displacement features keyed by the lower layer index.
    """
    sorted_layers = sorted(features.keys())
    displacement = {}

    for i in range(len(sorted_layers) - 1):
        l_low = sorted_layers[i]
        l_high = sorted_layers[i + 1]
        if l_high == l_low + 1:  # only for consecutive layers
            displacement[l_low] = features[l_high] - features[l_low]

    return displacement


def save_extracted(data: dict, output_dir: str):
    """Save extracted features and labels to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save labels
    torch.save(data["labels"], out / "labels.pt")

    # Save features per layer
    for layer_idx, feat in data["features"].items():
        torch.save(feat, out / f"features_layer{layer_idx}.pt")

    # Save metadata
    meta = {
        "rollout_boundaries": data["rollout_boundaries"],
        "metadata": data["metadata"],
        "target_layers": data["target_layers"],
        "hidden_dim": data["hidden_dim"],
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_size = sum(f.stat().st_size for f in out.iterdir()) / 1e9
    print(f"\nSaved to {output_dir} ({total_size:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Qwen3-1.7B hidden states for boundary probe training"
    )
    parser.add_argument("--annotations", required=True,
                        help="Path to boundary annotations JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", default="data/probe_features",
                        help="Output directory for extracted features")
    parser.add_argument("--num-rollouts", type=int, default=None,
                        help="Limit number of rollouts to process")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to extract (e.g., --layers 8 12 16 20)")
    parser.add_argument("--all-layers", action="store_true",
                        help="Extract all layers (for layer sweep)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load annotations
    print(f"Loading annotations from {args.annotations}...")
    with open(args.annotations) as f:
        annotations = [json.loads(line) for line in f]

    if args.num_rollouts and args.num_rollouts < len(annotations):
        annotations = annotations[:args.num_rollouts]
    print(f"  Processing {len(annotations)} rollouts")

    # Extract
    data = extract_hidden_states(
        args.model, annotations,
        layers=args.layers,
        all_layers=args.all_layers,
        device=args.device,
    )

    # Save
    save_extracted(data, args.output)


if __name__ == "__main__":
    main()
