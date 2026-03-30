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
    output_dir: str = "data/probe_features",
):
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

    # Prepare output directory for streaming writes
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open per-layer files for streaming writes
    layer_files = {}
    for l in target_layers:
        layer_files[l] = open(out_dir / f"features_layer{l}.bin", "wb")
    labels_file = open(out_dir / f"labels.bin", "wb")
    dist_labels_file = open(out_dir / f"dist_labels.bin", "wb")

    rollout_boundaries = []
    metadata_list = []
    global_idx = 0
    total_tokens = 0
    total_boundaries = 0

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

        # Create distance labels: forward-looking distance to NEXT boundary
        # For each token t, how many tokens until the next boundary ahead?
        # Boundary tokens themselves get 0.
        import math
        boundary_set = sorted(set(boundary_tok_indices))
        dist_labels = torch.zeros(num_tokens, dtype=torch.float)
        # Walk backwards: for each token, find the next boundary >= t
        next_boundary = num_tokens  # default: no boundary ahead
        for t in range(num_tokens - 1, -1, -1):
            if t in set(boundary_set):
                next_boundary = t
            dist_labels[t] = next_boundary - t

        # Log-normalize: compress long tail, keep 0 at boundary
        dist_labels = torch.tensor([math.log(d + 1) for d in dist_labels.tolist()],
                                   dtype=torch.float)

        # Forward pass
        input_ids = torch.tensor([token_ids], device=device)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # Stream hidden states to disk immediately, don't accumulate
        hidden_states = outputs.hidden_states
        for layer_idx in target_layers:
            if layer_idx < len(hidden_states):
                hs = hidden_states[layer_idx][0].cpu().half()  # [seq_len, hidden_dim], float16
                layer_files[layer_idx].write(hs.numpy().tobytes())

        # Stream labels to disk
        labels_file.write(labels.numpy().tobytes())
        dist_labels_file.write(dist_labels.numpy().tobytes())

        rollout_boundaries.append((global_idx, global_idx + num_tokens))
        global_idx += num_tokens
        total_tokens += num_tokens
        total_boundaries += labels.sum().item()

        metadata_list.append({
            "question": ann["question"][:100],
            "correct": ann["correct"],
            "num_tokens": num_tokens,
            "num_boundaries": len(boundary_tok_indices),
            "boundary_token_indices": boundary_tok_indices,
        })

        # Explicitly free GPU tensors
        del outputs, hidden_states, input_ids
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            elapsed = time.time() - t0
            print(f"  Processed {i+1}/{len(annotations)} "
                  f"({elapsed:.1f}s, {elapsed/(i+1):.2f}s/rollout)")

    elapsed = time.time() - t0
    print(f"  Done: {len(annotations)} rollouts in {elapsed:.1f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total boundaries: {total_boundaries} ({100*total_boundaries/total_tokens:.1f}%)")

    # Close streaming files
    for f in layer_files.values():
        f.close()
    labels_file.close()
    dist_labels_file.close()

    # Save metadata
    meta = {
        "rollout_boundaries": rollout_boundaries,
        "metadata": metadata_list,
        "target_layers": target_layers,
        "hidden_dim": hidden_dim,
        "total_tokens": total_tokens,
        "dtype": "float16",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_size = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e9
    print(f"\nSaved to {output_dir} ({total_size:.2f} GB)")

    # Free model
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


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


def load_features_from_disk(features_dir: str) -> dict:
    """Load streamed binary features and labels from disk.

    Returns dict with features (per-layer tensors), labels, and metadata.
    """
    d = Path(features_dir)
    with open(d / "meta.json") as f:
        meta = json.load(f)

    hidden_dim = meta["hidden_dim"]
    total_tokens = meta["total_tokens"]

    # Load labels
    labels_np = np.fromfile(d / "labels.bin", dtype=np.int64)
    labels = torch.from_numpy(labels_np)

    # Load distance labels (if available)
    dist_path = d / "dist_labels.bin"
    if dist_path.exists():
        dist_np = np.fromfile(dist_path, dtype=np.float32)
        dist_labels = torch.from_numpy(dist_np)
    else:
        dist_labels = None

    # Load per-layer features as memory-mapped (lazy, doesn't eat RAM)
    features = {}
    for layer_idx in meta["target_layers"]:
        path = d / f"features_layer{layer_idx}.bin"
        if path.exists():
            feat_mmap = np.memmap(path, dtype=np.float16, mode='r',
                                  shape=(total_tokens, hidden_dim))
            features[layer_idx] = feat_mmap

    return {
        "features": features,
        "labels": labels,
        "dist_labels": dist_labels,
        "meta": meta,
    }


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

    # Extract and stream to disk
    extract_hidden_states(
        args.model, annotations,
        layers=args.layers,
        all_layers=args.all_layers,
        device=args.device,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
