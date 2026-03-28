"""Train boundary probes on extracted hidden states.

Runs a sweep over layers, probe types (linear / MLP), and input types
(raw hidden state / displacement) to find the best configuration.

Usage:
    # Full sweep on extracted features
    python scripts/train_boundary_probe.py \
        --features-dir data/probe_features \
        --output-dir results/probe

    # Quick test with specific layer
    python scripts/train_boundary_probe.py \
        --features-dir data/probe_features \
        --layers 12 \
        --probe-type mlp
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np

from src.probe.boundary_probe import (
    LinearProbe,
    MLPProbe,
    ProbeTrainConfig,
    train_probe,
    evaluate_probe,
    compute_displacement,
)


def load_features(features_dir: str) -> dict:
    """Load extracted features and labels from disk."""
    d = Path(features_dir)

    labels = torch.load(d / "labels.pt", weights_only=True)

    with open(d / "meta.json") as f:
        meta = json.load(f)

    features = {}
    for layer_idx in meta["target_layers"]:
        path = d / f"features_layer{layer_idx}.pt"
        if path.exists():
            features[layer_idx] = torch.load(path, weights_only=True)

    return {
        "features": features,
        "labels": labels,
        "meta": meta,
    }


def split_by_rollout(
    data: dict, val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Split into train/val by rollout (not token) to avoid leakage.

    Returns train_indices and val_indices (token-level).
    """
    rng = np.random.RandomState(seed)
    boundaries = data["meta"]["rollout_boundaries"]
    n_rollouts = len(boundaries)

    perm = rng.permutation(n_rollouts)
    n_val = max(1, int(n_rollouts * val_ratio))
    val_rollouts = set(perm[:n_val].tolist())

    train_indices = []
    val_indices = []
    for i, (start, end) in enumerate(boundaries):
        indices = list(range(start, end))
        if i in val_rollouts:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return train_indices, val_indices


def run_sweep(
    data: dict,
    layers: list[int] | None = None,
    probe_types: list[str] | None = None,
    input_types: list[str] | None = None,
    config: ProbeTrainConfig | None = None,
) -> list[dict]:
    """Run full sweep over layers, probe types, and input types."""
    if config is None:
        config = ProbeTrainConfig()

    if layers is None:
        layers = sorted(data["features"].keys())
    if probe_types is None:
        probe_types = ["linear", "mlp"]
    if input_types is None:
        input_types = ["raw", "displacement"]

    hidden_dim = data["meta"]["hidden_dim"]
    labels = data["labels"]

    # Split
    train_idx, val_idx = split_by_rollout(data)
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    print(f"\nData split:")
    print(f"  Train: {len(train_idx)} tokens ({train_labels.sum().item()} boundaries)")
    print(f"  Val:   {len(val_idx)} tokens ({val_labels.sum().item()} boundaries)")

    # Compute displacement features
    disp_features = compute_displacement(data["features"])

    results = []
    total_configs = len(layers) * len(probe_types) * len(input_types)
    run_idx = 0

    for input_type in input_types:
        feat_dict = data["features"] if input_type == "raw" else disp_features
        available_layers = sorted(feat_dict.keys())

        for layer_idx in layers:
            if layer_idx not in feat_dict:
                continue

            feat = feat_dict[layer_idx]
            train_feat = feat[train_idx]
            val_feat = feat[val_idx]

            for probe_type in probe_types:
                run_idx += 1
                tag = f"{input_type}/layer{layer_idx}/{probe_type}"
                print(f"\n[{run_idx}/{total_configs}] {tag}")

                if probe_type == "linear":
                    probe = LinearProbe(hidden_dim)
                else:
                    probe = MLPProbe(hidden_dim, bottleneck=256)

                t0 = time.time()
                metrics = train_probe(
                    probe, train_feat, train_labels,
                    val_feat, val_labels, config,
                )
                elapsed = time.time() - t0

                result = {
                    "layer": layer_idx,
                    "probe_type": probe_type,
                    "input_type": input_type,
                    "auc": metrics["best_val_auc"],
                    "f1": metrics["best_val_f1"],
                    "accuracy": metrics["best_val_acc"],
                    "precision": metrics["best_val_precision"],
                    "recall": metrics["best_val_recall"],
                    "best_epoch": metrics["best_epoch"],
                    "total_epochs": metrics["total_epochs"],
                    "train_time_s": elapsed,
                }
                results.append(result)

                print(f"  AUC={result['auc']:.4f}  F1={result['f1']:.4f}  "
                      f"Acc={result['accuracy']:.4f}  "
                      f"P={result['precision']:.4f} R={result['recall']:.4f}  "
                      f"({elapsed:.1f}s, epoch {result['best_epoch']})")

    return results


def print_summary(results: list[dict]):
    """Print a sorted summary table of sweep results."""
    sorted_results = sorted(results, key=lambda r: -r["auc"])

    print(f"\n{'='*80}")
    print(f"SWEEP RESULTS (sorted by AUC)")
    print(f"{'='*80}")
    print(f"{'Input':<14} {'Layer':>5} {'Probe':>7} {'AUC':>7} {'F1':>7} "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'Epoch':>5}")
    print("-" * 80)

    for r in sorted_results:
        print(f"{r['input_type']:<14} {r['layer']:>5} {r['probe_type']:>7} "
              f"{r['auc']:>7.4f} {r['f1']:>7.4f} {r['accuracy']:>7.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['best_epoch']:>5}")

    # Best overall
    best = sorted_results[0]
    print(f"\nBest: {best['input_type']}/layer{best['layer']}/{best['probe_type']} "
          f"— AUC={best['auc']:.4f}, F1={best['f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train boundary probes with layer/architecture sweep"
    )
    parser.add_argument("--features-dir", required=True,
                        help="Directory with extracted features")
    parser.add_argument("--output-dir", default="results/probe",
                        help="Output directory for results")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to sweep (default: all available)")
    parser.add_argument("--probe-type", nargs="+", default=None,
                        choices=["linear", "mlp"],
                        help="Probe types to test (default: both)")
    parser.add_argument("--input-type", nargs="+", default=None,
                        choices=["raw", "displacement"],
                        help="Input types to test (default: both)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    # Load
    print(f"Loading features from {args.features_dir}...")
    data = load_features(args.features_dir)
    print(f"  Layers available: {sorted(data['features'].keys())}")
    print(f"  Total tokens: {len(data['labels'])}")
    print(f"  Boundaries: {data['labels'].sum().item()}")

    config = ProbeTrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # Sweep
    results = run_sweep(
        data,
        layers=args.layers,
        probe_types=args.probe_type,
        input_types=args.input_type,
        config=config,
    )

    # Summary
    print_summary(results)

    # Save results
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
