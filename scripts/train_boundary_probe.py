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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.probe.boundary_probe import (
    LinearProbe,
    MLPProbe,
    RegressionProbe,
    ProbeTrainConfig,
    train_probe,
    evaluate_probe,
    train_regression_probe,
)
from scripts.extract_hidden_states import load_features_from_disk, compute_displacement


def load_features(features_dir: str) -> dict:
    """Load extracted features and labels from disk (binary format)."""
    return load_features_from_disk(features_dir)


def _load_features_legacy(features_dir: str) -> dict:
    """Load extracted features from legacy .pt format (kept for compatibility)."""
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


def apply_context_window(features: torch.Tensor, labels: torch.Tensor,
                         rollout_boundaries: list, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate ±k neighboring hidden states for each token.

    Pads with zeros at rollout boundaries to avoid cross-rollout contamination.
    Returns (windowed_features [N, hidden_dim * (2k+1)], labels [N]).
    """
    N, D = features.shape
    # Pad features with zeros on both sides
    padded = torch.zeros(N + 2 * k, D, dtype=features.dtype)
    padded[k:k + N] = features

    # Zero out cross-rollout positions
    for start, end in rollout_boundaries:
        # Zero the padding zone around each rollout boundary
        for j in range(k):
            if start + j < N:
                padded[start + j] = 0  # can't look before rollout start
            if k + end + j < len(padded):
                padded[k + end + j] = 0  # can't look after rollout end

    # Build windowed features: for each position i, concat [i-k, ..., i, ..., i+k]
    windows = []
    for offset in range(-k, k + 1):
        windows.append(padded[k + offset:k + offset + N])

    windowed = torch.cat(windows, dim=1)  # [N, D * (2k+1)]
    return windowed, labels


def run_sweep(
    data: dict,
    layers: list[int] | None = None,
    probe_types: list[str] | None = None,
    input_types: list[str] | None = None,
    context_windows: list[int] | None = None,
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
        input_types = ["raw"]
    if context_windows is None:
        context_windows = [0]  # 0 = no context window

    hidden_dim = data["meta"]["hidden_dim"]
    labels = data["labels"]
    rollout_boundaries = data["meta"]["rollout_boundaries"]

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
    total_configs = len(layers) * len(probe_types) * len(input_types) * len(context_windows)
    run_idx = 0

    for input_type in input_types:
        feat_dict = data["features"] if input_type == "raw" else disp_features

        for layer_idx in layers:
            if layer_idx not in feat_dict:
                continue

            feat_full = feat_dict[layer_idx]

            # Pre-split before context window to avoid blowing up memory
            train_feat_raw = feat_full[train_idx]
            val_feat_raw = feat_full[val_idx]
            # Rollout boundaries mapped to train/val index space (approximate: treat as one big sequence)
            train_boundaries = [(0, len(train_idx))]
            val_boundaries = [(0, len(val_idx))]

            for ctx_k in context_windows:
                if ctx_k > 0:
                    train_feat, _ = apply_context_window(
                        train_feat_raw, train_labels, train_boundaries, ctx_k
                    )
                    val_feat, _ = apply_context_window(
                        val_feat_raw, val_labels, val_boundaries, ctx_k
                    )
                    input_dim = hidden_dim * (2 * ctx_k + 1)
                else:
                    train_feat = train_feat_raw
                    val_feat = val_feat_raw
                    input_dim = hidden_dim

                for probe_type in probe_types:
                    run_idx += 1
                    ctx_tag = f"/ctx{ctx_k}" if ctx_k > 0 else ""
                    tag = f"{input_type}/layer{layer_idx}{ctx_tag}/{probe_type}"
                    print(f"\n[{run_idx}/{total_configs}] {tag}")

                    if probe_type == "linear":
                        probe = LinearProbe(input_dim)
                    else:
                        probe = MLPProbe(input_dim, bottleneck=256)

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
                        "context_window": ctx_k,
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


def run_regression_sweep(
    data: dict,
    layers: list[int] | None = None,
    probe_types: list[str] | None = None,
    config: ProbeTrainConfig | None = None,
    save_dir: Path | None = None,
) -> list[dict]:
    """Run regression sweep: predict distance to nearest boundary.

    probe_types: list of "mlp" and/or "lstm"
    """
    from src.probe.boundary_probe import LSTMRegressionProbe

    if config is None:
        config = ProbeTrainConfig()
    if layers is None:
        layers = sorted(data["features"].keys())
    if probe_types is None:
        probe_types = ["lstm"]

    hidden_dim = data["meta"]["hidden_dim"]
    dist_labels = data["dist_labels"]
    rollout_boundaries = data["meta"]["rollout_boundaries"]
    if dist_labels is None:
        print("ERROR: No dist_labels found. Re-run extract_hidden_states.py.")
        return []

    # Split by rollout
    train_idx, val_idx = split_by_rollout(data)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)

    train_dist = dist_labels[train_idx_t]
    val_dist = dist_labels[val_idx_t]

    print(f"\nRegression mode — predict distance to nearest boundary")
    print(f"  Train: {len(train_idx)} tokens (mean dist={train_dist.mean():.2f})")
    print(f"  Val:   {len(val_idx)} tokens (mean dist={val_dist.mean():.2f})")
    print(f"  Probe types: {probe_types}")

    # For LSTM: build per-rollout sequences
    val_rollout_set = set()
    train_rollout_set = set()
    rng = np.random.RandomState(42)
    n_rollouts = len(rollout_boundaries)
    perm = rng.permutation(n_rollouts)
    n_val = max(1, int(n_rollouts * 0.2))
    val_rollout_set = set(perm[:n_val].tolist())
    train_rollout_set = set(perm[n_val:].tolist())

    # Check which layers support displacement (consecutive pairs)
    sorted_layers = sorted(data["features"].keys())
    disp_pairs = {}
    for i in range(len(sorted_layers) - 1):
        l_low, l_high = sorted_layers[i], sorted_layers[i + 1]
        if l_high == l_low + 1:
            disp_pairs[l_low] = (l_low, l_high)

    results = []
    for layer_idx in layers:
        for input_type in ["raw", "displacement"]:
            if input_type == "displacement":
                if layer_idx not in disp_pairs:
                    continue
                disp_pair = disp_pairs[layer_idx]
            else:
                if layer_idx not in data["features"]:
                    continue
                disp_pair = None

            for probe_type in probe_types:
                tag = f"{input_type}/layer{layer_idx}/{probe_type}"
                print(f"\n{tag}")

                # Wrap feature source: raw or displacement (computed lazily)
                feat_source = _FeatureSource(data["features"], layer_idx, disp_pair)

                if probe_type == "lstm":
                    metrics = _train_lstm_regression(
                        feat_source, dist_labels, rollout_boundaries,
                        train_rollout_set, val_rollout_set,
                        hidden_dim, config,
                    )
                else:
                    metrics = _train_mlp_regression_streaming(
                        feat_source, dist_labels, rollout_boundaries,
                        train_rollout_set, val_rollout_set,
                        hidden_dim, config,
                    )

                result = {
                    "layer": layer_idx,
                    "input_type": input_type,
                    "probe_type": probe_type,
                    "mae": metrics["best_val_mae"],
                    "rmse": metrics["best_val_rmse"],
                    "correlation": metrics["best_val_corr"],
                    "acc_at_3": metrics.get("best_val_acc_at_3", 0),
                    "best_epoch": metrics["best_epoch"],
                    "train_time_s": metrics.get("train_time_s", 0),
                }
                results.append(result)

                # Save probe checkpoint
                if "probe" in metrics and save_dir is not None:
                    ckpt_name = f"{input_type}_layer{layer_idx}_{probe_type}.pt"
                    ckpt_path = save_dir / ckpt_name
                    torch.save(metrics["probe"].state_dict(), ckpt_path)

                print(f"  MAE={result['mae']:.3f}  RMSE={result['rmse']:.3f}  "
                      f"Corr={result['correlation']:.4f}  "
                      f"({result['train_time_s']:.1f}s, epoch {result['best_epoch']})")

    # Summary
    sorted_r = sorted(results, key=lambda r: r["mae"])
    print(f"\n{'='*90}")
    print(f"REGRESSION RESULTS (sorted by MAE)")
    print(f"{'='*90}")
    print(f"{'Input':<14} {'Layer':>5} {'Probe':>6} {'MAE':>7} {'RMSE':>7} {'Corr':>7} {'Epoch':>5}")
    print("-" * 55)
    for r in sorted_r:
        print(f"{r['input_type']:<14} {r['layer']:>5} {r['probe_type']:>6} "
              f"{r['mae']:>7.3f} {r['rmse']:>7.3f} {r['correlation']:>7.4f} {r['best_epoch']:>5}")

    return results


class _FeatureSource:
    """Lazy feature source: supports raw or displacement slicing without full load."""

    def __init__(self, features_dict: dict, layer_idx: int, disp_pair: tuple | None = None):
        """
        If disp_pair is None: use features_dict[layer_idx] directly (raw).
        If disp_pair is (l_low, l_high): compute displacement per-slice.
        """
        self.disp_pair = disp_pair
        if disp_pair is not None:
            self.low = features_dict[disp_pair[0]]
            self.high = features_dict[disp_pair[1]]
        else:
            self.raw = features_dict[layer_idx]

    def slice(self, start: int, end: int) -> torch.Tensor:
        """Get features for token range [start:end] as float32 torch tensor."""
        if self.disp_pair is not None:
            low = np.array(self.low[start:end], dtype=np.float32)
            high = np.array(self.high[start:end], dtype=np.float32)
            return torch.from_numpy(high - low)
        else:
            raw = np.array(self.raw[start:end], dtype=np.float32)
            return torch.from_numpy(raw)


def _train_mlp_regression_streaming(
    feat_source: _FeatureSource,
    dist_labels: torch.Tensor,
    rollout_boundaries: list,
    train_rollouts: set,
    val_rollouts: set,
    hidden_dim: int,
    config: ProbeTrainConfig,
) -> dict:
    """Train MLP regression probe streaming per-rollout to save memory."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    probe = RegressionProbe(hidden_dim, bottleneck=512).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.lr)
    criterion = torch.nn.SmoothL1Loss()

    # Gather rollout indices
    train_idxs = sorted(train_rollouts)
    val_idxs = sorted(val_rollouts)

    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    t0 = time.time()
    for epoch in range(config.epochs):
        probe.train()
        np.random.shuffle(train_idxs)
        epoch_loss = 0.0
        n_tokens = 0

        for ridx in train_idxs:
            start, end = rollout_boundaries[ridx]
            feat = feat_source.slice(start, end).to(device)
            lab = dist_labels[start:end].to(device)  # [seq_len]

            pred = probe(feat)  # [seq_len]
            loss = criterion(pred, lab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(feat)
            n_tokens += len(feat)

        # Validation
        probe.eval()
        val_preds, val_labs = [], []
        with torch.no_grad():
            for ridx in val_idxs:
                start, end = rollout_boundaries[ridx]
                feat = feat_source.slice(start, end).to(device)
                lab = dist_labels[start:end]
                pred = probe(feat).cpu()
                val_preds.append(pred)
                val_labs.append(lab)

        preds_cat = torch.cat(val_preds)
        labels_cat = torch.cat(val_labs)
        diff = (preds_cat - labels_cat).abs()
        val_mae = diff.mean().item()
        val_rmse = ((preds_cat - labels_cat) ** 2).mean().sqrt().item()

        if preds_cat.std() > 0 and labels_cat.std() > 0:
            corr = torch.corrcoef(torch.stack([preds_cat, labels_cat]))[0, 1].item()
        else:
            corr = 0.0

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    elapsed = time.time() - t0

    # Restore best state
    if best_state is not None:
        probe.load_state_dict(best_state)

    return {
        "best_val_mae": best_val_mae,
        "best_val_rmse": val_rmse,
        "best_val_corr": corr,
        "best_val_acc_at_3": (diff <= 3).float().mean().item() if len(diff) > 0 else 0,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "train_time_s": elapsed,
        "train_losses": [],
        "probe": probe,
    }


def _train_lstm_regression(
    feat_source: _FeatureSource,
    dist_labels: torch.Tensor,
    rollout_boundaries: list,
    train_rollouts: set,
    val_rollouts: set,
    hidden_dim: int,
    config: ProbeTrainConfig,
) -> dict:
    """Train LSTM regression probe on per-rollout sequences."""
    from src.probe.boundary_probe import LSTMRegressionProbe, evaluate_regression_probe

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Build per-rollout sequences
    train_seqs, train_labels_seq = [], []
    val_seqs, val_labels_seq = [], []

    for i, (start, end) in enumerate(rollout_boundaries):
        seq = feat_source.slice(start, end)  # [seq_len, hidden_dim]
        lab = dist_labels[start:end]  # [seq_len]
        if i in val_rollouts:
            val_seqs.append(seq)
            val_labels_seq.append(lab)
        elif i in train_rollouts:
            train_seqs.append(seq)
            train_labels_seq.append(lab)

    probe = LSTMRegressionProbe(hidden_dim, lstm_dim=256, num_layers=1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.lr)
    criterion = torch.nn.SmoothL1Loss()

    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    t0 = time.time()
    for epoch in range(config.epochs):
        probe.train()
        # Shuffle training sequences
        perm = np.random.permutation(len(train_seqs))
        epoch_loss = 0.0

        for idx in perm:
            seq = train_seqs[idx].unsqueeze(0).to(device)  # [1, seq_len, hidden_dim]
            lab = train_labels_seq[idx].unsqueeze(0).to(device)  # [1, seq_len]

            pred = probe(seq)  # [1, seq_len]
            loss = criterion(pred, lab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        probe.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seq, lab in zip(val_seqs, val_labels_seq):
                pred = probe(seq.unsqueeze(0).to(device))
                all_preds.append(pred.squeeze(0).cpu())
                all_labels.append(lab)

        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        diff = (preds_cat - labels_cat).abs()
        val_mae = diff.mean().item()
        val_rmse = ((preds_cat - labels_cat) ** 2).mean().sqrt().item()

        if preds_cat.std() > 0 and labels_cat.std() > 0:
            corr = torch.corrcoef(torch.stack([preds_cat, labels_cat]))[0, 1].item()
        else:
            corr = 0.0

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    elapsed = time.time() - t0

    # Restore best state
    if best_state is not None:
        probe.load_state_dict(best_state)

    return {
        "best_val_mae": best_val_mae,
        "best_val_rmse": val_rmse,
        "best_val_corr": corr,
        "best_val_acc_at_3": (diff <= 3).float().mean().item() if len(diff) > 0 else 0,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "train_time_s": elapsed,
        "train_losses": [],
        "probe": probe,
    }


def print_summary(results: list[dict]):
    """Print a sorted summary table of sweep results."""
    sorted_results = sorted(results, key=lambda r: -r["auc"])

    print(f"\n{'='*90}")
    print(f"SWEEP RESULTS (sorted by AUC)")
    print(f"{'='*90}")
    print(f"{'Input':<10} {'Layer':>5} {'Ctx':>4} {'Probe':>7} {'AUC':>7} {'F1':>7} "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'Epoch':>5}")
    print("-" * 90)

    for r in sorted_results:
        ctx = r.get('context_window', 0)
        print(f"{r['input_type']:<10} {r['layer']:>5} {ctx:>4} {r['probe_type']:>7} "
              f"{r['auc']:>7.4f} {r['f1']:>7.4f} {r['accuracy']:>7.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['best_epoch']:>5}")

    # Best overall
    best = sorted_results[0]
    ctx = best.get('context_window', 0)
    print(f"\nBest: {best['input_type']}/layer{best['layer']}/ctx{ctx}/{best['probe_type']} "
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
                        choices=["linear", "mlp", "lstm"],
                        help="Probe types to test (default: both/all)")
    parser.add_argument("--input-type", nargs="+", default=None,
                        choices=["raw", "displacement"],
                        help="Input types to test (default: both)")
    parser.add_argument("--context-window", type=int, nargs="+", default=[0],
                        help="Context window sizes to test (e.g., --context-window 0 3 5)")
    parser.add_argument("--mode", default="regression", choices=["classify", "regression"],
                        help="Task mode: classify (binary) or regression (distance)")
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
    if args.mode == "regression":
        probe_save_dir = Path(args.output_dir) / "checkpoints"
        probe_save_dir.mkdir(parents=True, exist_ok=True)
        results = run_regression_sweep(
            data,
            layers=args.layers,
            probe_types=args.probe_type,
            config=config,
            save_dir=probe_save_dir,
        )
    else:
        results = run_sweep(
            data,
            layers=args.layers,
            probe_types=args.probe_type,
            input_types=args.input_type,
            context_windows=args.context_window,
            config=config,
        )
        print_summary(results)

    # Save results
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
