"""Lightweight probes for semantic boundary detection from hidden states.

Probes are trained on frozen LLM hidden states to predict whether a token
position is a semantic boundary (reasoning step transition). Two variants:
- LinearProbe: single linear layer (tests linear separability)
- MLPProbe: two-layer MLP (captures non-linear patterns)

Both support raw hidden states and displacement (h[l+1] - h[l]) inputs.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


class LinearProbe(nn.Module):
    """Linear probe: Linear(hidden_dim, 1) + sigmoid."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLPProbe(nn.Module):
    """Two-layer MLP probe: Linear → ReLU → Linear."""

    def __init__(self, hidden_dim: int, bottleneck: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class ProbeTrainConfig:
    lr: float = 5e-4
    epochs: int = 50
    batch_size: int = 512
    patience: int = 5           # early stopping
    pos_weight: float | None = None  # auto-computed if None
    device: str = "cuda"


def train_probe(
    probe: nn.Module,
    train_features: torch.Tensor,   # [N, hidden_dim]
    train_labels: torch.Tensor,     # [N] binary
    val_features: torch.Tensor,     # [M, hidden_dim]
    val_labels: torch.Tensor,       # [M] binary
    config: ProbeTrainConfig | None = None,
) -> dict:
    """Train a boundary probe and return metrics.

    Returns dict with: best_val_auc, best_val_f1, best_val_acc, best_epoch, train_losses.
    """
    if config is None:
        config = ProbeTrainConfig()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    probe = probe.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.float().to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.float().to(device)

    # Auto-compute pos_weight for class imbalance
    if config.pos_weight is None:
        num_pos = train_labels.sum().item()
        num_neg = len(train_labels) - num_pos
        pw = num_neg / max(num_pos, 1)
    else:
        pw = config.pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

    optimizer = torch.optim.Adam(probe.parameters(), lr=config.lr)
    n = len(train_features)

    best_val_auc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    train_losses = []

    for epoch in range(config.epochs):
        probe.train()
        # Shuffle
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, n, config.batch_size):
            idx = perm[i:i + config.batch_size]
            logits = probe(train_features[idx])
            loss = criterion(logits, train_labels[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Validation
        val_metrics = evaluate_probe(probe, val_features, val_labels, device)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    # Restore best
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe = probe.cpu()

    final_metrics = evaluate_probe(
        probe, val_features.cpu(), val_labels.cpu(), torch.device("cpu")
    )

    return {
        "best_val_auc": best_val_auc,
        "best_val_f1": final_metrics["f1"],
        "best_val_acc": final_metrics["accuracy"],
        "best_val_precision": final_metrics["precision"],
        "best_val_recall": final_metrics["recall"],
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "train_losses": train_losses,
    }


def evaluate_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict:
    """Evaluate probe on features/labels. Returns accuracy, AUC, F1, precision, recall."""
    probe.eval()
    with torch.no_grad():
        logits = probe(features.to(device))
        probs = torch.sigmoid(logits).cpu()
        preds = (probs > 0.5).long()
        labels_cpu = labels.long().cpu()

    # Accuracy
    accuracy = (preds == labels_cpu).float().mean().item()

    # Precision / Recall / F1
    tp = ((preds == 1) & (labels_cpu == 1)).sum().item()
    fp = ((preds == 1) & (labels_cpu == 0)).sum().item()
    fn = ((preds == 0) & (labels_cpu == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # AUC (simple trapezoidal)
    auc = _compute_auc(probs.numpy(), labels_cpu.numpy())

    return {
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def _compute_auc(probs, labels) -> float:
    """Compute ROC-AUC using numpy (avoid sklearn dependency)."""
    import numpy as np

    pos = probs[labels == 1]
    neg = probs[labels == 0]

    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    # Mann-Whitney U statistic
    n_pos = len(pos)
    n_neg = len(neg)

    # For efficiency, use sorted comparison
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    desc_idx = np.argsort(-all_scores)
    sorted_labels = all_labels[desc_idx]

    # Count concordant pairs
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)

    # Trapezoidal AUC
    tpr = np.concatenate([[0], tps / n_pos])
    fpr = np.concatenate([[0], fps / n_neg])

    auc = np.trapz(tpr, fpr)
    return float(auc)
