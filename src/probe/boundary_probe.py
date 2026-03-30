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


class RegressionProbe(nn.Module):
    """MLP probe for distance regression: predict distance to nearest boundary."""

    def __init__(self, hidden_dim: int, bottleneck: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, bottleneck // 2),
            nn.ReLU(),
            nn.Linear(bottleneck // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, hidden_dim] or [batch, seq_len, hidden_dim] (takes last dim)."""
        return self.net(x).squeeze(-1)


class LSTMRegressionProbe(nn.Module):
    """Causal (forward-only) LSTM probe for boundary distance prediction.

    Reads hidden states left-to-right and predicts distance to next
    boundary at each position. Unidirectional to match inference-time
    causality — token i only sees tokens 0..i.
    """

    def __init__(self, hidden_dim: int, lstm_dim: int = 256, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, hidden_dim] -> [batch, seq_len]"""
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_dim]
        return self.head(lstm_out).squeeze(-1)  # [batch, seq_len]


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal loss for binary classification with logits.

    Down-weights easy examples so the model focuses on hard boundary cases.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # Apply pos_weight to positive examples
        weight = torch.where(targets == 1, self.pos_weight, 1.0)
        # Focal modulation: down-weight easy examples
        p_t = torch.where(targets == 1, p, 1 - p)
        focal = (1 - p_t) ** self.gamma
        return (weight * focal * ce).mean()


@dataclass
class ProbeTrainConfig:
    lr: float = 5e-4
    epochs: int = 50
    batch_size: int = 512
    patience: int = 5           # early stopping
    pos_weight: float | None = None  # auto-computed if None
    focal_gamma: float = 2.0    # focal loss gamma (0 = no focal)
    undersample: bool = True    # undersample negatives to balance
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

    # Undersample negatives to balance classes
    if config.undersample:
        pos_mask = train_labels == 1
        neg_mask = train_labels == 0
        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()

        # Keep all positives, sample negatives at 3:1 ratio
        neg_indices = torch.where(neg_mask)[0]
        keep_neg = min(num_neg, num_pos * 3)
        perm = torch.randperm(num_neg)[:keep_neg]
        neg_keep = neg_indices[perm]

        pos_indices = torch.where(pos_mask)[0]
        keep_indices = torch.cat([pos_indices, neg_keep])
        keep_indices = keep_indices[torch.randperm(len(keep_indices))]

        train_features = train_features[keep_indices]
        train_labels = train_labels[keep_indices]
        print(f"  Undersampled: {num_pos} pos + {keep_neg} neg = {len(keep_indices)} "
              f"(was {num_pos + num_neg})")

    train_features = train_features.to(device)
    train_labels = train_labels.float().to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.float().to(device)

    # Loss function
    if config.pos_weight is None:
        num_pos = train_labels.sum().item()
        num_neg = len(train_labels) - num_pos
        pw = num_neg / max(num_pos, 1)
    else:
        pw = config.pos_weight

    if config.focal_gamma > 0:
        criterion = FocalBCEWithLogitsLoss(gamma=config.focal_gamma, pos_weight=pw)
    else:
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


def train_regression_probe(
    probe: nn.Module,
    train_features: torch.Tensor,   # [N, hidden_dim]
    train_labels: torch.Tensor,     # [N] float, distance to nearest boundary
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    config: ProbeTrainConfig | None = None,
) -> dict:
    """Train a regression probe to predict distance to nearest boundary."""
    if config is None:
        config = ProbeTrainConfig()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    probe = probe.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.float().to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.float().to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.lr)
    n = len(train_features)

    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    train_losses = []

    for epoch in range(config.epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, n, config.batch_size):
            idx = perm[i:i + config.batch_size]
            preds = probe(train_features[idx])
            loss = criterion(preds, train_labels[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Validation
        val_metrics = evaluate_regression_probe(probe, val_features, val_labels, device)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe = probe.cpu()

    final = evaluate_regression_probe(
        probe, val_features.cpu(), val_labels.cpu(), torch.device("cpu")
    )

    return {
        "best_val_mae": final["mae"],
        "best_val_rmse": final["rmse"],
        "best_val_corr": final["correlation"],
        "best_val_acc_at_3": final["acc_at_3"],
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "train_losses": train_losses,
    }


def evaluate_regression_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict:
    """Evaluate regression probe. Returns MAE, RMSE, correlation, acc@3."""
    probe.eval()
    with torch.no_grad():
        preds = probe(features.to(device)).cpu()
        labels_cpu = labels.cpu()

    diff = preds - labels_cpu
    mae = diff.abs().mean().item()
    rmse = (diff ** 2).mean().sqrt().item()

    # Correlation
    if preds.std() > 0 and labels_cpu.std() > 0:
        corr = torch.corrcoef(torch.stack([preds, labels_cpu]))[0, 1].item()
    else:
        corr = 0.0

    # Accuracy within ±3 tokens
    acc_at_3 = (diff.abs() <= 3).float().mean().item()

    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
        "acc_at_3": acc_at_3,
    }
