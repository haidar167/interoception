"""Evaluation metrics for uncertainty calibration and error detection."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(y_true_error: np.ndarray, y_score_error: np.ndarray) -> float:
    """Compute ROC-AUC for error detection.

    Args:
        y_true_error: Binary array where 1 indicates incorrect prediction and 0 indicates correct.
        y_score_error: Predicted score or probability of error (higher means more likely error).

    Returns:
        ROC-AUC score in [0.0, 1.0].
    """
    y_true = np.asarray(y_true_error)
    y_score = np.asarray(y_score_error)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def compute_ece(
    confidences: Union[np.ndarray, List[float]],
    accuracies: Union[np.ndarray, List[float]],
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins.

    Formula:
        ECE = sum_{b=1}^{B} (n_b / N) * |acc_b - conf_b|

    Args:
        confidences: Predicted confidence values in [0, 1].
        accuracies: Binary correctness indicators (1 for correct, 0 for incorrect).
        n_bins: Number of equal-width bins (default: 15).

    Returns:
        Scalar ECE value in [0.0, 1.0].
    """
    confs = np.asarray(confidences, dtype=np.float64)
    accs = np.asarray(accuracies, dtype=np.float64)

    if len(confs) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accs[in_bin])
            bin_conf = np.mean(confs[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(
    probs_or_confs: np.ndarray,
    targets_or_accs: np.ndarray,
) -> float:
    """Compute Brier score.

    Supports:
    1. Multi-class probability matrix [N, K] and integer target vector [N].
    2. 1D confidence vector [N] and 1D binary accuracy vector [N].

    Returns:
        Scalar Brier score (lower is better).
    """
    preds = np.asarray(probs_or_confs, dtype=np.float64)
    targets = np.asarray(targets_or_accs)

    if preds.ndim == 2:
        num_classes = preds.shape[1]
        if targets.ndim == 1:
            # One-hot encode targets
            one_hot = np.zeros_like(preds)
            one_hot[np.arange(len(targets)), targets.astype(int)] = 1.0
            return float(np.mean(np.sum((preds - one_hot) ** 2, axis=1)))
        else:
            return float(np.mean(np.sum((preds - targets) ** 2, axis=1)))
    else:
        # Confidence vs accuracy calibration Brier score: (conf - acc)^2
        return float(np.mean((preds - targets.astype(np.float64)) ** 2))


def plot_reliability_diagram(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Union[str, Path],
    title: str = "Reliability Diagram",
    n_bins: int = 15,
) -> None:
    """Plot reliability diagram (confidence vs accuracy) with diagonal reference.

    Args:
        models_data: Mapping of model_name -> (confidences, accuracies).
        save_path: Output file path for PNG.
        title: Plot title.
        n_bins: Number of equal-width bins.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (model_name, (confs, accs)) in enumerate(models_data.items()):
        confs = np.asarray(confs, dtype=np.float64)
        accs = np.asarray(accs, dtype=np.float64)

        bin_accs = []
        bin_counts = []
        valid_centers = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            if i == n_bins - 1:
                in_bin = (confs >= bin_lower) & (confs <= bin_upper)
            else:
                in_bin = (confs >= bin_lower) & (confs < bin_upper)

            count = np.sum(in_bin)
            if count > 0:
                bin_accs.append(np.mean(accs[in_bin]))
                bin_counts.append(count)
                valid_centers.append(bin_centers[i])

        color = colors[idx % len(colors)]
        ece = compute_ece(confs, accs, n_bins=n_bins)
        ax.plot(
            valid_centers,
            bin_accs,
            marker="o",
            linestyle="-",
            label=f"{model_name} (ECE={ece:.4f})",
            color=color,
            alpha=0.85,
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
