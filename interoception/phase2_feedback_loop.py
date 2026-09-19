"""Phase 2: The Feedback Loop - Comparing Baseline, Post-hoc Probe, and Interoceptive Nets."""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interoception.data import load_mnist_data
from interoception.features import extract_internal_stats
from interoception.metrics import compute_brier_score, compute_ece, plot_reliability_diagram
from interoception.models import BaselineMLP, InteroceptiveNet
from interoception.utils import save_results, set_seed


def train_baseline(
    train_loader: torch.utils.data.DataLoader,
    input_dim: int = 784,
    epochs: int = 3,
    lr: float = 1e-3,
) -> BaselineMLP:
    """Train standard Baseline MLP."""
    model = BaselineMLP(input_dim=input_dim, hidden_dim=256, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
    return model


def train_interoceptive_net(
    train_loader: torch.utils.data.DataLoader,
    input_dim: int = 784,
    detach_stats: bool = False,
    epochs: int = 3,
    lr: float = 1e-3,
) -> InteroceptiveNet:
    """Train InteroceptiveNet with internal state feedback."""
    model = InteroceptiveNet(
        input_dim=input_dim, hidden_dim=256, num_classes=10, detach_stats=detach_stats
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits, _, _ = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
    return model


def train_error_probe(
    baseline_model: BaselineMLP,
    train_loader: torch.utils.data.DataLoader,
) -> make_pipeline:
    """Train post-hoc error probe on training set activations."""
    baseline_model.eval()
    all_stats = []
    all_errors = []

    with torch.no_grad():
        for images, targets in train_loader:
            logits, hidden = baseline_model(images)
            stats = extract_internal_stats(hidden)
            preds = logits.argmax(dim=1)
            errors = (preds != targets).long()

            all_stats.append(stats.cpu().numpy())
            all_errors.append(errors.cpu().numpy())

    X_train = np.concatenate(all_stats, axis=0)
    y_train = np.concatenate(all_errors, axis=0)

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )
    probe.fit(X_train, y_train)
    return probe


def evaluate_baseline(
    model: BaselineMLP,
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate baseline model on test set."""
    model.eval()
    all_confs = []
    all_accs = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            logits, _ = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            accs = (preds == targets).float()

            all_confs.append(confs.cpu().numpy())
            all_accs.append(accs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    confs = np.concatenate(all_confs)
    accs = np.concatenate(all_accs)
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    accuracy = float(np.mean(accs))
    ece = compute_ece(confs, accs, n_bins=15)
    brier = compute_brier_score(probs, targets)

    return accuracy, ece, brier, confs, accs


def evaluate_post_hoc_probe(
    model: BaselineMLP,
    probe: make_pipeline,
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate post-hoc probe calibrated model on test set."""
    model.eval()
    all_confs_new = []
    all_accs = []
    all_probs_rescaled = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            logits, hidden = model(images)
            probs = torch.softmax(logits, dim=1)
            raw_confs, preds = probs.max(dim=1)
            accs = (preds == targets).float()

            stats = extract_internal_stats(hidden).cpu().numpy()
            p_error = probe.predict_proba(stats)[:, 1]

            # Rescale confidence: conf_new = softmax_max * (1 - p_error)
            conf_new = raw_confs.cpu().numpy() * (1.0 - p_error)
            conf_new = np.clip(conf_new, 0.0, 1.0)

            all_confs_new.append(conf_new)
            all_accs.append(accs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Rescale full probability distribution towards uniform by p_error
            probs_np = probs.cpu().numpy()
            rescaled_p = probs_np * (1.0 - p_error[:, None]) + (p_error[:, None] / probs_np.shape[1])
            all_probs_rescaled.append(rescaled_p)

    confs = np.concatenate(all_confs_new)
    accs = np.concatenate(all_accs)
    probs = np.concatenate(all_probs_rescaled)
    targets = np.concatenate(all_targets)

    accuracy = float(np.mean(accs))
    ece = compute_ece(confs, accs, n_bins=15)
    brier = compute_brier_score(probs, targets)

    return accuracy, ece, brier, confs, accs


def evaluate_interoceptive_net(
    model: InteroceptiveNet,
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate InteroceptiveNet on test set."""
    model.eval()
    all_confs = []
    all_accs = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            logits, _, _ = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            accs = (preds == targets).float()

            all_confs.append(confs.cpu().numpy())
            all_accs.append(accs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    confs = np.concatenate(all_confs)
    accs = np.concatenate(all_accs)
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    accuracy = float(np.mean(accs))
    ece = compute_ece(confs, accs, n_bins=15)
    brier = compute_brier_score(probs, targets)

    return accuracy, ece, brier, confs, accs


def run_phase2(seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Execute full Phase 2 experiment and generate comparisons and reliability diagram."""
    start_time = time.time()
    set_seed(seed)
    print("=" * 75)
    print("PHASE 2: The Feedback Loop - Calibration and Introspective Architectures")
    print("=" * 75)

    print("Loading dataset...")
    train_loader, test_loader, input_dim = load_mnist_data(batch_size=128)

    # 1. Train Model A: Baseline MLP
    print("\n[1/4] Training Model A (Baseline MLP)...")
    baseline_model = train_baseline(train_loader, input_dim=input_dim, epochs=3)
    acc_a, ece_a, brier_a, confs_a, accs_a = evaluate_baseline(baseline_model, test_loader)
    print(f"Model A -> Accuracy: {acc_a:.4%}, ECE: {ece_a:.4f}, Brier: {brier_a:.4f}")

    # 2. Train Model B: Post-hoc Probe
    print("\n[2/4] Training Model B (Post-hoc Error Probe on hidden stats)...")
    probe = train_error_probe(baseline_model, train_loader)
    acc_b, ece_b, brier_b, confs_b, accs_b = evaluate_post_hoc_probe(
        baseline_model, probe, test_loader
    )
    print(f"Model B -> Accuracy: {acc_b:.4%}, ECE: {ece_b:.4f}, Brier: {brier_b:.4f}")

    # 3. Train Model C1: Interoceptive Net (Detached Stats)
    print("\n[3/4] Training Model C1 (Interoceptive Net - Detached Stats)...")
    model_c1 = train_interoceptive_net(
        train_loader, input_dim=input_dim, detach_stats=True, epochs=3
    )
    acc_c1, ece_c1, brier_c1, confs_c1, accs_c1 = evaluate_interoceptive_net(
        model_c1, test_loader
    )
    print(f"Model C1 -> Accuracy: {acc_c1:.4%}, ECE: {ece_c1:.4f}, Brier: {brier_c1:.4f}")

    # 4. Train Model C2: Interoceptive Net (End-to-End Gradients)
    print("\n[4/4] Training Model C2 (Interoceptive Net - End-to-End Gradients)...")
    model_c2 = train_interoceptive_net(
        train_loader, input_dim=input_dim, detach_stats=False, epochs=3
    )
    acc_c2, ece_c2, brier_c2, confs_c2, accs_c2 = evaluate_interoceptive_net(
        model_c2, test_loader
    )
    print(f"Model C2 -> Accuracy: {acc_c2:.4%}, ECE: {ece_c2:.4f}, Brier: {brier_c2:.4f}")

    elapsed_time = time.time() - start_time

    # Print comparative results table
    print("\n" + "=" * 75)
    print("PHASE 2 MODEL COMPARISON TABLE:")
    header = f"{'Model':<38} | {'Accuracy':<10} | {'ECE (15 bins)':<14} | {'Brier':<10}"
    print(header)
    print("-" * 75)
    print(f"{'A) Baseline MLP':<38} | {acc_a*100:>8.2f}% | {ece_a:>14.4f} | {brier_a:>10.4f}")
    print(f"{'B) Post-hoc Probe (conf_new)':<38} | {acc_b*100:>8.2f}% | {ece_b:>14.4f} | {brier_b:>10.4f}")
    print(f"{'C1) Interoceptive Net (Detached)':<38} | {acc_c1*100:>8.2f}% | {ece_c1:>14.4f} | {brier_c1:>10.4f}")
    print(f"{'C2) Interoceptive Net (End-to-End)':<38} | {acc_c2*100:>8.2f}% | {ece_c2:>14.4f} | {brier_c2:>10.4f}")
    print("=" * 75)
    print(f"Total Phase 2 Runtime: {elapsed_time:.2f}s (< 900s requirement)")

    # Save Reliability Diagrams
    fig_path = Path("figures/phase2_reliability.png")
    plot_reliability_diagram(
        {
            "Baseline MLP": (confs_a, accs_a),
            "Post-hoc Probe": (confs_b, accs_b),
            "Interoceptive (Detached)": (confs_c1, accs_c1),
            "Interoceptive (End-to-End)": (confs_c2, accs_c2),
        },
        save_path=fig_path,
        title="Phase 2 Calibration: Reliability Diagrams (MNIST)",
        n_bins=15,
    )
    print(f"Reliability diagram saved to: {fig_path.resolve()}")

    results = {
        "phase": 2,
        "models": {
            "baseline": {"accuracy": acc_a, "ece": ece_a, "brier": brier_a},
            "post_hoc_probe": {"accuracy": acc_b, "ece": ece_b, "brier": brier_b},
            "interoceptive_detached": {"accuracy": acc_c1, "ece": ece_c1, "brier": brier_c1},
            "interoceptive_end_to_end": {"accuracy": acc_c2, "ece": ece_c2, "brier": brier_c2},
        },
        "execution_time_seconds": elapsed_time,
    }

    save_path = Path("results/phase2.json")
    save_results(results, save_path, seed=seed)
    print(f"Phase 2 results saved to: {save_path.resolve()}\n")

    return results


if __name__ == "__main__":
    run_phase2()
