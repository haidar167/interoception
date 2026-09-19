"""Phase 1: Prove internal activation statistics detect neural network errors."""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interoception.data import load_mnist_data
from interoception.features import extract_features
from interoception.metrics import compute_auc
from interoception.models import BaselineMLP
from interoception.utils import save_results, set_seed


def train_baseline_model(
    train_loader: torch.utils.data.DataLoader,
    input_dim: int = 784,
    hidden_dim: int = 256,
    num_classes: int = 10,
    lr: float = 1e-3,
    epochs: int = 3,
) -> BaselineMLP:
    """Train Baseline MLP on MNIST for 3 epochs with Adam."""
    model = BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4%}")

    return model


def extract_test_features_and_labels(
    model: BaselineMLP,
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract 5 uncertainty features and binary error labels from test set.

    Label = 1 if prediction is wrong, 0 if correct.
    """
    model.eval()
    all_features = []
    all_error_labels = []
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for images, targets in test_loader:
            logits, hidden = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            feats = extract_features(hidden, probs)
            errors = (preds != targets).long()

            correct_count += (preds == targets).sum().item()
            total_count += targets.size(0)

            all_features.append(feats.cpu().numpy())
            all_error_labels.append(errors.cpu().numpy())

    features_matrix = np.concatenate(all_features, axis=0)
    error_labels = np.concatenate(all_error_labels, axis=0)
    accuracy = correct_count / total_count

    return features_matrix, error_labels, accuracy


def run_phase1(seed: int = 42) -> Dict[str, float]:
    """Run full Phase 1 experiment."""
    start_time = time.time()
    set_seed(seed)
    print("=" * 70)
    print("PHASE 1: Internal Activation Statistics for Error Detection")
    print("=" * 70)

    # 1. Load data
    print("Loading dataset...")
    train_loader, test_loader, input_dim = load_mnist_data(batch_size=128)

    # 2. Train baseline MLP
    print(f"Training Baseline MLP ({input_dim} -> 256 -> 10) on CPU...")
    model = train_baseline_model(train_loader, input_dim=input_dim, hidden_dim=256, epochs=3)

    # 3. Extract test set features and error labels
    print("Extracting features from test set...")
    features, error_labels, test_acc = extract_test_features_and_labels(model, test_loader)
    num_samples = len(error_labels)
    num_errors = int(np.sum(error_labels))
    print(f"Test Accuracy: {test_acc:.4%} ({num_samples - num_errors}/{num_samples} correct, {num_errors} errors)")

    # 4. Train 3 Logistic Regression models with identical train/test split
    # a) Confidence only (feature 5, index 4)
    # b) Internal stats only (features 1-4, indices 0:4)
    # c) All features (indices 0:5)
    X_train, X_test, y_train, y_test = train_test_split(
        features, error_labels, test_size=0.3, random_state=0, stratify=error_labels
    )

    X_train_conf, X_test_conf = X_train[:, [4]], X_test[:, [4]]
    X_train_internal, X_test_internal = X_train[:, 0:4], X_test[:, 0:4]
    X_train_all, X_test_all = X_train[:, 0:5], X_test[:, 0:5]

    # Fit classifiers using StandardScaler pipeline for fast numerical convergence
    clf_conf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    clf_conf.fit(X_train_conf, y_train)
    p_err_conf = clf_conf.predict_proba(X_test_conf)[:, 1]
    auc_conf = compute_auc(y_test, p_err_conf)

    clf_internal = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    clf_internal.fit(X_train_internal, y_train)
    p_err_internal = clf_internal.predict_proba(X_test_internal)[:, 1]
    auc_internal = compute_auc(y_test, p_err_internal)

    clf_all = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    clf_all.fit(X_train_all, y_train)
    p_err_all = clf_all.predict_proba(X_test_all)[:, 1]
    auc_all = compute_auc(y_test, p_err_all)

    elapsed_time = time.time() - start_time

    # 5. Display and save results
    print("\n" + "-" * 70)
    print("PHASE 1 RESULTS SUMMARY:")
    print(f"{'Feature Configuration':<35} | {'ROC-AUC':<10}")
    print("-" * 50)
    print(f"{'(a) Softmax Confidence Only':<35} | {auc_conf:.4f}")
    print(f"{'(b) Internal Stats Only (Features 1-4)':<35} | {auc_internal:.4f}")
    print(f"{'(c) All Features (Internal + Confidence)':<35} | {auc_all:.4f}")
    print("-" * 50)
    print(f"Execution Time: {elapsed_time:.2f}s (< 600s requirement)")
    print("-" * 70)

    results = {
        "phase": 1,
        "test_accuracy": float(test_acc),
        "total_test_samples": int(num_samples),
        "total_errors": int(num_errors),
        "auc_confidence_only": float(auc_conf),
        "auc_internal_stats_only": float(auc_internal),
        "auc_all_features": float(auc_all),
        "execution_time_seconds": float(elapsed_time),
    }

    save_path = Path("results/phase1.json")
    save_results(results, save_path, seed=seed)
    print(f"Results successfully saved to {save_path.resolve()}\n")

    return results


if __name__ == "__main__":
    run_phase1()
