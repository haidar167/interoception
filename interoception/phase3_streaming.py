"""Phase 3: Introspection under distribution drift (streaming simulation across 12 months)."""

import collections
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interoception.data import load_mnist_data
from interoception.features import extract_internal_stats
from interoception.models import BaselineMLP, FIFOReplayBuffer, ReservoirReplayBuffer
from interoception.utils import save_results, set_seed


def generate_permutations(input_dim: int, num_months: int = 12, seed: int = 42) -> List[np.ndarray]:
    """Generate deterministic pixel permutations for 12 months of drift."""
    rng = np.random.RandomState(seed)
    perms = [np.arange(input_dim)]  # Month 1: Identity
    for _ in range(num_months - 1):
        perms.append(rng.permutation(input_dim))
    return perms


def apply_permutation(x: torch.Tensor, perm: np.ndarray) -> torch.Tensor:
    """Apply pixel permutation to input tensor."""
    return x[:, perm]


def evaluate_accuracy(
    model: BaselineMLP,
    test_loader: torch.utils.data.DataLoader,
    perm: np.ndarray,
    max_eval_samples: int = 1500,
) -> float:
    """Evaluate classification accuracy on permuted test distribution."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            perm_images = apply_permutation(images, perm)
            logits, _ = model(perm_images)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            if total >= max_eval_samples:
                break
    return float(correct / total)


def run_streaming_simulation(
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    input_dim: int,
    permutations: List[np.ndarray],
    condition: str = "baseline",  # "baseline", "intro_fifo", "intro_reservoir"
    samples_per_month: int = 600,
    chunk_size: int = 30,
    lr: float = 1e-3,
) -> Tuple[List[float], int]:
    """Simulate streaming adaptation across 12 months under one condition."""
    model = BaselineMLP(input_dim=input_dim, hidden_dim=256, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Buffer setup
    buffer = None
    if condition == "intro_fifo":
        buffer = FIFOReplayBuffer(capacity=500)
    elif condition == "intro_reservoir":
        buffer = ReservoirReplayBuffer(capacity=500)

    # Simple online error detector head trained on past activations
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
    probe_fitted = False
    past_stats_list = []
    past_errors_list = []

    # Initial pretrain on small clean unpermuted batch (Month 1 calibration)
    initial_images, initial_targets = next(iter(train_loader))
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        logits, h = model(initial_images)
        loss = criterion(logits, initial_targets)
        loss.backward()
        optimizer.step()

    # Pre-fit probe on initial batch
    with torch.no_grad():
        logits, h = model(initial_images)
        stats = extract_internal_stats(h).cpu().numpy()
        errors = (logits.argmax(dim=1) != initial_targets).long().cpu().numpy()
        # Ensure at least two classes for logistic probe
        if len(np.unique(errors)) < 2:
            errors[0] = 1
            errors[1] = 0
        probe.fit(stats, errors)
        probe_fitted = True

    monthly_accuracies = []
    total_triggers = 0
    train_iter = iter(train_loader)

    for month_idx, perm in enumerate(permutations):
        # 1. Measure accuracy on the current month's drifted distribution
        acc = evaluate_accuracy(model, test_loader, perm)
        monthly_accuracies.append(acc)

        # 2. Stream online data for this month in tiny chunks
        samples_streamed = 0
        while samples_streamed < samples_per_month:
            try:
                images, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, targets = next(train_iter)

            # Take tiny chunk
            chunk_x = apply_permutation(images[:chunk_size], perm)
            chunk_y = targets[:chunk_size]
            samples_streamed += len(chunk_x)

            model.train()
            optimizer.zero_grad()
            logits, h = model(chunk_x)
            main_loss = criterion(logits, chunk_y)

            # Introspection Check
            trigger_replay = False
            if condition != "baseline" and probe_fitted:
                stats = extract_internal_stats(h).detach().cpu().numpy()
                p_err = probe.predict_proba(stats)[:, 1]
                # If mean error probability exceeds 0.5, model senses confusion
                if np.mean(p_err) > 0.5:
                    trigger_replay = True
                    total_triggers += 1

            if trigger_replay and buffer is not None and len(buffer) > 0:
                # Replay / consolidation step from internal memory
                replay_x, replay_y = buffer.sample(batch_size=32)
                if replay_x is not None:
                    replay_logits, _ = model(replay_x)
                    replay_loss = criterion(replay_logits, replay_y)
                    combined_loss = 0.5 * main_loss + 0.5 * replay_loss
                    combined_loss.backward()
                else:
                    main_loss.backward()
            else:
                main_loss.backward()

            optimizer.step()

            # Store to buffer and maintain introspection memory
            if buffer is not None:
                buffer.add(chunk_x, chunk_y)

            # Track error stats for probe retraining
            with torch.no_grad():
                stats = extract_internal_stats(h).cpu().numpy()
                errs = (logits.argmax(dim=1) != chunk_y).long().cpu().numpy()
                past_stats_list.append(stats)
                past_errors_list.append(errs)
                if len(past_stats_list) > 20:
                    past_stats_list.pop(0)
                    past_errors_list.pop(0)

                # Periodically re-fit probe with recent stream stats
                if samples_streamed % 300 == 0:
                    X_cat = np.concatenate(past_stats_list, axis=0)
                    y_cat = np.concatenate(past_errors_list, axis=0)
                    if len(np.unique(y_cat)) >= 2:
                        probe.fit(X_cat, y_cat)

    return monthly_accuracies, total_triggers


def run_phase3(seed: int = 42) -> Dict[str, any]:
    """Execute full Phase 3 experiment under 12 months distribution drift."""
    start_time = time.time()
    set_seed(seed)
    print("=" * 75)
    print("PHASE 3: Introspection Under Distribution Drift (Streaming Permuted-MNIST)")
    print("=" * 75)

    print("Loading dataset and generating 12 monthly pixel permutations (seed 42)...")
    train_loader, test_loader, input_dim = load_mnist_data(batch_size=128)
    permutations = generate_permutations(input_dim, num_months=12, seed=42)

    # Condition 1: Baseline (No introspection / naive streaming)
    print("\n[1/3] Running Condition 1: Baseline (No introspection triggers)...")
    acc_baseline, _ = run_streaming_simulation(
        train_loader, test_loader, input_dim, permutations, condition="baseline"
    )

    # Condition 2: Introspective Streaming with FIFO Buffer
    print("\n[2/3] Running Condition 2: Introspective Streaming (FIFO Replay Buffer, max 500)...")
    acc_fifo, triggers_fifo = run_streaming_simulation(
        train_loader, test_loader, input_dim, permutations, condition="intro_fifo"
    )

    # Condition 3: Introspective Streaming with Reservoir Buffer
    print("\n[3/3] Running Condition 3: Introspective Streaming (Reservoir Replay Buffer, max 500)...")
    acc_reservoir, triggers_reservoir = run_streaming_simulation(
        train_loader, test_loader, input_dim, permutations, condition="intro_reservoir"
    )

    elapsed_time = time.time() - start_time

    # Print comparative results table
    print("\n" + "=" * 75)
    print("PHASE 3: 12-MONTH DRIFT ACCURACY TRAJECTORY:")
    print(f"{'Month':<7} | {'Baseline':<12} | {'Intro (FIFO)':<14} | {'Intro (Reservoir)':<18}")
    print("-" * 75)
    for m in range(12):
        print(
            f"Month {m+1:<2} | {acc_baseline[m]*100:>10.2f}% | {acc_fifo[m]*100:>12.2f}% | {acc_reservoir[m]*100:>16.2f}%"
        )
    print("-" * 75)
    mean_base = np.mean(acc_baseline) * 100
    mean_fifo = np.mean(acc_fifo) * 100
    mean_res = np.mean(acc_reservoir) * 100
    print(f"{'MEAN':<7} | {mean_base:>10.2f}% | {mean_fifo:>12.2f}% | {mean_res:>16.2f}%")
    print(f"Triggers Activated -> FIFO: {triggers_fifo}, Reservoir: {triggers_reservoir}")
    print(f"Total Phase 3 Runtime: {elapsed_time:.2f}s (< 900s requirement)")
    print("=" * 75)

    # Plot Accuracy Curves
    fig_path = Path("figures/phase3_drift.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    months = np.arange(1, 13)

    ax.plot(months, [a * 100 for a in acc_baseline], "r--o", label=f"Baseline (Mean: {mean_base:.1f}%)", alpha=0.8)
    ax.plot(months, [a * 100 for a in acc_fifo], "b-s", label=f"Intro FIFO Replay (Mean: {mean_fifo:.1f}%)", alpha=0.85)
    ax.plot(months, [a * 100 for a in acc_reservoir], "g-^", label=f"Intro Reservoir Replay (Mean: {mean_res:.1f}%)", alpha=0.85)

    ax.set_xlabel("Month (Drift Permutation)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Phase 3: Accuracy Under 12 Months Distribution Drift", fontsize=12, pad=10)
    ax.set_xticks(months)
    ax.set_ylim([0, 100])
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Drift trajectory plot saved to: {fig_path.resolve()}")

    results = {
        "phase": 3,
        "months": 12,
        "seed": seed,
        "baseline_accuracy_per_month": acc_baseline,
        "intro_fifo_accuracy_per_month": acc_fifo,
        "intro_reservoir_accuracy_per_month": acc_reservoir,
        "mean_accuracy": {
            "baseline": float(mean_base / 100.0),
            "intro_fifo": float(mean_fifo / 100.0),
            "intro_reservoir": float(mean_res / 100.0),
        },
        "triggers_activated": {
            "fifo": int(triggers_fifo),
            "reservoir": int(triggers_reservoir),
        },
        "buffer_max_capacity": 500,
        "execution_time_seconds": float(elapsed_time),
    }

    save_path = Path("results/phase3.json")
    save_results(results, save_path, seed=seed)
    print(f"Phase 3 results saved to: {save_path.resolve()}\n")

    return results


if __name__ == "__main__":
    run_phase3()
