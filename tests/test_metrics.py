"""Unit tests for calibration and error detection metrics."""

import numpy as np
import pytest
from interoception.metrics import (
    compute_auc,
    compute_brier_score,
    compute_ece,
)


def test_ece_perfect_calibration():
    # Exactly calibrated bins
    # Bin 1: conf = 0.2, acc = 0.2
    # Bin 2: conf = 0.5, acc = 0.5
    # Bin 3: conf = 0.8, acc = 0.8
    confs = np.concatenate([
        np.full(100, 0.2),
        np.full(100, 0.5),
        np.full(100, 0.8),
    ])
    accs = np.concatenate([
        np.array([1] * 20 + [0] * 80),
        np.array([1] * 50 + [0] * 50),
        np.array([1] * 80 + [0] * 20),
    ])
    ece = compute_ece(confs, accs, n_bins=15)
    assert ece < 0.02, f"Expected ECE near 0, got {ece}"


def test_ece_always_confident_half_wrong():
    # 100% confidence on all samples, but only 50% correct
    confs = np.ones(500)
    accs = np.array([1] * 250 + [0] * 250)
    ece = compute_ece(confs, accs, n_bins=15)
    assert ece >= 0.45, f"Expected high ECE near 0.5, got {ece}"


def test_auc_perfect_and_inverse():
    y_true_error = np.array([0, 0, 0, 1, 1, 1])
    # Higher score = higher probability of error
    perfect_score = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    assert compute_auc(y_true_error, perfect_score) == 1.0

    inverted_score = 1.0 - perfect_score
    assert compute_auc(y_true_error, inverted_score) == 0.0


def test_brier_score_binary_and_multiclass():
    # Binary
    confs = np.array([1.0, 1.0])
    accs = np.array([1.0, 0.0])
    # ((1-1)^2 + (1-0)^2)/2 = 0.5
    assert compute_brier_score(confs, accs) == 0.5

    # Multi-class
    probs = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
    ])
    targets = np.array([0, 1])
    # Sample 0: (1-1)^2 + (0-0)^2 = 0
    # Sample 1: (0.5-0)^2 + (0.5-1)^2 = 0.25 + 0.25 = 0.5
    # Mean: 0.25
    brier = compute_brier_score(probs, targets)
    assert np.isclose(brier, 0.25)
