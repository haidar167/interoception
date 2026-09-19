"""Unit tests for feature extraction."""

import numpy as np
import pytest
import torch
from interoception.features import extract_features, extract_internal_stats


def test_extract_features_shape_and_values():
    batch_size = 16
    hidden_dim = 256
    num_classes = 10

    torch.manual_seed(42)
    h = torch.randn(batch_size, hidden_dim)
    logits = torch.randn(batch_size, num_classes)
    p = torch.softmax(logits, dim=1)

    feats = extract_features(h, p)

    # Check shape [batch, 5]
    assert feats.shape == (batch_size, 5)

    # Check no NaNs and all finite
    assert not torch.isnan(feats).any()
    assert torch.isfinite(feats).all()

    # Check awake fraction is in [0, 1]
    awake = feats[:, 2]
    assert (awake >= 0.0).all() and (awake <= 1.0).all()

    # Check confidence is in [0, 1]
    conf = feats[:, 4]
    assert (conf >= 0.0).all() and (conf <= 1.0).all()


def test_extract_features_single_sample_and_numpy():
    hidden_dim = 64
    num_classes = 10

    h_np = np.random.randn(hidden_dim)
    p_np = np.exp(np.random.randn(num_classes))
    p_np = p_np / p_np.sum()

    feats = extract_features(h_np, p_np)
    assert feats.shape == (1, 5)
    assert not torch.isnan(feats).any()
    assert torch.isfinite(feats).all()


def test_extract_internal_stats():
    batch_size = 8
    hidden_dim = 128
    h = torch.randn(batch_size, hidden_dim)

    stats = extract_internal_stats(h)
    assert stats.shape == (batch_size, 4)
    assert not torch.isnan(stats).any()
    assert torch.isfinite(stats).all()
