"""Unit tests for model definitions and replay buffers."""

import pytest
import torch
from interoception.models import (
    BaselineMLP,
    ErrorProbe,
    FIFOReplayBuffer,
    InteroceptiveNet,
    ReservoirReplayBuffer,
)


def test_baseline_mlp_forward_and_backward():
    model = BaselineMLP(input_dim=64, hidden_dim=32, num_classes=5)
    x = torch.randn(8, 64)
    logits, hidden = model(x)

    assert logits.shape == (8, 5)
    assert hidden.shape == (8, 32)

    loss = logits.sum()
    loss.backward()
    assert model.fc1.weight.grad is not None
    assert model.fc2.weight.grad is not None


def test_error_probe():
    probe = ErrorProbe(in_features=4, hidden_dim=8)
    stats = torch.randn(10, 4)
    p_err = probe(stats)
    assert p_err.shape == (10, 1)
    assert (p_err >= 0.0).all() and (p_err <= 1.0).all()


def test_interoceptive_net_forward_and_backward():
    model_e2e = InteroceptiveNet(input_dim=64, hidden_dim=32, num_classes=5, detach_stats=False)
    x = torch.randn(8, 64, requires_grad=True)
    logits, hidden, stats = model_e2e(x)

    assert logits.shape == (8, 5)
    assert hidden.shape == (8, 32)
    assert stats.shape == (8, 4)

    loss = logits.sum()
    loss.backward()
    assert model_e2e.fc1.weight.grad is not None
    assert model_e2e.fc2.weight.grad is not None

    model_det = InteroceptiveNet(input_dim=64, hidden_dim=32, num_classes=5, detach_stats=True)
    x2 = torch.randn(8, 64)
    logits2, hidden2, stats2 = model_det(x2)
    assert logits2.shape == (8, 5)


def test_replay_buffers_strict_memory_cap():
    # Enforce buffer capacity = 500
    fifo = FIFOReplayBuffer(capacity=500)
    reservoir = ReservoirReplayBuffer(capacity=500)

    # Feed 2000 samples into each buffer
    for i in range(20):
        x = torch.randn(100, 64)
        y = torch.randint(0, 10, (100,))
        fifo.add(x, y)
        reservoir.add(x, y)
        # Invariant: buffer size never exceeds 500
        assert len(fifo) <= 500
        assert len(reservoir) <= 500

    assert len(fifo) == 500
    assert len(reservoir) == 500

    # Sampling works as expected
    x_samp, y_samp = fifo.sample(32)
    assert x_samp.shape == (32, 64)
    assert y_samp.shape == (32,)

    xr_samp, yr_samp = reservoir.sample(32)
    assert xr_samp.shape == (32, 64)
    assert yr_samp.shape == (32,)
