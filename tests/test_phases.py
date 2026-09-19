"""Tests for phase module functions and data utilities."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from interoception.models import BaselineMLP
from interoception.phase1_error_detection import train_baseline_model
from interoception.data import load_mnist_data


def test_train_baseline_model_quick():
    """Verify train_baseline_model executes a quick epoch without errors."""
    dummy_x = torch.randn(10, 784)
    dummy_y = torch.randint(0, 10, (10,))
    ds = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(ds, batch_size=5)

    model = train_baseline_model(loader, input_dim=784, hidden_dim=32, num_classes=10, epochs=1)
    assert isinstance(model, BaselineMLP)


def test_load_mnist_data_shapes():
    """Verify load_mnist_data returns valid train/test data loaders."""
    train_loader, test_loader = load_mnist_data(batch_size=32)
    assert train_loader is not None
    assert test_loader is not None
    images, labels = next(iter(test_loader))
    assert images.shape[0] <= 32
    assert images.shape[1] == 784
