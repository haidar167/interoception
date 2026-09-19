"""Dataset loading for MNIST with automatic fallback to sklearn load_digits."""

import os
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_mnist_data(
    batch_size: int = 128,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader, int]:
    """Load MNIST dataset with torchvision, falling back to sklearn digits if download fails.

    Args:
        batch_size: Mini-batch size for DataLoaders.
        data_dir: Directory path for downloading/storing torchvision datasets.

    Returns:
        (train_loader, test_loader, input_dim) where input_dim is 784 (MNIST) or 64 (digits).
    """
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ])
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, 784
    except Exception as e:
        print(f"torchvision MNIST download/load failed ({e}). Falling back to sklearn digits.")
        digits = load_digits()
        X = digits.data / 16.0  # normalize [0, 1]
        y = digits.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test).long(),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, 64
