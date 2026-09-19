"""Neural network architectures for baseline and interoceptive models."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from interoception.features import extract_internal_stats


class BaselineMLP(nn.Module):
    """Standard Multi-Layer Perceptron baseline.

    Architecture:
        Linear(input_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> num_classes)
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and hidden activations.

        Args:
            x: Input tensor of shape [batch_size, input_dim].

        Returns:
            Tuple of (logits [batch, num_classes], hidden [batch, hidden_dim]).
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        hidden = self.relu(self.fc1(x))
        logits = self.fc2(hidden)
        return logits, hidden


class ErrorProbe(nn.Module):
    """Small post-hoc probe predicting error probability from internal hidden stats."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """Predict error probability.

        Args:
            stats: Tensor of shape [batch, 4].

        Returns:
            Error probabilities of shape [batch, 1].
        """
        return self.net(stats)


class InteroceptiveNet(nn.Module):
    """End-to-end Interoceptive MLP.

    Internal activation statistics (mean, std, awake fraction, mean absolute)
    are appended as 4 extra inputs to the final classification layer.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        num_classes: int = 10,
        detach_stats: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.detach_stats = detach_stats

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Hidden dimension + 4 internal stats
        self.fc2 = nn.Linear(hidden_dim + 4, num_classes)

    def compute_stats(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the 4 internal activation statistics with torch ops."""
        m = h.mean(dim=1, keepdim=True)
        s = h.std(dim=1, keepdim=True, unbiased=False)
        awake = (h > 0).float().mean(dim=1, keepdim=True)
        mag = h.abs().mean(dim=1, keepdim=True)
        stats = torch.cat([m, s, awake, mag], dim=1)
        if self.detach_stats:
            stats = stats.detach()
        return stats

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with internal state feedback.

        Args:
            x: Input tensor of shape [batch, input_dim].

        Returns:
            Tuple of (logits [batch, num_classes], hidden [batch, hidden_dim], stats [batch, 4]).
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        hidden = self.relu(self.fc1(x))
        stats = self.compute_stats(hidden)
        h_augmented = torch.cat([hidden, stats], dim=1)
        logits = self.fc2(h_augmented)
        return logits, hidden, stats
