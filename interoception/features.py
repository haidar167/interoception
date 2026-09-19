"""Feature extraction from neural network internal hidden activations and output probabilities."""

from typing import Union
import numpy as np
import torch


def extract_features(
    h: Union[torch.Tensor, np.ndarray],
    p: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Extract 5 uncertainty/interoception features for each sample in a batch.

    Features:
    1. Mean activation: h.mean(axis=1)
    2. Activation spread: h.std(axis=1)
    3. Fraction of awake neurons: (h > 0).float().mean(axis=1)
    4. Activation magnitude: h.abs().mean(axis=1)
    5. Softmax confidence: p.max(axis=1)

    Args:
        h: Hidden activation matrix of shape [batch, hidden_dim] or [hidden_dim].
        p: Softmax probability matrix of shape [batch, num_classes] or [num_classes].

    Returns:
        Tensor of shape [batch, 5] containing the 5 features.
    """
    if isinstance(h, np.ndarray):
        h_t = torch.from_numpy(h).float()
    else:
        h_t = h.float()

    if isinstance(p, np.ndarray):
        p_t = torch.from_numpy(p).float()
    else:
        p_t = p.float()

    if h_t.dim() == 1:
        h_t = h_t.unsqueeze(0)
    if p_t.dim() == 1:
        p_t = p_t.unsqueeze(0)

    # 1. Mean activation
    feat1 = h_t.mean(dim=1, keepdim=True)

    # 2. Activation spread (unbiased=False avoids NaN when hidden_dim=1)
    feat2 = h_t.std(dim=1, keepdim=True, unbiased=False)

    # 3. Fraction of awake neurons
    feat3 = (h_t > 0).float().mean(dim=1, keepdim=True)

    # 4. Activation magnitude
    feat4 = h_t.abs().mean(dim=1, keepdim=True)

    # 5. Softmax confidence
    feat5 = p_t.max(dim=1, keepdim=True).values

    features = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
    return features


def extract_internal_stats(
    h: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Extract only the 4 internal activation statistics [batch, 4].

    Features:
    1. Mean activation: h.mean(axis=1)
    2. Activation spread: h.std(axis=1)
    3. Fraction of awake neurons: (h > 0).float().mean(axis=1)
    4. Activation magnitude: h.abs().mean(axis=1)

    Args:
        h: Hidden activation matrix of shape [batch, hidden_dim] or [hidden_dim].

    Returns:
        Tensor of shape [batch, 4] containing internal stats.
    """
    if isinstance(h, np.ndarray):
        h_t = torch.from_numpy(h).float()
    else:
        h_t = h.float()

    if h_t.dim() == 1:
        h_t = h_t.unsqueeze(0)

    feat1 = h_t.mean(dim=1, keepdim=True)
    feat2 = h_t.std(dim=1, keepdim=True, unbiased=False)
    feat3 = (h_t > 0).float().mean(dim=1, keepdim=True)
    feat4 = h_t.abs().mean(dim=1, keepdim=True)

    return torch.cat([feat1, feat2, feat3, feat4], dim=1)
