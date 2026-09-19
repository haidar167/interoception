"""Project INTEROCEPTION: Neural Networks That Sense Their Own Confusion.

A second channel for neural uncertainty via internal activation statistics.
"""

from interoception.features import extract_features, extract_internal_stats
from interoception.metrics import compute_auc, compute_brier_score, compute_ece
from interoception.models import (
    BaselineMLP,
    ErrorProbe,
    FIFOReplayBuffer,
    InteroceptiveNet,
    ReservoirReplayBuffer,
)
from interoception.utils import save_results, set_seed

__version__ = "0.1.0"

__all__ = [
    "extract_features",
    "extract_internal_stats",
    "compute_auc",
    "compute_ece",
    "compute_brier_score",
    "BaselineMLP",
    "ErrorProbe",
    "InteroceptiveNet",
    "FIFOReplayBuffer",
    "ReservoirReplayBuffer",
    "set_seed",
    "save_results",
]
