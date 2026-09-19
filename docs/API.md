# Interoception API Reference

This document provides detailed API specifications for the modules in the `interoception` library.

---

## 1. Feature Extraction (`interoception.features`)

### `extract_internal_stats(outputs: torch.Tensor, hidden_activations: list[torch.Tensor]) -> np.ndarray`
Extracts statistical features from intermediate model activation tensors and logits.
- **Parameters:**
  - `outputs`: Logit tensor from final model classification head.
  - `hidden_activations`: List of intermediate hidden feature map tensors.
- **Returns:** 1D NumPy array containing layer activation norms, variance drift, and entropy features.

### `extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
Runs model inference over a DataLoader and extracts joint feature matrices.
- **Returns:** `(confidence_features, internal_stats_features, ground_truth_labels)`

---

## 2. Calibration & Evaluation Metrics (`interoception.metrics`)

### `compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float`
Calculates ROC AUC score for binary classification or error detection.

### `ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float`
Calculates Expected Calibration Error (ECE) across binned confidence intervals.

### `brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float`
Calculates Brier Score (mean squared error of predicted probabilities).

---

## 3. Models (`interoception.models`)

### `BaselineMLP(input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10)`
Standard multi-layer perceptron architecture with forward activation hooks.

### `InteroceptiveProbe(input_dim: int)`
Auxiliary classification head predicting representation stability and model error probability.
