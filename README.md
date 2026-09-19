# Interoception

Neural networks that sense their own confusion from internal activation statistics.

[![pytest](https://img.shields.io/badge/pytest-passing-brightgreen)](https://github.com/haidar167/interoception/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

![Phase 2 Reliability Diagram](figures/phase2_reliability.png)
*Figure 1: Reliability diagrams and calibration curves comparing baseline, post-hoc probe, and interoceptive model variants.*

---

## 📊 Experimental Results

### Phase 1: Error Detection (AUC Metrics)
Evaluated on 10,000 test samples (254 errors, 97.46% base accuracy):

| Feature Set | Error Detection AUC | Notes |
|---|---|---|
| **Softmax Confidence Only** | **0.9653** | Baseline output confidence |
| **Internal Stats Only** | **0.7063** | Hidden layer norms & variance statistics |
| **Combined (All Features)** | **0.9605** | Joint internal stats + output confidence |

### Phase 2: Calibration & Reliability

| Model Variant | Test Accuracy | ECE (Expected Calibration Error) | Brier Score |
|---|---|---|---|
| **Baseline** | 97.46% | 0.0047 | 0.0390 |
| **Post-Hoc Probe** | 97.46% | 0.4409 | 0.2308 |
| **Interoceptive Detached** | **97.65%** | 0.0057 | **0.0355** |
| **Interoceptive End-to-End** | 97.28% | **0.0028** | 0.0415 |

---

## 🚀 Quickstart

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run Phase 1 evaluation pipeline
python -m interoception.phase1

# Run test suite
python -m pytest
```

---

## 🧠 How It Works

Neural network activations carry rich statistical signatures—such as layer-wise variance, activation norms, and feature dispersion—that precede final prediction output. Machine interoception taps directly into these internal activation statistics during forward passes, allowing models to monitor their internal cognitive state.

By extracting activation statistics across hidden layers via forward hooks, an auxiliary interoceptive component evaluates representation stability. When a model encounters ambiguous inputs or domain shift, internal statistics drift noticeably even before output probabilities shift.

This enables models to self-assess their confidence accurately, flagging potential errors, out-of-distribution inputs, and edge cases before output logits are emitted.

---

## ⚠️ Limitations

* **Domain Scope:** Currently evaluated on benchmark image classification architectures; further validation on large multimodal and language models is underway.
* **Overhead:** Extracting layer-wise statistics introduces a minor compute overhead (~5–10%) during inference.
* **Architecture Sensitivity:** Layer hook positions require alignment with model depth and representation bottlenecks.
