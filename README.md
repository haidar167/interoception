# INTEROCEPTION: Networks That Sense Their Own Internal Activity

Giving artificial neural networks a sense of their own internal cognitive state by tapping directly into hidden layer activation statistics to detect classification errors, calibrate uncertainty, and regulate learning dynamics.

[![pytest](https://img.shields.io/badge/pytest-passing-brightgreen)](https://github.com/haidar167/interoception/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

![Phase 2 Reliability Diagram](figures/phase2_reliability.png)
*Figure 1: Reliability diagrams and calibration curves comparing baseline, post-hoc probe, and interoceptive model variants.*

---

## 1. Executive Summary & Biological Motivation

In biological organisms, **interoception** is the perception of internal physiological signals—cardiovascular pulse, respiratory rate, metabolic fatigue, and visceral tension. Organisms do not perceive the world solely through external sensors; they continuously monitor the stability of their internal physiological substrate.

In standard artificial neural networks, prediction confidence is typically derived exclusively from the **output layer** (e.g. maximum softmax probability). However:
1. **Output Overconfidence**: Softmax probabilities are notoriously uncalibrated and overconfident on ambiguous or out-of-distribution inputs.
2. **Hidden Information Discard**: Layers preceding the logits carry rich internal signals—such as activation dispersion, layer-wise variance, and fraction of active units—that signal representation instability long before logits are produced.

**Machine Interoception** equips neural architectures with an internal monitoring loop that reads internal hidden layer statistics $\mathbf{h} \in \mathbb{R}^H$ during forward passes, enabling the network to know when it is confused.

```
Input Sample x ──► [ Hidden Layer 1 (h1) ] ──► [ Hidden Layer 2 (h2) ] ──► Logits z ──► Softmax p
                             │                               │
                             ▼                               ▼
                 [ Forward Hook Monitor ]        [ Forward Hook Monitor ]
                 Extract: Mean, Std, Awake%, Mag Extract: Mean, Std, Awake%, Mag
                             │                               │
                             └───────────────┬───────────────┘
                                             ▼
                             [ Interoceptive Feature Vector ]
                                     z_stats in R^4
                                             │
                                             ▼
                               [ Error Detection / Probe ]
                                 s = P(error | z_stats)
```

---

## 2. Mathematical Formulation

### 2.1 Model Architecture & Forward Propagation

Let input sample $\mathbf{x} \in \mathbb{R}^D$ ($D=784$ for MNIST) with ground-truth categorical target $y \in \{0, 1, \dots, K-1\}$ ($K=10$). The feedforward network is parameterized by $\boldsymbol{\theta} = \{\mathbf{W}_1 \in \mathbb{R}^{H_1 \times D}, \mathbf{b}_1 \in \mathbb{R}^{H_1}, \mathbf{W}_2 \in \mathbb{R}^{H_2 \times H_1}, \mathbf{b}_2 \in \mathbb{R}^{H_2}, \mathbf{W}_3 \in \mathbb{R}^{K \times H_2}, \mathbf{b}_3 \in \mathbb{R}^K\}$ with $H_1=256, H_2=128$:

1. **Hidden Representations**:
   $$\mathbf{h}_1(\mathbf{x}) = \operatorname{ReLU}\left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right) \in \mathbb{R}_{\ge 0}^{H_1}$$
   $$\mathbf{h}_2(\mathbf{x}) = \operatorname{ReLU}\left(\mathbf{W}_2 \mathbf{h}_1(\mathbf{x}) + \mathbf{b}_2\right) \in \mathbb{R}_{\ge 0}^{H_2}$$

2. **Logit Vector & Softmax Distribution**:
   $$\mathbf{z}(\mathbf{x}) = \mathbf{W}_3 \mathbf{h}_2(\mathbf{x}) + \mathbf{b}_3 \in \mathbb{R}^K$$
   $$\mathbf{p}(\mathbf{x}) = \operatorname{softmax}(\mathbf{z}(\mathbf{x})) \in \Delta^{K-1}, \quad p_k(\mathbf{x}) = \frac{\exp(z_k)}{\sum_{j=0}^{K-1} \exp(z_j)}$$

3. **Classification Decision & Error Indicator**:
   $$\hat{y}(\mathbf{x}) = \operatorname{argmax}_{k \in \{0,\dots,K-1\}} p_k(\mathbf{x}), \quad e(\mathbf{x}, y) = \mathbb{I}\left(\hat{y}(\mathbf{x}) \ne y\right) \in \{0, 1\}$$

---

### 2.2 The 4 Internal Activation Statistics

For hidden activation vector $\mathbf{h} = [h_1, h_2, \dots, h_H]^\top \in \mathbb{R}^H$, the interoceptive extractor computes a 4-dimensional summary vector $\mathbf{z}_{\text{stats}} \in \mathbb{R}^4$:

$$\mathbf{z}_{\text{stats}}(\mathbf{h}) = \begin{bmatrix}
\mu(\mathbf{h}) \\
\sigma(\mathbf{h}) \\
\rho_{>0}(\mathbf{h}) \\
\nu(\mathbf{h})
\end{bmatrix} \in \mathbb{R}^4$$

where:
1. **Mean Activation**:
   $$\mu(\mathbf{h}) = \frac{1}{H} \sum_{d=1}^H h_d$$
2. **Activation Spread / Standard Deviation**:
   $$\sigma(\mathbf{h}) = \sqrt{\frac{1}{H} \sum_{d=1}^H \left(h_d - \mu(\mathbf{h})\right)^2}$$
3. **Fraction of Awake / Active Neurons**:
   $$\rho_{>0}(\mathbf{h}) = \frac{1}{H} \sum_{d=1}^H \mathbb{I}\left(h_d > 0\right)$$
4. **Mean Absolute Activation Magnitude**:
   $$\nu(\mathbf{h}) = \frac{1}{H} \sum_{d=1}^H |h_d|$$

*(Note: Post-ReLU $\mathbf{h} \ge 0$, which mathematically implies $\nu(\mathbf{h}) \equiv \mu(\mathbf{h})$; under pre-activation or centered representations, the two measures diverge).*

### 2.3 The Combined 5-Feature Interoceptive Vector

Appending the output softmax confidence $\kappa(\mathbf{x}) = \max_{k} p_k(\mathbf{x}) = \|\mathbf{p}(\mathbf{x})\|_\infty$ yields the complete 5-dimensional uncertainty vector:

$$\mathbf{z}_{\text{all}}(\mathbf{x}) = \begin{bmatrix}
\mu(\mathbf{h}) \\
\sigma(\mathbf{h}) \\
\rho_{>0}(\mathbf{h}) \\
\nu(\mathbf{h}) \\
\max_k p_k(\mathbf{x})
\end{bmatrix} \in \mathbb{R}^5$$

---

### 2.4 Calibration & Evaluation Metrics

#### Expected Calibration Error (ECE)
Samples are partitioned into $B$ equal-width confidence bins $B_1, B_2, \dots, B_B$ across $[0, 1]$ (default $B=15$):

$$\operatorname{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} \left| \operatorname{acc}(B_b) - \operatorname{conf}(B_b) \right|$$

where:
$$\operatorname{acc}(B_b) = \frac{1}{|B_b|} \sum_{i \in B_b} \mathbb{I}\left(\hat{y}_i = y_i\right), \quad \operatorname{conf}(B_b) = \frac{1}{|B_b|} \sum_{i \in B_b} \max_k p_{i,k}$$

#### Brier Score
Strictly proper scoring rule measuring probability calibration across all $K$ classes:

$$\operatorname{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{k=0}^{K-1} \left(p_{i,k} - \mathbb{I}(y_i = k)\right)^2 \in [0, 2]$$

#### Error Detection ROC-AUC
Measures ranking ability to assign higher predicted error scores $s(\mathbf{x}) \in [0, 1]$ to misclassified samples ($e_i = 1$) than to correct samples ($e_i = 0$):

$$\operatorname{AUC} = \frac{\sum_{i: e_i = 1} \sum_{j: e_j = 0} \mathbb{I}\left(s(\mathbf{x}_i) > s(\mathbf{x}_j)\right) + \frac{1}{2}\mathbb{I}\left(s(\mathbf{x}_i) = s(\mathbf{x}_j)\right)}{N_{\text{err}} \cdot N_{\text{corr}}}$$

---

## 3. Experimental Results

### Phase 1: Error Detection (AUC Metrics)
Evaluated on 10,000 MNIST test samples (254 errors, **97.46%** base test accuracy):

| Feature Set | Dimensionality | Error Detection ROC-AUC | Description |
| :--- | :---: | :---: | :--- |
| **Softmax Confidence Only** | $\mathbb{R}^1$ | **0.9653** | Baseline output confidence $1 - \max_k p_k$ |
| **Internal Stats Only** | $\mathbb{R}^4$ | **0.7063** | 4 activation statistics from hidden layer |
| **Combined (All Features)** | $\mathbb{R}^5$ | **0.9605** | Joint internal stats + output confidence |

> **Finding**: On pristine in-distribution test sets, output softmax confidence is strong for error detection (0.9653 AUC). However, the internal 4-stat vector independently achieves 0.7063 AUC **without access to class logits**, demonstrating that internal activations encode error probability before output projection.

---

### Phase 2: Calibration & Reliability

Comparing four training and inference variants:

| Model Variant | Test Accuracy | ECE (Expected Calibration Error) | Brier Score |
| :--- | :---: | :---: | :---: |
| **Baseline MLP** | 97.46% | 0.0047 | 0.0390 |
| **Post-Hoc Error Probe** | 97.46% | 0.4409 | 0.2308 |
| **Interoceptive Detached** | **97.65%** | 0.0057 | **0.0355** |
| **Interoceptive End-to-End** | 97.28% | **0.0028** | 0.0415 |

> **Key Findings**:
> - **End-to-End Joint Training** achieves a **40.4% reduction in ECE** ($0.0047 \to 0.0028$), yielding near-perfect calibration.
> - **Interoceptive Detached** achieves the lowest overall Brier score (**0.0355**) and highest test accuracy (**97.65%**).

---

## 4. The Interoceptive Research Continuum

This project is the foundational first chapter in our four-part neural self-awareness series:

1. **[INTEROCEPTION](https://github.com/haidar167/interoception)**: Sensing internal cognitive confusion and calibrating uncertainty from 4 internal activation statistics.
2. **[PROPRIOCEPTION](https://github.com/haidar167/proprioception)**: Sensing physical substrate weight damage, localizing damaged layers, and triggering targeted self-repair reflexes.
3. **[META-INTEROCEPTION](https://github.com/haidar167/meta-interoception)**: Reading another network's mind—external observer networks vs internal introspection under 12-month drift.
4. **[NOCICEPTION](https://github.com/haidar167/nociception)**: Introspection-driven help-seeking under drift and pain-triggered consolidation ("resting when hurt").

---

## 5. Quickstart & Test Suite

All experiments are CPU-compatible, deterministic, and validated with unit tests:

```bash
# Clone and install
pip install -e .

# Run complete pytest suite (14 passing tests)
pytest -v

# Run Phase 1 evaluation
python -m interoception.phase1

# Run Phase 2 calibration benchmark
python -m interoception.phase2
```

---

## 🌐 The Neural Self-Awareness Continuum

| # | Project | Biological Analogy | Core Capability | Live Link |
|---|---|---|---|---|
| **1** | **[Interoception](https://haidar167.github.io/interoception/)** | Internal visceral sensing | Senses internal confusion via hidden activation stats | [GitHub](https://github.com/haidar167/interoception) |
| **2** | **[Proprioception](https://haidar167.github.io/proprioception/)** | Body substrate awareness | Senses weight damage & localizes corrupted layers | [GitHub](https://github.com/haidar167/proprioception) |
| **3** | **[Meta-Interoception](https://haidar167.github.io/meta-interoception/)** | Metacognitive monitoring | Monitors the calibration of its own self-monitors | [GitHub](https://github.com/haidar167/meta-interoception) |
| **4** | **[Nociception](https://haidar167.github.io/nociception/)** | Pain-driven help seeking | Spends limited human supervision budget on likely errors | [GitHub](https://github.com/haidar167/nociception) |
| **5** | **[SOMNIA](https://haidar167.github.io/somnia/)** | Targeted sleep consolidation | Dreams targeted examples to patch its own weak spots | [GitHub](https://github.com/haidar167/somnia) |

---
*Part of the Neural Self-Awareness research continuum by [haidar167](https://github.com/haidar167).*
