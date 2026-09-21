# Social Media Posts Drafts (Local Only - Not Committed)

---

## 1. Reddit Post for r/MachineLearning

**Title:** `[R] Interoception in Neural Networks: Error detection and calibration via internal layer activation statistics`

**Body:**
Most uncertainty estimation techniques rely on output logits, temperature scaling, or ensembling. We explored an alternative hypothesis: can a neural network detect its own impending errors solely by monitoring its internal activation statistics during forward passes?

In our experiments with the `interoception` repository (https://github.com/haidar167/interoception), we extracted layer-wise activation norms, variance drift, and inter-layer covariances across hidden representations. On test classification tasks, output softmax confidence alone achieved an error detection AUC of 0.9653, whereas internal layer statistics alone reached an AUC of 0.7063 (and 0.9605 combined). Furthermore, evaluating interoceptive loss heads in Phase 2 demonstrated improved expected calibration error (ECE down to 0.0028 in end-to-end setups vs 0.0047 baseline).

The key finding is that while output confidence dominates single-domain error detection, internal activation statistics capture representational instability and subtle drift before final output softmax probabilities shift.

Code, benchmark tables, and reproducibility scripts are available open-source on GitHub: https://github.com/haidar167/interoception. We would appreciate feedback, criticism, or ideas on scaling this to deeper transformer architectures and out-of-distribution benchmarks!

---

## 2. Reddit Post for r/learnmachinelearning

**Title:** `I built a project exploring if neural networks can "sense" when they are confused from hidden layer stats`

**Body:**
Hey everyone! I've been working on an open-source project called **Interoception** to test if neural networks can detect their own mistakes by looking at their internal hidden activations rather than just output probabilities.

By attaching light hooks to intermediate layers and computing activation norms and variance, the model learns to track when internal representations start drifting. In our benchmarks, output confidence reached 96.5% error-detection AUC, while internal stats alone achieved 70.6% AUC and helped lower expected calibration error (ECE down to 0.0028).

Check out the code, setup instructions, and reliability graphs here: https://github.com/haidar167/interoception. Would love any thoughts, questions, or feedback on the approach!

---

## 3. Show HN Post

**Title:** `Show HN: Interoception – Neural networks that sense their own confusion from activation statistics`

**Body:**
Interoception is an open-source PyTorch project evaluating whether neural networks can detect prediction errors and calibration drift directly from internal activation statistics (layer norms, variance drift, and representation stability) during forward passes. Code, benchmark results, and reliability diagrams are available at https://github.com/haidar167/interoception.
