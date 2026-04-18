# Self-Pruning Neural Network

## Overview

This project implements a self-pruning neural network where each weight is associated with a learnable gate parameter. The network learns to dynamically remove unnecessary connections during training, leading to a sparse and efficient model.

---

## Key Idea

Each weight is multiplied by a learnable gate value between 0 and 1:

* Gate ≈ 1 → Connection is active
* Gate ≈ 0 → Connection is pruned

This allows the network to **adaptively reduce its complexity during training** rather than relying on post-processing pruning.

---

## Why L1 Regularization Encourages Sparsity

An L1 penalty is applied to all gate values. Since L1 regularization minimizes the sum of absolute values, it pushes many gate values toward zero.

As a result:

* Less important connections receive lower gate values
* Many gates approach zero
* Corresponding weights effectively get removed

This leads to a **sparse neural network** with fewer active parameters.

---

## Experimental Setup

* Dataset: CIFAR-10
* Model: Fully connected neural network with custom prunable layers
* Optimizer: Adam
* Epochs: 5
* Sparsity threshold: 1e-2
* Loss Function:
  Total Loss = CrossEntropyLoss + λ × L1(Gates)

---

## Results

| Lambda | Test Accuracy (%) | Sparsity (%) |
| ------ | ----------------- | ------------ |
| 1e-5   | 46.50             | 1.15         |
| 1e-4   | 45.32             | 1.57         |
| 1e-3   | 41.34             | 1.70         |

---

## Observations

* Increasing λ slightly increases sparsity in the network
* Higher λ leads to a small drop in accuracy
* The sparsity values are relatively low, indicating that:

  * The regularization strength may need to be increased further
  * Or more training epochs are required for stronger pruning

This demonstrates the expected **trade-off between model sparsity and performance**.

---

## Gate Distribution Analysis

The histogram of gate values shows:

* Most gates are clustered away from zero
* Only a small fraction approach zero

This aligns with the observed low sparsity levels.

---

## Conclusion

The model successfully integrates pruning into the training process. While the sparsity achieved is modest, the approach demonstrates how neural networks can learn to optimize their own structure dynamically.

Future improvements could include:

* Increasing training epochs
* Using stronger regularization (higher λ)
* Applying pruning-aware initialization

---

## How to Run

```bash
python src/train.py
```

---

## Repository Structure

```
src/
 ├── model.py
 ├── utils.py
 └── train.py
```
