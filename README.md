# 🧠 Self-Pruning Neural Network

## 📌 Overview

This project implements a **self-pruning neural network** that learns to remove its own unnecessary connections during training. Unlike traditional pruning (done after training), this approach integrates pruning directly into the learning process.

The model is trained on the **CIFAR-10 dataset** and uses a custom gating mechanism to dynamically control which weights remain active.

---

## 🚀 Key Idea

Each weight in the network is associated with a learnable **gate parameter**:

* Gate ≈ 1 → Weight is active
* Gate ≈ 0 → Weight is pruned

The effective weight becomes:

```
effective_weight = weight × gate
```

This allows the network to **adaptively shrink itself** while training.

---

## ⚙️ Model Architecture

* Custom `PrunableLinear` layer
* Fully connected network:

  * Input → 512 → 256 → Output (10 classes)
* Activation: ReLU
* Dataset: CIFAR-10

---

## 🧩 Pruning Mechanism

* Each weight has a corresponding gate (via sigmoid)
* L1 regularization is applied on gate values
* Encourages many gates to approach zero → sparse network

---

## 📉 Loss Function

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

Where:

* **CrossEntropyLoss** → classification performance
* **SparsityLoss (L1)** → encourages pruning

---

## 📊 Results

| Lambda | Accuracy (%) | Sparsity (%) |
| ------ | ------------ | ------------ |
| 1e-5   | 46.50        | 1.15         |
| 1e-4   | 45.32        | 1.57         |
| 1e-3   | 41.34        | 1.70         |

---

## 🔍 Observations

* Increasing λ increases sparsity
* Higher sparsity slightly reduces accuracy
* Demonstrates trade-off between model size and performance
* Current sparsity is low → can be improved with:

  * More epochs
  * Higher λ

---

## 📈 Visualization

The distribution of gate values shows:

* Most gates remain active
* Few gates approach zero

This aligns with observed sparsity levels.

---

## 🛠️ How to Run

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run training

```
python src/train.py
```

---

## 📂 Project Structure

```
self-pruning-nn/
│
├── src/
│   ├── model.py
│   ├── utils.py
│   └── train.py
│
├── notebooks/
│   └── experiment.ipynb
│
├── report.md
├── README.md
└── requirements.txt
```

---

## 🎯 Key Highlights

* Custom neural layer with learnable pruning gates
* Integrated pruning during training
* Clean modular code (model, utils, training)
* Experimental analysis with multiple λ values

---

## 🚀 Future Improvements

* Increase training epochs for better pruning
* Experiment with stronger regularization
* Extend to convolutional architectures
* Apply structured pruning techniques

---

## 📌 Note

Dataset files are **not included in the repository**. CIFAR-10 will be automatically downloaded when running the code.

---

## 👨‍💻 Author

Pooja Ellanti
