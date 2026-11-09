# Machine Learning & Neural Networks

**A hands-on comparison of classical machine learning, feature importance, neural-network activations, and MLP training configurations using Iris and MNIST.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Networks-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

## Overview

This repository contains **Assignment 4: Machine Learning and Neural Networks**, implemented in Python in a single Google Colab / Jupyter notebook. It progresses from interpretable baseline classifiers to a fully connected neural network trained on handwritten digits. Each part includes reproducible data preparation, experiments, quantitative evaluation, visualizations, and short conclusions.

**Notebook:** [`ProgForAI4.ipynb`](./ProgForAI4.ipynb)

### Project highlights

- **4 tasks** covering both classical machine learning and PyTorch neural networks.
- **2 datasets:** Iris (150 flower samples, 3 species) and MNIST (70,000 handwritten digit images, 10 classes).
- **3 classifiers** compared using Accuracy, macro Precision/Recall/F1, and multiclass ROC-AUC.
- **Feature importance and hyperparameter experiments** for KNN and Decision Tree.
- **3 activation functions** evaluated with the same single-hidden-layer network.
- **9 MLP configurations** evaluated by varying training epochs and batch size.
- **97.81% test accuracy** for the selected MNIST MLP in the recorded run.

## Contents

1. [Tasks at a glance](#tasks-at-a-glance)
2. [Technology stack](#technology-stack)
3. [Datasets](#datasets)
4. [Task 1 — Iris classification](#task-1--iris-classification)
5. [Task 2 — Feature importance and hyperparameters](#task-2--feature-importance-and-hyperparameters)
6. [Task 3 — PyTorch perceptron](#task-3--pytorch-perceptron)
7. [Task 4 — MNIST MLP](#task-4--mnist-mlp)
8. [How to run](#how-to-run)
9. [Reproducibility and evaluation notes](#reproducibility-and-evaluation-notes)
10. [Lessons learned](#lessons-learned)
11. [Repository structure](#repository-structure)
12. [References](#references)

## Tasks at a glance

| Task | Focus | Dataset | Main output |
|---|---|---|---|
| **1** | Compare Decision Tree, Logistic Regression, and KNN | Iris | Classification reports, confusion matrices, ROC curves, metrics table |
| **2** | Investigate important features and hyperparameters | Iris | Feature-importance chart, KNN `k` sweep, Decision Tree depth sweep |
| **3** | Compare ReLU, Sigmoid, and Tanh activations | Iris | Learning curves, accuracy comparison, confusion matrices |
| **4** | Train an MLP and study epochs / batch size | MNIST | Nine-run experiment table, learning curves, digit predictions |

## Technology stack

| Category | Libraries / tools | Purpose |
|---|---|---|
| Programming | Python | All experiments |
| Classical ML | scikit-learn | Iris dataset, splitting, preprocessing, classifiers, metrics |
| Deep learning | PyTorch, torchvision | Neural networks, training loops, MNIST loader |
| Data handling | NumPy, pandas | Arrays, tables, experiment summaries |
| Visualization | Matplotlib, Seaborn | Learning curves, bar charts, heatmaps, ROC plots |
| Environment | Google Colab / Jupyter Notebook | Interactive execution and visualization |

The notebook selects **CUDA automatically when available** for Task 4; the recorded MNIST run used **CPU**. A GPU is optional, not required.

## Datasets

### Iris

The built-in `load_iris()` dataset from scikit-learn contains **150 rows**, **4 numeric measurements**, and **3 balanced species** (50 examples per species):

- **Setosa**
- **Versicolor**
- **Virginica**

Features: sepal length, sepal width, petal length, and petal width (centimeters).

Tasks 1 and 2 use a stratified **80/20 train/test split**, resulting in 120 training and 30 test examples. Task 3 uses a separate stratified **60/20/20 train/validation/test split** (90/30/30). Feature scaling is fitted on training data only when required.

### MNIST

MNIST comprises **70,000 grayscale images** of handwritten digits `0`–`9`, each **28 × 28 pixels**. Images are converted to tensors with pixel values in the **[0, 1]** range.

| Split | Samples | Purpose |
|---|---:|---|
| Training | 54,000 | Fit network weights |
| Validation | 6,000 | Compare training configurations |
| Test | 10,000 | Final evaluation of selected model |

The training/validation partition uses a fixed random seed. MNIST is downloaded automatically by `torchvision` if not already cached.

---

## Task 1 — Iris classification

### Objective

Train three traditional supervised classifiers on the same stratified split and compare both their predicted classes and class-separation performance.

### Models and preprocessing

| Model | Configuration | Scaling |
|---|---|---|
| Decision Tree | `max_depth=3`, `random_state=42` | Not required |
| Logistic Regression | `max_iter=1000`, `random_state=42` | `StandardScaler` |
| K-Nearest Neighbors | `n_neighbors=5` | `StandardScaler` |

Scaling is encapsulated within scikit-learn `Pipeline`s to avoid fitting the scaler on test data.

### Metrics

- **Accuracy:** fraction of test observations classified correctly.
- **Precision (macro):** average per-class precision.
- **Recall (macro):** average per-class recall.
- **F1-score (macro):** average per-class F1.
- **ROC-AUC (macro):** mean of per-class One-vs-Rest ROC area values, calculated from predicted class probabilities.
- **Confusion matrices and classification reports:** show class-level mistakes and support.

For multiclass ROC analysis, each target class is binarized against the other two classes (**One-vs-Rest**).

### Results

| Classifier | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | AUC (macro, OvR) |
|---|---:|---:|---:|---:|---:|
| **Decision Tree** | **0.9667** | **0.9697** | **0.9667** | **0.9666** | 0.9717 |
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 | **0.9967** |
| KNN | 0.9333 | 0.9444 | 0.9333 | 0.9327 | 0.9933 |

**Interpretation:** The Decision Tree correctly classified **29/30** test samples and had the best test accuracy and macro F1. Logistic Regression recorded the best ROC-AUC despite making two classification mistakes, showing that ranking/class separability and final class predictions are not identical concepts. Setosa was classified perfectly by all three models, while confusion occurred between Versicolor and Virginica.

**Visualizations included in the notebook:** three confusion-matrix heatmaps, per-class ROC curves for each model, and a grouped metric comparison chart.

---

## Task 2 — Feature importance and hyperparameters

### Objective

Inspect what the Decision Tree learns from flower measurements and test how complexity settings influence KNN and Decision Tree accuracy.

### Feature importance

A Decision Tree with `max_depth=3` provides impurity-based `feature_importances_`:

| Input feature | Importance |
|---|---:|
| **Petal length** | **0.5791 (57.91%)** |
| **Petal width** | **0.4209 (42.09%)** |
| Sepal width | 0.0000 |
| Sepal length | 0.0000 |

**Finding:** The fitted tree relied entirely on **petal measurements**. A zero importance value means that this *particular fitted tree* did not use a feature to split its nodes; it does not imply the feature has no predictive information in general.

### Confusion matrix

The chosen tree reached **96.67%** test accuracy, making only **one mistake**: a Versicolor example was predicted as Virginica. All Setosa and all Virginica test examples were correctly classified.

### KNN neighbors experiment

The notebook tests `n_neighbors` from **1 to 20**, training a standardized KNN model for each value and displaying both train and test accuracy.

- **Best observed test accuracy:** 0.9667.
- **First maximizing setting:** `k=1` (ties are possible).
- At `k=1`, the model also perfectly fits the training samples; this is consistent with the higher variance/overfitting risk of very small neighborhoods.
- Several larger `k` values achieved comparable test accuracy, indicating that the result is not unique to `k=1`.

### Decision Tree depth experiment

The notebook tests `max_depth` from **1 to 10**, recording training and test accuracy.

- **Best observed test accuracy:** 0.9667.
- **Best observed depth:** `max_depth=3`.
- The deepest trees achieve very high or perfect training accuracy, without improving test accuracy; this is consistent with overfitting.

**Visualizations included:** feature-importance horizontal bar chart, confusion-matrix heatmap, KNN accuracy versus `k`, and Decision Tree train/test accuracy versus depth.

> **Evaluation caveat:** Task 2 uses the test set to inspect and select hyperparameters, as prescribed in the assignment experiment. In a production workflow, use cross-validation or a separate validation set for hyperparameter selection and reserve the untouched test set for final evaluation.

---

## Task 3 — PyTorch perceptron: activation functions

### Objective

Implement a fully connected classifier with **one hidden layer** and investigate how activation function choice affects training dynamics and accuracy on Iris.

### Architecture

```text
Input (4 Iris features)
        │
    Linear(4, 32)
        │
ReLU / Sigmoid / Tanh   ← only changed component
        │
    Linear(32, 3)
        │
   Class logits
```

The network outputs **logits** and uses `CrossEntropyLoss`, which handles the softmax operation internally.

### Training protocol

| Setting | Value |
|---|---|
| Train / validation / test | 60% / 20% / 20% |
| Hidden units | 32 |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Batch size | 16 |
| Maximum epochs | 200 |
| Weight decay | 0.0001 |
| Early stopping patience | 25 epochs |
| Seed | 42 |
| Model selection | Lowest validation loss |

All three runs start with the same weight initialization and training configuration. Early stopping restores the best weights according to validation loss. Train/validation loss and accuracy are recorded after each epoch.

### Results

| Activation | Test accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Stopped after |
|---|---:|---:|---:|---:|---:|
| ReLU | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 58 epochs |
| Sigmoid | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 57 epochs |
| Tanh | 0.9333 | 0.9333 | 0.9333 | 0.9333 | **43 epochs** |

**Interpretation:** All activations reached **28/30 correct classifications**, and their confusion matrices were identical. **Tanh** showed rapid early loss reduction and reached its early-stopping condition after fewer epochs in this run. The program prints “Best activation: ReLU” because ReLU appears first in a tie on test Accuracy and F1; **there is no unique test-metric winner**. The validation-loss curves flatten or rise while training loss keeps declining, suggesting that continued training can overfit this small dataset.

**Visualizations included:** train/validation loss curves, train/validation accuracy curves, activation accuracy bar chart, and three confusion matrices.

The notebook also exports `task3_metrics.csv` when the corresponding cell runs.

---

## Task 4 — MNIST MLP

### Objective

Classify handwritten digits using a multilayer perceptron, then study how **number of epochs** and **batch size** influence generalization and training time.

### Architecture

```text
28 × 28 grayscale image
          │
      Flatten
          │
    Linear(784, 256) → ReLU
          │
    Linear(256, 128) → ReLU
          │
    Linear(128, 64)  → ReLU
          │
    Linear(64, 10)
          │
     Class logits
```

- **Loss:** `CrossEntropyLoss` (softmax is handled internally).
- **Optimizer:** Adam (`lr=0.001`, `weight_decay=0.0001`).
- **Varied epochs:** 5, 10, 20.
- **Varied batch sizes:** 32, 64, 128.
- **Total configurations:** 3 × 3 = **9**.
- **Fair comparison:** all runs start from the same initial weights and use the same data split.

### Results: all nine experiments

Numbers below are from the output saved in [`ProgForAI4.ipynb`](./ProgForAI4.ipynb). Times were recorded during a **CPU** run and will differ by machine.

| Epochs | Batch | Train acc. | Validation acc. | Train loss | Validation loss | Time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 32 | 0.9838 | 0.9710 | 0.0509 | 0.1055 | 134.1 |
| 10 | 32 | 0.9898 | **0.9772** | 0.0306 | 0.0873 | 230.8 |
| 20 | 32 | 0.9927 | 0.9718 | 0.0218 | 0.1063 | 517.3 |
| 5 | 64 | 0.9834 | 0.9745 | 0.0518 | 0.0914 | 75.5 |
| 10 | 64 | 0.9917 | 0.9770 | 0.0259 | **0.0806** | 162.8 |
| 20 | 64 | 0.9953 | **0.9772** | 0.0137 | 0.0866 | 358.8 |
| 5 | 128 | 0.9808 | 0.9707 | 0.0634 | 0.0990 | 51.1 |
| 10 | 128 | 0.9925 | 0.9687 | 0.0242 | 0.1093 | 118.9 |
| 20 | 128 | 0.9954 | 0.9735 | 0.0138 | 0.1076 | 247.6 |

### Selected model and test performance

The saved notebook's selection code sorts by **validation accuracy** and then **validation loss**. It selected:

| Item | Recorded result |
|---|---:|
| **Epochs** | **20** |
| **Batch size** | **64** |
| Validation accuracy | **0.9772 (97.72%)** |
| Test accuracy | **0.9781 (97.81%)** |
| Test loss | **0.0837** |

**Important nuance:** `10 epochs / batch 32` also displays **0.9772** validation accuracy after rounding. The code selected `20 / 64` using unrounded numerical values. The `10 / 64` run had the **lowest validation loss (0.0806)** and trained considerably faster than `20 / 64`, making it a sensible efficiency-oriented alternative. The highest displayed accuracy and the lowest loss do not necessarily occur in the same run.

### Findings

1. **Epochs help at first:** most configurations gain accuracy during the first several epochs.
2. **More epochs do not guarantee better validation accuracy:** for batch 32, validation accuracy falls from **97.72% at 10 epochs** to **97.18% at 20 epochs**, even as training accuracy rises. This is a signal of possible overfitting.
3. **Batch size affects the trade-off:** larger batches generally complete epochs with fewer optimizer updates and showed lower total runtime in these CPU measurements, but did not consistently achieve the best validation scores.
4. **Validation data matters:** model configuration is selected on validation results; the held-out MNIST test set is used for the selected model's final score.
5. **Typical errors are visually plausible:** some handwritten digits are confused because of similar shapes, such as `4 → 9`, `6 → 0`, or `5 → 3` in the displayed examples.

**Visualizations included:** nine validation learning curves, final validation Accuracy and Loss comparisons by batch size, and five correct plus five incorrect digit predictions with ground-truth and predicted labels.

The notebook exports `task4_results.csv` when the final cell runs.

---

## How to run

### Option A — Google Colab (recommended)

1. Open [`ProgForAI4.ipynb`](./ProgForAI4.ipynb) in GitHub.
2. Click **Open in Colab** if available, or upload the notebook to [Google Colab](https://colab.research.google.com/).
3. Select **Runtime → Run all** to execute the notebook in sequence.
4. Optionally select **Runtime → Change runtime type → T4 GPU** before running Task 4; the notebook also supports CPU.
5. Examine the inline metric tables, charts, and prediction images.

**Note:** MNIST downloads on the first run. Task 4 trains nine models, so running the entire notebook may be computationally intensive, especially on CPU.

### Option B — Local Jupyter

Make sure you have Python 3 and a working PyTorch installation, then install the notebook dependencies:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision jupyter
jupyter notebook ProgForAI4.ipynb
```

PyTorch installation options vary by operating system and GPU/CUDA version. Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) if needed.

### Expected outputs

The notebook itself contains the printed reports and inline visualizations. When run, it also writes the following files to the **current runtime working directory**:

```text
task3_metrics.csv
task4_results.csv
```

These generated CSVs are **not assumed to be committed** to this repository. In Colab, download them explicitly if you want to add them to GitHub. The MNIST dataset is cached under `./data/` and does not need to be committed.

## Reproducibility and evaluation notes

- **Random seeds:** `42` is used throughout for Python/NumPy/PyTorch where applicable.
- **Stratification:** used for the Iris splits to preserve class proportions.
- **Training-only scaling:** `StandardScaler` is fitted on the training data, avoiding test-set preprocessing leakage.
- **Consistent weight initialization:** Task 3 and Task 4 use the same initial weights across compared configurations.
- **Metric differences:** Task 1 uses macro-averaged multiclass metrics and macro OvR AUC; Task 4 reports accuracy/loss rather than per-class F1.
- **Stochastic variation:** actual scores and timings may differ slightly across runtime types, platforms, and library versions, even with fixed seeds.
- **Task 2 limitation:** comparing many hyperparameters on the test set can bias reported performance; a dedicated validation set or cross-validation is preferable in applied research.
- **Task 3 tie:** the printed “best” activation is a tie-breaking artifact, not evidence that ReLU outperformed Sigmoid or Tanh on test accuracy.
- **Task 4 selection:** saved outputs show the selected final configuration as **20 epochs / batch 64**. Statements based on rounded tables should account for near-ties.
- **Scope:** This is an educational assignment, not a production deployment or a statistically robust benchmark across multiple random seeds.

## Lessons learned

- Traditional classifiers can perform strongly on small, well-structured tabular datasets.
- Good **Accuracy** does not automatically imply the best **ROC-AUC**.
- Feature importance depends on the fitted model and should be interpreted carefully.
- Hyperparameter tuning balances model complexity and the risk of overfitting.
- Activation functions can influence learning dynamics even when final test accuracy is identical.
- More training epochs can improve training metrics while degrading validation performance.
- Batch size affects optimization, time, and generalization, making fair controlled experiments important.
- Visual diagnostics—including confusion matrices, ROC plots, learning curves, and example predictions—make raw metric tables easier to interpret.

## Repository structure

The essential repository can be as simple as:

```text
Machine-Learning-and-Neural-Networks/
├── README.md
└── ProgForAI4.ipynb
```

Optionally, add the two exported metrics CSVs after downloading them from the notebook runtime:

```text
Machine-Learning-and-Neural-Networks/
├── README.md
├── ProgForAI4.ipynb
└── results/
    ├── task3_metrics.csv
    └── task4_results.csv
```

## References

- [scikit-learn — Iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [scikit-learn — Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [PyTorch — Documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision — MNIST dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [Google Colab](https://colab.research.google.com/)

---

**Project type:** Academic coursework · Python · Machine Learning · Neural Networks  
**Status:** All four tasks completed and evaluated in the submitted notebook.
