# comparison_analysis/comparison_analysis_report.md

# Parzen Window Classifier: Train-Test Split Comparison Analysis

**CS 7063 – Advanced Machine Learning Spring 2026**  
**Assignment #1: Parzen Classifiers – Distances in ML**  
**Author:** Ryan Kamp  
**Due Date:** January 27, 2026  
**Generated:** November 12, 2026

---

## Executive Summary

This report analyzes the effect of different train-test splits on the performance of Parzen window classifiers using various kernel functions on the Iris dataset.

**Key Findings:**
- **Best performing kernel overall:** Gaussian (avg test accuracy on Problems 2, 3, 4: 0.9682)
- **Split ratio:** average test accuracy for 50/50 to 90/10 is 0.8354, 0.8459, 0.8239, 0.8423, 0.8615 and does not change steadily with training share; each ratio is one split from the same shuffle (random state 42), so these averages do not identify a best ratio.
- **Total experiments conducted:** 65

---

## Experimental Setup

### Dataset
- **Dataset:** Iris Dataset (150 samples, 4 features, 3 classes)
- **Features:** Sepal length, Sepal width, Petal length, Petal width
- **Classes:** Setosa (0), Versicolor (1), Virginica (2)

### Train-Test Splits Evaluated

| Split | Train % | Test % | Train Samples | Test Samples |
|-------|---------|--------|---------------|-------------|
| 90/10 | 90% | 10% | 135 | 15 |
| 80/20 | 80% | 20% | 120 | 30 |
| 70/30 | 70% | 30% | 105 | 45 |
| 60/40 | 60% | 40% | 90 | 60 |
| 50/50 | 50% | 50% | 75 | 75 |

**Random State:** 42

### Kernel Functions

1. **Dot Product Kernel:** $K(x, y) = x \cdot y$
2. **Polynomial Kernel (d=2):** $K(x, y) = (x \cdot y + 1)^2$
3. **Polynomial Kernel (d=3):** $K(x, y) = (x \cdot y + 1)^3$
4. **Gaussian Kernel:** $K(x, y) = \exp(-\|x - y\|^2 / (2\sigma^2))$, where $\sigma = 1/\sqrt{2}$

### Classification Problems

- **Problem 1:** Setosa vs. (Versicolor + Virginica)
- **Problem 2:** Virginica vs. (Setosa + Versicolor)
- **Problem 3:** Versicolor vs. (Setosa + Virginica)
- **Problem 4:** Versicolor vs. Virginica

---

## Results Summary

### Average Accuracy by Train-Test Split

| Split | Train Mean | Train Std | Test Mean | Test Std |
|-------|------------|-----------|-----------|----------|
| 50/50 | 0.8079 | 0.1495 | 0.8354 | 0.1439 |
| 60/40 | 0.8138 | 0.1419 | 0.8459 | 0.1583 |
| 70/30 | 0.8283 | 0.1323 | 0.8239 | 0.1717 |
| 80/20 | 0.8244 | 0.1336 | 0.8423 | 0.1620 |
| 90/10 | 0.8292 | 0.1329 | 0.8615 | 0.1159 |

### Average Accuracy by Kernel

Averaged over Problems 2, 3, 4, the problems every kernel ran on.

| Kernel | Train Mean | Train Std | Test Mean | Test Std |
|--------|------------|-----------|-----------|----------|
| Dot Product | 0.7685 | 0.1191 | 0.7952 | 0.1115 |
| Gaussian | 0.9350 | 0.0143 | 0.9682 | 0.0299 |
| Polynomial (d=2) | 0.7658 | 0.1311 | 0.7768 | 0.1198 |
| Polynomial (d=3) | 0.7577 | 0.1271 | 0.7743 | 0.1837 |

---

## Detailed Analysis

### Train-Test Splits (one split per ratio, all from one shuffle)

Each ratio is evaluated on a single random split, and all five use random state 42, so they come from one shuffle: each smaller test set lies inside the larger ones. Differences between ratios mix training-set size with which samples happen to be in the test set, and each is one draw, so this section describes the ratios without ranking them.

1. **Average test accuracy by training share** (50/50 to 90/10): 0.8354, 0.8459, 0.8239, 0.8423, 0.8615; it does not change steadily with training share.

2. **Spread within a split:** the standard deviation of test accuracy across problems and kernels ranges from 0.1159 (90/10) to 0.1717 (70/30). It measures spread across configurations, not variation between runs.

3. **Training and test accuracy:** mean training accuracy is 0.8079 for 50/50 and 0.8292 for 90/10; mean test accuracy is 0.8354 and 0.8615.

### Kernel Performance Comparison

1. **Best Kernel:** The Gaussian kernel achieved the highest average test accuracy (0.9682) on Problems 2, 3, 4 across all splits.

2. **Kernel Ranking by Test Accuracy:**
   1. Gaussian: 0.9682
   2. Dot Product: 0.7952
   3. Polynomial (d=2): 0.7768
   4. Polynomial (d=3): 0.7743

3. **By problem (mean test accuracy over all splits):**
   - Problem 1: Dot Product only (1.0000).
   - Problem 2: highest Gaussian (0.9513), lowest Dot Product (0.7876).
   - Problem 3: highest Gaussian (0.9662), lowest Polynomial (d=3) (0.5553).
   - Problem 4: highest Gaussian (0.9870), lowest Polynomial (d=3) (0.8586).

### Problem-Specific Analysis

#### Problem 1: Setosa vs. (Versicolor + Virginica)

| Kernel | 50/50 | 60/40 | 70/30 | 80/20 | 90/10 |
|--------|-------|-------|-------|-------|-------|
| Dot Product | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

#### Problem 2: Virginica vs. (Setosa + Versicolor)

| Kernel | 50/50 | 60/40 | 70/30 | 80/20 | 90/10 |
|--------|-------|-------|-------|-------|-------|
| Dot Product | 0.8267 | 0.8000 | 0.7778 | 0.8000 | 0.7333 |
| Gaussian | 0.9733 | 0.9833 | 0.9333 | 0.9333 | 0.9333 |
| Polynomial (d=2) | 0.8933 | 0.8333 | 0.8444 | 0.8333 | 0.7333 |
| Polynomial (d=3) | 0.9067 | 0.8833 | 0.8889 | 0.9333 | 0.9333 |

#### Problem 3: Versicolor vs. (Setosa + Virginica)

| Kernel | 50/50 | 60/40 | 70/30 | 80/20 | 90/10 |
|--------|-------|-------|-------|-------|-------|
| Dot Product | 0.6800 | 0.6667 | 0.6889 | 0.6000 | 0.7333 |
| Gaussian | 0.9200 | 0.9667 | 0.9778 | 0.9667 | 1.0000 |
| Polynomial (d=2) | 0.6667 | 0.6500 | 0.6222 | 0.5333 | 0.6667 |
| Polynomial (d=3) | 0.4933 | 0.4833 | 0.4000 | 0.6000 | 0.8000 |

#### Problem 4: Versicolor vs. Virginica

| Kernel | 50/50 | 60/40 | 70/30 | 80/20 | 90/10 |
|--------|-------|-------|-------|-------|-------|
| Dot Product | 0.9130 | 0.9459 | 0.9231 | 0.9500 | 0.8889 |
| Gaussian | 0.9348 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Polynomial (d=2) | 0.8478 | 0.8919 | 0.8462 | 0.9000 | 0.8889 |
| Polynomial (d=3) | 0.8043 | 0.8919 | 0.8077 | 0.9000 | 0.8889 |

---

## Visualizations

The following plots are available in this folder:

1. **comparison_test_accuracy_by_problem.png** - Bar charts showing test accuracy across all splits for each problem
2. **comparison_train_accuracy_by_problem.png** - Bar charts showing train accuracy across all splits for each problem
3. **comparison_test_accuracy_trends.png** - Line plots showing how test accuracy changes with different splits
4. **comparison_heatmap.png** - Heatmap visualization of test accuracy (Kernel × Split) for each problem
5. **comparison_average_accuracy.png** - Average accuracy over Problems 2, 3, 4 for each kernel and split

---

## Conclusions

1. **Kernel:** Gaussian has the highest average test accuracy on Problems 2, 3, 4 (0.9682); the kernel averages span 0.1939.

2. **Split:** average test accuracy differs by 0.0376 across the ratios and does not change steadily with training share. With one split per ratio, all from the same shuffle, this comparison does not identify a best ratio; repeated_splits.py evaluates each ratio over 500 random splits.

---

*Report generated automatically by the Parzen Window Classifier analysis script, kamp_hw1.py.*
