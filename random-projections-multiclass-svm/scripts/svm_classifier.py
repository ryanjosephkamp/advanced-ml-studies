#!/usr/bin/env python3
"""
svm_classifier.py - Multi-Class SVM Classification (Part II)

This module implements multi-class SVM classification on projected (k=2) data using
four kernels: linear, polynomial degree 2, polynomial degree 3, and
Gaussian (RBF). 

Scikit-learn's SVC implementation is used for training and evaluation, and the module 
computes various performance metrics, including confusion matrices, accuracy, precision, 
recall, and F1 scores for each class as well as macro-averaged metrics.

The results are saved in structured formats (CSV for metrics and PNG for plots) for later analysis.

Reference:
    Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
    Journal of Machine Learning Research, 12, 2825-2830.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 3 – Random Projections and Multiclass Classification
Date: February 13, 2026
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)
from tabulate import tabulate

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


# =============================================================================
# KERNEL CONFIGURATIONS
# =============================================================================

# The linear kernel is the simplest SVM kernel, which finds a linear hyperplane to separate classes in the feature space. 
# It is effective when the data is linearly separable or when the number of features is large compared to the number of samples.

# The polynomial kernel of degree 2 allows the SVM to find quadratic decision boundaries, capturing more complex relationships between features.

# The polynomial kernel of degree 3 allows the SVM to find cubic decision boundaries, capturing even more complex relationships between features.

# The Gaussian (RBF) kernel maps the data into an infinite-dimensional space, allowing the SVM to find highly flexible decision boundaries. 
# It is effective for non-linear classification problems.

KERNEL_CONFIGS = [
    {"name": "Linear",              "kernel": "linear"}, # Linear kernel configuration for SVM (finds linear decision boundaries)
    {"name": "Polynomial (d=2)",    "kernel": "poly", "degree": 2}, # Polynomial kernel configuration for SVM with degree 2 (finds quadratic decision boundaries)
    {"name": "Polynomial (d=3)",    "kernel": "poly", "degree": 3}, # Polynomial kernel configuration for SVM with degree 3 (finds cubic decision boundaries)
    {"name": "Gaussian (RBF)",      "kernel": "rbf"}, # Gaussian (RBF) kernel configuration for SVM (finds highly flexible, non-linear decision boundaries)
]


# =============================================================================
# TIES
# =============================================================================

def top_kernels(names, scores, tol: float = 1e-12) -> list:
    """Return every kernel name whose score equals the maximum (within tol), in input order."""
    names = list(names)
    scores = [float(s) for s in scores]
    best = max(scores)
    return [n for n, s in zip(names, scores) if s >= best - tol]


def describe_top_kernels(names, scores) -> str:
    """Name the best kernel, or all of them with "(tie)" when several share the best score."""
    top = top_kernels(names, scores)
    if len(top) == 1:
        return top[0]
    if len(top) == 2:
        return f"{top[0]} and {top[1]} (tie)"
    return ", ".join(top[:-1]) + f", and {top[-1]} (tie)"


# =============================================================================
# SINGLE SVM TRAIN + EVALUATE
# =============================================================================

def train_and_evaluate_svm(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           kernel_config: dict, class_names: list) -> dict:
    """
    Train an SVM with the given kernel and evaluate on train and test sets.

    This function also computes various performance metrics, including confusion matrices, 
    accuracy, precision, recall, and F1 scores for each class as well as macro-averaged metrics. 
    The results are returned in a structured dictionary format for further analysis and reporting.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Testing data.
        kernel_config: Dict with 'name', 'kernel', and optionally 'degree'.
        class_names: List of class label strings.

    Returns:
        Dictionary with all metrics for train and test sets.
    """
    class_names = [str(c) for c in class_names]
    kernel_name = kernel_config["name"]

    # Build SVC kwargs based on kernel configuration; include degree parameter for polynomial kernels
    svc_kwargs = {"kernel": kernel_config["kernel"], "random_state": 42}
    if "degree" in kernel_config:
        svc_kwargs["degree"] = kernel_config["degree"]

    # Train
    model = SVC(**svc_kwargs) # Initialize the SVM model with the specified kernel and parameters
    model.fit(X_train, y_train) # Fit the SVM model to the training data (X_train, y_train), learning the decision boundaries that separate the classes in the projected feature space

    # Predict
    y_train_pred = model.predict(X_train) # Use the trained SVM model to predict class labels for the training data, which will be used to evaluate how well the model has fit the training data and to compute training metrics
    y_test_pred = model.predict(X_test) # Use the trained SVM model to predict class labels for the test data, which will be used to evaluate the model's generalization performance on unseen data and to compute test metrics

    # Compute metrics for both sets
    results = {"kernel_name": kernel_name, "model": model}
    labels = list(range(len(class_names)))

    for split_name, y_true, y_pred in [
        ("train", y_train, y_train_pred), # Compare true train labels (y_train) with predicted train labels (y_train_pred) to assess how well the model has learned from the training data
        ("test", y_test, y_test_pred),   # Compare true test labels (y_test) with predicted test labels (y_test_pred) to evaluate the model's generalization performance on unseen data
    ]:
        # Compute the confusion matrix for the given true labels (y_true) and predicted labels (y_pred) using the specified class label indices (labels):
        # This will show the counts of true positives, false positives, true negatives, and false negatives for each class, providing insight into the model's 
        # performance in terms of correctly and incorrectly classified samples for each class.
        cm = confusion_matrix(y_true, y_pred, labels=labels) 
        # Compute the accuracy score:
        # This is the ratio of correctly predicted samples to the total number of samples, providing a simple measure of overall model performance.
        acc = accuracy_score(y_true, y_pred) 
        # Compute the precision for each class:
        # This is the ratio of true positives to the sum of true positives and false positives, indicating how many of the predicted positive samples are actually positive.
        prec = precision_score(y_true, y_pred, labels=labels, average=None,
                               zero_division=0) 
        # Compute the recall for each class:
        # This is the ratio of true positives to the sum of true positives and false negatives, indicating how many of the actual positive samples are correctly identified.
        rec = recall_score(y_true, y_pred, labels=labels, average=None,
                           zero_division=0) 
        # Compute the F1 score for each class:
        # This is the harmonic mean (2 * (precision * recall) / (precision + recall)) of precision and recall, providing a single metric that balances both aspects of model performance.
        f1 = f1_score(y_true, y_pred, labels=labels, average=None,
                      zero_division=0)

        # Compute macro averages:
        # These are the unweighted mean of the metrics across all classes, providing a single value that summarizes the overall performance without considering class imbalance.
        
        # Compute the macro-averaged precision:
        # This is the average of the precision scores for all classes, treating each class equally regardless of its frequency in the dataset.
        prec_macro = precision_score(y_true, y_pred, labels=labels,
                                     average="macro", zero_division=0) 
        # Compute the macro-averaged recall:
        # This is the average of the recall scores for all classes, treating each class equally regardless of its frequency in the dataset.
        rec_macro = recall_score(y_true, y_pred, labels=labels,
                                 average="macro", zero_division=0)
        # Compute the macro-averaged F1 score:
        # This is the average of the F1 scores for all classes, treating each class equally regardless of its frequency in the dataset.
        f1_macro = f1_score(y_true, y_pred, labels=labels,
                            average="macro", zero_division=0)
        
        # Store all computed metrics in the results dictionary under the corresponding split name (train/test) for later analysis and reporting:
        results[split_name] = {
            "confusion_matrix": cm, # Confusion matrix for the current split, showing the counts of true positives, false positives, true negatives, and false negatives for each class
            "accuracy": acc, # Accuracy score for the current split, representing the ratio of correctly predicted samples to the total number of samples
            "precision_per_class": prec, # Precision scores for each class, indicating how many of the predicted positive samples are actually positive
            "recall_per_class": rec, # Recall scores for each class, indicating how many of the actual positive samples are correctly identified
            "f1_per_class": f1, # F1 scores for each class, providing a single metric that balances precision and recall
            "precision_macro": prec_macro, # Macro-averaged precision, treating each class equally regardless of its frequency in the dataset
            "recall_macro": rec_macro, # Macro-averaged recall, treating each class equally regardless of its frequency in the dataset
            "f1_macro": f1_macro, # Macro-averaged F1 score, treating each class equally regardless of its frequency in the dataset
            "y_true": y_true, # The true class labels for the current split, used for further analysis and reporting
            "y_pred": y_pred, # The predicted class labels for the current split, used for further analysis and reporting
            "classification_report": classification_report( 
                y_true, y_pred, target_names=class_names, zero_division=0
            ), # Detailed classification report for the current split, including precision, recall, F1 score, and support for each class
        }

    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str,
                          save_path: str) -> None:
    """
    Plot a confusion matrix heatmap.
    
    This confusion matrix heatmap visualizes the performance of the classification model by showing
    the counts of true positives, false positives, true negatives, and false negatives for each class.
    The x-axis represents the predicted class labels, while the y-axis represents the actual class labels.
    The color intensity of each cell corresponds to the count of samples in that category, allowing for
    easy identification of which classes are being correctly classified and which are being misclassified by the model.
    The class names are used as labels for the axes, and the title provides context for the confusion matrix being displayed.
    
    The resulting plot is saved to the specified path for later reference and analysis.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2),
                                    max(5, len(class_names))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray,
                           class_names: list, title: str,
                           save_path: str) -> None:
    """
    Plot SVM decision boundaries for 2-D data.
    Creates a colored mesh showing predicted regions and overlays data points.
    
    This function visualizes the decision boundaries of a trained SVM model in a 2D feature space.
    It creates a mesh grid over the range of the data, uses the model to predict the class label for each point in the grid,
    and then colors the regions according to the predicted class. The data points are then overlaid on the decision regions, 
    with different colors representing different classes.
    This visualization helps in understanding how the SVM model separates different classes in the feature space, 
    showing the shape of the decision boundaries and how well they align with the distribution of the data points.

    The resulting plot is saved to the specified path for later reference and analysis.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("Set2", len(class_names))

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    h = (x_max - x_min) / 200
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap([(*c, 0.25) for c in colors[:len(class_names)]])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light, levels=np.arange(
        len(class_names) + 1) - 0.5)

    # Plot data points
    for ci, cn in enumerate(class_names):
        mask = y == ci
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[ci]], label=cn,
                   edgecolors="k", linewidth=0.3, s=40, alpha=0.8)

    ax.set_xlabel("Projected Dimension 1", fontsize=12)
    ax.set_ylabel("Projected Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =============================================================================
# METRICS SAVING
# =============================================================================

def save_metrics_csv(results: dict, class_names: list, save_path: str,
                     split_name: str) -> None:
    """Save per-class and overall metrics to a CSV file."""
    split = results[split_name]
    rows = []
    # For each class, we create a row in the CSV with the class name and its corresponding precision, recall, and F1 score.
    for ci, cn in enumerate(class_names):
        rows.append({
            "class": cn, # Name of the class for which the metrics are being recorded, used for identification in the CSV file
            "precision": split["precision_per_class"][ci], # Precision score for the current class, indicating how many of the predicted positive samples for this class are actually positive
            "recall": split["recall_per_class"][ci], # Recall score for the current class, indicating how many of the actual positive samples for this class are correctly identified
            "f1_score": split["f1_per_class"][ci], # F1 score for the current class, harmonic mean of precision and recall
        })
    rows.append({
        "class": "MACRO_AVG", # Macro-average metrics across all classes, providing a single value that summarizes the overall performance without considering class imbalance
        "precision": split["precision_macro"], # Macro-averaged precision, treating each class equally regardless of its frequency in the dataset
        "recall": split["recall_macro"], # Macro-averaged recall, treating each class equally regardless of its frequency in the dataset
        "f1_score": split["f1_macro"], # Macro-averaged F1 score, treating each class equally regardless of its frequency in the dataset
    })
    rows.append({
        "class": "OVERALL", # Overall metrics, typically representing the accuracy across all classes
        "precision": split["accuracy"], # Overall accuracy, representing the ratio of correctly predicted samples to the total number of samples, used here as a general performance metric for the overall classification task
        "recall": split["accuracy"], # Overall accuracy is used here as a proxy for recall in the overall row, since recall is not typically defined for the overall case; this is a simplification for reporting purposes
        "f1_score": split["accuracy"], # Overall accuracy is used here as a proxy for F1 score in the overall row, since F1 score is not typically defined for the overall case; this is a simplification for reporting purposes
    })
    df = pd.DataFrame(rows)
    df["accuracy"] = split["accuracy"] # Overall accuracy, representing the ratio of correctly predicted samples to the total number of samples
    df["kernel"] = results["kernel_name"] # Name of the kernel used for this set of results, used for identification in the CSV file
    df.to_csv(save_path, index=False) # Save the metrics DataFrame to a CSV file at the specified path, allowing for later analysis and reporting of the performance metrics for each class and overall


# =============================================================================
# MAIN ORCHESTRATOR FOR ONE DATASET + ONE PROJECTION
# =============================================================================

def run_svm_classification(E: np.ndarray, y: np.ndarray, class_names: list,
                           dataset_key: str, dataset_name: str,
                           projection_label: str, output_dir: str,
                           test_size: float = 0.3, seed: int = 42) -> dict:
    """
    Run SVM classification with all 4 kernels on projected data.

    Args:
        E: Projected data matrix (n, k).
        y: Labels (n,).
        class_names: Class name strings.
        dataset_key: e.g., 'iris'.
        dataset_name: e.g., 'Iris'.
        projection_label: e.g., 'eq3' or 'eq4'.
        output_dir: Directory to save results.
        test_size: Fraction of data for test set.
        seed: Random seed.

    Returns:
        Dictionary of all 4 kernel results + train/test split info.
    """
    class_names = [str(c) for c in class_names]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  SVM CLASSIFICATION — {dataset_name} ({projection_label.upper()})")
    print(f"  Samples: {len(y)}, Dimensions: {E.shape[1]}, Test size: {test_size}")
    print(f"{'─' * 60}")

    # === Single train/test split for ALL kernels ===
    # We perform a single train/test split of the projected data (E) and labels (y) 
    # to ensure that all kernel configurations are evaluated on the same training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        E, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"\n  Train set: {len(y_train)} samples")
    print(f"  Test set:  {len(y_test)} samples")

    all_kernel_results = {}

    # For each kernel configuration defined in KERNEL_CONFIGS, we train an SVM model using the specified kernel 
    # and evaluate its performance on both the training and testing sets.
    for kc in KERNEL_CONFIGS:
        kernel_name = kc["name"]
        kernel_tag = kc["kernel"]
        if "degree" in kc:
            kernel_tag += str(kc["degree"])

        print(f"\n  ── Training: {kernel_name} ──")
        results = train_and_evaluate_svm(
            X_train, y_train, X_test, y_test, kc, class_names
        )

        # Print test results
        test_r = results["test"]
        print(f"    Test Accuracy: {test_r['accuracy']:.4f}")
        print(f"    Test Precision (macro): {test_r['precision_macro']:.4f}")
        print(f"    Test Recall (macro): {test_r['recall_macro']:.4f}")
        print(f"\n    Confusion Matrix (Test):")
        cm_table = []
        for i, cn in enumerate(class_names):
            row = [cn] + [str(test_r["confusion_matrix"][i][j])
                          for j in range(len(class_names))]
            cm_table.append(row)
        print(tabulate(cm_table, headers=[""] + class_names, tablefmt="grid"))

        # Save metrics CSVs
        prefix = f"{dataset_key}_{projection_label}_{kernel_tag}"
        save_metrics_csv(results, class_names,
                         os.path.join(output_dir, f"{prefix}_train_metrics.csv"),
                         "train")
        save_metrics_csv(results, class_names,
                         os.path.join(output_dir, f"{prefix}_test_metrics.csv"),
                         "test")

        # Save confusion matrix plot
        plot_confusion_matrix(
            test_r["confusion_matrix"], class_names,
            f"{dataset_name} ({projection_label.upper()}) — {kernel_name}\nTest Confusion Matrix",
            os.path.join(output_dir, f"{prefix}_confusion_matrix.png"),
        )

        # Save decision boundary plot (only for 2-D data)
        if E.shape[1] == 2:
            plot_decision_boundary(
                results["model"], X_test, y_test, class_names,
                f"{dataset_name} ({projection_label.upper()}) — {kernel_name}\nDecision Boundary (Test Set)",
                os.path.join(output_dir, f"{prefix}_decision_boundary.png"),
            )

        all_kernel_results[kernel_tag] = results

    print(f"\n  ✓ SVM classification complete for {dataset_name} ({projection_label.upper()})")

    # Return a dictionary containing the results for all kernels, as well as the train/test split information and dataset metadata, 
    # which can be used for further analysis and reporting in the within-dataset analysis step.
    return {
        "kernel_results": all_kernel_results, # Dictionary containing the results for each kernel configuration, including performance metrics and confusion matrices for both train and test sets
        "X_train": X_train, # Training data matrix (n_train, k) used for training the SVM models, where n_train is the number of training samples and k is the number of projected dimensions (2 in this case)
        "X_test": X_test, # Testing data matrix (n_test, k) used for evaluating the SVM models, where n_test is the number of testing samples and k is the number of projected dimensions (2 in this case)
        "y_train": y_train, # Training labels (n_train,) corresponding to the training data, used for training the SVM models and computing training metrics
        "y_test": y_test, # Testing labels (n_test,) corresponding to the testing data, used for evaluating the SVM models and computing test metrics
        "class_names": class_names, # List of class name strings corresponding to the class labels, used for labeling plots and reports
        "dataset_key": dataset_key, # Short key for the dataset (e.g., 'iris'), used for naming files and organizing results
        "dataset_name": dataset_name, # Full name of the dataset (e.g., 'Iris'), used for labeling plots and reports
        "projection_label": projection_label, # Label for the projection method used (e.g., 'eq3' or 'eq4'), used for labeling plots and reports
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    from data_loader import load_dataset
    from random_projection import run_random_projection

    HW3_DIR = SCRIPT_DIR.parent # Assuming this script is in HW3/scripts/, we go up one level to HW3/ to access the data_loader and results directory
    
    # Load the Iris dataset using the data_loader module: 
    # This provides the original data matrix (X), labels (y), class names, and other metadata about the dataset. 
    # The loaded dataset will be used as input for the random projection step, which will then be followed by SVM classification on the projected data.
    ds = load_dataset("iris") 
    
    # Run random projection on the loaded dataset: 
    # This will project the original data into a lower-dimensional space (k=2) while preserving distances according to the Johnson-Lindenstrauss lemma. 
    # The results include the projected data matrix (E), labels (y), class names, and other relevant information needed for SVM classification.
    proj = run_random_projection(ds, str(HW3_DIR / "results" / "test_svm")) 

    # Run SVM classification on the projected data (E_eq3) using the labels (y) and class names from the dataset, 
    # specifying the dataset key, name, projection label, and output directory for results:
    # This will train and evaluate SVM models with all 4 kernel configurations on the projected data, compute performance metrics, generate plots, 
    # and save the results in structured formats for later analysis and reporting in the within-dataset analysis step.
    svm_results = run_svm_classification(
        proj["E_eq3"], proj["y"], proj["class_names"],
        "iris", "Iris", "eq3",
        str(HW3_DIR / "results" / "test_svm"),
    )
