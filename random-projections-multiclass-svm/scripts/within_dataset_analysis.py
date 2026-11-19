#!/usr/bin/env python3
"""
within_dataset_analysis.py - Within-Dataset Kernel Comparison

This module performs the "within-dataset" analysis for HW3. 

For a single dataset (Iris, Breast Cancer, Wine, Seeds, or Glass Identification)
and a single random projection type (either Eq.3 or Eq.4), 
it compares the performance of all four SVM kernels (linear, poly2, poly3, rbf) against each other.

The analysis includes generating a comparison table of performance metrics, plotting 
accuracy and per-class precision/recall, and creating a grid of confusion matrices for 
all kernels. 

Finally, it compiles all results into a comprehensive markdown report for the dataset. 

The report includes tables summarizing the performance of each kernel, key findings, and references to all generated plots. 

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 3 – Random Projections and Multiclass Classification
Date: February 13, 2026
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from author_info import get_markdown_header, get_markdown_footer
from svm_classifier import describe_top_kernels

# Project paths
HW3_DIR = SCRIPT_DIR.parent
REPORTS_DIR = HW3_DIR / "reports"


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_within_dataset_analysis(svm_results: dict, projection_results: dict,
                                dataset_dict: dict, output_dir: str) -> dict:
    """
    Compare all four kernel classifiers for one dataset.

    Args:
        svm_results: Output from svm_classifier.run_svm_classification.
        projection_results: Output from random_projection.run_random_projection.
        dataset_dict: Original dataset dictionary.
        output_dir: Directory to save analysis outputs.

    Returns:
        Analysis results dictionary.
    """
    # Extract dataset and projection info:
    ds_key = dataset_dict["dataset_key"] # e.g., "dataset1"
    ds_name = dataset_dict["dataset_name"] # e.g., "Dataset 1"
    class_names = [str(c) for c in dataset_dict["class_names"]] # e.g., ["Class 0", "Class 1", "Class 2"]
    n_classes = dataset_dict["n_classes"] # e.g., 3
    proj_label = svm_results["projection_label"] # e.g., "eq3" or "eq4"

    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    print(f"\n{'═' * 60}")
    print(f"  WITHIN-DATASET ANALYSIS — {ds_name} ({proj_label.upper()})")
    print(f"{'═' * 60}")

    kernel_results = svm_results["kernel_results"] # Dictionary with keys "linear", "poly2", "poly3", "rbf" and values containing performance metrics and confusion matrices for each kernel

    # === Build comparison table ===
    rows = []
    # Iterate over each kernel's results:
    for ktag, kr in kernel_results.items():
        # Extract the test performance metrics (accuracy, precision_macro, recall_macro, f1_macro) for the current dataset:
        test_r = kr["test"] # Test set results for this kernel
        row = {
            "kernel": kr["kernel_name"], # Kernel name, e.g., "Linear", "Poly2", "Poly3", "RBF"
            "accuracy": test_r["accuracy"], # Test accuracy, e.g., 0.85
            "precision_macro": test_r["precision_macro"], # Test precision (macro), e.g., 0.83
            "recall_macro": test_r["recall_macro"], # Test recall (macro), e.g., 0.84
            "f1_macro": test_r["f1_macro"], # Test F1 score (macro), e.g., 0.835
        }
        # Iterate over each class and add per-class precision, recall, and F1 scores to the row dictionary:
        for ci, cn in enumerate(class_names):
            row[f"precision_{cn}"] = test_r["precision_per_class"][ci] # Test precision for class cn, e.g., 0.80 for Class 0, 0.85 for Class 1
            row[f"recall_{cn}"] = test_r["recall_per_class"][ci] # Test recall for class cn, e.g., 0.78 for Class 0, 0.88 for Class 1
            row[f"f1_{cn}"] = test_r["f1_per_class"][ci] # Test F1 score for class cn, e.g., 0.79 for Class 0, 0.86 for Class 1
        # Append the row dictionary to the rows list for later conversion into a DataFrame:
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir,
                            f"{ds_key}_{proj_label}_within_dataset_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)

    # Print summary
    summary_table = [[r["kernel"], f"{r['accuracy']:.4f}",
                       f"{r['precision_macro']:.4f}", f"{r['recall_macro']:.4f}",
                       f"{r['f1_macro']:.4f}"]
                      for _, r in comparison_df.iterrows()]
    print("\n  Kernel Comparison (Test Set):")
    print(tabulate(summary_table,
                   headers=["Kernel", "Accuracy", "Prec (macro)", "Rec (macro)", "F1 (macro)"],
                   tablefmt="grid"))

    # Identify best kernel
    # Identify the best kernel, naming every kernel that ties for the highest accuracy
    best_kernel = describe_top_kernels(comparison_df["kernel"], comparison_df["accuracy"]) # e.g., "RBF", or "Polynomial (d=2) and Polynomial (d=3) (tie)"
    best_acc = comparison_df["accuracy"].max() # Highest test accuracy, e.g., 0.85
    print(f"\n  Highest test accuracy in this run: {best_kernel} (accuracy = {best_acc:.4f})")

    # === Generate plots ===
    # Plots include:
    # 1. Bar chart comparing accuracy across kernels
    _plot_accuracy_comparison(comparison_df, ds_name, proj_label, output_dir, ds_key)
    # 2. Grouped bar charts for precision and recall per class per kernel
    _plot_precision_recall_comparison(comparison_df, class_names, ds_name,
                                     proj_label, output_dir, ds_key)
    # 3. 2x2 grid of confusion matrices for all 4 kernels
    _plot_confusion_matrices_grid(kernel_results, class_names, ds_name,
                                  proj_label, output_dir, ds_key)

    # === Generate markdown report ===
    _generate_within_report(comparison_df, kernel_results, projection_results,
                            dataset_dict, svm_results, proj_label, output_dir, ds_key)

    print(f"\n  ✓ Within-dataset analysis complete for {ds_name}")

    # Return results for potential further use (e.g., in between-dataset analysis):
    return {
        "comparison_df": comparison_df, # DataFrame containing the performance metrics for each kernel
        "best_kernel": best_kernel, # Name of the best performing kernel based on test accuracy
        "best_accuracy": best_acc, # Accuracy value of the best performing kernel
    }


# =============================================================================
# PLOTTING
# =============================================================================

def _plot_accuracy_comparison(df: pd.DataFrame, ds_name: str, proj_label: str,
                              output_dir: str, ds_key: str) -> None:
    """
    Bar chart comparing accuracy across kernels.

    This bar chart visualizes the test accuracy of each kernel for the given dataset and projection type.
    Each bar represents a kernel, and the height corresponds to its accuracy. The plot includes annotations 
    of the accuracy values on top of each bar for clarity. 
    The x-axis lists the kernel names, and the y-axis shows the accuracy values. 
    A title indicates the dataset name and projection type for context. 
    
    The plot is saved as a PNG file in the specified output directory.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(df))
    bars = ax.bar(df["kernel"], df["accuracy"], color=colors, edgecolor="black",
                  linewidth=0.5)
    for bar, val in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{ds_name} ({proj_label.upper()}) — Kernel Accuracy Comparison",
                 fontsize=14)
    ax.set_ylim(0, min(1.15, df["accuracy"].max() + 0.1))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f"{ds_key}_{proj_label}_kernel_accuracy_comparison.png"),
                dpi=150)
    plt.close(fig)


def _plot_precision_recall_comparison(df: pd.DataFrame, class_names: list,
                                      ds_name: str, proj_label: str,
                                      output_dir: str, ds_key: str) -> None:
    """
    Grouped bar charts for precision and recall per class per kernel.   

    These two grouped bar charts visualize the per-class precision and recall for each kernel on the test set.
    
    Each group of bars corresponds to a kernel, and within each group, there are bars for each class. 
    The height of each bar represents the precision or recall value for that class and kernel. 
    The x-axis lists the class names, and the y-axis shows the precision or recall values. 
    Titles indicate the dataset name, projection type, and whether the chart is for precision or recall. 
    Legends differentiate the kernels by color. 

    The plots are saved as PNG files in the specified output directory.
    """
    n_kernels = len(df)
    n_classes = len(class_names)

    for metric_name in ["precision", "recall"]:
        fig, ax = plt.subplots(figsize=(max(8, n_classes * 2), 5))
        x = np.arange(n_classes)
        width = 0.8 / n_kernels
        colors = sns.color_palette("Set2", n_kernels)

        for ki, (_, row) in enumerate(df.iterrows()):
            vals = [row[f"{metric_name}_{cn}"] for cn in class_names]
            offset = (ki - n_kernels / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=row["kernel"],
                   color=colors[ki], edgecolor="black", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f"{ds_name} ({proj_label.upper()}) — Per-Class {metric_name.capitalize()} by Kernel",
                     fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(os.path.join(
            output_dir,
            f"{ds_key}_{proj_label}_kernel_{metric_name}_comparison.png"
        ), dpi=150)
        plt.close(fig)


def _plot_confusion_matrices_grid(kernel_results: dict, class_names: list,
                                   ds_name: str, proj_label: str,
                                   output_dir: str, ds_key: str) -> None:
    """
    2x2 grid of confusion matrices for all 4 kernels.
    
    This 2x2 grid of confusion matrices visualizes the performance of all four kernels on the test set 
    for the given dataset and projection type.
    Each subplot corresponds to one kernel and displays its confusion matrix, which shows the counts of 
    true positives, false positives, true negatives, and false negatives for each class.
    The x-axis of each confusion matrix represents the predicted classes, while the y-axis represents the actual classes.
    The cells of the confusion matrix are annotated with the count values for clarity.
    The title of each subplot indicates the kernel name, and the overall figure title includes the dataset name 
    and projection type for context.
    
    The grid of confusion matrices is saved as a PNG file in the specified output directory.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (ktag, kr) in enumerate(kernel_results.items()):
        ax = axes[idx]
        cm = kr["test"]["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5,
                    annot_kws={"size": 11})
        ax.set_title(kr["kernel_name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle(f"{ds_name} ({proj_label.upper()}) — Confusion Matrices (Test Set)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir,
                             f"{ds_key}_{proj_label}_confusion_matrices_grid.png"),
                dpi=150)
    plt.close(fig)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def _generate_within_report(comparison_df: pd.DataFrame, kernel_results: dict,
                            projection_results: dict, dataset_dict: dict,
                            svm_results: dict, proj_label: str,
                            output_dir: str, ds_key: str) -> None:
    """
    Generate a detailed within-dataset analysis markdown report.
    
    A comprehensive report is created that includes:
        - Dataset overview and properties.
        - Summary of random projection distortion metrics (Eq.3 and Eq.4).
        - A comparison table of all kernels with accuracy, precision, recall, and F1 scores.
        - Per-class performance tables for each kernel.
        - Key findings and insights based on the results.
        - References to all generated plots (accuracy comparison, precision/recall charts, confusion matrices).
    
    The report is saved in the "within_dataset_reports" subdirectory of the main reports folder 
    and copied to the output directory for easy access alongside the CSV and PNG files.

    Args:
        comparison_df: DataFrame containing kernel performance metrics.
        kernel_results: Detailed results for each kernel (including confusion matrices).
        projection_results: Distortion metrics from the random projection step.
        dataset_dict: Original dataset information.
        svm_results: Original SVM results dictionary.
        proj_label: Label for the projection type (e.g., "eq3" or "eq4").
        output_dir: Directory to save the report.
        ds_key: Dataset key for naming the report and plots.

    Returns:
        None (the report is saved as a markdown file in the specified output directory).
    """
    ds_name = dataset_dict["dataset_name"]
    class_names = [str(c) for c in dataset_dict["class_names"]]
    dist_eq3 = projection_results["distortion_eq3"]
    dist_eq4 = projection_results["distortion_eq4"]

    report = get_markdown_header(
        f"Within-Dataset Analysis — {ds_name} ({proj_label.upper()} Projection)"
    )

    report += f"""
## 1. Dataset Overview

| Property | Value |
|----------|-------|
| **Dataset** | {ds_name} |
| **Samples** | {dataset_dict['n_samples']} |
| **Original Dimensions** | {dataset_dict['n_features']} |
| **Projected Dimensions** | {projection_results['k']} |
| **Classes** | {', '.join(class_names)} ({dataset_dict['n_classes']} classes) |
| **Train Set Size** | {len(svm_results['y_train'])} |
| **Test Set Size** | {len(svm_results['y_test'])} |
| **Projection** | {proj_label.upper()} |

## 2. Random Projection — Distortion Summary

| Distribution | m | M | ε | Mean Ratio | Std Ratio |
|-------------|---|---|---|-----------|----------|
| Eq.3 (±1) | {dist_eq3['m']:.6f} | {dist_eq3['M']:.6f} | {dist_eq3['epsilon']:.6f} | {dist_eq3['mean_ratio']:.6f} | {dist_eq3['std_ratio']:.6f} |
| Eq.4 (sparse) | {dist_eq4['m']:.6f} | {dist_eq4['M']:.6f} | {dist_eq4['epsilon']:.6f} | {dist_eq4['mean_ratio']:.6f} | {dist_eq4['std_ratio']:.6f} |

## 3. SVM Kernel Comparison (Test Set)

{comparison_df[['kernel', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']].round(4).to_markdown(index=False)}

"""

    # Per-class tables
    report += "## 4. Per-Class Metrics\n\n"
    for _, row in comparison_df.iterrows():
        report += f"### {row['kernel']}\n\n"
        report += "| Class | Precision | Recall | F1-Score |\n"
        report += "|-------|-----------|--------|----------|\n"
        for cn in class_names:
            report += (f"| {cn} | {row[f'precision_{cn}']:.4f} | "
                       f"{row[f'recall_{cn}']:.4f} | {row[f'f1_{cn}']:.4f} |\n")
        report += "\n"

    # Best kernel
    # Name every kernel that ties for the best score
    best_acc_kernels = describe_top_kernels(comparison_df["kernel"], comparison_df["accuracy"])
    best_f1_kernels = describe_top_kernels(comparison_df["kernel"], comparison_df["f1_macro"])
    report += f"""
## 5. Key Findings

- **Highest test accuracy in this run:** {best_acc_kernels} ({comparison_df['accuracy'].max():.4f}); lowest {comparison_df['accuracy'].min():.4f}
- **Highest macro F1-score in this run:** {comparison_df['f1_macro'].max():.4f} ({best_f1_kernels})

These come from one random matrix and one train/test split, so they do not rank the kernels in general.

"""

    # Confusion matrices reference
    report += f"""## 6. Figures

### Accuracy Comparison
![Accuracy Comparison]({ds_key}_{proj_label}_kernel_accuracy_comparison.png)

### Precision by Kernel
![Precision Comparison]({ds_key}_{proj_label}_kernel_precision_comparison.png)

### Recall by Kernel
![Recall Comparison]({ds_key}_{proj_label}_kernel_recall_comparison.png)

### Confusion Matrices
![Confusion Matrices Grid]({ds_key}_{proj_label}_confusion_matrices_grid.png)

"""

    report += get_markdown_footer()

    # Save to reports folder AND to the output directory
    report_filename = f"{ds_key}_{proj_label}_within_dataset_report.md"
    within_reports_dir = os.path.join(str(REPORTS_DIR), "within_dataset_reports", f"{ds_key}_reports")
    for save_dir in [within_reports_dir, output_dir]:
        report_path = os.path.join(save_dir, report_filename)
        os.makedirs(save_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

    # Copy plot images alongside the reports in the within_reports_dir
    image_files = [
        f"{ds_key}_{proj_label}_kernel_accuracy_comparison.png",
        f"{ds_key}_{proj_label}_kernel_precision_comparison.png",
        f"{ds_key}_{proj_label}_kernel_recall_comparison.png",
        f"{ds_key}_{proj_label}_confusion_matrices_grid.png",
    ]
    for img in image_files:
        src = os.path.join(output_dir, img)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(within_reports_dir, img))
