#!/usr/bin/env python3
"""
dataset_stats.py - Dataset Statistics and Analysis

This module computes descriptive statistics, generates plots, and creates
summary reports for each dataset used in HW3.

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
from tabulate import tabulate

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from author_info import get_markdown_header, get_markdown_footer

# Project paths
HW3_DIR = SCRIPT_DIR.parent
STATS_DIR = HW3_DIR / "dataset_stats"


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def compute_and_save_statistics(dataset_dict: dict) -> str:
    """
    Compute descriptive statistics for a dataset, generate plots, and save
    a summary markdown report.

    The function computes summary statistics (mean, std, min, max, quartiles) for each feature,
    calculates class distribution, generates a correlation heatmap, feature distribution plots,
    and a class distribution bar chart. 
    
    If the dataset has 10 or fewer features, it also creates a pairwise scatter matrix.

    All results are saved in a structured output directory, and a comprehensive markdown report 
    is generated that includes tables and references to the generated plots. 
    
    The report is saved in the "dataset_stats" subdirectory of the main reports folder and copied 
    to the output directory for easy access alongside the CSV and PNG files.

    Args:
        dataset_dict: Dataset dictionary from data_loader.

    Returns:
        Path to the output directory.
    """
    # Extract dataset information from the input dictionary:
    ds_key = dataset_dict["dataset_key"] # Unique key for the dataset, e.g., "iris", "breast_cancer", etc.
    ds_name = dataset_dict["dataset_name"] # Human-readable name of the dataset, e.g., "Iris", "Breast Cancer", etc.
    X = dataset_dict["X"] # Feature matrix (n_samples x n_features)
    y = dataset_dict["y"] # Target labels (n_samples,)
    feature_names = dataset_dict["feature_names"] # List of feature names
    class_names = dataset_dict["class_names"] # List of class names
    n_samples = dataset_dict["n_samples"] # Number of samples
    n_features = dataset_dict["n_features"] # Number of features
    n_classes = dataset_dict["n_classes"] # Number of classes

    out_dir = STATS_DIR / ds_key
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─' * 50}")
    print(f"  Computing statistics for: {ds_name}")
    print(f"{'─' * 50}")

    # --- Build DataFrame ---
    df = pd.DataFrame(X, columns=feature_names)
    df["class"] = y
    df["class_name"] = [class_names[int(c)] for c in y]

    # === 1. Summary Statistics ===
    # Compute descriptive statistics for each feature and save to CSV:
    # The statistics include count, mean, std, min, 25th percentile, 50th percentile (median), 
    # 75th percentile, and max for each feature. 
    # The resulting DataFrame is transposed for better readability and saved as "{ds_key}_summary_stats.csv" 
    # in the output directory.
    stats_df = df[feature_names].describe().T # Compute summary statistics and transpose for better readability
    stats_df.index.name = "feature"
    stats_df.to_csv(out_dir / f"{ds_key}_summary_stats.csv")

    print(f"\n  Summary Statistics ({ds_name}):")
    print(tabulate(stats_df.round(4), headers="keys", tablefmt="grid",
                   showindex=True))

    # === 2. Class Distribution ===
    # Calculate the distribution of classes in the dataset by counting the number of samples belonging to each class:
    # The class distribution is saved as a CSV file named "{ds_key}_class_distribution.csv" in the output directory. 
    # The CSV file contains columns for class index, class name, count of samples in each class, and the proportion of 
    # samples in each class relative to the total number of samples.
    class_counts = pd.Series(y).value_counts().sort_index()
    class_dist_df = pd.DataFrame({
        "class_index": class_counts.index, # Class index (e.g., 0, 1, 2 for a 3-class dataset)
        "class_name": [class_names[int(i)] for i in class_counts.index], # Class name corresponding to each class index
        "count": class_counts.values, # Count of samples in each class (e.g., 50 for Class 0, 50 for Class 1)
        "proportion": (class_counts.values / n_samples).round(4), # Proportion of samples in each class (e.g., 0.3333 for Class 0, 0.3333 for Class 1)
    })
    class_dist_df.to_csv(out_dir / f"{ds_key}_class_distribution.csv", index=False)

    print(f"\n  Class Distribution ({ds_name}):")
    print(tabulate(class_dist_df, headers="keys", tablefmt="grid",
                   showindex=False))

    # === 3. Correlation Heatmap ===
    # Generate a correlation heatmap for the features in the dataset:
    # The heatmap visualizes the pairwise correlation coefficients between features, with color intensity indicating the strength and direction of the correlation. 
    # The heatmap is saved as a PNG file named "{ds_key}_correlation_heatmap.png" in the output directory. 
    # If the number of features is greater than 15, annotations are disabled for better readability, and the figure size is adjusted to accommodate more features. 
    # If any error occurs during heatmap generation (e.g., due to a large number of features), a warning message is printed instead of the heatmap.

    try: # Try to generate the correlation heatmap, but catch any exceptions that may occur (e.g., due to too many features or memory issues)
        fig, ax = plt.subplots(figsize=(min(12, n_features + 2), min(10, n_features + 1)))
        corr = df[feature_names].corr()
        sns.heatmap(corr, annot=(n_features <= 15), fmt=".2f", cmap="coolwarm",
                    center=0, square=True, ax=ax,
                    annot_kws={"size": max(6, 10 - n_features // 5)})
        ax.set_title(f"Feature Correlation Heatmap — {ds_name}", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / f"{ds_key}_correlation_heatmap.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] Could not generate correlation heatmap: {e}")

    # === 4. Feature Distributions (box plots) ===
    try: # Try to generate feature distributions, but catch any exceptions that may occur (e.g., due to too many features or memory issues)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, feat in enumerate(feature_names):
            ax = axes[i]
            for ci, cn in enumerate(class_names):
                mask = y == ci
                ax.hist(X[mask, i], bins=20, alpha=0.5, label=str(cn))
            ax.set_title(feat, fontsize=9)
            ax.set_xlabel("")
            if i == 0:
                ax.legend(fontsize=7)

        # Hide extra axes
        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Feature Distributions — {ds_name}", fontsize=14, y=1.01)
        plt.tight_layout()
        fig.savefig(out_dir / f"{ds_key}_feature_distributions.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] Could not generate feature distributions: {e}")

    # === 5. Class Distribution Bar Chart ===
    try: # Try to generate class distribution bar chart, but catch any exceptions that may occur (e.g., due to too many classes or memory issues)
        fig, ax = plt.subplots(figsize=(max(6, n_classes * 1.5), 4))
        colors = sns.color_palette("Set2", n_classes)
        bars = ax.bar(range(n_classes), class_dist_df["count"], color=colors,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(n_classes))
        cn_strs = [str(c) for c in class_names]
        ax.set_xticklabels(cn_strs, rotation=30 if max(len(c) for c in cn_strs) > 10 else 0,
                           ha="right" if max(len(c) for c in cn_strs) > 10 else "center")
        ax.set_ylabel("Count")
        ax.set_title(f"Class Distribution — {ds_name}", fontsize=14)
        for bar, count in zip(bars, class_dist_df["count"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(count), ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        fig.savefig(out_dir / f"{ds_key}_class_distribution.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] Could not generate class distribution plot: {e}")

    # === 6. Pairwise Scatter Matrix (if features <= 10) ===
    if n_features <= 10:
        try: # Try to generate pairwise scatter matrix, but catch any exceptions that may occur (e.g., due to too many features or memory issues)
            g = sns.pairplot(df, vars=feature_names[:min(n_features, 6)],
                             hue="class_name", palette="Set2",
                             plot_kws={"alpha": 0.5, "s": 20},
                             diag_kind="hist")
            g.figure.suptitle(f"Pairwise Scatter — {ds_name}", y=1.01, fontsize=14)
            g.figure.savefig(out_dir / f"{ds_key}_pairwise_scatter.png", dpi=120,
                             bbox_inches="tight")
            plt.close(g.figure)
        except Exception as e:
            print(f"  [WARNING] Could not generate pairwise scatter: {e}")

    # === 7. Markdown Summary Report ===
    # Generate a comprehensive markdown report that includes:
    # - Dataset overview and properties.
    # - Summary of random projection distortion metrics (Eq.3 and Eq.4).
    # - A comparison table of all kernels with accuracy, precision, recall, and F1 scores.
    # - Per-class performance tables for each kernel.
    # - Key findings and insights based on the results.
    # - References to all generated plots (accuracy comparison, precision/recall charts, confusion matrices).
    # The report is saved in the "dataset_stats" subdirectory of the main reports folder and copied to the 
    # output directory for easy access alongside the CSV and PNG files.
    _generate_stats_report(dataset_dict, stats_df, class_dist_df, out_dir, ds_key)

    print(f"\n  ✓ Statistics saved to: {out_dir}")
    return str(out_dir)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def _generate_stats_report(dataset_dict: dict, stats_df: pd.DataFrame,
                           class_dist_df: pd.DataFrame, out_dir: Path,
                           ds_key: str) -> None:
    """
    Generate a markdown statistics report for the dataset.
    
    A comprehensive markdown report is created that includes:
        - Dataset overview and properties.
        - Summary of feature statistics (mean, std, min, max, quartiles).
        - Class distribution table with counts and proportions.
        - References to all generated plots (correlation heatmap, feature distributions, class distribution).
    
    The report is saved in the "dataset_stats" subdirectory of the main reports folder and copied to the 
    output directory for easy access alongside the CSV and PNG files.
    
    Args:
        dataset_dict: Original dataset information.
        stats_df: DataFrame containing summary statistics for each feature.
        class_dist_df: DataFrame containing class distribution information.
        out_dir: Directory to save the report.
        ds_key: Dataset key for naming the report and plots.
    
    Returns:
        None (the report is saved as a markdown file in the specified output directory).
    """
    ds_name = dataset_dict["dataset_name"]
    n_samples = dataset_dict["n_samples"]
    n_features = dataset_dict["n_features"]
    n_classes = dataset_dict["n_classes"]

    report = get_markdown_header(f"Dataset Statistics — {ds_name}")
    report += f"""
## Dataset Overview

| Property | Value |
|----------|-------|
| **Name** | {ds_name} |
| **Samples** | {n_samples} |
| **Features** | {n_features} |
| **Classes** | {n_classes} |
| **Feature Names** | {', '.join(dataset_dict['feature_names'][:10])}{'...' if n_features > 10 else ''} |
| **Class Names** | {', '.join(str(c) for c in dataset_dict['class_names'])} |

## Summary Statistics

{stats_df.round(4).to_markdown()}

## Class Distribution

{class_dist_df.to_markdown(index=False)}

## Figures

### Correlation Heatmap
![Correlation Heatmap]({ds_key}_correlation_heatmap.png)

### Feature Distributions
![Feature Distributions]({ds_key}_feature_distributions.png)

### Class Distribution
![Class Distribution]({ds_key}_class_distribution.png)

"""
    if n_features <= 10:
        report += f"""### Pairwise Scatter
![Pairwise Scatter]({ds_key}_pairwise_scatter.png)

"""
    report += get_markdown_footer()

    report_path = out_dir / f"{ds_key}_stats_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    from data_loader import load_dataset, DEFAULT_DATASETS
    # Compute statistics and generate reports for all datasets defined in DEFAULT_DATASETS:
    for key in DEFAULT_DATASETS:
        ds = load_dataset(key)
        compute_and_save_statistics(ds)
