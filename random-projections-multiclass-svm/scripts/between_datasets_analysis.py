#!/usr/bin/env python3
"""
between_datasets_analysis.py - Between-Datasets Kernel Comparison

This module performs the "between-datasets" analysis for HW3.

For a single random projection type (either Eq.3 or Eq.4), 
it compares the performance of all four SVM kernels (linear, poly2, poly3, rbf)
across all five datasets (Iris, Breast Cancer, Wine, Seeds, Glass Identification).

The analysis includes generating a comparison table of performance metrics, plotting
comparative bar charts and heatmaps, and creating a detailed markdown report summarizing the findings.

Finally, it plots the distortion (epsilon) of the random projections across datasets.

The report includes tables, figures, and, per dataset, the kernel(s) with the highest test accuracy in this run
(one projection matrix and one split, so not a ranking of the kernels in general).

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

def run_between_datasets_analysis(all_results: dict, output_dir: str,
                                  proj_label: str = "eq3") -> dict:
    """
    Compare each kernel's performance across all datasets.

    Args:
        all_results: Dict mapping dataset_key -> {
            "dataset": dataset_dict,
            "projection": projection_results,
            "svm_eq3": svm_results_eq3,
            "svm_eq4": svm_results_eq4,
        }.
        output_dir: Directory to save analysis results.
        proj_label: Which projection to use for comparison ('eq3' or 'eq4').

    Returns:
        Analysis results dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract dataset keys and names for reporting and analysis
    dataset_keys = list(all_results.keys())
    dataset_names = {k: all_results[k]["dataset"]["dataset_name"] for k in dataset_keys}

    print(f"\n{'═' * 60}")
    print(f"  BETWEEN-DATASETS ANALYSIS ({proj_label.upper()} Projection)")
    print(f"  Datasets: {', '.join(dataset_names.values())}")
    print(f"{'═' * 60}")

    svm_key = f"svm_{proj_label}" # Key to access the SVM results for the specified projection type (eq3 or eq4)

    # === Build comparison table ===
    # Rows: one per (kernel, dataset) combination
    rows = []
    # Iterate over each dataset:
    for ds_key in dataset_keys:
        # Extract the dataset name and SVM results for the current dataset and specified projection type:
        ds_name = dataset_names[ds_key]
        svm_res = all_results[ds_key][svm_key]
        kernel_results = svm_res["kernel_results"]
        # Iterate over each kernel's results:
        for ktag, kr in kernel_results.items():
            # Extract the test performance metrics (accuracy, precision_macro, recall_macro, f1_macro) for the current dataset:
            test_r = kr["test"]
            # Append a dictionary containing the dataset name, kernel name, and performance metrics to the rows list for later conversion into a DataFrame:
            rows.append({
                "dataset": ds_name, # Dataset name, e.g., "Iris", "Breast Cancer", "Wine", "Seeds", "Glass Identification"
                "dataset_key": ds_key, # Dataset key, e.g., "iris", "breast_cancer", "wine", "seeds", "glass"
                "kernel": kr["kernel_name"], # Kernel name, e.g., "Linear", "Poly2", "Poly3", "RBF"
                "kernel_tag": ktag, # Kernel tag, e.g., "linear", "poly2", "poly3", "rbf"
                "accuracy": test_r["accuracy"], # Test accuracy, e.g., 0.85
                "precision_macro": test_r["precision_macro"], # Test precision (macro), e.g., 0.84
                "recall_macro": test_r["recall_macro"], # Test recall (macro), e.g., 0.84
                "f1_macro": test_r["f1_macro"], # Test F1 score (macro), e.g., 0.835
            })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(output_dir, f"between_datasets_{proj_label}_comparison.csv"),
                         index=False)

    # Print summary
    pivot = comparison_df.pivot_table(
        index="kernel", columns="dataset", values="accuracy"
    )
    print("\n  Accuracy Matrix (Kernel × Dataset):")
    print(tabulate(pivot.round(4), headers="keys", tablefmt="grid",
                   showindex=True))

    # === Distortion comparison ===
    dist_rows = []
    # Iterate over each dataset: 
    for ds_key in dataset_keys:
        # Extract the dataset name and projection results for the current dataset:
        ds_name = dataset_names[ds_key]
        proj = all_results[ds_key]["projection"]
        # Use only the projection type this analysis covers, so the Eq.3 and Eq.4 outputs
        # (CSV, table, and figure) each show their own distribution:
        for eq_label in [proj_label]:
            # Extract the distortion metrics (epsilon, m, M, mean_ratio) for the current projection type:
            dist = proj[f"distortion_{eq_label}"]
            # Append a dictionary containing the dataset name, projection type, and distortion metrics to the dist_rows list for later conversion into a DataFrame:
            dist_rows.append({
                "dataset": ds_name, # Dataset name, e.g., "Iris", "Breast Cancer", "Wine", "Seeds", "Glass Identification"
                "dataset_key": ds_key, # Dataset key, e.g., "iris", "breast_cancer", "wine", "seeds", "glass"
                "distribution": eq_label.upper(), # Projection type, e.g., "EQ3", "EQ4"
                "epsilon": dist["epsilon"], # Distortion epsilon, e.g., 0.1
                "m": dist["m"], # Distortion m, e.g., 5
                "M": dist["M"], # Distortion M, e.g., 10
                "mean_ratio": dist["mean_ratio"], # Mean ratio, e.g., 0.95
            })
    dist_df = pd.DataFrame(dist_rows) 
    dist_df.to_csv(os.path.join(output_dir, f"distortion_comparison_{proj_label}.csv"), index=False)

    print("\n  Distortion Comparison:")
    print(tabulate(dist_df.round(4), headers="keys", tablefmt="grid",
                   showindex=False))

    # === Generate plots ===
    _plot_kernel_accuracy_across_datasets(comparison_df, output_dir, proj_label)
    _plot_accuracy_heatmap(comparison_df, output_dir, proj_label)
    _plot_precision_recall_across_datasets(comparison_df, output_dir, proj_label)
    _plot_distortion_comparison(dist_df, output_dir, proj_label)

    # === Generate report ===
    _generate_between_report(comparison_df, dist_df, all_results, output_dir,
                             proj_label, dataset_keys, dataset_names)

    print(f"\n  ✓ Between-datasets analysis complete")

    # Return the comparison DataFrame, distortion DataFrame, and the pivot table of accuracies for use in the report generation and discussion:
    return {
        "comparison_df": comparison_df, # DataFrame containing the performance metrics for each kernel and dataset combination
        "distortion_df": dist_df, # DataFrame containing the distortion metrics for each dataset and projection type
        "pivot_accuracy": pivot, # Pivot table of accuracies with kernels as rows and datasets as columns for easy analysis and reporting
    }


# =============================================================================
# PLOTTING
# =============================================================================

def _plot_kernel_accuracy_across_datasets(df: pd.DataFrame, output_dir: str, proj_label: str = "eq3") -> None:
    """
    Grouped bar chart: for each kernel, accuracy across datasets.
    
    This grouped bar chart visualizes the test accuracy of each kernel across all datasets for the specified projection type (eq3 or eq4).
    The x-axis lists the kernel names, and for each kernel, there are grouped bars representing the accuracy on each dataset. 
    The y-axis shows the accuracy values, and a title indicates the projection type for context. 
    Each bar is annotated with its accuracy value for clarity. 

    The plot is saved as a PNG file in the specified output directory.
    """
    kernels = df["kernel"].unique()
    datasets = df["dataset"].unique()
    n_kernels = len(kernels)
    n_datasets = len(datasets)

    fig, ax = plt.subplots(figsize=(max(8, n_kernels * 2), 5))
    x = np.arange(n_kernels)
    width = 0.8 / n_datasets
    colors = sns.color_palette("Set2", n_datasets)

    for di, ds in enumerate(datasets):
        vals = []
        for k in kernels:
            row = df[(df["kernel"] == k) & (df["dataset"] == ds)]
            vals.append(row["accuracy"].values[0] if len(row) > 0 else 0)
        offset = (di - n_datasets / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=ds, color=colors[di],
                      edgecolor="black", linewidth=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(kernels, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Kernel Accuracy Across Datasets ({proj_label.upper()})", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"kernel_accuracy_across_datasets_{proj_label}.png"),
                dpi=150)
    plt.close(fig)


def _plot_accuracy_heatmap(df: pd.DataFrame, output_dir: str, proj_label: str = "eq3") -> None:
    """
    Heatmap: kernel × dataset → accuracy.
    
    This heatmap visualizes the test accuracy of each kernel on each dataset for the specified projection type (eq3 or eq4).
    The x-axis lists the dataset names, and the y-axis lists the kernel names.
    Each cell is annotated with the accuracy value for clarity.

    The plot is saved as a PNG file in the specified output directory.
    """
    pivot = df.pivot_table(index="kernel", columns="dataset", values="accuracy")

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2),
                                    max(4, len(pivot.index) * 1.2)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, vmin=0, vmax=1)
    ax.set_title(f"Kernel × Dataset Accuracy Heatmap ({proj_label.upper()})", fontsize=14)
    ax.set_ylabel("Kernel", fontsize=12)
    ax.set_xlabel("Dataset", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"kernel_dataset_accuracy_heatmap_{proj_label}.png"),
                dpi=150)
    plt.close(fig)


def _plot_precision_recall_across_datasets(df: pd.DataFrame, output_dir: str, proj_label: str = "eq3") -> None:
    """
    Grouped bar charts for macro precision and recall across datasets.

    These grouped bar charts visualize the macro precision and macro recall of each kernel across all datasets for the specified projection type (eq3 or eq4).
    The x-axis lists the kernel names, and for each kernel, there are grouped bars representing the metric value on each dataset.
    The y-axis shows the metric values, and a title indicates the projection type for context.
    Each bar is annotated with its metric value for clarity.

    The plot is saved as a PNG file in the specified output directory.
    """
    for metric in ["precision_macro", "recall_macro"]:
        metric_label = "Precision" if "precision" in metric else "Recall"
        kernels = df["kernel"].unique()
        datasets = df["dataset"].unique()
        n_kernels = len(kernels)
        n_datasets = len(datasets)

        fig, ax = plt.subplots(figsize=(max(8, n_kernels * 2), 5))
        x = np.arange(n_kernels)
        width = 0.8 / n_datasets
        colors = sns.color_palette("Set2", n_datasets)

        for di, ds in enumerate(datasets):
            vals = []
            for k in kernels:
                row = df[(df["kernel"] == k) & (df["dataset"] == ds)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            offset = (di - n_datasets / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=ds, color=colors[di],
                   edgecolor="black", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(kernels, fontsize=10)
        ax.set_ylabel(f"Macro {metric_label}", fontsize=12)
        ax.set_title(f"Macro {metric_label} Across Datasets ({proj_label.upper()})", fontsize=14)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                 f"kernel_{metric}_across_datasets_{proj_label}.png"),
                    dpi=150)
        plt.close(fig)


def _plot_distortion_comparison(dist_df: pd.DataFrame, output_dir: str, proj_label: str = "eq3") -> None:
    """
    Bar chart comparing distortion (epsilon) across datasets and distributions.

    This bar chart visualizes the distortion (epsilon) for each distribution across all datasets for the specified projection type (eq3 or eq4).
    The x-axis lists the dataset names, and for each dataset, there are grouped bars representing the distortion for each distribution.
    The y-axis shows the distortion values, and a title indicates the projection type for context.
    Each bar is annotated with its distortion value for clarity.
    
    The plot is saved as a PNG file in the specified output directory.
    """
    # Keep only the distribution this figure is labeled for
    dist_df = dist_df[dist_df["distribution"] == proj_label.upper()]
    fig, ax = plt.subplots(figsize=(max(8, dist_df["dataset"].nunique() * 3.0), 5))

    datasets = dist_df["dataset"].unique()
    distributions = dist_df["distribution"].unique()
    n_datasets = len(datasets)
    n_dist = len(distributions)

    x = np.arange(n_datasets)
    width = 0.35
    colors = sns.color_palette("Set1", n_dist)

    for di, dist_type in enumerate(distributions):
        vals = []
        for ds in datasets:
            row = dist_df[(dist_df["dataset"] == ds) & (dist_df["distribution"] == dist_type)]
            vals.append(row["epsilon"].values[0] if len(row) > 0 else 0)
        offset = (di - n_dist / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=dist_type, color=colors[di],
                      edgecolor="black", linewidth=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Epsilon (ε)", fontsize=12)
    ax.set_title(f"Distortion (ε) Across Datasets ({proj_label.upper()})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"distortion_comparison_across_datasets_{proj_label}.png"),
                dpi=150)
    plt.close(fig)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def _generate_between_report(comparison_df: pd.DataFrame, dist_df: pd.DataFrame,
                             all_results: dict, output_dir: str,
                             proj_label: str, dataset_keys: list,
                             dataset_names: dict) -> None:
    """
    Generate a detailed between-datasets analysis markdown report.
    
    A comprehensive markdown report is generated that includes:
        - An overview of the datasets and projection type.
        - A comparison table of accuracy for each kernel across all datasets.
        - Per-kernel analysis showing precision, recall, and F1 scores for each dataset.
        - A comparison of distortion (epsilon) across datasets and projection types.
        - Figures visualizing the accuracy, precision, recall, and distortion comparisons.
        - A discussion section listing, per dataset, the kernel(s) with the highest test accuracy in this run and
          the range across kernels, with a note that one draw does not rank the kernels in general.
    
    The report is saved as a markdown file in the specified output directory and copied to a central "between_datasets_reports" 
    directory for easy access alongside the generated figures.
          
    Args:
        comparison_df: DataFrame containing the performance metrics for each kernel and dataset combination.
        dist_df: DataFrame containing the distortion metrics for each dataset and projection type.
        all_results: Original results dictionary containing detailed information about datasets, projections, and SVM results for report generation.
        output_dir: Directory to save the generated report and associated figures.
        proj_label: Label for the projection type used in the analysis (e.g., "eq3" or "eq4") to contextualize the report.
        dataset_keys: List of dataset keys used in the analysis for referencing in the report.
        dataset_names: Dictionary mapping dataset keys to their human-readable names for use in the report.

    Returns:
        None (the report is saved as a markdown file in the specified output directory).

    """
    report = get_markdown_header(
        f"Between-Datasets Analysis ({proj_label.upper()} Projection)"
    )

    # Datasets overview
    report += "## 1. Datasets Overview\n\n"
    report += "| Dataset | Samples | Features | Classes | Projected Dim |\n"
    report += "|---------|---------|----------|---------|---------------|\n"
    for dk in dataset_keys:
        ds = all_results[dk]["dataset"]
        k = all_results[dk]["projection"]["k"]
        report += (f"| {ds['dataset_name']} | {ds['n_samples']} | "
                   f"{ds['n_features']} | {ds['n_classes']} | {k} |\n")

    # Accuracy pivot
    pivot = comparison_df.pivot_table(index="kernel", columns="dataset", values="accuracy")
    report += f"\n## 2. Accuracy Comparison (Kernel × Dataset)\n\n"
    report += pivot.round(4).to_markdown() + "\n\n"

    # Per-kernel analysis
    report += "## 3. Per-Kernel Cross-Dataset Analysis\n\n"
    for kernel in comparison_df["kernel"].unique():
        kdf = comparison_df[comparison_df["kernel"] == kernel]
        report += f"### {kernel}\n\n"
        report += kdf[["dataset", "accuracy", "precision_macro", "recall_macro",
                        "f1_macro"]].round(4).to_markdown(index=False)
        report += "\n\n"

    # Distortion comparison
    report += "## 4. Distortion Comparison\n\n"
    report += dist_df.round(4).to_markdown(index=False)
    report += "\n\n"

    # Figures
    report += f"""## 5. Figures

### Kernel Accuracy Across Datasets
![Accuracy Comparison](kernel_accuracy_across_datasets_{proj_label}.png)

### Accuracy Heatmap
![Accuracy Heatmap](kernel_dataset_accuracy_heatmap_{proj_label}.png)

### Macro Precision Across Datasets
![Precision](kernel_precision_macro_across_datasets_{proj_label}.png)

### Macro Recall Across Datasets
![Recall](kernel_recall_macro_across_datasets_{proj_label}.png)

### Distortion Comparison
![Distortion](distortion_comparison_across_datasets_{proj_label}.png)

"""

    # Discussion
    report += "## 6. Discussion\n\n"
    report += ("Each data set was projected with one random matrix and split once, so these numbers "
               "describe this run; they do not rank the kernels in general.\n\n")

    # Kernel(s) with the highest test accuracy per dataset in this run
    for dk in dataset_keys:
        ds_name = dataset_names[dk]
        ds_df = comparison_df[comparison_df["dataset_key"] == dk]
        best_names = describe_top_kernels(ds_df["kernel"], ds_df["accuracy"])
        lo, hi = ds_df["accuracy"].min(), ds_df["accuracy"].max()
        if lo == hi:
            report += f"- **{ds_name}:** all four kernels {hi:.4f}\n"
        else:
            report += (f"- **{ds_name}:** highest test accuracy in this run: **{best_names}** "
                       f"({hi:.4f}; the four kernels range from {lo:.4f} to {hi:.4f})\n")

    # Highest mean accuracy across datasets (one projection draw per dataset, so not a robustness measure)
    mean_acc = comparison_df.groupby("kernel", sort=False)["accuracy"].mean()
    best_overall = describe_top_kernels(mean_acc.index, mean_acc.values)
    report += (f"\n- **Highest mean accuracy across datasets:** **{best_overall}** "
               f"with mean accuracy {mean_acc.max():.4f} (one projection draw per dataset)\n")

    report += get_markdown_footer()

    # Save
    report_filename = f"between_datasets_{proj_label}_analysis_report.md"
    between_reports_dir = os.path.join(str(REPORTS_DIR), "between_datasets_reports")
    for save_dir in [between_reports_dir, output_dir]:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, report_filename), "w", encoding="utf-8") as f:
            f.write(report)

    # Copy plot images alongside the reports in the between_reports_dir
    image_files = [
        f"kernel_accuracy_across_datasets_{proj_label}.png",
        f"kernel_dataset_accuracy_heatmap_{proj_label}.png",
        f"kernel_precision_macro_across_datasets_{proj_label}.png",
        f"kernel_recall_macro_across_datasets_{proj_label}.png",
        f"distortion_comparison_across_datasets_{proj_label}.png",
    ]
    for img in image_files:
        src = os.path.join(output_dir, img)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(between_reports_dir, img))
