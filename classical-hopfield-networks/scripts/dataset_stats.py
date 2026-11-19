#!/usr/bin/env python3
"""
dataset_stats.py - Descriptive Statistics and Visualization Script

*** NOT REQUIRED BY ASSIGNMENT ***

This script computes descriptive statistics for the HW2 binary datasets,
generates visualizations, and creates a markdown report.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 2 – Hopfield Networks
Date: February 2, 2026
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

# Import author info for consistent headers
try:
    from author_info import get_markdown_header, get_markdown_footer
except ImportError:
    # Fallback if author_info not available
    def get_markdown_header(title, date=None):
        if date is None:
            date = datetime.now().strftime("%B %d, %Y")
        return f"# {title}\n\n**Date:** {date}\n\n---\n\n"
    def get_markdown_footer():
        return "\n\n---\n"


def get_project_paths() -> dict:
    """Get the standard project paths."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    stats_dir = project_dir / "dataset_stats"
    
    return {
        "project": project_dir,
        "data": data_dir,
        "dataset_stats": stats_dir,
        "scripts": script_dir
    }


def create_stats_folder(seed: int, noise_percent: float, paths: dict) -> Tuple[Path, str]:
    """
    Create a timestamped folder for statistics output.
    
    Args:
        seed: Random seed used for validation dataset
        noise_percent: Noise percentage used for validation dataset
        paths: Dictionary of project paths
        
    Returns:
        Tuple of (folder path, timestamp string)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"dataset_stats_{seed}_{noise_percent}_{timestamp}"
    stats_folder = paths["dataset_stats"] / folder_name
    
    stats_folder.mkdir(parents=True, exist_ok=True)
    
    # Create plots subfolder
    plots_folder = stats_folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    
    return stats_folder, timestamp


def compute_binary_stats(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """
    Compute descriptive statistics for a binary dataset.
    
    Args:
        df: DataFrame with binary values
        name: Name of the dataset
        
    Returns:
        Dictionary with computed statistics
    """
    data = df.values
    
    # Basic shape info
    n_samples, n_features = data.shape
    
    # Count statistics
    total_values = data.size
    ones_count = np.sum(data)
    zeros_count = total_values - ones_count
    
    # Proportions
    ones_proportion = ones_count / total_values
    zeros_proportion = zeros_count / total_values
    
    # Per-sample statistics (how many 1s per sample)
    ones_per_sample = np.sum(data, axis=1)
    ones_per_sample_mean = np.mean(ones_per_sample)
    ones_per_sample_std = np.std(ones_per_sample)
    ones_per_sample_min = np.min(ones_per_sample)
    ones_per_sample_max = np.max(ones_per_sample)
    
    # Per-feature statistics (how many 1s per feature/pixel position)
    ones_per_feature = np.sum(data, axis=0)
    ones_per_feature_mean = np.mean(ones_per_feature)
    ones_per_feature_std = np.std(ones_per_feature)
    
    # Density (proportion of 1s)
    sample_densities = ones_per_sample / n_features
    density_mean = np.mean(sample_densities)
    density_std = np.std(sample_densities)
    
    # Sparsity analysis
    fully_zero_samples = np.sum(ones_per_sample == 0)
    fully_one_samples = np.sum(ones_per_sample == n_features)
    
    # Feature variance (binary variance = p*(1-p))
    feature_proportions = ones_per_feature / n_samples
    feature_variance = feature_proportions * (1 - feature_proportions)
    avg_feature_variance = np.mean(feature_variance)
    
    # Entropy per feature (for binary: -p*log(p) - (1-p)*log(1-p))
    # Avoid log(0) by clipping
    p = np.clip(feature_proportions, 1e-10, 1 - 1e-10)
    feature_entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    avg_feature_entropy = np.mean(feature_entropy)
    
    # Compile all stats into a dictionary
    return {
        "dataset_name": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "total_values": total_values,
        "ones_count": int(ones_count),
        "zeros_count": int(zeros_count),
        "ones_proportion": ones_proportion,
        "zeros_proportion": zeros_proportion,
        "ones_per_sample_mean": ones_per_sample_mean,
        "ones_per_sample_std": ones_per_sample_std,
        "ones_per_sample_min": int(ones_per_sample_min),
        "ones_per_sample_max": int(ones_per_sample_max),
        "ones_per_feature_mean": ones_per_feature_mean,
        "ones_per_feature_std": ones_per_feature_std,
        "density_mean": density_mean,
        "density_std": density_std,
        "fully_zero_samples": fully_zero_samples,
        "fully_one_samples": fully_one_samples,
        "avg_feature_variance": avg_feature_variance,
        "avg_feature_entropy": avg_feature_entropy,
        "ones_per_sample": ones_per_sample,
        "ones_per_feature": ones_per_feature,
        "sample_densities": sample_densities,
        "feature_variance": feature_variance
    }


def compute_comparison_stats(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                              valid_df: pd.DataFrame, noise_percent: float) -> Dict[str, Any]:
    """
    Compute comparison statistics between datasets.
    This is used to assess how the test (10% noise) and validation (20% noise by default) datasets differ from the training dataset.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        valid_df: Validation DataFrame
        noise_percent: Noise percentage used for validation
        
    Returns:
        Dictionary with comparison statistics
    """
    train_data = train_df.values
    test_data = test_df.values
    valid_data = valid_df.values
    
    # Hamming distances (number of differing bits)
    # Between training and testing
    train_test_diff = np.sum(train_data != test_data)
    train_test_diff_percent = (train_test_diff / train_data.size) * 100
    
    # Between training and validation
    train_valid_diff = np.sum(train_data != valid_data)
    train_valid_diff_percent = (train_valid_diff / train_data.size) * 100
    
    # Per-sample Hamming distances
    train_test_hamming_per_sample = np.sum(train_data != test_data, axis=1)
    train_valid_hamming_per_sample = np.sum(train_data != valid_data, axis=1)
    
    # Correlation between feature activations
    train_feature_sums = np.sum(train_data, axis=0)
    test_feature_sums = np.sum(test_data, axis=0)
    valid_feature_sums = np.sum(valid_data, axis=0)
    
    train_test_corr = np.corrcoef(train_feature_sums, test_feature_sums)[0, 1]
    train_valid_corr = np.corrcoef(train_feature_sums, valid_feature_sums)[0, 1]
    
    return {
        "train_test_diff_count": train_test_diff,
        "train_test_diff_percent": train_test_diff_percent,
        "train_valid_diff_count": train_valid_diff,
        "train_valid_diff_percent": train_valid_diff_percent,
        "expected_valid_diff_percent": noise_percent,
        "train_test_hamming_mean": np.mean(train_test_hamming_per_sample),
        "train_test_hamming_std": np.std(train_test_hamming_per_sample),
        "train_valid_hamming_mean": np.mean(train_valid_hamming_per_sample),
        "train_valid_hamming_std": np.std(train_valid_hamming_per_sample),
        "train_test_feature_correlation": train_test_corr,
        "train_valid_feature_correlation": train_valid_corr
    }


# The plot style applies only while this function runs, so later plots in the same process
# (for example the Hopfield experiment in kamp_hw2.py) keep matplotlib's defaults.
@plt.style.context('seaborn-v0_8-whitegrid')
def create_visualizations(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          valid_df: pd.DataFrame, train_stats: dict, 
                          test_stats: dict, valid_stats: dict,
                          plots_folder: Path) -> list:
    """
    Create visualizations for the datasets.
    
    Args:
        train_df, test_df, valid_df: DataFrames for each dataset
        train_stats, test_stats, valid_stats: Statistics dictionaries
        plots_folder: Path to save plots
        
    Returns:
        List of created plot filenames
    """
    plot_files = []
    
    # 1. Sample Matrices Visualization (show first 6 samples from each dataset)
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    datasets = [("Training", train_df), ("Testing (10% noise)", test_df), 
                ("Validation", valid_df)]
    
    for row, (name, df) in enumerate(datasets):
        for col in range(min(6, df.shape[0])):
            matrix = df.iloc[col].values.reshape(12, 12)
            axes[row, col].imshow(matrix, cmap='binary', interpolation='nearest')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(name, fontsize=10)
            if row == 0:
                axes[row, col].set_title(f"Sample {col+1}", fontsize=9)
    
    plt.suptitle("Sample Binary Matrices from Each Dataset", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = plots_folder / "01_sample_matrices.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append("01_sample_matrices.png")
    
    # 2. Ones Proportion Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ['Training', 'Testing', 'Validation']
    ones_props = [train_stats['ones_proportion'], test_stats['ones_proportion'], 
                  valid_stats['ones_proportion']]
    zeros_props = [train_stats['zeros_proportion'], test_stats['zeros_proportion'],
                   valid_stats['zeros_proportion']]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ones_props, width, label='Ones (1)', color='#2ecc71')
    bars2 = ax.bar(x + width/2, zeros_props, width, label='Zeros (0)', color='#3498db')
    
    ax.set_ylabel('Proportion')
    ax.set_title('Distribution of Binary Values Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = plots_folder / "02_binary_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append("02_binary_distribution.png")
    
    # 3. Ones per Sample Distribution
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    stats_list = [train_stats, test_stats, valid_stats]
    colors = ['#e74c3c', '#9b59b6', '#f39c12']
    
    for i, (name, stats, color) in enumerate(zip(names, stats_list, colors)):
        ax = axes[i]
        ax.hist(stats['ones_per_sample'], bins=15, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(stats['ones_per_sample_mean'], color='black', linestyle='--', 
                   label=f'Mean: {stats["ones_per_sample_mean"]:.1f}')
        ax.set_xlabel('Number of Ones per Sample')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Dataset')
        ax.legend()
    
    plt.suptitle("Distribution of Ones per Sample (Character)", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = plots_folder / "03_ones_per_sample.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append("03_ones_per_sample.png")
    
    # 4. Feature Heatmap (Average activation per pixel position)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for i, (name, df) in enumerate(datasets):
        ax = axes[i]
        avg_matrix = df.mean().values.reshape(12, 12)
        im = ax.imshow(avg_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'{name[0] if i < 2 else "Validation"}\nAvg Activation')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle("Average Pixel Activation Across All Samples", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = plots_folder / "04_feature_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append("04_feature_heatmap.png")
    
    # 5. Difference Heatmap (Train vs Test, Train vs Valid)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Train vs Test absolute difference in mean activation
    train_avg = train_df.mean().values.reshape(12, 12)
    test_avg = test_df.mean().values.reshape(12, 12)
    valid_avg = valid_df.mean().values.reshape(12, 12)
    
    diff_test = np.abs(train_avg - test_avg)
    diff_valid = np.abs(train_avg - valid_avg)
    
    im1 = axes[0].imshow(diff_test, cmap='Reds', vmin=0)
    axes[0].set_title('|Training - Testing|\n(10% noise applied)')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(diff_valid, cmap='Reds', vmin=0)
    axes[1].set_title('|Training - Validation|\n(Noise applied)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.suptitle("Absolute Difference in Average Activation", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = plots_folder / "05_difference_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append("05_difference_heatmap.png")
    
    # 6. Feature Entropy Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for name, stats, color in zip(names, stats_list, colors):
        # Compute entropy for each feature
        feature_proportions = stats['ones_per_feature'] / stats['n_samples']
        p = np.clip(feature_proportions, 1e-10, 1 - 1e-10)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        ax.hist(entropy, bins=20, alpha=0.5, label=name, color=color, edgecolor='black')
    
    ax.set_xlabel('Feature Entropy (bits)')
    ax.set_ylabel('Number of Features')
    ax.set_title('Distribution of Feature Entropy Across Datasets')
    ax.legend()
    
    plt.tight_layout()
    plot_path = plots_folder / "06_feature_entropy.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append("06_feature_entropy.png")
    
    # 7. Density Comparison Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    density_data = [train_stats['sample_densities'], 
                    test_stats['sample_densities'], 
                    valid_stats['sample_densities']]
    
    bp = ax.boxplot(density_data, labels=names, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Sample Density (proportion of 1s)')
    ax.set_title('Distribution of Sample Densities Across Datasets')
    
    plt.tight_layout()
    plot_path = plots_folder / "07_density_boxplot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append("07_density_boxplot.png")
    
    return plot_files


def save_stats_csv(train_stats: dict, test_stats: dict, valid_stats: dict,
                   comparison_stats: dict, stats_folder: Path, 
                   seed: int, noise_percent: float, timestamp: str) -> str:
    """
    Save statistics to a CSV file.
    
    Returns:
        Filename of the saved CSV
    """
    # Create a summary DataFrame
    summary_rows = []
    
    for name, stats in [("Training", train_stats), ("Testing", test_stats), 
                        ("Validation", valid_stats)]:
        row = {
            "Dataset": name,
            "N_Samples": stats["n_samples"],
            "N_Features": stats["n_features"],
            "Total_Values": stats["total_values"],
            "Ones_Count": stats["ones_count"],
            "Zeros_Count": stats["zeros_count"],
            "Ones_Proportion": round(stats["ones_proportion"], 4),
            "Zeros_Proportion": round(stats["zeros_proportion"], 4),
            "Ones_Per_Sample_Mean": round(stats["ones_per_sample_mean"], 2),
            "Ones_Per_Sample_Std": round(stats["ones_per_sample_std"], 2),
            "Ones_Per_Sample_Min": stats["ones_per_sample_min"],
            "Ones_Per_Sample_Max": stats["ones_per_sample_max"],
            "Density_Mean": round(stats["density_mean"], 4),
            "Density_Std": round(stats["density_std"], 4),
            "Avg_Feature_Variance": round(stats["avg_feature_variance"], 4),
            "Avg_Feature_Entropy": round(stats["avg_feature_entropy"], 4)
        }
        summary_rows.append(row)
    
    # Add comparison stats as separate rows
    comparison_row = {
        "Dataset": "Comparison",
        "Train_Test_Diff_Percent": round(comparison_stats["train_test_diff_percent"], 2),
        "Train_Valid_Diff_Percent": round(comparison_stats["train_valid_diff_percent"], 2),
        "Expected_Valid_Diff_Percent": comparison_stats["expected_valid_diff_percent"],
        "Train_Test_Feature_Corr": round(comparison_stats["train_test_feature_correlation"], 4),
        "Train_Valid_Feature_Corr": round(comparison_stats["train_valid_feature_correlation"], 4)
    }
    
    summary_df = pd.DataFrame(summary_rows)
    
    filename = f"dataset_stats_{seed}_{noise_percent}_{timestamp}.csv"
    filepath = stats_folder / filename
    summary_df.to_csv(filepath, index=False)
    
    return filename


def generate_markdown_report(train_stats: dict, test_stats: dict, valid_stats: dict,
                             comparison_stats: dict, plot_files: list,
                             stats_folder: Path, seed: int, noise_percent: float, 
                             timestamp: str) -> str:
    """
    Generate a markdown report summarizing the statistics.
    
    Returns:
        Filename of the saved report
    """
    report_lines = []
    
    # Author Header
    header = get_markdown_header("Dataset Statistics Report")
    report_lines.append(header)
    
    # Generation info
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Random Seed:** {seed}")
    report_lines.append(f"**Validation Noise Percentage:** {noise_percent}%")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append("This report presents descriptive statistics for three binary image datasets:")
    report_lines.append("1. **Training Dataset** (`characters.mat`): Original binary character matrices")
    report_lines.append(f"2. **Testing Dataset** (`test.mat`): Training data with noise applied; "
                        f"{comparison_stats['train_test_diff_percent']:.2f}% of pixels differ from it")
    report_lines.append(f"3. **Validation Dataset**: Training data with {noise_percent}% noise applied (seed={seed}); "
                        f"{comparison_stats['train_valid_diff_percent']:.2f}% of pixels differ from it")
    report_lines.append("")
    
    # Dataset Overview
    report_lines.append("## Dataset Overview")
    report_lines.append("")
    report_lines.append("| Metric | Training | Testing | Validation |")
    report_lines.append("|--------|----------|---------|------------|")
    report_lines.append(f"| Samples | {train_stats['n_samples']} | {test_stats['n_samples']} | {valid_stats['n_samples']} |")
    report_lines.append(f"| Features (pixels) | {train_stats['n_features']} | {test_stats['n_features']} | {valid_stats['n_features']} |")
    report_lines.append(f"| Matrix Size | 12×12 | 12×12 | 12×12 |")
    report_lines.append(f"| Total Binary Values | {train_stats['total_values']:,} | {test_stats['total_values']:,} | {valid_stats['total_values']:,} |")
    report_lines.append("")
    
    # Binary Value Distribution
    report_lines.append("## Binary Value Distribution")
    report_lines.append("")
    report_lines.append("| Dataset | Ones Count | Zeros Count | Ones % | Zeros % |")
    report_lines.append("|---------|------------|-------------|--------|---------|")
    for name, stats in [("Training", train_stats), ("Testing", test_stats), ("Validation", valid_stats)]:
        report_lines.append(f"| {name} | {stats['ones_count']:,} | {stats['zeros_count']:,} | "
                           f"{stats['ones_proportion']*100:.2f}% | {stats['zeros_proportion']*100:.2f}% |")
    report_lines.append("")
    
    # Sample-Level Statistics
    report_lines.append("## Sample-Level Statistics")
    report_lines.append("")
    report_lines.append("Statistics on the number of '1' values per sample (character):")
    report_lines.append("")
    report_lines.append("| Dataset | Mean | Std Dev | Min | Max |")
    report_lines.append("|---------|------|---------|-----|-----|")
    for name, stats in [("Training", train_stats), ("Testing", test_stats), ("Validation", valid_stats)]:
        report_lines.append(f"| {name} | {stats['ones_per_sample_mean']:.2f} | "
                           f"{stats['ones_per_sample_std']:.2f} | {stats['ones_per_sample_min']} | "
                           f"{stats['ones_per_sample_max']} |")
    report_lines.append("")
    
    # Density Statistics
    report_lines.append("## Density Statistics")
    report_lines.append("")
    report_lines.append("Sample density represents the proportion of '1' values in each sample:")
    report_lines.append("")
    report_lines.append("| Dataset | Mean Density | Std Dev |")
    report_lines.append("|---------|--------------|---------|")
    for name, stats in [("Training", train_stats), ("Testing", test_stats), ("Validation", valid_stats)]:
        report_lines.append(f"| {name} | {stats['density_mean']:.4f} | {stats['density_std']:.4f} |")
    report_lines.append("")
    
    # Information Theory Metrics
    report_lines.append("## Information Theory Metrics")
    report_lines.append("")
    report_lines.append("| Dataset | Avg Feature Variance | Avg Feature Entropy (bits) |")
    report_lines.append("|---------|---------------------|---------------------------|")
    for name, stats in [("Training", train_stats), ("Testing", test_stats), ("Validation", valid_stats)]:
        report_lines.append(f"| {name} | {stats['avg_feature_variance']:.4f} | "
                           f"{stats['avg_feature_entropy']:.4f} |")
    report_lines.append("")
    
    # Dataset Comparison
    report_lines.append("## Dataset Comparison")
    report_lines.append("")
    report_lines.append("### Noise Analysis")
    report_lines.append("")
    report_lines.append("| Comparison | Differing Values | Difference % | Expected % |")
    report_lines.append("|------------|-----------------|--------------|------------|")
    report_lines.append(f"| Training vs Testing | {comparison_stats['train_test_diff_count']:,} | "
                       f"{comparison_stats['train_test_diff_percent']:.2f}% | ~10% |")
    report_lines.append(f"| Training vs Validation | {comparison_stats['train_valid_diff_count']:,} | "
                       f"{comparison_stats['train_valid_diff_percent']:.2f}% | {noise_percent}% |")
    report_lines.append("")
    
    report_lines.append("### Hamming Distance (per sample)")
    report_lines.append("")
    report_lines.append("| Comparison | Mean | Std Dev |")
    report_lines.append("|------------|------|---------|")
    report_lines.append(f"| Training vs Testing | {comparison_stats['train_test_hamming_mean']:.2f} | "
                       f"{comparison_stats['train_test_hamming_std']:.2f} |")
    report_lines.append(f"| Training vs Validation | {comparison_stats['train_valid_hamming_mean']:.2f} | "
                       f"{comparison_stats['train_valid_hamming_std']:.2f} |")
    report_lines.append("")
    
    report_lines.append("### Feature Activation Correlation")
    report_lines.append("")
    report_lines.append(f"- Training vs Testing: {comparison_stats['train_test_feature_correlation']:.4f}")
    report_lines.append(f"- Training vs Validation: {comparison_stats['train_valid_feature_correlation']:.4f}")
    report_lines.append("")
    
    # Visualizations
    report_lines.append("## Visualizations")
    report_lines.append("")
    report_lines.append("The following plots have been generated in the `plots/` subfolder:")
    report_lines.append("")
    for plot_file in plot_files:
        plot_name = plot_file.replace("_", " ").replace(".png", "").split(" ", 1)[1].title()
        report_lines.append(f"- **{plot_name}**: `plots/{plot_file}`")
        report_lines.append(f"  ![{plot_name}](plots/{plot_file})")
        report_lines.append("")
    
    # Key Findings, computed from the statistics above
    report_lines.append("## Key Findings")
    report_lines.append("")

    noise_gap = comparison_stats['train_valid_diff_percent'] - noise_percent
    report_lines.append(f"1. **Noise:** the test set differs from the training patterns in "
                        f"{comparison_stats['train_test_diff_percent']:.2f}% of pixels "
                        f"({comparison_stats['train_test_hamming_mean']:.2f} per pattern on average, "
                        f"SD {comparison_stats['train_test_hamming_std']:.2f}); the validation set in "
                        f"{comparison_stats['train_valid_diff_percent']:.2f}% "
                        f"({comparison_stats['train_valid_hamming_mean']:.2f} per pattern, "
                        f"SD {comparison_stats['train_valid_hamming_std']:.2f}), "
                        f"{abs(noise_gap):.2f} points {'above' if noise_gap > 0 else 'below'} the "
                        f"{noise_percent}% requested.")
    report_lines.append("")

    report_lines.append("2. **Density:** share of pixels that are 1: "
                        f"{train_stats['ones_proportion']*100:.2f}% (training), "
                        f"{test_stats['ones_proportion']*100:.2f}% (testing), and "
                        f"{valid_stats['ones_proportion']*100:.2f}% (validation). Per training pattern: "
                        f"{train_stats['ones_per_sample_min']} to {train_stats['ones_per_sample_max']} "
                        f"of {train_stats['n_features']} pixels.")
    report_lines.append("")

    report_lines.append("3. **Per-pixel agreement:** correlation between the training set's per-pixel counts "
                        "of 1s and the noisy set's: "
                        f"{comparison_stats['train_test_feature_correlation']:.4f} (testing) and "
                        f"{comparison_stats['train_valid_feature_correlation']:.4f} (validation).")
    report_lines.append("")

    # Footer
    report_lines.append("---")
    report_lines.append(f"*Report generated with seed={seed}, noise={noise_percent}%*")
    
    # Write report
    filename = f"dataset_stats_report_{seed}_{noise_percent}_{timestamp}.md"
    filepath = stats_folder / filename
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return filename


def print_stats_to_console(train_stats: dict, test_stats: dict, valid_stats: dict,
                           comparison_stats: dict, seed: int, noise_percent: float):
    """Print statistics summary to console."""
    print("\n" + "="*70)
    print("DATASET DESCRIPTIVE STATISTICS")
    print("="*70)
    print(f"Random Seed: {seed} | Validation Noise: {noise_percent}%")
    print("-"*70)
    
    # Dataset Overview
    print("\n📊 DATASET OVERVIEW")
    print("-"*40)
    print(f"{'Dataset':<15} {'Samples':<10} {'Features':<10} {'Total Values':<15}")
    for name, stats in [("Training", train_stats), ("Testing", test_stats), ("Validation", valid_stats)]:
        print(f"{name:<15} {stats['n_samples']:<10} {stats['n_features']:<10} {stats['total_values']:<15,}")
    
    # Binary Distribution
    print("\n📈 BINARY VALUE DISTRIBUTION")
    print("-"*40)
    print(f"{'Dataset':<15} {'Ones':<12} {'Zeros':<12} {'Ones %':<10}")
    for name, stats in [("Training", train_stats), ("Testing", test_stats), ("Validation", valid_stats)]:
        print(f"{name:<15} {stats['ones_count']:<12,} {stats['zeros_count']:<12,} "
              f"{stats['ones_proportion']*100:.2f}%")
    
    # Sample Statistics
    print("\n📐 ONES PER SAMPLE STATISTICS")
    print("-"*40)
    print(f"{'Dataset':<15} {'Mean':<10} {'Std':<10} {'Min':<8} {'Max':<8}")
    for name, stats in [("Training", train_stats), ("Testing", test_stats), ("Validation", valid_stats)]:
        print(f"{name:<15} {stats['ones_per_sample_mean']:<10.2f} {stats['ones_per_sample_std']:<10.2f} "
              f"{stats['ones_per_sample_min']:<8} {stats['ones_per_sample_max']:<8}")
    
    # Density
    print("\n🎯 SAMPLE DENSITY (proportion of 1s)")
    print("-"*40)
    print(f"{'Dataset':<15} {'Mean':<12} {'Std Dev':<12}")
    for name, stats in [("Training", train_stats), ("Testing", test_stats), ("Validation", valid_stats)]:
        print(f"{name:<15} {stats['density_mean']:<12.4f} {stats['density_std']:<12.4f}")
    
    # Comparison
    print("\n🔍 DATASET COMPARISON")
    print("-"*40)
    print(f"Training vs Testing:")
    print(f"  - Differing values: {comparison_stats['train_test_diff_count']:,} "
          f"({comparison_stats['train_test_diff_percent']:.2f}%)")
    print(f"  - Hamming distance per sample: {comparison_stats['train_test_hamming_mean']:.2f} ± "
          f"{comparison_stats['train_test_hamming_std']:.2f}")
    
    print(f"Training vs Validation:")
    print(f"  - Differing values: {comparison_stats['train_valid_diff_count']:,} "
          f"({comparison_stats['train_valid_diff_percent']:.2f}%)")
    print(f"  - Expected: {noise_percent}%")
    print(f"  - Hamming distance per sample: {comparison_stats['train_valid_hamming_mean']:.2f} ± "
          f"{comparison_stats['train_valid_hamming_std']:.2f}")
    
    print("\n" + "="*70)


def compute_and_save_statistics(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                 valid_df: pd.DataFrame, seed: int, 
                                 noise_percent: float, verbose: bool = True) -> dict:
    """
    Main function to compute all statistics and save outputs.
    
    Args:
        train_df, test_df, valid_df: DataFrames for each dataset
        seed: Random seed used for validation
        noise_percent: Noise percentage used for validation
        verbose: Whether to print to console
        
    Returns:
        Dictionary with paths to generated files
    """
    paths = get_project_paths()
    
    # Create output folder
    stats_folder, timestamp = create_stats_folder(seed, noise_percent, paths)
    plots_folder = stats_folder / "plots"
    
    if verbose:
        print("\n" + "="*60)
        print("COMPUTING DESCRIPTIVE STATISTICS")
        print("="*60)
        print(f"Output folder: {stats_folder.name}")
    
    # Compute statistics for each dataset
    if verbose:
        print("\n[1/5] Computing training dataset statistics...")
    train_stats = compute_binary_stats(train_df, "Training")
    
    if verbose:
        print("[2/5] Computing testing dataset statistics...")
    test_stats = compute_binary_stats(test_df, "Testing")
    
    if verbose:
        print("[3/5] Computing validation dataset statistics...")
    valid_stats = compute_binary_stats(valid_df, "Validation")
    
    if verbose:
        print("[4/5] Computing comparison statistics...")
    comparison_stats = compute_comparison_stats(train_df, test_df, valid_df, noise_percent)
    
    # Generate visualizations
    if verbose:
        print("[5/5] Generating visualizations...")
    plot_files = create_visualizations(
        train_df, test_df, valid_df,
        train_stats, test_stats, valid_stats,
        plots_folder
    )
    
    # Save CSV
    csv_filename = save_stats_csv(
        train_stats, test_stats, valid_stats, comparison_stats,
        stats_folder, seed, noise_percent, timestamp
    )
    
    # Generate markdown report
    report_filename = generate_markdown_report(
        train_stats, test_stats, valid_stats, comparison_stats,
        plot_files, stats_folder, seed, noise_percent, timestamp
    )
    
    # Print to console
    if verbose:
        print_stats_to_console(train_stats, test_stats, valid_stats, 
                               comparison_stats, seed, noise_percent)
        print(f"\n📁 Output saved to: {stats_folder}")
        print(f"   - Statistics CSV: {csv_filename}")
        print(f"   - Report: {report_filename}")
        print(f"   - Plots: {len(plot_files)} visualizations in plots/")
    
    return {
        "stats_folder": stats_folder,
        "csv_file": stats_folder / csv_filename,
        "report_file": stats_folder / report_filename,
        "plot_files": [plots_folder / pf for pf in plot_files],
        "train_stats": train_stats,
        "test_stats": test_stats,
        "valid_stats": valid_stats,
        "comparison_stats": comparison_stats
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute descriptive statistics for HW2 datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument( # Seed argument
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed used for validation dataset"
    )
    
    parser.add_argument( # Noise percentage argument
        "--noise-percent", "-n",
        type=float,
        default=20.0,
        help="Noise percentage used for validation dataset"
    )
    
    parser.add_argument( # Quiet mode argument
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point when script is run directly."""
    args = parse_args()
    
    # Print console header
    if not args.quiet:
        try:
            from author_info import print_console_header
            print_console_header("dataset_stats.py")
        except ImportError:
            pass
    
    # Import get_data to load datasets
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from get_data import load_all_datasets
    
    # Load datasets (now also returns matrix arrays, which we don't need here)
    train_df, test_df, valid_df, train_data, test_data, valid_data, info = load_all_datasets(
        noise_percent=args.noise_percent,
        random_seed=args.seed,
        verbose=not args.quiet
    )
    
    # Compute and save statistics
    results = compute_and_save_statistics(
        train_df, test_df, valid_df,
        args.seed, args.noise_percent,
        verbose=not args.quiet
    )
    
    return results


if __name__ == "__main__":
    main()
