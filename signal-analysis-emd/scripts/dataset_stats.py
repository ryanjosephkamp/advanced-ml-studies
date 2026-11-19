#!/usr/bin/env python3
"""
dataset_stats.py - Dataset Statistics Computation and Reporting

Computes and saves descriptive statistics for each EEG dataset including
signal statistics, class distribution, and summary tables.

Generates and saves console output, CSV files, and markdown reports with embedded figures.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 4 – ML for Signal Analysis using EMD
Date: March 8, 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def compute_and_save_statistics(dataset_dict, output_dir, verbose=True):
    """
    Compute and save descriptive statistics for a loaded dataset.

    Args:
        dataset_dict: Dictionary from load_dataset()
        output_dir: Directory to save stats (e.g., dataset_stats/eeg_mmi/)
        verbose: Print stats to console

    Returns:
        dict with computed statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract relevant data from dataset_dict:
    signals = dataset_dict['signals'] # Main data array of shape (n_segments, n_samples)
    labels = dataset_dict['labels'] # Class labels for each segment
    label_names = dataset_dict['label_names'] # Mapping from label integers to class names
    dataset_name = dataset_dict['dataset_name'] # Human-readable name of the dataset
    dataset_key = dataset_dict['dataset_key'] # Unique key identifier for the dataset

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Dataset Statistics: {dataset_name}")
        print(f"{'─' * 60}")

    stats = {}

    # ---- Dataset metadata ----
    # This section compiles key metadata about the dataset, such as the number of subjects, channels, sampling frequency, and segment information:
    stats['metadata'] = {
        'dataset_name': dataset_name, # Human-readable name of the dataset (e.g., "EEG Motor Movement/Imagery Dataset")
        'dataset_key': dataset_key, # Unique key identifier for the dataset (e.g., "eeg_mmi")
        'source_url': dataset_dict['source_url'], # Original URL where the dataset was obtained
        'n_subjects': dataset_dict['n_subjects'], # Total number of subjects included in the dataset
        'n_channels': len(dataset_dict['channel_names']), # Number of EEG channels recorded
        'channel_names': dataset_dict['channel_names'], # List of channel names (e.g., ['C3', 'C4', 'Pz'])
        'sampling_frequency': dataset_dict['sfreq'], # Sampling frequency in Hz (e.g., 160 Hz)
        'n_segments': len(signals), # Total number of segments extracted from the raw data
        'samples_per_segment': signals.shape[1] if len(signals) > 0 else 0, # Number of samples in each segment (e.g., 1600 samples for 10-second segments at 160 Hz)
        'window_duration_sec': signals.shape[1] / dataset_dict['sfreq'] if len(signals) > 0 else 0, # Duration of each segment in seconds (e.g., 10 seconds)
    }

    # ---- Signal statistics ----
    # This section computes global statistics across all segments and samples, as well as ranges of per-segment means and standard deviations:
    stats['signal_stats'] = {
        'global_mean': float(np.mean(signals)), # Overall mean amplitude across all segments and samples
        'global_std': float(np.std(signals)), # Overall standard deviation of amplitude across all segments and samples
        'global_min': float(np.min(signals)), # Minimum amplitude value across all segments and samples
        'global_max': float(np.max(signals)), # Maximum amplitude value across all segments and samples
        'global_median': float(np.median(signals)), # Median amplitude value across all segments and samples
        'per_segment_mean_range': [float(np.min(np.mean(signals, axis=1))), # Minimum of the mean amplitude across segments
                                    float(np.max(np.mean(signals, axis=1)))], # Maximum of the mean amplitude across segments
        'per_segment_std_range': [float(np.min(np.std(signals, axis=1))), # Minimum of the standard deviation across segments
                                   float(np.max(np.std(signals, axis=1)))], # Maximum of the standard deviation across segments
    }

    # ---- Class distribution ----
    # This section calculates the distribution of class labels in the dataset, including both counts and percentages for each class:
    unique_labels, counts = np.unique(labels, return_counts=True) # Unique class labels and their corresponding counts
    class_dist = {}
    # Iterate over each unique label and its count, mapping the label to a human-readable class name using label_names. 
    # Store both the count and percentage of segments for each class in the class_dist dictionary:
    for lbl, cnt in zip(unique_labels, counts):
        name = label_names.get(int(lbl), f'class_{lbl}')
        class_dist[name] = {'count': int(cnt), 'percentage': float(cnt / len(labels) * 100)}
    stats['class_distribution'] = class_dist

    # ---- Print to console ----
    if verbose:
        _print_stats(stats)

    # ---- Save CSV ----
    _save_stats_csv(stats, output_dir, dataset_key)

    # ---- Save plots ----
    _save_stats_plots(signals, labels, label_names, dataset_name, output_dir)

    # ---- Save markdown report ----
    _save_stats_markdown(stats, dataset_name, output_dir, dataset_key)

    if verbose:
        print(f"  Statistics saved to: {os.path.join(os.path.basename(os.path.dirname(output_dir)), os.path.basename(output_dir))}")

    return stats


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def _print_stats(stats):
    """Print statistics to console in tabular format."""
    meta = stats['metadata']
    sig = stats['signal_stats']

    # Metadata table
    meta_table = [
        ['Dataset', meta['dataset_name']],
        ['Source URL', meta['source_url']],
        ['Subjects Loaded', meta['n_subjects']],
        ['Channels', meta['n_channels']],
        ['Sampling Frequency', f"{meta['sampling_frequency']} Hz"],
        ['Total Segments', meta['n_segments']],
        ['Samples per Segment', meta['samples_per_segment']],
        ['Window Duration', f"{meta['window_duration_sec']:.2f} sec"],
    ]
    print(f"\n  {tabulate(meta_table, headers=['Property', 'Value'], tablefmt='grid')}")

    # Signal stats table
    sig_table = [
        ['Mean', f"{sig['global_mean']:.6f}"],
        ['Std Dev', f"{sig['global_std']:.6f}"],
        ['Min', f"{sig['global_min']:.6f}"],
        ['Max', f"{sig['global_max']:.6f}"],
        ['Median', f"{sig['global_median']:.6f}"],
    ]
    print(f"\n  Signal Statistics:")
    print(f"  {tabulate(sig_table, headers=['Statistic', 'Value'], tablefmt='grid')}")

    # Class distribution table
    class_table = []
    for name, info in stats['class_distribution'].items():
        class_table.append([name, info['count'], f"{info['percentage']:.1f}%"])
    print(f"\n  Class Distribution:")
    print(f"  {tabulate(class_table, headers=['Class', 'Count', 'Percentage'], tablefmt='grid')}")


# =============================================================================
# FILE OUTPUTS
# =============================================================================

def _save_stats_csv(stats, output_dir, dataset_key):
    """Save statistics to CSV."""
    rows = []
    for key, val in stats['signal_stats'].items():
        rows.append({'statistic': key, 'value': str(val)})
    for name, info in stats['class_distribution'].items():
        rows.append({'statistic': f'class_{name}_count', 'value': str(info['count'])})
        rows.append({'statistic': f'class_{name}_pct', 'value': f"{info['percentage']:.1f}"})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f'{dataset_key}_summary_statistics.csv'), index=False)


def _save_stats_plots(signals, labels, label_names, dataset_name, output_dir):
    """Generate and save statistical plots."""
    # 1. Class distribution bar chart
    unique_labels, counts = np.unique(labels, return_counts=True)
    names = [label_names.get(int(l), f'class_{l}') for l in unique_labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, counts, color=['#2196F3', '#FF5722', '#4CAF50', '#FFC107'][:len(names)])
    ax.set_title(f'{dataset_name} — Class Distribution')
    ax.set_ylabel('Number of Segments')
    ax.set_xlabel('Class')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
    plt.close()

    # 2. Signal amplitude distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = signals.flatten()
    if len(sample) > 100000:
        sample = np.random.choice(sample, 100000, replace=False)
    ax.hist(sample, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.set_title(f'{dataset_name} — Signal Amplitude Distribution')
    ax.set_xlabel('Amplitude (z-scored)')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'amplitude_distribution.png'), dpi=150)
    plt.close()

    # 3. Example signals per class
    n_examples = min(3, len(signals))
    n_classes = len(np.unique(labels))
    fig, axes = plt.subplots(n_classes, n_examples, figsize=(15, 3 * n_classes),
                              squeeze=False)
    for i, lbl in enumerate(sorted(np.unique(labels))):
        class_signals = signals[labels == lbl]
        name = label_names.get(int(lbl), f'class_{lbl}')
        for j in range(min(n_examples, len(class_signals))):
            axes[i, j].plot(class_signals[j], linewidth=0.5, color='steelblue')
            axes[i, j].set_title(f'{name} — Example {j + 1}', fontsize=9)
            axes[i, j].set_xlabel('Sample')
            axes[i, j].set_ylabel('Amplitude')
    plt.suptitle(f'{dataset_name} — Example Signals by Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_signals.png'), dpi=150)
    plt.close()


def _save_stats_markdown(stats, dataset_name, output_dir, dataset_key):
    """Save statistics as markdown report."""
    meta = stats['metadata']
    sig = stats['signal_stats']

    md = f"# Dataset Statistics: {dataset_name}\n\n"
    md += "## Metadata\n\n"
    md += f"| Property | Value |\n|---|---|\n"
    md += f"| Dataset | {meta['dataset_name']} |\n"
    md += f"| Source | {meta['source_url']} |\n"
    md += f"| Subjects | {meta['n_subjects']} |\n"
    md += f"| Channels | {meta['n_channels']} ({', '.join(meta['channel_names'])}) |\n"
    md += f"| Sampling Rate | {meta['sampling_frequency']} Hz |\n"
    md += f"| Segments | {meta['n_segments']} |\n"
    md += f"| Samples/Segment | {meta['samples_per_segment']} |\n"
    md += f"| Window Duration | {meta['window_duration_sec']:.2f} sec |\n\n"

    md += "## Signal Statistics\n\n"
    md += f"| Statistic | Value |\n|---|---|\n"
    md += f"| Mean | {sig['global_mean']:.6f} |\n"
    md += f"| Std Dev | {sig['global_std']:.6f} |\n"
    md += f"| Min | {sig['global_min']:.6f} |\n"
    md += f"| Max | {sig['global_max']:.6f} |\n"
    md += f"| Median | {sig['global_median']:.6f} |\n\n"

    md += "## Class Distribution\n\n"
    md += f"| Class | Count | Percentage |\n|---|---|---|\n"
    for name, info in stats['class_distribution'].items():
        md += f"| {name} | {info['count']} | {info['percentage']:.1f}% |\n"

    md += "\n## Figures\n\n"
    md += "![Class Distribution](class_distribution.png)\n\n"
    md += "![Amplitude Distribution](amplitude_distribution.png)\n\n"
    md += "![Example Signals](example_signals.png)\n\n"

    with open(os.path.join(output_dir, f'{dataset_key}_statistics_report.md'), 'w') as f:
        f.write(md)
