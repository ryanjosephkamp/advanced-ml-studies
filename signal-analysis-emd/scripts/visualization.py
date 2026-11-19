#!/usr/bin/env python3
"""
visualization.py - Plotting and Visualization

Generates all plots for the HW4 assignment including signal plots,
IMF decompositions, confusion matrices, and clustering visualizations.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 4 – ML for Signal Analysis using EMD
Date: March 8, 2026
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =============================================================================
# SIGNAL VISUALIZATION
# =============================================================================

def plot_representative_signals(signals, labels, label_names, sfreq,
                                 output_path, n_examples=3):
    """Plot example EEG signals per class (Task 1 / Deliverable 2)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    unique_labels = sorted(np.unique(labels))
    n_classes = len(unique_labels)
    n_examples = min(n_examples, min(np.sum(labels == l) for l in unique_labels))
    if n_examples < 1:
        n_examples = 1

    fig, axes = plt.subplots(n_classes, n_examples,
                              figsize=(5 * n_examples, 3 * n_classes),
                              squeeze=False)

    time_axis = np.arange(signals.shape[1]) / sfreq

    for i, lbl in enumerate(unique_labels):
        class_signals = signals[labels == lbl]
        name = label_names.get(int(lbl), f'Class {lbl}')

        for j in range(n_examples):
            ax = axes[i, j]
            ax.plot(time_axis, class_signals[j], linewidth=0.6, color='steelblue')
            ax.set_title(f'{name} — Example {j + 1}', fontsize=9)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.tick_params(labelsize=7)

    plt.suptitle('Representative EEG Signals by Class', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_signal_with_imfs(signal, imfs, residual, sfreq, output_path, title=""):
    """Plot original signal and its IMFs (Task 2 / Deliverable 2)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_imfs = imfs.shape[0]
    n_plots = n_imfs + 2  # original + IMFs + residue
    time_axis = np.arange(len(signal)) / sfreq

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=True)

    # Original signal
    axes[0].plot(time_axis, signal, 'k', linewidth=0.7)
    axes[0].set_ylabel('Original', fontsize=8)
    axes[0].set_title(f'EMD Decomposition{" — " + title if title else ""}', fontsize=11)

    # IMFs
    for k in range(n_imfs):
        axes[k + 1].plot(time_axis, imfs[k], linewidth=0.6, color='steelblue')
        axes[k + 1].set_ylabel(f'IMF {k + 1}', fontsize=8)

    # Residue (the trend left after the last IMF; not an IMF)
    axes[-1].plot(time_axis, residual, 'r', linewidth=0.7)
    axes[-1].set_ylabel('Residue', fontsize=8)
    axes[-1].set_xlabel('Time (s)', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# RESULTS VISUALIZATION
# =============================================================================

def plot_confusion_matrix(cm, label_names, output_path, title=""):
    """Plot confusion matrix heatmap (Deliverable 4)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if cm.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Labels
    if label_names:
        names = [label_names.get(i, f'Class {i}') for i in range(len(cm))]
    else:
        names = [f'Class {i}' for i in range(len(cm))]

    ax.set(xticks=np.arange(len(cm)),
           yticks=np.arange(len(cm)),
           xticklabels=names,
           yticklabels=names,
           ylabel='True Label',
           xlabel='Predicted Label')

    ax.set_title(f'Confusion Matrix{" — " + title if title else ""}')

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_distributions(features, labels, label_names, feature_type,
                                output_path):
    """Plot feature distributions per class."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    unique_labels = sorted(np.unique(labels))
    n_features = features.shape[1]
    n_show = min(n_features, 4)  # Show first 4 IMF features

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4), squeeze=False)

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107']

    for f_idx in range(n_show):
        ax = axes[0, f_idx]
        for c_idx, lbl in enumerate(unique_labels):
            name = label_names.get(int(lbl), f'Class {lbl}')
            vals = features[labels == lbl, f_idx]
            ax.hist(vals, bins=20, alpha=0.5, label=name,
                    color=colors[c_idx % len(colors)])
        ax.set_title(f'{feature_type} — IMF {f_idx + 1}', fontsize=9)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=7)

    plt.suptitle(f'Feature Distribution: {feature_type}', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_distance_matrix_heatmap(dist_matrix, labels, measure_name, output_path):
    """Heatmap of pairwise distance matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sort by label for visual clarity
    sort_idx = np.argsort(labels, kind='stable')
    sorted_matrix = dist_matrix[np.ix_(sort_idx, sort_idx)]

    # Large matrices are block-averaged for display (at most 300 x 300 cells),
    # which keeps the class blocks visible and the file small
    n = sorted_matrix.shape[0]
    note = ''
    if n > 300:
        edges = np.linspace(0, n, 301).astype(int)
        sums = np.add.reduceat(np.add.reduceat(sorted_matrix, edges[:-1], axis=0), edges[:-1], axis=1)
        sizes = np.diff(edges)
        sorted_matrix = sums / np.outer(sizes, sizes)
        note = ' (block-averaged for display)'

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sorted_matrix, cmap='viridis', aspect='auto', interpolation='nearest',
                   extent=(0, n, n, 0))
    ax.figure.colorbar(im, ax=ax, label='Distance')
    ax.set_title(f'Distance Matrix — {measure_name}{note}')
    ax.set_xlabel('Segment Index (sorted by class)')
    ax.set_ylabel('Segment Index (sorted by class)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_clustering_results(features, cluster_labels, true_labels, label_names,
                             output_path, title=""):
    """2D visualization of clustering using PCA."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # PCA to 2D
    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(features)
    else:
        coords = features

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0']

    # Plot 1: Cluster assignments
    for k in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == k
        ax1.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.6,
                    color=colors[k % len(colors)], label=f'Cluster {k}')
    ax1.set_title('K-Means Cluster Assignments')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend(fontsize=8)

    # Plot 2: True labels
    for lbl in sorted(np.unique(true_labels)):
        mask = true_labels == lbl
        name = label_names.get(int(lbl), f'Class {lbl}') if label_names else f'Class {lbl}'
        ax2.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.6,
                    color=colors[int(lbl) % len(colors)], label=name)
    ax2.set_title('True Class Labels')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend(fontsize=8)

    plt.suptitle(f'Clustering Visualization{" — " + title if title else ""}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# COMPARISON FIGURES
# =============================================================================

def generate_comparison_figures(all_results, output_dir):
    """Generate comprehensive cross-dataset comparison figures.

    Creates grouped bar charts and heatmaps for classification F1 scores
    and clustering silhouette scores across all datasets.

    Args:
        all_results: dict of {dataset_key: {dataset_info, experiments}}
        output_dir: Directory to save figures (e.g., comparative_analysis/figures/)

    Returns:
        List of generated figure filenames (relative to output_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    # Collect data into structured form
    ds_keys = list(all_results.keys())
    ds_names = {k: all_results[k].get('dataset_info', {}).get('name', k)
                for k in ds_keys}

    class_data = {}  # {(feat, sim): {ds_key: f1}}
    clust_data = {}  # {(feat, sim): {ds_key: sil}}

    for ds_key, ds_data in all_results.items():
        for config_key, result in ds_data.get('experiments', {}).items():
            feat, sim, task = _parse_config(config_key)
            if not feat:
                continue
            if task == 'classification':
                class_data.setdefault((feat, sim), {})[ds_key] = result.get('f1_score', 0)
            elif task == 'clustering':
                clust_data.setdefault((feat, sim), {})[ds_key] = result.get('silhouette_score', -1)

    if not class_data and not clust_data:
        return generated

    colors = ['#2196F3', '#FF5722', '#4CAF50']

    # --- 1. Classification F1 grouped bar chart ---
    if class_data:
        fig, ax = plt.subplots(figsize=(14, 6))
        configs = sorted(class_data.keys())
        x = np.arange(len(configs))
        width = 0.25
        for i, ds_key in enumerate(ds_keys):
            vals = [class_data[c].get(ds_key, 0) for c in configs]
            ax.bar(x + i * width, vals, width, label=ds_names[ds_key],
                   color=colors[i % len(colors)], alpha=0.85)
        ax.set_xlabel('Feature + Similarity', fontsize=10)
        ax.set_ylabel('F1 Score (Weighted)', fontsize=10)
        ax.set_title('Classification F1 Score — Cross-Dataset Comparison', fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{f}\n{s}' for f, s in configs], fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fname = 'classification_f1_comparison.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        generated.append(fname)

    # --- 2. Clustering silhouette grouped bar chart ---
    if clust_data:
        fig, ax = plt.subplots(figsize=(14, 6))
        configs = sorted(clust_data.keys())
        x = np.arange(len(configs))
        width = 0.25
        for i, ds_key in enumerate(ds_keys):
            vals = [clust_data[c].get(ds_key, -1) for c in configs]
            ax.bar(x + i * width, vals, width, label=ds_names[ds_key],
                   color=colors[i % len(colors)], alpha=0.85)
        ax.set_xlabel('Feature + Similarity', fontsize=10)
        ax.set_ylabel('Silhouette Score', fontsize=10)
        ax.set_title('Clustering Silhouette Score — Cross-Dataset Comparison', fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{f}\n{s}' for f, s in configs], fontsize=8)
        ax.legend(fontsize=9)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fname = 'clustering_silhouette_comparison.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        generated.append(fname)

    # --- 3. Per-dataset classification F1 heatmaps ---
    features_list = sorted(set(f for f, s in class_data.keys()))
    sims_list = sorted(set(s for f, s in class_data.keys()))

    for ds_key in ds_keys:
        if class_data:
            fig, ax = plt.subplots(figsize=(8, 5))
            matrix = np.zeros((len(features_list), len(sims_list)))
            for i, feat in enumerate(features_list):
                for j, sim in enumerate(sims_list):
                    matrix[i, j] = class_data.get((feat, sim), {}).get(ds_key, 0)
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(sims_list)))
            ax.set_xticklabels(sims_list, fontsize=9)
            ax.set_yticks(range(len(features_list)))
            ax.set_yticklabels(features_list, fontsize=9)
            ax.set_title(f'Classification F1 — {ds_names[ds_key]}', fontsize=11)
            fig.colorbar(im, ax=ax, label='F1 Score')
            for i in range(len(features_list)):
                for j in range(len(sims_list)):
                    ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                            fontsize=9, color='white' if matrix[i, j] > matrix.max() * 0.6 else 'black')
            plt.tight_layout()
            fname = f'classification_f1_heatmap_{ds_key}.png'
            plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()
            generated.append(fname)

    # --- 4. Per-dataset clustering silhouette heatmaps ---
    for ds_key in ds_keys:
        if clust_data:
            fig, ax = plt.subplots(figsize=(8, 5))
            matrix = np.zeros((len(features_list), len(sims_list)))
            for i, feat in enumerate(features_list):
                for j, sim in enumerate(sims_list):
                    matrix[i, j] = clust_data.get((feat, sim), {}).get(ds_key, -1)
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1.0)
            ax.set_xticks(range(len(sims_list)))
            ax.set_xticklabels(sims_list, fontsize=9)
            ax.set_yticks(range(len(features_list)))
            ax.set_yticklabels(features_list, fontsize=9)
            ax.set_title(f'Clustering Silhouette — {ds_names[ds_key]}', fontsize=11)
            fig.colorbar(im, ax=ax, label='Silhouette Score')
            for i in range(len(features_list)):
                for j in range(len(sims_list)):
                    ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                            fontsize=9, color='white' if matrix[i, j] > 0.5 else 'black')
            plt.tight_layout()
            fname = f'clustering_silhouette_heatmap_{ds_key}.png'
            plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()
            generated.append(fname)

    # --- 5. Best results summary bar chart ---
    if class_data and clust_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        # Best F1 per dataset
        best_f1 = {}
        best_f1_config = {}
        for (feat, sim), ds_vals in class_data.items():
            for ds_key, val in ds_vals.items():
                if ds_key not in best_f1 or val > best_f1[ds_key]:
                    best_f1[ds_key] = val
                    best_f1_config[ds_key] = f'{feat}+{sim}'
        names_ordered = [ds_names[k] for k in ds_keys]
        f1_vals = [best_f1.get(k, 0) for k in ds_keys]
        bars = ax1.bar(names_ordered, f1_vals, color=colors[:len(ds_keys)], alpha=0.85)
        for bar, val, cfg in zip(bars, f1_vals, [best_f1_config.get(k, '') for k in ds_keys]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}\n({cfg})', ha='center', va='bottom', fontsize=8)
        ax1.set_ylabel('F1 Score', fontsize=10)
        ax1.set_title('Best Classification F1 per Dataset', fontsize=11)
        ax1.set_ylim(0, max(f1_vals) * 1.25 if f1_vals else 1)

        # Best silhouette per dataset
        best_sil = {}
        best_sil_config = {}
        for (feat, sim), ds_vals in clust_data.items():
            for ds_key, val in ds_vals.items():
                if ds_key not in best_sil or val > best_sil[ds_key]:
                    best_sil[ds_key] = val
                    best_sil_config[ds_key] = f'{feat}+{sim}'
        sil_vals = [best_sil.get(k, -1) for k in ds_keys]
        bars = ax2.bar(names_ordered, sil_vals, color=colors[:len(ds_keys)], alpha=0.85)
        for bar, val, cfg in zip(bars, sil_vals, [best_sil_config.get(k, '') for k in ds_keys]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}\n({cfg})', ha='center', va='bottom', fontsize=8)
        ax2.set_ylabel('Silhouette Score', fontsize=10)
        ax2.set_title('Best Clustering Silhouette per Dataset', fontsize=11)
        ax2.set_ylim(min(0, min(sil_vals) - 0.1) if sil_vals else -0.5,
                     max(sil_vals) * 1.2 if sil_vals else 1)

        plt.suptitle('Best Results Summary — Cross-Dataset', fontsize=13)
        plt.tight_layout()
        fname = 'best_results_summary.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        generated.append(fname)

    return generated


def _parse_config(config_key):
    """Parse feature_similarity_task from config_key."""
    for t in ('classification', 'clustering'):
        if config_key.endswith('_' + t):
            remainder = config_key[:-(len(t) + 1)]
            for f in ('inst_freq', 'variance', 'entropy'):
                if remainder.startswith(f + '_'):
                    return f, remainder[len(f) + 1:], t
    return '', '', ''


def copy_figures_for_reports(all_results, run_dir):
    """Copy relevant figures to a shared figures/ folder in the run directory.

    Creates run_dir/figures/ with representative images from each dataset
    for use by methodology_report, experiment_summary, and similarity_description.

    Returns:
        dict mapping figure keys to relative paths (e.g. 'figures/eeg_ma_signals.png')
    """
    figures_dir = os.path.join(run_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    figure_map = {}

    for ds_key in all_results:
        ds_dir = os.path.join(run_dir, ds_key)
        signals_dir = os.path.join(ds_dir, 'signals')

        # Representative signals
        src = os.path.join(signals_dir, 'representative_signals.png')
        if os.path.exists(src):
            dst_name = f'{ds_key}_representative_signals.png'
            shutil.copy2(src, os.path.join(figures_dir, dst_name))
            figure_map[f'{ds_key}_signals'] = f'figures/{dst_name}'

        # IMF decomposition (first example)
        src = os.path.join(signals_dir, 'signal_with_imfs_0.png')
        if os.path.exists(src):
            dst_name = f'{ds_key}_imf_decomposition.png'
            shutil.copy2(src, os.path.join(figures_dir, dst_name))
            figure_map[f'{ds_key}_imfs'] = f'figures/{dst_name}'

        # Find best classification config and copy confusion matrix + distance matrix
        experiments = all_results[ds_key].get('experiments', {})
        best_f1, best_class_config = -1, None
        best_sil, best_clust_config = -2, None
        for config_key, result in experiments.items():
            feat, sim, task = _parse_config(config_key)
            if task == 'classification' and result.get('f1_score', -1) > best_f1:
                best_f1 = result['f1_score']
                best_class_config = config_key
            elif task == 'clustering' and result.get('silhouette_score', -2) > best_sil:
                best_sil = result['silhouette_score']
                best_clust_config = config_key

        if best_class_config:
            exp_dir = os.path.join(ds_dir, best_class_config)
            for fname, key_suffix in [('confusion_matrix.png', 'cm'),
                                       ('distance_matrix.png', 'class_dist'),
                                       ('feature_distributions.png', 'class_features')]:
                src = os.path.join(exp_dir, fname)
                if os.path.exists(src):
                    dst_name = f'{ds_key}_best_{key_suffix}.png'
                    shutil.copy2(src, os.path.join(figures_dir, dst_name))
                    figure_map[f'{ds_key}_{key_suffix}'] = f'figures/{dst_name}'

        if best_clust_config:
            exp_dir = os.path.join(ds_dir, best_clust_config)
            for fname, key_suffix in [('clustering_results.png', 'clustering'),
                                       ('distance_matrix.png', 'clust_dist')]:
                src = os.path.join(exp_dir, fname)
                if os.path.exists(src):
                    dst_name = f'{ds_key}_best_{key_suffix}.png'
                    shutil.copy2(src, os.path.join(figures_dir, dst_name))
                    figure_map[f'{ds_key}_{key_suffix}'] = f'figures/{dst_name}'

        # Copy one distance matrix per similarity type for similarity_description
        for sim_type in ('pearson', 'euclidean', 'itakura_saito'):
            for config_key in experiments:
                if sim_type in config_key and 'classification' in config_key:
                    src = os.path.join(ds_dir, config_key, 'distance_matrix.png')
                    if os.path.exists(src):
                        dst_name = f'{sim_type}_distance_example_{ds_key}.png'
                        shutil.copy2(src, os.path.join(figures_dir, dst_name))
                        figure_map[f'{sim_type}_dist_{ds_key}'] = f'figures/{dst_name}'
                    break

    return figure_map
