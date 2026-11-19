#!/usr/bin/env python3
"""
report_generator.py - Automated Report Generation

Automatically generates methodology reports, similarity measure descriptions, and
experiment summary reports in markdown format.

This module enables the creation of markdown reports that summarize the methodology, describe the similarity measures used, 
and present the results of the experiments conducted on the EEG datasets, *for the specific experiment run chosen by the user 
via command-line arguments*.
By automatically generating these reports, we ensure that the documentation of the experiment is consistent, comprehensive, 
and directly tied to the actual results obtained.

The reports are saved in the specified output directory and include embedded figures from the experiment results.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 4 – ML for Signal Analysis using EMD
Date: March 8, 2026
"""

import os
import json
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from author_info import get_markdown_header, get_markdown_footer


# =============================================================================
# METHODOLOGY REPORT (Deliverable 1)
# =============================================================================

def generate_methodology_report(all_results, output_dir, figure_map=None):
    """
    Generate brief methodology markdown report (Deliverable 1).

    Args:
        all_results: dict of {dataset_key: dataset_results}
        output_dir: Directory to save the report
        figure_map: dict of figure keys to relative paths (optional)

    Returns:
        Path to the generated report
    """
    if figure_map is None:
        figure_map = {}
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'methodology_report.md')

    md = get_markdown_header("Methodology Report — ML for Signal Analysis using EMD")

    md += "## 1. Overview\n\n"
    md += ("This report describes the methodology used for the EMD-based "
           "signal analysis pipeline applied to EEG datasets. The pipeline "
           "follows five sequential tasks: signal preprocessing, EMD decomposition, "
           "feature extraction, similarity computation, and learning tasks.\n\n")

    md += '<div style="page-break-after: always;"></div>\n\n'

    md += "## 2. Datasets\n\n"
    for ds_key, ds_results in all_results.items():
        info = ds_results.get('dataset_info', {})
        md += f"### {info.get('name', ds_key)}\n\n"
        md += f"- **Source:** {info.get('source_url', 'N/A')}\n"
        md += f"- **Subjects loaded:** {info.get('n_subjects', 'N/A')}\n"
        md += f"- **Channels:** {', '.join(info.get('channel_names', []))}\n"
        md += f"- **Sampling rate:** {info.get('sfreq', 'N/A')} Hz\n"
        md += f"- **Total segments:** {info.get('n_segments', 'N/A')}\n\n"
        # Embed representative signals for this dataset
        sig_key = f'{ds_key}_signals'
        if sig_key in figure_map:
            md += f"![Representative Signals — {info.get('name', ds_key)}]({figure_map[sig_key]})\n\n"
    
    md += '<div style="page-break-after: always;"></div>\n\n'

    md += "## 3. Preprocessing (Task 1)\n\n"
    md += ("- **Channel selection:** Motor cortex channels (C3, Cz, C4) for motor imagery "
           "datasets; frontal channels (Fp1, Fp2, F3, F4, Fz) for mental arithmetic.\n"
           "- **Segmentation:** Fixed-length windows (2.0 seconds) starting at each cue "
           "(EEG MMI, BCI CIV 2a) or consecutive non-overlapping windows (EEG MA).\n"
           "- **Quality checks:** EEG MMI files not sampled at 160 Hz are skipped; windows "
           "in which any channel is flat for 0.1 s or longer (clipping, or the padded end of "
           "each EEG MA recording) are dropped.\n"
           "- **Normalization:** Z-score normalization per segment to ensure zero mean "
           "and unit variance.\n"
           "- **Channel averaging:** Signals from selected channels averaged into "
           "a single time series per segment.\n\n")

    md += "## 4. EMD Decomposition (Task 2)\n\n"
    md += ("- **Method:** Empirical Mode Decomposition (EMD) using the PyEMD library.\n"
           "- **IMFs used:** The first n IMFs of every segment (default: 4), so each feature "
           "position always describes the same IMF; segments with fewer IMFs are skipped. "
           "No zero padding.\n"
           "- **Residue:** The trend left after the last IMF, kept separately and not used "
           "as a feature.\n\n")

    # Embed IMF decomposition examples
    for ds_key in all_results:
        imf_key = f'{ds_key}_imfs'
        if imf_key in figure_map:
            ds_name = all_results[ds_key].get('dataset_info', {}).get('name', ds_key)
            md += f"![EMD Decomposition Example — {ds_name}]({figure_map[imf_key]})\n\n"

    md += "## 5. Feature Extraction (Task 3)\n\n"
    md += "Three feature types are computed from each IMF:\n\n"
    md += ("1. **Variance:** Measures the spread/power of each IMF. "
           "Computed as `np.var(imf)`.\n")
    md += ("2. **Entropy:** Shannon entropy of a 50-bin histogram of each IMF's "
           "amplitudes (the shape of the amplitude distribution, not the spectrum).\n")
    md += ("3. **Instantaneous Frequency:** Mean absolute instantaneous frequency "
           "derived from the Hilbert transform of each IMF.\n\n")
    md += ("Each feature type produces a vector of length `n_imfs` per segment.\n\n")

    md += "## 6. Similarity Measures (Task 4)\n\n"
    md += "Three distance/similarity measures are implemented:\n\n"
    md += "1. **Pearson Correlation Distance:** $d = 1 - \\rho$, range [0, 2].\n"
    md += "2. **Euclidean Distance:** Standard L2 norm between feature vectors.\n"
    md += ("3. **Itakura–Saito Distance:** Symmetrized divergence measure, "
           "originally from speech processing. Requires strictly positive inputs.\n\n")

    # Embed example distance matrices (one per similarity type, using first dataset)
    first_ds = list(all_results.keys())[0] if all_results else None
    if first_ds:
        for sim_type, sim_label in [('pearson', 'Pearson Correlation'),
                                     ('euclidean', 'Euclidean'),
                                     ('itakura_saito', 'Itakura–Saito')]:
            dist_key = f'{sim_type}_dist_{first_ds}'
            if dist_key in figure_map:
                md += f"![{sim_label} Distance Matrix Example]({figure_map[dist_key]})\n\n"

    md += '<div style="page-break-after: always;"></div>\n\n'

    md += "## 7. Learning Tasks (Task 5)\n\n"
    md += "### Classification\n"
    md += ("- **Algorithm:** K-Nearest Neighbors (KNN) with `metric='precomputed'`\n"
           "- **Parameters:** k=5, 70/30 train/test split with stratification\n"
           "- **Metrics:** Weighted F1-score, accuracy, balanced accuracy, macro F1, "
           "confusion matrix\n\n")
    md += "### Clustering\n"
    md += ("- **Algorithm:** K-Means on feature vectors\n"
           "- **Parameters:** n_clusters = number of true classes, n_init=10\n"
           "- **Metrics:** Silhouette score under each distance matrix (cluster separation), "
           "and the adjusted Rand index and normalized mutual information against the true "
           "labels (class agreement). K-Means itself does not use the distance measure.\n\n")

    md += get_markdown_footer()

    with open(path, 'w') as f:
        f.write(md)

    return path


# =============================================================================
# SIMILARITY DESCRIPTION (Deliverable 3)
# =============================================================================

def generate_similarity_description(output_dir, figure_map=None, all_results=None):
    """
    Generate similarity measure description (Deliverable 3).

    Args:
        output_dir: Directory to save the report
        figure_map: dict of figure keys to relative paths (optional)
        all_results: dict of {dataset_key: dataset_results} for display names

    Returns:
        Path to the generated report
    """
    if figure_map is None:
        figure_map = {}
    if all_results is None:
        all_results = {}
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'similarity_description.md')

    # Build dataset display name mapping
    ds_display = {}
    for k, v in all_results.items():
        ds_display[k] = v.get('dataset_info', {}).get('name', k)

    def _get_ds_name(ds_key):
        return ds_display.get(ds_key, ds_key.replace('_', ' ').upper())

    # Ordered list of dataset keys referenced in distance matrix figures
    dist_ds_keys = sorted(set(
        k.replace('pearson_dist_', '')
        for k in figure_map if k.startswith('pearson_dist_')
    ))

    md = get_markdown_header("Similarity Measures Description")

    md += "## Overview\n\n"
    md += ("Three distance/similarity measures are used to quantify the similarity "
           "between EEG signals based on their IMF-derived feature vectors. Each "
           "measure produces a pairwise distance matrix used for classification "
           "(KNN) and clustering evaluation (silhouette score).\n\n")
    if dist_ds_keys:
        md += ("The distance matrix visualizations shown below are computed for "
               "each of the three EEG datasets used in this experiment. "
               "Each heatmap shows the pairwise distances between all signal "
               "segments in a given dataset, sorted by class label. Darker "
               "colors (purple) indicate smaller distances (more similar), "
               "while brighter colors (yellow) indicate larger distances "
               "(more dissimilar). The datasets are:\n\n")
        for ds_key in dist_ds_keys:
            md += f"- **{_get_ds_name(ds_key)}** (`{ds_key}`)\n"
        md += "\n"

    md += "## 1. Pearson Correlation Distance\n\n"
    md += "### Definition\n\n"
    md += ("The Pearson correlation coefficient measures the linear relationship "
           "between two feature vectors $\\mathbf{x}$ and $\\mathbf{y}$:\n\n")
    md += ("$$\\rho_{xy} = \\frac{\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})}"
           "{\\sqrt{\\sum_{i=1}^{n}(x_i - \\bar{x})^2 \\cdot \\sum_{i=1}^{n}(y_i - \\bar{y})^2}}$$\n\n")
    md += "Converted to distance: $d = 1 - \\rho$\n\n"
    md += "### Rationale\n\n"
    md += ("Pearson correlation captures the *shape similarity* between IMF feature "
           "profiles regardless of absolute magnitude. Two signals with similar "
           "IMF energy distributions (even at different scales) will have high "
           "correlation and thus low distance. Range: [0, 2].\n\n")

    # Embed Pearson distance matrix examples with explanatory text
    pearson_keys = [k for k in sorted(figure_map) if k.startswith('pearson_dist_')]
    if pearson_keys:
        md += ("The following distance matrices show the Pearson correlation distance "
               "between all pairs of signal segments for each dataset. Each matrix "
               "corresponds to one of the three datasets:\n\n")
        for ds_key in pearson_keys:
            ds_id = ds_key.replace('pearson_dist_', '')
            md += f"![Pearson Distance Matrix — {_get_ds_name(ds_id)}]({figure_map[ds_key]})\n\n"
    
    md += '<div style="page-break-after: always;"></div>\n\n'
    
    md += "## 2. Euclidean Distance\n\n"
    md += "### Definition\n\n"
    md += ("$$d_{\\text{Euc}}(\\mathbf{x}, \\mathbf{y}) = "
           "\\sqrt{\\sum_{i=1}^{n}(x_i - y_i)^2}$$\n\n")
    md += "### Rationale\n\n"
    md += ("Euclidean distance measures the absolute difference in feature space. "
           "It is sensitive to both shape and magnitude differences. Signals with "
           "similar IMF variance/entropy/frequency profiles across all IMFs will "
           "have small Euclidean distance.\n\n")

    # Embed Euclidean distance matrix examples with explanatory text
    euclidean_keys = [k for k in sorted(figure_map) if k.startswith('euclidean_dist_')]
    if euclidean_keys:
        md += ("The following distance matrices show the Euclidean distance "
               "between all pairs of signal segments for each dataset:\n\n")
        for ds_key in euclidean_keys:
            ds_id = ds_key.replace('euclidean_dist_', '')
            md += f"![Euclidean Distance Matrix — {_get_ds_name(ds_id)}]({figure_map[ds_key]})\n\n"
    
    md += '<div style="page-break-after: always;"></div>\n\n'

    md += "## 3. Itakura–Saito Distance\n\n"
    md += "### Definition\n\n"
    md += ("The Itakura–Saito divergence from $\\mathbf{x}$ to $\\mathbf{y}$:\n\n")
    md += ("$$D_{IS}(\\mathbf{x} \\| \\mathbf{y}) = "
           "\\sum_{i=1}^{n}\\left[\\frac{x_i}{y_i} - \\log\\frac{x_i}{y_i} - 1\\right]$$\n\n")
    md += "Symmetrized version used as the distance:\n\n"
    md += ("$$d_{IS}(\\mathbf{x}, \\mathbf{y}) = "
           "\\frac{D_{IS}(\\mathbf{x} \\| \\mathbf{y}) + D_{IS}(\\mathbf{y} \\| \\mathbf{x})}{2}$$\n\n")
    md += "### Rationale\n\n"
    md += ("The Itakura–Saito distance originates from speech/audio signal processing "
           "and is particularly suited for comparing spectral or energy-based features. "
           "It is scale-dependent and penalizes underestimation more heavily than "
           "overestimation, making it sensitive to the *ratio* between corresponding "
           "feature values rather than their absolute difference.\n\n")
    md += ("**Positivity handling:** The Itakura–Saito divergence requires strictly "
           "positive inputs. A zero-valued feature (for example the entropy of a constant "
           "IMF) is replaced with the smallest non-zero value in that feature dimension. "
           "In the symmetrized form the log terms cancel, leaving "
           "$\\frac{1}{2}\\sum_i (x_i/y_i + y_i/x_i - 2)$. The divergence is defined for "
           "power-like vectors such as IMF variances; on entropies and frequencies it is "
           "computable but has no spectral meaning.\n\n")

    # Embed Itakura-Saito distance matrix examples with explanatory text
    is_keys = [k for k in sorted(figure_map) if k.startswith('itakura_saito_dist_')]
    if is_keys:
        md += ("The following distance matrices show the symmetrized Itakura–Saito "
               "distance between all pairs of signal segments for each dataset:\n\n")
        for ds_key in is_keys:
            ds_id = ds_key.replace('itakura_saito_dist_', '')
            md += f"![Itakura–Saito Distance Matrix — {_get_ds_name(ds_id)}]({figure_map[ds_key]})\n\n"
    
    md += '<div style="page-break-after: always;"></div>\n\n'

    md += "## Comparison\n\n"
    md += "| Property | Pearson | Euclidean | Itakura–Saito |\n"
    md += "|---|---|---|---|\n"
    md += "| Scale invariant | Yes | No | No |\n"
    md += "| Range | [0, 2] | [0, ∞) | [0, ∞) |\n"
    md += "| Symmetric | Yes | Yes | Yes (symmetrized) |\n"
    md += "| Positivity required | No | No | Yes |\n"
    md += "| Best for | Shape similarity | Magnitude similarity | Spectral ratios |\n\n"

    md += get_markdown_footer()

    with open(path, 'w') as f:
        f.write(md)

    return path


# =============================================================================
# EXPERIMENT SUMMARY
# =============================================================================

def _parse_config_key(config_key):
    """Parse feature_similarity_task from config_key, handling multi-word names."""
    for t in ('classification', 'clustering'):
        if config_key.endswith('_' + t):
            remainder = config_key[:-(len(t) + 1)]
            for f in ('inst_freq', 'variance', 'entropy'):
                if remainder.startswith(f + '_'):
                    return f, remainder[len(f) + 1:], t
    return config_key, '', ''

def generate_experiment_summary(all_results, output_dir, figure_map=None):
    """
    Generate per-dataset and cross-dataset experiment summary.

    Args:
        all_results: dict of {dataset_key: {config_key: result_dict}}
        output_dir: Directory to save the report
        figure_map: dict of figure keys to relative paths (optional)

    Returns:
        Path to the generated report
    """
    if figure_map is None:
        figure_map = {}
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'experiment_summary.md')

    md = get_markdown_header("Experiment Summary — EMD Signal Analysis")

    md += "## Results Overview\n\n"

    for ds_key, ds_data in all_results.items():
        ds_info = ds_data.get('dataset_info', {})
        experiments = ds_data.get('experiments', {})

        md += '<div style="page-break-after: always;"></div>\n\n'

        md += f"### {ds_info.get('name', ds_key)}\n\n" 

        if not experiments:
            md += "*No experiments completed.*\n\n"
            continue

        # Classification results table
        class_results = {k: v for k, v in experiments.items() if 'classification' in k}
        if class_results:
            md += "#### Classification Results\n\n"
            md += "| Feature | Similarity | F1 Score | Accuracy |\n"
            md += "|---|---|---|---|\n"
            for config_key, result in sorted(class_results.items()):
                feat, sim, _ = _parse_config_key(config_key)
                f1 = result.get('f1_score', 0)
                acc = result.get('accuracy', 0)
                md += f"| {feat} | {sim} | {f1:.4f} | {acc:.4f} |\n"
            md += "\n"

        # Clustering results table
        clust_results = {k: v for k, v in experiments.items() if 'clustering' in k}
        if clust_results:
            md += '<div style="page-break-after: always;"></div>\n\n'
            md += "#### Clustering Results\n\n"
            md += "| Feature | Similarity | Silhouette Score |\n"
            md += "|---|---|---|\n"
            for config_key, result in sorted(clust_results.items()):
                feat, sim, _ = _parse_config_key(config_key)
                sil = result.get('silhouette_score', -1)
                md += f"| {feat} | {sim} | {sil:.4f} |\n"
            md += "\n"

        # Embed best result figures for this dataset
        md += f"#### Key Visualizations\n\n"
        cm_key = f'{ds_key}_cm'
        if cm_key in figure_map:
            md += f"![Best Classification Confusion Matrix — {ds_info.get('name', ds_key)}]({figure_map[cm_key]})\n\n"
        clustering_key = f'{ds_key}_clustering'
        if clustering_key in figure_map:
            md += f"![Best Clustering Results — {ds_info.get('name', ds_key)}]({figure_map[clustering_key]})\n\n"
        feat_key = f'{ds_key}_class_features'
        if feat_key in figure_map:
            md += f"![Feature Distributions — {ds_info.get('name', ds_key)}]({figure_map[feat_key]})\n\n"

    md += get_markdown_footer()

    with open(path, 'w') as f:
        f.write(md)

    return path


def save_experiment_result_json(result, output_path):
    """Save a single experiment result as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy types for JSON serialization
    serializable = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            serializable[key] = val.tolist()
        elif isinstance(val, (np.integer, np.int64)):
            serializable[key] = int(val)
        elif isinstance(val, (np.floating, np.float64)):
            serializable[key] = float(val)
        elif isinstance(val, dict):
            serializable[key] = {
                str(k): (v.tolist() if isinstance(v, np.ndarray)
                         else float(v) if isinstance(v, (np.floating, np.float64))
                         else int(v) if isinstance(v, (np.integer, np.int64))
                         else v)
                for k, v in val.items()
            }
        else:
            serializable[key] = val

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
