#!/usr/bin/env python3
"""
kamp_hw4.py - Main Orchestrator for HW4

Entry point for the EMD-based EEG signal analysis pipeline.
Orchestrates data loading, EMD, feature extraction, similarity
computation, and learning tasks across all three datasets.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 4 – ML for Signal Analysis using EMD
Date: March 8, 2026
Revised: November 2026, for this repository
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from author_info import print_console_header, TeeLogger
from data_loader import load_dataset, DEFAULT_DATASETS, DATASET_REGISTRY
from dataset_stats import compute_and_save_statistics
from emd_processor import apply_emd_to_signals
from feature_extraction import extract_imf_features, FEATURE_TYPES
from similarity import compute_distance_matrix, SIMILARITY_MEASURES
from learning_tasks import run_classification, run_clustering
from visualization import (
    plot_representative_signals, plot_signal_with_imfs,
    plot_confusion_matrix, plot_feature_distributions,
    plot_distance_matrix_heatmap, plot_clustering_results,
    generate_comparison_figures, copy_figures_for_reports
)
from report_generator import (
    generate_methodology_report, generate_similarity_description,
    generate_experiment_summary, save_experiment_result_json
)


# Metrics saved per experiment
METRIC_KEYS = ['f1_score', 'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_per_class',
               'silhouette_score', 'ari', 'nmi']


# =============================================================================
# COMMAND-LINE ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="HW4 — ML for Signal Analysis using EMD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (single dataset, single config)
  python kamp_hw4.py --raw-dir /path/to/raw --datasets eeg_ma --features variance --similarity euclidean --task classification --n-subjects 5 --no-stats

  # Medium test (one dataset, all configs)
  python kamp_hw4.py --raw-dir /path/to/raw --datasets bci_civ_2a

  # Full run (all datasets, all configs), raw files in /path/to/raw/<dataset>/
  # (as scripts/download_data.py lays them out), cache files written to
  # /path/to/cache/<dataset>/
  python kamp_hw4.py --raw-dir /path/to/raw --cache-dir /path/to/cache
        """
    )
    # Command-line options:

    # Dataset and configuration options:
    # This option allows users to specify which datasets to process. 
    # By default, all three datasets will be processed, 
    # but users can select one or more specific datasets to focus their analysis or reduce runtime for testing purposes.
    parser.add_argument('--datasets', nargs='+',
                        choices=['eeg_mmi', 'bci_civ_2a', 'eeg_ma'],
                        default=None,
                        help='Dataset(s) to process (default: all three)')
    
    # Feature types, similarity measures, and learning tasks to run (default: all):
    # These options allow users to customize which features, similarity measures, and learning tasks to execute.
    # By default, all combinations will be run for comprehensive analysis, 
    # but users can select specific options to focus on particular aspects or reduce runtime for testing.
    parser.add_argument('--features', nargs='+',
                        choices=FEATURE_TYPES,
                        default=FEATURE_TYPES,
                        help='IMF feature type(s)')
    
    # Similarity measures to compute (default: all three) --- IGNORE ---:
    # This option allows users to select which similarity measures to compute for the feature vectors.
    # The default includes all three measures, but users can specify a subset for quicker runs or focused analysis.
    parser.add_argument('--similarity', nargs='+',
                        choices=SIMILARITY_MEASURES,
                        default=SIMILARITY_MEASURES,
                        help='Similarity measure(s)')
    
    # Learning tasks to perform (default: both classification and clustering) --- IGNORE ---:
    # This option allows users to choose which learning tasks to execute. 
    # By default, both classification and clustering will be performed for each feature-similarity combination, 
    # but users can select only one type of task if they wish to focus their analysis or reduce runtime.
    parser.add_argument('--task', nargs='+',
                        choices=['classification', 'clustering'],
                        default=['classification', 'clustering'],
                        help='Learning task(s)')
    
    # Additional options:
    # Disable dataset statistics computation (for faster runs)
    parser.add_argument('--no-stats', action='store_true',
                        help='Disable dataset statistics')
    
    # Limit number of subjects per dataset (for quicker testing):
    # This option allows users to limit the number of subjects included from each dataset, which can significantly 
    # reduce processing time for testing or debugging purposes. 
    # By default, all subjects will be included, but setting this to a specific number (e.g., 5) keeps the first
    # that many subjects of each dataset, in sorted order.
    parser.add_argument('--n-subjects', type=int, default=None,
                        help='Limit number of subjects per dataset')
    
    # Number of IMFs used per segment (default: 4):
    # Each segment is decomposed fully; its first n IMFs (residue excluded) form the feature vector,
    # so position k always means IMF k. Segments with fewer than n IMFs are skipped and counted.
    parser.add_argument('--n-imfs', type=int, default=4,
                        help='Number of IMFs used per segment, residue excluded (default: 4)')

    # Data locations:
    # The raw recordings are not in the repository. --raw-dir points to a folder holding one subfolder
    # per dataset (eeg_mmi/, bci_iv_2a/ or bci_civ_2a/, eegmat/ or eeg_ma/). --cache-dir sets where the
    # preprocessed windows are cached (default: data/<dataset>/preprocessed_data/).
    parser.add_argument('--raw-dir', default=None,
                        help='Folder with one subfolder of raw files per dataset')
    parser.add_argument('--cache-dir', default=None,
                        help='Folder for preprocessed cache files')
    
    # Window length for segmenting signals (default: 2.0 seconds):
    # This option specifies the length of time windows (in seconds) used to segment the continuous EEG signals before applying EMD.
    # Shorter windows may capture more transient features but can lead to fewer samples for analysis, while longer windows may provide 
    # more stable features at the cost of temporal resolution. 
    # The default is set to 2.0 seconds, which is a common choice for EEG analysis, but users can adjust this based on the 
    # characteristics of their data and analysis goals.
    parser.add_argument('--window-length', type=float, default=2.0,
                        help='Window length in seconds (default: 2.0)')
    # Random seed for reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Paths
    data_dir = os.path.join(PROJECT_DIR, 'data')
    results_base = os.path.join(PROJECT_DIR, 'results')
    stats_base = os.path.join(PROJECT_DIR, 'dataset_stats')
    printouts_dir = os.path.join(PROJECT_DIR, 'printouts')
    reports_dir = os.path.join(PROJECT_DIR, 'reports')

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(results_base, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Printout log path
    log_path = os.path.join(printouts_dir, f"hw4_printout__{timestamp}.txt")

    # Datasets
    datasets = args.datasets or DEFAULT_DATASETS

    # Start logging and print header
    with TeeLogger(log_path):
        start_time = time.time()
        print_console_header("kamp_hw4.py")

        # Print run configuration
        print(f"\n{'─' * 70}")
        print(f"  RUN CONFIGURATION")
        print(f"{'─' * 70}")
        print(f"  Timestamp:    {timestamp}")
        print(f"  Output dir:   {os.path.relpath(run_dir, PROJECT_DIR)}")
        print(f"  Datasets:     {', '.join(datasets)}")
        print(f"  Features:     {', '.join(args.features)}")
        print(f"  Similarity:   {', '.join(args.similarity)}")
        print(f"  Tasks:        {', '.join(args.task)}")
        print(f"  IMFs used:    first {args.n_imfs} per segment (residue excluded)")
        print(f"  Window:       {args.window_length}s")
        print(f"  N subjects:   {args.n_subjects or 'all'}")
        print(f"  Stats:        {'disabled' if args.no_stats else 'enabled'}")
        print(f"  Random seed:  {args.seed}")
        print(f"{'─' * 70}\n")

        total_experiments = (len(datasets) * len(args.features) *
                             len(args.similarity) * len(args.task))
        print(f"  Total experiment configurations: {total_experiments}")

        all_results = {}

        # ================================================================
        # MAIN LOOP: Process each dataset
        # ================================================================
        for ds_idx, dataset_key in enumerate(datasets):
            ds_info = DATASET_REGISTRY[dataset_key]
            ds_start = time.time()

            print(f"\n{'=' * 70}")
            print(f"  DATASET {ds_idx + 1}/{len(datasets)}: {ds_info['name']}")
            print(f"  ({dataset_key})")
            print(f"{'=' * 70}\n")

            # Dataset output directory
            ds_run_dir = os.path.join(run_dir, dataset_key)
            os.makedirs(ds_run_dir, exist_ok=True)

            # ---- TASK 1: Load & preprocess data ----
            print(f"  [Task 1] Loading and preprocessing data...")
            try:
                dataset_dict = load_dataset(
                    dataset_key, data_dir,
                    n_subjects=args.n_subjects,
                    window_length=args.window_length,
                    verbose=True,
                    raw_dir=args.raw_dir,
                    cache_dir=args.cache_dir,
                )
            except Exception as e:
                print(f"  ERROR: Failed to load {dataset_key}: {e}")
                continue

            signals = dataset_dict['signals']
            labels = dataset_dict['labels']
            label_names = dataset_dict['label_names']
            sfreq = dataset_dict['sfreq']

            print(f"  Loaded: {signals.shape[0]} segments, "
                  f"{signals.shape[1]} samples each, "
                  f"sfreq={sfreq} Hz")
            print(f"  Classes: {label_names}")
            class_counts = {label_names.get(int(l), str(l)): int(np.sum(labels == l))
                            for l in np.unique(labels)}
            print(f"  Distribution: {class_counts}")

            # Store dataset info for reports
            all_results[dataset_key] = {
                'dataset_info': {
                    'name': dataset_dict['dataset_name'], # Name of the dataset for reporting
                    'source_url': dataset_dict['source_url'], # URL for dataset reference in reports
                    'n_subjects': dataset_dict['n_subjects'], # Number of subjects included in the analysis
                    'channel_names': dataset_dict['channel_names'], # List of EEG channel names for reference in reports
                    'sfreq': sfreq, # Sampling frequency of the signals, important for feature extraction and reporting
                    'n_segments': len(signals), # Total number of signal segments processed, useful for reporting and analysis context
                },
                'experiments': {},
            }
            dropped = dataset_dict.get('metadata', {}).get('dropped', {})
            if dropped:
                print(f"  Dropped while windowing: {json.dumps(dropped)}")
            all_results[dataset_key]['dataset_info']['dropped'] = dropped

            # ---- DATASET STATS ----
            if not args.no_stats:
                print(f"\n  [Stats] Computing dataset statistics...")
                ds_stats_dir = os.path.join(stats_base, dataset_key)
                compute_and_save_statistics(dataset_dict, ds_stats_dir, verbose=True)

            # ---- TASK 1 VISUALIZATION ----
            print(f"\n  [Task 1] Generating signal visualizations...")
            signals_dir = os.path.join(ds_run_dir, 'signals')
            os.makedirs(signals_dir, exist_ok=True)
            plot_representative_signals(
                signals, labels, label_names, sfreq,
                os.path.join(signals_dir, 'representative_signals.png')
            )
            print(f"  Saved representative signals plot.")

            # ---- TASK 2: EMD ----
            print(f"\n  [Task 2] Applying EMD to {len(signals)} segments...")
            emd_results, n_imfs, failed_indices = apply_emd_to_signals(
                signals, n_imfs=args.n_imfs, verbose=True
            )

            # Filter labels to match the segments kept after EMD
            successful_indices = [r['original_index'] for r in emd_results]
            filtered_labels = labels[successful_indices]
            imf_totals = np.array([r['n_imfs_total'] for r in emd_results])
            all_results[dataset_key]['dataset_info']['n_segments_used'] = len(emd_results)
            all_results[dataset_key]['dataset_info']['n_segments_skipped_emd'] = len(failed_indices)
            all_results[dataset_key]['dataset_info']['imf_count_range'] = [int(imf_totals.min()), int(imf_totals.max())]
            all_results[dataset_key]['dataset_info']['imf_count_median'] = float(np.median(imf_totals))

            # ---- TASK 2 VISUALIZATION ----
            print(f"\n  [Task 2] Generating IMF decomposition plots...")
            n_examples = min(3, len(emd_results))
            for ex_idx in range(n_examples):
                item = emd_results[ex_idx]
                orig_idx = item['original_index']
                lbl = labels[orig_idx]
                name = label_names.get(int(lbl), f'class_{lbl}')
                plot_signal_with_imfs(
                    signals[orig_idx], item['imfs_all'], item['residual'], sfreq,
                    os.path.join(signals_dir, f'signal_with_imfs_{ex_idx}.png'),
                    title=f'{name} (segment {orig_idx})'
                )
            print(f"  Saved {n_examples} IMF decomposition plots.")

            # ---- TASKS 3-5: Feature × Similarity × Task loop ----
            print(f"\n  [Tasks 3-5] Running experiment configurations...")
            exp_count = 0
            total_ds_exp = len(args.features) * len(args.similarity) * len(args.task)

            for feat_type in args.features:
                # TASK 3: Feature extraction
                print(f"\n  ── Feature: {feat_type} ──")
                print(f"    Extracting {feat_type} features...")
                features = extract_imf_features(
                    emd_results, feat_type, n_imfs=n_imfs, sfreq=sfreq
                )
                print(f"    Feature matrix shape: {features.shape}")

                for sim_measure in args.similarity:
                    # TASK 4: Distance matrix
                    print(f"    Computing {sim_measure} distance matrix...")
                    dist_matrix = compute_distance_matrix(features, sim_measure)
                    print(f"    Distance matrix shape: {dist_matrix.shape}")

                    for task_type in args.task:
                        exp_count += 1
                        config_key = f"{feat_type}_{sim_measure}_{task_type}"
                        exp_dir = os.path.join(ds_run_dir, config_key)
                        os.makedirs(exp_dir, exist_ok=True)

                        print(f"\n    [{exp_count}/{total_ds_exp}] "
                              f"{feat_type} + {sim_measure} + {task_type}")

                        # TASK 5: Learning task
                        if task_type == 'classification':
                            result = run_classification(
                                features, filtered_labels, dist_matrix,
                                sim_measure, label_names,
                                random_state=args.seed
                            )

                            print(f"      F1 Score: {result['f1_score']:.4f}")
                            print(f"      Accuracy: {result['accuracy']:.4f}")
                            print(f"      Balanced accuracy: {result['balanced_accuracy']:.4f}")
                            print(f"      Per-class F1: {result['f1_per_class']}")

                            # Save confusion matrix plot
                            if result['confusion_matrix'].size > 0:
                                plot_confusion_matrix(
                                    result['confusion_matrix'], label_names,
                                    os.path.join(exp_dir, 'confusion_matrix.png'),
                                    title=f'{feat_type} + {sim_measure}'
                                )

                            # Save classification report
                            with open(os.path.join(exp_dir, 'classification_report.txt'), 'w') as f:
                                f.write(result['classification_report'])

                        elif task_type == 'clustering':
                            result = run_clustering(
                                features, filtered_labels, dist_matrix,
                                sim_measure, random_state=args.seed
                            )

                            print(f"      Silhouette Score: {result['silhouette_score']:.4f}")
                            print(f"      ARI / NMI vs. labels: {result['ari']:.4f} / {result['nmi']:.4f}")

                            # Save clustering visualization
                            plot_clustering_results(
                                features, result['cluster_labels'],
                                filtered_labels, label_names,
                                os.path.join(exp_dir, 'clustering_results.png'),
                                title=f'{feat_type} + {sim_measure}'
                            )

                        # Save additional visualizations
                        plot_distance_matrix_heatmap(
                            dist_matrix, filtered_labels, sim_measure,
                            os.path.join(exp_dir, 'distance_matrix.png')
                        )
                        plot_feature_distributions(
                            features, filtered_labels, label_names, feat_type,
                            os.path.join(exp_dir, 'feature_distributions.png')
                        )

                        # Save result JSON
                        save_experiment_result_json(
                            {
                                'dataset': dataset_key,
                                'feature_type': feat_type,
                                'similarity_measure': sim_measure,
                                'learning_task': task_type,
                                'metrics': {k: v for k, v in result.items()
                                            if k in METRIC_KEYS},
                                'parameters': {
                                    'n_segments': len(features),
                                    'n_imfs': n_imfs,
                                    'window_length': args.window_length,
                                    'seed': args.seed,
                                },
                            },
                            os.path.join(exp_dir, 'results.json')
                        )

                        # Store for summary
                        all_results[dataset_key]['experiments'][config_key] = {
                            k: v for k, v in result.items()
                            if k in METRIC_KEYS
                        }

            ds_elapsed = time.time() - ds_start
            print(f"\n  Dataset {dataset_key} completed in {ds_elapsed:.1f}s")

        # ================================================================
        # COMPARATIVE ANALYSIS & REPORTS
        # ================================================================
        if len(datasets) >= 2:
            print(f"\n{'=' * 70}")
            print(f"  COMPARATIVE ANALYSIS")
            print(f"{'=' * 70}\n")

            comp_dir = os.path.join(run_dir, 'comparative_analysis')
            os.makedirs(comp_dir, exist_ok=True)

            _generate_comparison_table(all_results, comp_dir)

        # Copy figures for reports and generate comparison figures
        print(f"\n{'=' * 70}")
        print(f"  GENERATING FIGURES & REPORTS")
        print(f"{'=' * 70}\n")

        figure_map = copy_figures_for_reports(all_results, run_dir)
        print(f"  Copied {len(figure_map)} figures for report embedding.")

        generate_methodology_report(all_results, run_dir, figure_map=figure_map)
        print(f"  Generated methodology report.")

        generate_similarity_description(run_dir, figure_map=figure_map, all_results=all_results)
        print(f"  Generated similarity description.")

        generate_experiment_summary(all_results, run_dir, figure_map=figure_map)
        print(f"  Generated experiment summary.")

        # ================================================================
        # COMPLETION SUMMARY
        # ================================================================
        total_elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Total time:    {total_elapsed:.1f}s")
        print(f"  Results saved: {os.path.relpath(run_dir, PROJECT_DIR)}")
        print(f"  Printout log:  {os.path.relpath(log_path, PROJECT_DIR)}")
        print(f"  Experiments:   {sum(len(v.get('experiments', {})) for v in all_results.values())}")
        print(f"{'=' * 70}\n")


# =============================================================================
# COMPARATIVE ANALYSIS
# =============================================================================

def _generate_comparison_table(all_results, output_dir):
    """Generate cross-dataset comparison tables and save as CSV + markdown."""
    rows = []
    for ds_key, ds_data in all_results.items():
        ds_name = ds_data.get('dataset_info', {}).get('name', ds_key)
        for config_key, result in ds_data.get('experiments', {}).items():
            # Parse config_key = feature_similarity_task
            # Handle multi-word names (inst_freq, itakura_saito)
            feat, sim, task = '', '', ''
            for t in ('classification', 'clustering'):
                if config_key.endswith('_' + t):
                    task = t
                    remainder = config_key[:-(len(t) + 1)]
                    for f in ('inst_freq', 'variance', 'entropy'):
                        if remainder.startswith(f + '_'):
                            feat = f
                            sim = remainder[len(f) + 1:]
                            break
                    break

            row = {
                'dataset': ds_name, # Human-readable dataset name for reporting
                'dataset_key': ds_key, # Original dataset key for reference
                'feature': feat, # Feature type used in the experiment
                'similarity': sim, # Similarity measure used in the experiment
                'task': task, # Learning task performed (classification or clustering)
            }

            # Add metrics based on task type:
            # For classification, we focus on F1 score and accuracy; for clustering, we focus on silhouette score.
            if task == 'classification':
                row['f1_score'] = result.get('f1_score', 0) # Overall weighted F1 score; convert to float for JSON serialization
                row['accuracy'] = result.get('accuracy', 0) # Accuracy score; convert to float for JSON serialization
                row['balanced_accuracy'] = result.get('balanced_accuracy', 0) # Mean per-class recall
                row['f1_macro'] = result.get('f1_macro', 0) # Unweighted mean of per-class F1
                row['metric_value'] = result.get('f1_score', 0) # Primary metric for classification is F1 score
                row['metric_name'] = 'F1' # Primary metric name for classification
            # For clustering, we use silhouette score as the primary metric.
            else:
                row['silhouette'] = result.get('silhouette_score', -1) # Silhouette score; set to -1 if not available
                row['ari'] = result.get('ari', 0) # Agreement of the clusters with the true labels
                row['nmi'] = result.get('nmi', 0) # Normalized mutual information with the true labels
                row['metric_value'] = result.get('silhouette_score', -1) # Primary metric for clustering is silhouette score
                row['metric_name'] = 'Silhouette' # Primary metric name for clustering
            rows.append(row)

    # Handle case where no experiments were run (e.g., due to errors or filtering)
    if not rows: 
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)

    # Summary markdown
    md = "# Cross-Dataset Comparison\n\n"
    md += "## Classification Results (F1 Score)\n\n"
    class_df = df[df['task'] == 'classification'] if 'task' in df.columns else pd.DataFrame()
    if not class_df.empty:
        pivot = class_df.pivot_table(
            values='f1_score', index=['feature', 'similarity'],
            columns='dataset_key', aggfunc='first'
        )
        md += pivot.to_markdown() + "\n\n"

    md += "## Clustering Results (Silhouette Score)\n\n"
    clust_df = df[df['task'] == 'clustering'] if 'task' in df.columns else pd.DataFrame()
    if not clust_df.empty:
        pivot = clust_df.pivot_table(
            values='silhouette', index=['feature', 'similarity'],
            columns='dataset_key', aggfunc='first'
        )
        md += pivot.to_markdown() + "\n\n"

    if not class_df.empty:
        md += "## Classification Results (Balanced Accuracy)\n\n"
        md += ("Balanced accuracy is the mean per-class recall; chance is 0.50 for two classes "
               "and 0.25 for four, whatever the class balance.\n\n")
        pivot = class_df.pivot_table(
            values='balanced_accuracy', index=['feature', 'similarity'],
            columns='dataset_key', aggfunc='first'
        )
        md += pivot.to_markdown() + "\n\n"

    if not clust_df.empty:
        md += "## Clustering Agreement with the Classes (ARI)\n\n"
        md += ("K-Means does not use the distance measure, so the three rows of a feature share one "
               "partition; the adjusted Rand index (0 = chance) compares it with the true labels.\n\n")
        pivot = clust_df.pivot_table(
            values='ari', index=['feature', 'similarity'],
            columns='dataset_key', aggfunc='first'
        )
        md += pivot.to_markdown() + "\n\n"

    # Generate comparison figures
    comp_fig_dir = os.path.join(output_dir, 'figures')
    generated_figs = generate_comparison_figures(all_results, comp_fig_dir)

    if generated_figs:
        md += "## Comparison Figures\n\n"
        for fig_name in generated_figs:
            caption = fig_name.replace('.png', '').replace('_', ' ').title()
            md += f"![{caption}](figures/{fig_name})\n\n"

    with open(os.path.join(output_dir, 'comparison_summary.md'), 'w') as f:
        f.write(md)

    print("  Generated comparison table, summary, and figures.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
