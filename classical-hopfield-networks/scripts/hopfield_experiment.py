#!/usr/bin/env python3
"""
hopfield_experiment.py - Complete Hopfield Network Experiment

This script runs the complete Hopfield network experiment as specified in the assignment:
1. Train the network on characters.mat
2. Test with original training patterns (modeling power evaluation)
3. Test with test.mat (10% noise)
4. Test with validation set (20% noise)
5. Generate visualizations, energy plots, and comparative analysis

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 2 – Hopfield Networks
Date: February 2, 2026
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import author info for consistent headers
try:
    from author_info import get_markdown_header, get_markdown_footer
except ImportError:
    def get_markdown_header(title, date=None):
        if date is None:
            date = datetime.now().strftime("%B %d, %Y")
        return f"# {title}\n\n**Date:** {date}\n\n---\n\n"
    def get_markdown_footer():
        return "\n\n---\n"

# Import Hopfield network functions
from get_data import load_all_datasets, get_project_paths
from hopfield import (
    train_hopfield_network, # Train the Hopfield network
    recall_all_patterns, # Recall patterns using the trained network
    evaluate_recall_results, # Evaluate recall results
    compute_pattern_error, # Compute pattern error metrics
    find_closest_training_pattern, # Find closest training pattern
    UpdateResult # Data class for update results
)


def create_output_folder(seed: int, noise_percent: float) -> Path:
    """Create timestamped output folder for experiment results."""
    paths = get_project_paths()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"hopfield_results_{seed}_{noise_percent}_{timestamp}"
    output_folder = paths["project"] / "hopfield_results" / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "plots").mkdir(exist_ok=True)
    (output_folder / "energy_plots").mkdir(exist_ok=True)
    (output_folder / "pattern_comparisons").mkdir(exist_ok=True)
    return output_folder


def plot_pattern_comparison(original: np.ndarray, recalled: np.ndarray,
                            pattern_idx: int, dataset_name: str,
                            save_path: Path, training_pattern: np.ndarray = None,
                            closest_idx: int = None):
    """
    Plot side-by-side comparison of original query and recalled pattern.
    
    Args:
        original: Original query pattern (12x12)
        recalled: Recalled pattern (12x12)
        pattern_idx: Index of the pattern
        dataset_name: Name of the dataset
        save_path: Path to save the figure
        training_pattern: Closest training pattern (optional)
        closest_idx: Index of closest training pattern (optional)
    """
    n_cols = 3 if training_pattern is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Original query
    axes[0].imshow(original, cmap='binary', interpolation='nearest')
    axes[0].set_title(f'Query (q)\n{dataset_name} #{pattern_idx}', fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Recalled pattern
    hamming = np.sum(original != recalled)
    axes[1].imshow(recalled, cmap='binary', interpolation='nearest')
    axes[1].set_title(f"Recalled (q')\nHamming dist: {hamming}", fontsize=10)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Closest training pattern (if provided)
    if training_pattern is not None:
        training_2d = training_pattern.reshape(12, 12)
        hamming_train = np.sum(recalled != training_2d)
        axes[2].imshow(training_2d, cmap='binary', interpolation='nearest')
        axes[2].set_title(f'Closest Training #{closest_idx}\nHamming to recalled: {hamming_train}', 
                         fontsize=10)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_patterns_grid(results: List[UpdateResult], dataset_name: str,
                           save_path: Path, max_patterns: int = 18):
    """Plot grid of all original vs recalled patterns."""
    n_patterns = min(len(results), max_patterns)
    n_cols = 6
    n_rows = (n_patterns * 2 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for i, result in enumerate(results[:n_patterns]):
        # Original
        ax_orig = axes[i * 2] if i * 2 < len(axes) else None
        ax_rec = axes[i * 2 + 1] if i * 2 + 1 < len(axes) else None
        
        if ax_orig:
            ax_orig.imshow(result.query_original_2d, cmap='binary', interpolation='nearest')
            ax_orig.set_title(f'q[{i}]', fontsize=8)
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
        
        if ax_rec:
            ax_rec.imshow(result.query_final_2d, cmap='binary', interpolation='nearest')
            status = "✓" if result.hamming_distance == 0 else f"d={result.hamming_distance}"
            ax_rec.set_title(f"q'[{i}] {status}", fontsize=8)
            ax_rec.set_xticks([])
            ax_rec.set_yticks([])
    
    # Hide unused axes
    for i in range(n_patterns * 2, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{dataset_name}: Query (q) vs Recalled (q\')', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_energy_curve(result: UpdateResult, pattern_idx: int, dataset_name: str,
                      save_path: Path):
    """Plot energy curve for a single pattern recall."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    iterations = list(range(len(result.energy_history)))
    ax.plot(iterations, result.energy_history, 'b-o', markersize=4, linewidth=1.5)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(f'{dataset_name} Pattern #{pattern_idx} - Energy vs Iterations\n'
                 f'(Converged: {result.converged}, Iterations: {result.n_iterations}, '
                 f'Hamming: {result.hamming_distance})', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Mark start and end
    ax.scatter([0], [result.energy_history[0]], color='green', s=100, zorder=5, label='Start')
    ax.scatter([len(result.energy_history)-1], [result.energy_history[-1]], 
               color='red', s=100, zorder=5, label='End')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_energy_curves(results: List[UpdateResult], dataset_name: str, save_path: Path):
    """Plot all energy curves on one figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        iterations = list(range(len(result.energy_history)))
        ax.plot(iterations, result.energy_history, color=colors[i], 
                alpha=0.7, linewidth=1.5, label=f'Pattern {i}')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(f'{dataset_name} - Energy Curves for All Patterns', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_comparison(evaluations: Dict[str, Dict], save_path: Path):
    """Plot comparison of errors across datasets."""
    datasets = list(evaluations.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Hamming distance to original query
    ax = axes[0, 0]
    for name in datasets:
        eval_data = evaluations[name]
        ax.bar(name, eval_data['avg_hamming_to_original'], 
               color=['#2ecc71', '#3498db', '#e74c3c'][datasets.index(name)])
    ax.set_ylabel('Avg Hamming Distance')
    ax.set_title('Average Hamming Distance: Query vs Recalled')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Hamming distance to closest training pattern
    ax = axes[0, 1]
    for name in datasets:
        eval_data = evaluations[name]
        ax.bar(name, eval_data['avg_hamming_to_training'],
               color=['#2ecc71', '#3498db', '#e74c3c'][datasets.index(name)])
    ax.set_ylabel('Avg Hamming Distance')
    ax.set_title('Average Hamming Distance: Recalled vs Closest Training')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Exact match rate
    ax = axes[1, 0]
    exact_rates = [evaluations[name]['exact_match_rate'] for name in datasets]
    bars = ax.bar(datasets, exact_rates, 
                  color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Exact Match Rate (%)')
    ax.set_title('Exact Match Rate to Training Patterns')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars, exact_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 4. Convergence rate
    ax = axes[1, 1]
    conv_rates = [evaluations[name]['convergence_rate'] for name in datasets]
    bars = ax.bar(datasets, conv_rates,
                  color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Convergence Rate (%)')
    ax.set_title('Network Convergence Rate')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars, conv_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Hopfield Network Performance Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hamming_distribution(evaluations: Dict[str, Dict], save_path: Path):
    """Plot distribution of Hamming distances for each dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (name, eval_data) in enumerate(evaluations.items()):
        ax = axes[i]
        
        # Hamming to original
        hamming_orig = eval_data['hamming_distances_to_original']
        hamming_train = eval_data['hamming_distances_to_training']
        
        x = np.arange(len(hamming_orig))
        width = 0.35
        
        ax.bar(x - width/2, hamming_orig, width, label='To Query', color=colors[i], alpha=0.7)
        ax.bar(x + width/2, hamming_train, width, label='To Training', color=colors[i], alpha=0.4)
        
        ax.set_xlabel('Pattern Index')
        ax.set_ylabel('Hamming Distance')
        ax.set_title(f'{name}')
        ax.legend()
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Hamming Distance per Pattern', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _count(k: int, n: int) -> str:
    """Format k of n as 'k/n (p%)'."""
    return f"{k}/{n} ({100.0 * k / n:.1f}%)"


def _series(values: List[str], names: List[str]) -> str:
    """'a (A), b (B), and c (C)', or 'a on all three sets' when the values agree."""
    if len(set(values)) == 1:
        if len(values) == 2:
            return f"{values[0]} on both sets"
        count = {3: "three", 4: "four", 5: "five"}.get(len(values), str(len(values)))
        return f"{values[0]} on all {count} sets"
    parts = [f"{v} ({n})" for v, n in zip(values, names)]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}" if len(parts) > 2 else " and ".join(parts)


def summarize_runs(evaluations: Dict[str, Dict], train_patterns: np.ndarray) -> Dict[str, Any]:
    """
    Facts the report states, computed from the recall results.

    Query k of every dataset is a noisy (or clean) copy of stored pattern k, so stored
    pattern k is that query's own target.
    """
    train_patterns = np.asarray(train_patterns)
    per_set = {}
    finals_all = []
    for name, ev in evaluations.items():
        results = ev['results']
        n = len(results)
        targets = train_patterns[:n]
        finals = np.array([np.asarray(r.query_final).ravel() for r in results])
        queries = np.array([np.asarray(r.query_original).ravel() for r in results])
        to_target = (finals != targets).sum(axis=1)
        rises, sweeps = 0, 0
        for r in results:
            h = np.asarray(r.energy_history, dtype=float)
            sweeps += len(h) - 1
            rises += int(np.sum(np.diff(h) > 1e-9 * max(1.0, np.abs(h).max())))
        per_set[name] = {
            "n": n,
            "flipped": (queries != targets).sum(axis=1),
            "to_target": to_target,
            "own_exact": int(np.sum(to_target == 0)),
            "energy_rises": rises,
            "sweeps": sweeps,
            "final_energy": np.array([r.energy_history[-1] for r in results], dtype=float),
        }
        finals_all.extend(tuple(int(v) for v in f) for f in finals)

    # Distinct final states over all queries, most common first
    counts: Dict[tuple, int] = {}
    for f in finals_all:
        counts[f] = counts.get(f, 0) + 1
    states = []
    for f, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        f = np.array(f)
        d = (f != train_patterns).sum(axis=1)
        states.append({"queries": c, "pixels_on": int(f.sum()), "nearest": int(d.argmin()),
                       "nearest_dist": int(d.min())})
    return {"per_set": per_set, "states": states, "n_queries": len(finals_all)}


def generate_comparative_report(evaluations: Dict[str, Dict],
                                 output_folder: Path,
                                 seed: int, noise_percent: float,
                                 train_patterns: np.ndarray,
                                 weights: np.ndarray,
                                 max_iterations: int) -> str:
    """Generate the markdown report. Every number and comparison in it is computed from the results."""

    facts = summarize_runs(evaluations, train_patterns)
    per_set = facts["per_set"]
    states = facts["states"]
    n_queries = facts["n_queries"]
    datasets = list(evaluations.keys())
    n_stored, n_units = np.asarray(train_patterns).shape
    side = int(round(np.sqrt(n_units)))

    report_lines = []

    # Author Header
    header = get_markdown_header("Hopfield Network Experiment Report")
    report_lines.append(header)

    # Generation info
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Random Seed:** {seed}")
    report_lines.append(f"**Validation Noise Percentage:** {noise_percent}%")
    report_lines.append("")

    # Summary of the setup
    report_lines.append("## Setup")
    report_lines.append("")
    report_lines.append(f"The network stores {n_stored} binary character patterns ({side}×{side} = {n_units} units) "
                        "and is queried with three sets of 18 patterns, where query k is a copy of stored "
                        "pattern k:")
    report_lines.append("")
    for name in datasets:
        fl = per_set[name]["flipped"]
        report_lines.append(f"- **{name}:** {fl.mean():.2f} of {n_units} pixels differ from the stored pattern "
                            f"on average ({100.0 * fl.mean() / n_units:.2f}%), range {fl.min()} to {fl.max()}.")
    report_lines.append("")

    # Network Configuration
    report_lines.append("## Network Configuration")
    report_lines.append("")
    report_lines.append("| Parameter | Value |")
    report_lines.append("|-----------|-------|")
    report_lines.append(f"| Number of units | {n_units} ({side}×{side}) |")
    report_lines.append(f"| Number of stored patterns | {n_stored} |")
    report_lines.append("| Weights | Sum over stored patterns of (2x_i − 1)(2x_j − 1); zero diagonal |")
    report_lines.append(f"| Weight range | {weights.min():g} to {weights.max():g} |")
    report_lines.append("| Unit states | 0 and 1 |")
    report_lines.append("| Update rule | q_i ← 1 if W[:, i] · q ≥ 0, else 0; one unit at a time, random order |")
    report_lines.append(f"| Stopping rule | A full sweep changes no unit, or {max_iterations} sweeps |")
    report_lines.append("")

    # Results Summary Table
    report_lines.append("## Results Summary")
    report_lines.append("")
    report_lines.append("| Metric | " + " | ".join(datasets) + " |")
    report_lines.append("|--------|" + "|".join(["----------"] * len(datasets)) + "|")

    def row(label, values):
        report_lines.append(f"| {label} | " + " | ".join(values) + " |")

    row("Recalled own pattern exactly", [_count(per_set[d]["own_exact"], per_set[d]["n"]) for d in datasets])
    row("Final state equals some stored pattern",
        [_count(evaluations[d]['exact_matches_to_training'], per_set[d]["n"]) for d in datasets])
    row("Converged", [_count(evaluations[d]['n_converged'], per_set[d]["n"]) for d in datasets])
    row("Mean sweeps (including the last, unchanged sweep)",
        [f"{evaluations[d]['avg_iterations']:.2f}" for d in datasets])
    row("Mean Hamming, final state to query (q vs. q')",
        [f"{evaluations[d]['avg_hamming_to_original']:.2f}" for d in datasets])
    row("Mean Hamming, final state to own pattern", [f"{per_set[d]['to_target'].mean():.2f}" for d in datasets])
    row("Mean Hamming, final state to nearest stored pattern",
        [f"{evaluations[d]['avg_hamming_to_training']:.2f}" for d in datasets])
    report_lines.append("")

    # Detailed Analysis per Dataset
    report_lines.append("## Detailed Analysis")
    report_lines.append("")

    for name, eval_data in evaluations.items():
        report_lines.append(f"### {name}")
        report_lines.append("")
        report_lines.append(f"**Convergence:** {eval_data['n_converged']}/{eval_data['n_patterns']} "
                           f"queries converged ({eval_data['convergence_rate']:.1f}%)")
        report_lines.append("")
        report_lines.append(f"**Sweeps:** Average {eval_data['avg_iterations']:.2f}, "
                           f"range [{eval_data['min_iterations']}, {eval_data['max_iterations']}]")
        report_lines.append("")
        report_lines.append(f"**Exact recall of own pattern:** {per_set[name]['own_exact']}/{eval_data['n_patterns']}")
        report_lines.append("")

        # Per-pattern breakdown
        report_lines.append("**Per-query Hamming distances from the final state:**")
        report_lines.append("")
        report_lines.append("| Query | To Query | To Own Pattern | To Nearest Stored | Nearest Stored Idx |")
        report_lines.append("|-------|----------|----------------|-------------------|--------------------|")

        for i in range(eval_data['n_patterns']):
            h_orig = eval_data['hamming_distances_to_original'][i]
            h_target = per_set[name]['to_target'][i]
            h_train = eval_data['hamming_distances_to_training'][i]
            closest = eval_data['closest_training_indices'][i]
            report_lines.append(f"| {i} | {h_orig} | {h_target} | {h_train} | {closest} |")

        report_lines.append("")

    # Comparative Analysis
    report_lines.append("## Comparative Analysis")
    report_lines.append("")

    train_name, noisy_names = datasets[0], datasets[1:]

    # 1. Recall of the stored patterns
    report_lines.append("### 1. Recall of the Stored Patterns")
    report_lines.append("")
    a = per_set[train_name]["own_exact"]
    report_lines.append(f"Presented unchanged, {a} of {n_stored} stored patterns were recalled exactly. A stored "
                        "pattern is recalled from itself only if the update leaves it unchanged (a fixed point), so "
                        + (f"all {n_stored} are fixed points of this network." if a == n_stored
                           else f"none of the {n_stored} is a fixed point of this network." if a == 0
                           else f"{n_stored - a} of the {n_stored} are not fixed points of this network."))
    report_lines.append("")
    noisy = [f"{per_set[d]['own_exact']} of {per_set[d]['n']} from the {d.lower()} queries" for d in noisy_names]
    report_lines.append("Exact recall from the noisy queries: " + "; ".join(noisy) + ".")
    report_lines.append("")

    # 2. Final states
    report_lines.append("### 2. Final States")
    report_lines.append("")
    report_lines.append(f"The {n_queries} queries ended in {len(states)} distinct final "
                        f"state{'s' if len(states) != 1 else ''}.")
    report_lines.append("")
    shown = states[:10]
    report_lines.append("| Final state | Queries ending there | Pixels on | Nearest stored pattern | Hamming to it |")
    report_lines.append("|-------------|----------------------|-----------|------------------------|---------------|")
    for k, s in enumerate(shown, start=1):
        report_lines.append(f"| {k} | {s['queries']} | {s['pixels_on']} | {s['nearest']} | {s['nearest_dist']} |")
    if len(states) > len(shown):
        report_lines.append(f"| {len(shown) + 1} to {len(states)} | "
                            f"{sum(s['queries'] for s in states[len(shown):])} | | | |")
    report_lines.append("")

    # 3. Noise comparison
    report_lines.append("### 3. Noise Comparison")
    report_lines.append("")
    report_lines.append("| Query set | Mean pixels flipped | Recalled own pattern | Mean Hamming to own pattern "
                        "| Mean Hamming to nearest stored |")
    report_lines.append("|-----------|---------------------|----------------------|-----------------------------"
                        "|--------------------------------|")
    for d in datasets:
        report_lines.append(f"| {d} | {per_set[d]['flipped'].mean():.2f} | "
                            f"{_count(per_set[d]['own_exact'], per_set[d]['n'])} | "
                            f"{per_set[d]['to_target'].mean():.2f} | "
                            f"{evaluations[d]['avg_hamming_to_training']:.2f} |")
    report_lines.append("")
    short = [d.split(" (")[0].lower() for d in datasets]
    report_lines.append("- Exact recall of the own pattern: "
                        + _series([f"{per_set[d]['own_exact']}/{per_set[d]['n']}" for d in datasets], short) + ".")
    report_lines.append("- Mean Hamming distance from the final state to the own pattern: "
                        + _series([f"{per_set[d]['to_target'].mean():.2f}" for d in datasets], short) + ".")
    if len(states) == 1:
        report_lines.append("- Every query ends in the same state, so neither distance depends on the noise in "
                            "the query.")
    report_lines.append("")

    # 4. Energy
    report_lines.append("### 4. Energy")
    report_lines.append("")
    report_lines.append("The energy is computed on the 0/1 state q, as the assignment defines it:")
    report_lines.append("")
    report_lines.append("$$E = -\\frac{1}{2} \\sum_i \\sum_j w_{ij} q_i q_j$$")
    report_lines.append("")
    report_lines.append("It is recorded once before the first sweep and once after every sweep. With symmetric "
                        "weights and a zero diagonal, a single-unit update under the rule above cannot raise E.")
    report_lines.append("")
    rises = sum(per_set[d]["energy_rises"] for d in datasets)
    sweeps = sum(per_set[d]["sweeps"] for d in datasets)
    report_lines.append(f"- Recorded sweeps in which E rose: {rises} of {sweeps}, over all {n_queries} queries.")
    final_e = np.concatenate([per_set[d]["final_energy"] for d in datasets])
    uniq = np.unique(final_e)
    if len(uniq) <= 5:
        report_lines.append("- Final energies: " + ", ".join(f"{v:,.0f}" for v in uniq) + ".")
    else:
        report_lines.append(f"- Final energies range from {final_e.min():,.0f} to {final_e.max():,.0f}.")
    report_lines.append("")

    # 5. Convergence Behavior
    report_lines.append("### 5. Convergence")
    report_lines.append("")
    n_conv = sum(evaluations[d]['n_converged'] for d in datasets)
    report_lines.append(f"- {n_conv} of {n_queries} queries reached a state that a full sweep left unchanged, "
                        f"within the limit of {max_iterations} sweeps.")
    report_lines.append("- Mean sweeps, counting the last, unchanged sweep: "
                        + _series([f"{evaluations[d]['avg_iterations']:.2f}" for d in datasets], short)
                        + f"; the most any query took was {max(evaluations[d]['max_iterations'] for d in datasets)}.")
    report_lines.append("")

    # Summary of results
    report_lines.append("## Summary of Results")
    report_lines.append("")
    report_lines.append("1. **Exact recall of the own pattern:** "
                        + ", ".join(f"{per_set[d]['own_exact']} of {per_set[d]['n']} ({s})"
                                    for d, s in zip(datasets, short)) + ".")
    top = states[0]
    top_text = f"{top['pixels_on']} pixels on, {top['nearest_dist']} bits from stored pattern {top['nearest']}"
    if len(states) == 1:
        report_lines.append(f"2. **Final states:** all {n_queries} queries end in one state ({top_text}).")
    else:
        report_lines.append(f"2. **Final states:** {len(states)} distinct; the most common, reached by "
                            f"{top['queries']} of {n_queries} queries: {top_text}.")
    report_lines.append(f"3. **Energy:** rose in {rises} of {sweeps} recorded sweeps.")
    report_lines.append(f"4. **Convergence:** {n_conv} of {n_queries} queries converged within {max_iterations} sweeps.")
    report_lines.append("")

    # Visualizations reference
    report_lines.append("## Visualizations")
    report_lines.append("")
    report_lines.append("The following visualizations have been generated:")
    report_lines.append("")
    report_lines.append("- `plots/comparison_metrics.png` - Performance comparison across datasets")
    report_lines.append("- `plots/hamming_distribution.png` - Hamming distance per pattern")
    report_lines.append("- `plots/training_patterns_grid.png` - Grid of training q vs q'")
    report_lines.append("- `plots/test_patterns_grid.png` - Grid of test q vs q'")
    report_lines.append("- `plots/validation_patterns_grid.png` - Grid of validation q vs q'")
    report_lines.append("- `plots/*_all_energy_curves.png` - Energy curves of all queries in each set")
    report_lines.append("- `energy_plots/` - Individual energy curves for each test and validation query")
    report_lines.append("- `pattern_comparisons/` - Individual pattern comparison images")
    report_lines.append("")

    # Footer
    report_lines.append("---")
    report_lines.append(f"*Report generated with seed={seed}, validation noise={noise_percent}%*")

    # Write report
    report_path = output_folder / "hopfield_experiment_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    return str(report_path)


def save_results_csv(evaluations: Dict[str, Dict], output_folder: Path) -> str:
    """Save numerical results to CSV."""
    rows = []
    
    for name, eval_data in evaluations.items():
        for i in range(eval_data['n_patterns']):
            rows.append({
                'dataset': name,
                'pattern_idx': i,
                'converged': eval_data['results'][i].converged,
                'iterations': eval_data['results'][i].n_iterations,
                'hamming_to_query': eval_data['hamming_distances_to_original'][i],
                'hamming_to_training': eval_data['hamming_distances_to_training'][i],
                'closest_training_idx': eval_data['closest_training_indices'][i],
                'final_energy': eval_data['results'][i].energy_history[-1]
            })
    
    df = pd.DataFrame(rows)
    csv_path = output_folder / "experiment_results.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def run_experiment(noise_percent: float = 20.0, random_seed: int = 42,
                   max_iterations: int = 100, verbose: bool = True,
                   train_data: np.ndarray = None, test_data: np.ndarray = None,
                   valid_data: np.ndarray = None) -> Dict[str, Any]:
    """
    Run the complete Hopfield network experiment.
    
    Args:
        noise_percent: Noise percentage for validation dataset
        random_seed: Random seed for reproducibility
        max_iterations: Maximum iterations for pattern recall
        verbose: Whether to print progress
        train_data: Pre-loaded training data (3D array), if None will load from files
        test_data: Pre-loaded test data (3D array), if None will load from files
        valid_data: Pre-loaded validation data (3D array), if None will load from files
        
    Returns:
        Dictionary with all experiment results
    """
    if verbose:
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " "*18 + "HOPFIELD NETWORK EXPERIMENT" + " "*23 + "║")
        print("╚" + "═"*68 + "╝")
        print()
    
    # Create output folder
    output_folder = create_output_folder(random_seed, noise_percent)
    if verbose:
        print(f"📁 Output folder: {output_folder.name}")
        print()
    
    # =========================================================================
    # STEP 1: Load Data (or use pre-loaded data)
    # =========================================================================
    if verbose:
        print("=" * 60)
        print("STEP 1: LOADING DATASETS")
        print("=" * 60)
    
    if train_data is not None and test_data is not None and valid_data is not None:
        # Use pre-loaded data - flatten 3D arrays to 2D
        if verbose:
            print("  Using pre-loaded data arrays...")
        n_patterns = train_data.shape[0]
        train_patterns = train_data.reshape(n_patterns, -1)
        test_patterns = test_data.reshape(n_patterns, -1)
        valid_patterns = valid_data.reshape(n_patterns, -1)
        if verbose:
            print(f"  Training patterns:   {train_patterns.shape}")
            print(f"  Test patterns:       {test_patterns.shape}")
            print(f"  Validation patterns: {valid_patterns.shape}")
    else:
        # Load from files
        train_df, test_df, valid_df, train_data_3d, test_data_3d, valid_data_3d, data_info = load_all_datasets(
            noise_percent=noise_percent,
            random_seed=random_seed,
            verbose=verbose
        )
        
        train_patterns = train_df.values
        test_patterns = test_df.values
        valid_patterns = valid_df.values
    
    # =========================================================================
    # STEP 2: Train Network
    # =========================================================================
    if verbose:
        print()
    
    combined_weights, individual_weights = train_hopfield_network(
        train_patterns, verbose=verbose
    )
    
    # =========================================================================
    # STEP 3: Test with Training Patterns (Modeling Power)
    # =========================================================================
    if verbose:
        print()
        print("=" * 60)
        print("STEP 3: TESTING WITH TRAINING PATTERNS (Modeling Power)")
        print("=" * 60)
    
    # Recall all training patterns
    train_results = recall_all_patterns(
        combined_weights, train_patterns,
        max_iterations=max_iterations,
        random_seed=random_seed,
        dataset_name="Training",
        verbose=verbose
    )
    
    # Evaluate results
    train_eval = evaluate_recall_results(
        train_results, train_patterns, train_patterns,
        dataset_name="Training", verbose=verbose
    )
    
    # =========================================================================
    # STEP 4: Test with Test.mat (10% noise)
    # =========================================================================
    if verbose:
        print()
        print("=" * 60)
        print("STEP 4: TESTING WITH TEST.MAT (10% noise)")
        print("=" * 60)
    
    # Recall all test (10% noise) patterns
    test_results = recall_all_patterns(
        combined_weights, test_patterns,
        max_iterations=max_iterations,
        random_seed=random_seed,
        dataset_name="Test (10% noise)",
        verbose=verbose
    )
    
    # Evaluate results
    test_eval = evaluate_recall_results(
        test_results, train_patterns, test_patterns,
        dataset_name="Test (10% noise)", verbose=verbose
    )
    
    # =========================================================================
    # STEP 5: Test with Validation Set (20% noise)
    # =========================================================================
    if verbose:
        print()
        print("=" * 60)
        print(f"STEP 5: TESTING WITH VALIDATION SET ({noise_percent}% noise)")
        print("=" * 60)
    
    # Recall all validation (20% noise by default) patterns
    valid_results = recall_all_patterns(
        combined_weights, valid_patterns,
        max_iterations=max_iterations,
        random_seed=random_seed,
        dataset_name=f"Validation ({noise_percent}% noise)",
        verbose=verbose
    )
    
    # Evaluate results
    valid_eval = evaluate_recall_results(
        valid_results, train_patterns, valid_patterns,
        dataset_name=f"Validation ({noise_percent}% noise)", verbose=verbose
    )
    
    # =========================================================================
    # STEP 6: Generate Visualizations
    # =========================================================================
    if verbose:
        print()
        print("=" * 60)
        print("STEP 6: GENERATING VISUALIZATIONS")
        print("=" * 60)
        print()
    
    evaluations = {
        "Training": train_eval,
        "Test (10% noise)": test_eval,
        f"Validation ({noise_percent}% noise)": valid_eval
    }
    
    # Plot comparison metrics
    if verbose:
        print("  Generating comparison metrics plot...")
    plot_error_comparison(evaluations, output_folder / "plots" / "comparison_metrics.png")
    
    # Plot Hamming distribution
    if verbose:
        print("  Generating Hamming distribution plot...")
    plot_hamming_distribution(evaluations, output_folder / "plots" / "hamming_distribution.png")
    
    # Plot pattern grids
    all_results = [
        ("training", "Training", train_results),
        ("test", "Test (10% noise)", test_results),
        ("validation", f"Validation ({noise_percent}% noise)", valid_results)
    ]
    
    for prefix, name, results in all_results:
        if verbose:
            print(f"  Generating {prefix} pattern grid...")
        plot_all_patterns_grid(
            results, name,
            output_folder / "plots" / f"{prefix}_patterns_grid.png"
        )
        
        # Plot all energy curves for this dataset
        if verbose:
            print(f"  Generating {prefix} energy curves...")
        plot_all_energy_curves(
            results, name,
            output_folder / "plots" / f"{prefix}_all_energy_curves.png"
        )
    
    # Individual pattern comparisons and energy plots (for test and validation)
    for prefix, name, results in [("test", "Test", test_results), 
                                   ("validation", "Validation", valid_results)]:
        if verbose:
            print(f"  Generating individual {prefix} pattern comparisons and energy plots...")
        
        for i, result in enumerate(results):
            # Pattern comparison
            closest_idx, _ = find_closest_training_pattern(result.query_final, train_patterns)
            plot_pattern_comparison(
                result.query_original_2d, result.query_final_2d,
                i, name,
                output_folder / "pattern_comparisons" / f"{prefix}_pattern_{i:02d}.png",
                training_pattern=train_patterns[closest_idx],
                closest_idx=closest_idx
            )
            
            # Energy curve
            plot_energy_curve(
                result, i, name,
                output_folder / "energy_plots" / f"{prefix}_energy_{i:02d}.png"
            )
    
    # =========================================================================
    # STEP 7: Generate Report and Save Results
    # =========================================================================
    if verbose:
        print()
        print("=" * 60)
        print("STEP 7: GENERATING REPORT")
        print("=" * 60)
        print()
    
    # Save CSV results
    csv_path = save_results_csv(evaluations, output_folder)
    if verbose:
        print(f"  Saved results CSV: {Path(csv_path).name}")
    
    # Generate markdown report
    report_path = generate_comparative_report(
        evaluations, output_folder, random_seed, noise_percent,
        train_patterns=train_patterns, weights=combined_weights,
        max_iterations=max_iterations
    )
    if verbose:
        print(f"  Saved report: {Path(report_path).name}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    if verbose:
        print()
        print("╔" + "═"*68 + "╗")
        print("║" + " "*22 + "EXPERIMENT COMPLETE" + " "*27 + "║")
        print("╚" + "═"*68 + "╝")
        print()
        print("📊 RESULTS SUMMARY:")
        print()
        print(f"{'Dataset':<30} {'Exact Match':<15} {'Avg Hamming':<15}")
        print("-" * 60)
        for name, eval_data in evaluations.items():
            exact = f"{eval_data['exact_matches_to_training']}/18 ({eval_data['exact_match_rate']:.1f}%)"
            hamming = f"{eval_data['avg_hamming_to_training']:.2f}"
            print(f"{name:<30} {exact:<15} {hamming:<15}")
        print()
        print(f"📁 All outputs saved to: {output_folder}")
        print()
    
    # Prepare proper return structure for kamp_hw2.py (the main script)
    return {
        "evaluations": evaluations,
        "combined_weights": combined_weights,
        "output_folder": output_folder,
        "report_file": Path(report_path),
        "train_results": train_results,
        "test_results": test_results,
        "valid_results": valid_results,
        # Simplified result summaries for kamp_hw2.py
        "training_recall": {
            "accuracy": train_eval['exact_match_rate'] / 100,
            "mean_hamming": train_eval['avg_hamming_to_training'],
        },
        "test_recall": {
            "accuracy": test_eval['exact_match_rate'] / 100,
            "mean_hamming": test_eval['avg_hamming_to_training'],
        },
        "valid_recall": {
            "accuracy": valid_eval['exact_match_rate'] / 100,
            "mean_hamming": valid_eval['avg_hamming_to_training'],
        }
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Hopfield Network Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument( # Noise percentage for validation dataset
        "--noise-percent", "-n",
        type=float,
        default=20.0,
        help="Noise percentage for validation dataset"
    )
    
    parser.add_argument( # Random seed for reproducibility
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument( # Maximum iterations for pattern recall
        "--max-iterations", "-m",
        type=int,
        default=100,
        help="Maximum iterations for pattern recall"
    )
    
    parser.add_argument( # Suppress verbose output
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print console header
    if not args.quiet:
        try:
            from author_info import print_console_header
            print_console_header("hopfield_experiment.py")
        except ImportError:
            pass
    
    # Run the experiment
    results = run_experiment(
        noise_percent=args.noise_percent,
        random_seed=args.seed,
        max_iterations=args.max_iterations,
        verbose=not args.quiet
    )
    
    return results


if __name__ == "__main__":
    main()
