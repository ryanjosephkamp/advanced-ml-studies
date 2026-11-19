#!/usr/bin/env python3
"""
report_generator.py - Final Report Generation (Markdown + LaTeX)

This module generates the comprehensive HW3_REPORT.md and hw3_report_ieee.tex reports
from the collected results of all analyses.

It compiles the results into a structured format, including:
- Dataset statistics
- Class distributions
- Correlation heatmaps
- Feature distribution plots
- Projection scatter plots
- Distance comparisons
- Confusion matrices
- Accuracy comparisons
- Between-dataset analyses
- Key findings and insights
- References to all generated figures and tables

The generated reports are saved in the "reports" directory, with figures organized in "reports/latex/figures".

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 3 – Random Projections and Multiclass Classification
Date: February 13, 2026
"""

import math
import os
import sys
import shutil
from pathlib import Path

import pandas as pd

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from author_info import (
    get_markdown_header, get_markdown_footer,
    AUTHOR_NAME, AUTHOR_AFFILIATION,
    COURSE_NAME, COURSE_ASSIGNMENT,
)
from svm_classifier import describe_top_kernels

# Project paths
HW3_DIR = SCRIPT_DIR.parent
REPORTS_DIR = HW3_DIR / "reports"
LATEX_DIR = REPORTS_DIR / "latex"
FIGURES_DIR = LATEX_DIR / "figures"


def target_k(all_results: dict) -> int:
    """The target dimension k of the run's projections (the same for every dataset)."""
    return next(iter(all_results.values()))["projection"].get("k", 2)


# =============================================================================
# FIGURE COLLECTION
# =============================================================================

def collect_figures(all_results: dict, run_dir: str, proj_label: str = "eq3") -> list:
    """
    Select key figures from the run and copy them to the latex/figures directory.

    The function looks for specific figure types (projection scatter, distance comparisons, confusion matrices, 
    accuracy comparisons) for each dataset and projection type. 
    
    It also collects between-dataset comparison figures if they exist. 
    
    Each figure is copied to the latex/figures directory with a standardized filename, and a list of (filename, caption) 
    tuples is returned for inclusion in the report. 

    Args:
        all_results: The dictionary containing all results from the analyses.
        run_dir: The directory where the results are stored.
        proj_label: The label for the projection type (e.g., "eq3" or "eq4") to identify the correct figures.

    Returns:
        List of (figure_filename, caption) tuples
    """
    os.makedirs(str(FIGURES_DIR), exist_ok=True)
    figures = []

    # Iterate through each dataset's results to find and copy relevant figures:
    for ds_key, res in all_results.items():
        ds_name = res["dataset"]["dataset_name"]
        ds_dir = os.path.join(run_dir, ds_key)

        # Projection scatter
        for eq in ["eq3", "eq4"]:
            src = os.path.join(ds_dir, f"{ds_key}_projection_{eq}.png")
            if os.path.exists(src):
                dest_name = f"{ds_key}_projection_{eq}.png"
                shutil.copy2(src, str(FIGURES_DIR / dest_name))
                eq_label = "±1" if eq == "eq3" else "Sparse Ternary"
                k_run = res["projection"].get("k", 2)
                view = "2-D Projection" if k_run == 2 else f"Projection, first 2 of k = {k_run} dimensions"
                figures.append((dest_name, f"{ds_name} — {view} ({eq_label})"))

        # D1 vs D2
        src = os.path.join(ds_dir, f"{ds_key}_d1_vs_d2_{proj_label}.png")
        if os.path.exists(src):
            dest_name = f"{ds_key}_d1_vs_d2_{proj_label}.png"
            shutil.copy2(src, str(FIGURES_DIR / dest_name))
            figures.append((dest_name, f"{ds_name} — Original vs Projected Distances"))

        # Confusion matrices grid
        src = os.path.join(ds_dir, f"{ds_key}_{proj_label}_confusion_matrices_grid.png")
        if os.path.exists(src):
            dest_name = f"{ds_key}_{proj_label}_confusion_matrices_grid.png"
            shutil.copy2(src, str(FIGURES_DIR / dest_name))
            figures.append((dest_name, f"{ds_name} — Confusion Matrices (All Kernels)"))

        # Accuracy comparison
        src = os.path.join(ds_dir, f"{ds_key}_{proj_label}_kernel_accuracy_comparison.png")
        if os.path.exists(src):
            dest_name = f"{ds_key}_{proj_label}_kernel_accuracy_comparison.png"
            shutil.copy2(src, str(FIGURES_DIR / dest_name))
            figures.append((dest_name, f"{ds_name} — Kernel Accuracy Comparison"))

    # Between-datasets figures (saved directly in run_dir by between_datasets_analysis)
    # Iterate through expected between-dataset figure types and copy them if they exist:
    for eq in ["eq3", "eq4"]:
        eq_label = "EQ3 (±1)" if eq == "eq3" else "EQ4 (Sparse)"
        for fname_template, cap_template in [
            ("kernel_accuracy_across_datasets_{eq}.png", "Kernel Accuracy Across Datasets ({eql})"),
            ("kernel_dataset_accuracy_heatmap_{eq}.png", "Kernel × Dataset Accuracy Heatmap ({eql})"),
            ("distortion_comparison_across_datasets_{eq}.png", "Distortion (ε) Across Datasets ({eql})"),
        ]:
            fname = fname_template.format(eq=eq)
            cap = cap_template.format(eql=eq_label)
            src = os.path.join(run_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, str(FIGURES_DIR / fname))
                figures.append((fname, cap))

    return figures


# =============================================================================
# DISCUSSION TEXT COMPUTED FROM THE RESULTS
# =============================================================================

PROJ_NAMES = {"eq3": "Eq.3", "eq4": "Eq.4"}

SINGLE_DRAW_CAVEAT = (
    "Each data set was projected with one random matrix per distribution and split once into "
    "training and test sets, so these numbers describe this run. They do not rank the kernels or "
    "the two distributions in general; that needs repeated draws."
)


def summarize_run(all_results: dict, svm_key: str) -> list:
    """Per-dataset facts for the discussion, computed from the results (no fixed claims).

    For each dataset: the range of test accuracy across kernels, the kernel(s) with the highest
    test accuracy in this run, the distortion summary for both distributions, and 24 ln n, the
    smallest target size Achlioptas's bound can certify (its limit as epsilon -> 1 and beta -> 0).
    """
    rows = []
    for dk in all_results:
        ds = all_results[dk]["dataset"]
        proj = all_results[dk]["projection"]
        kr = all_results[dk][svm_key]["kernel_results"]
        accs = {r["kernel_name"]: r["test"]["accuracy"] for r in kr.values()}
        rows.append({
            "name": ds["dataset_name"],
            "k": proj.get("k", 2),
            "acc_min": min(accs.values()),
            "acc_max": max(accs.values()),
            "top": describe_top_kernels(accs.keys(), accs.values()),
            "eps": {eq: proj[f"distortion_{eq}"]["epsilon"] for eq in ["eq3", "eq4"]},
            "m": {eq: proj[f"distortion_{eq}"]["m"] for eq in ["eq3", "eq4"]},
            "k0_min": 24 * math.log(ds["n_samples"]),
        })
    return rows


def discussion_markdown(rows: list) -> str:
    """Discussion paragraphs (Markdown); every comparison is computed from rows."""
    text = SINGLE_DRAW_CAVEAT + "\n\n"
    text += "Test accuracy across the four kernels, per data set:\n\n"
    for r in rows:
        if r["acc_min"] == r["acc_max"]:
            text += f"- **{r['name']}:** all four kernels {r['acc_max']:.4f}\n"
        else:
            text += (f"- **{r['name']}:** {r['acc_min']:.4f} to {r['acc_max']:.4f} "
                     f"(highest in this run: {r['top']})\n")
    text += "\nDistortion (Eq.3, Eq.4) per data set:\n\n"
    for r in rows:
        text += (f"- **{r['name']}:** ε = {r['eps']['eq3']:.4f}, {r['eps']['eq4']:.4f}; "
                 f"smallest ratio m = {r['m']['eq3']:.4f}, {r['m']['eq4']:.4f}\n")
    text += ("\nWhen ε ≥ 1 the lower Johnson–Lindenstrauss inequality is vacuous (1 − ε ≤ 0), so ε "
             "cannot show how well the lemma held. Achlioptas's sufficient target size is at least "
             "24 ln n: ")
    text += "; ".join(f"{r['k0_min']:.0f} for {r['name']} (k used: {r['k']})" for r in rows)
    text += ". Where k is below that, the lemma gives no guarantee for these projections.\n\n"
    return text


def discussion_latex(rows: list) -> str:
    """Discussion paragraphs (LaTeX); every comparison is computed from rows."""
    text = SINGLE_DRAW_CAVEAT + "\n\n"
    text += "Test accuracy across the four kernels ranged from "
    text += "; ".join(f"{r['acc_max']:.4f} on {r['name']} (all four kernels)" if r["acc_min"] == r["acc_max"]
                      else f"{r['acc_min']:.4f} to {r['acc_max']:.4f} on {r['name']} "
                           f"(highest in this run: {r['top']})" for r in rows)
    text += ".\n\n"
    text += (r"When $\varepsilon \geq 1$ the lower Johnson--Lindenstrauss inequality is vacuous, "
             r"so $\varepsilon$ cannot show how well the lemma held. Achlioptas's sufficient "
             r"target size is at least $24 \ln n$: ")
    text += "; ".join(f"{r['k0_min']:.0f} for {r['name']} ($k = {r['k']}$)" for r in rows)
    text += ". Where $k$ is below that, the lemma gives no guarantee for these projections.\n"
    return text


# =============================================================================
# MARKDOWN REPORT
# =============================================================================

def generate_comprehensive_markdown_report(all_results: dict, run_dir: str,
                                           between_analysis: dict,
                                           proj_label: str = "eq3") -> str:
    """
    Generate the comprehensive HW3_REPORT.md.

    This report will be generated automatically by the main script after all analyses are complete. 
    
    It will compile all results into a single, well-structured markdown document that provides a comprehensive 
    overview of the findings from the random projection and SVM classification experiments.

    A comprehensive markdown report is generated that includes:
        - Dataset overview and properties.
        - Summary of random projection distortion metrics (Eq.3 and Eq.4).
        - A comparison table of all kernels with accuracy, precision, recall, and F1 scores.
        - Per-class performance tables for each kernel.
        - Key findings and insights based on the results.
        - References to all generated plots (accuracy comparison, precision/recall charts, confusion matrices).

    The report is saved in the "reports" subdirectory of the main reports folder and copied to the output directory 
    for easy access alongside the CSV and PNG files.

    Args:
        all_results: The dictionary containing all results from the analyses.
        run_dir: The directory where the results are stored.
        between_analysis: The dictionary containing between-dataset analysis results (if any).
        proj_label: The label for the projection type (e.g., "eq3" or "eq4") to identify the correct figures.

    Returns:
        Path to the saved report file (HW3_REPORT.md).
    """
    figures = collect_figures(all_results, run_dir, proj_label)
    svm_key = f"svm_{proj_label}"
    k_target = target_k(all_results)

    report = get_markdown_header(
        "HW3 Report — Random Projections and Multiclass Classification"
    )

    # === Abstract ===
    dataset_names = [all_results[k]["dataset"]["dataset_name"] for k in all_results]
    report += f"""## Abstract

This report presents an implementation and analysis of random projections for
dimensionality reduction combined with multi-class Support Vector Machine (SVM)
classification. Following the Johnson–Lindenstrauss lemma and Achlioptas's
random projection framework, we reduce high-dimensional datasets to k = {k_target} dimensions
using two random matrix distributions (symmetric ±1 and sparse ternary). We then
train and evaluate SVM classifiers with four kernel types (linear, polynomial
degree 2, polynomial degree 3, and Gaussian RBF) on the projected data.
Experiments are conducted on {len(all_results)} dataset(s): {', '.join(dataset_names)}.
We report distortion analysis, classification accuracy, precision, recall, and
F1-scores, along with within-dataset and between-dataset comparisons.

"""

    # === 1. Introduction ===
    report += """## 1. Introduction

Dimensionality reduction is a fundamental preprocessing step in machine learning
that addresses the curse of dimensionality. The Johnson–Lindenstrauss (JL) lemma
provides a theoretical guarantee that points in high-dimensional space can be
projected into a lower-dimensional space while approximately preserving pairwise
distances. Achlioptas (2003) showed that simple random matrices with entries drawn
from specific discrete distributions can achieve this projection efficiently.

In this assignment, we implement Achlioptas's random projection method to reduce
dataset dimensionality to k=""" + str(k_target) + """, then apply multi-class SVM classification with
four different kernels to evaluate classification performance in the projected space.

"""

    # === 2. Background ===
    report += r"""## 2. Background

### 2.1 Johnson–Lindenstrauss Lemma

Given $\varepsilon > 0$ and an integer $n$, for every set $P$ of $n$ points in
$\mathbb{R}^d$, there exists a mapping $f: \mathbb{R}^d \to \mathbb{R}^k$ where
$k = O(\varepsilon^{-2} \log n)$ such that for all $u, v \in P$:

$$(1 - \varepsilon)\|u - v\|^2 \leq \|f(u) - f(v)\|^2 \leq (1 + \varepsilon)\|u - v\|^2$$

### 2.2 Achlioptas's Random Projections

The projection is computed as $E = \frac{1}{\sqrt{k}} A R$ where $R$ is a $d \times k$
random matrix with entries from one of two distributions:

- **Equation (3):** $r_{ij} = \pm 1$ each with probability $1/2$
- **Equation (4):** $r_{ij} = \sqrt{3} \times \{+1, 0, -1\}$ with probabilities $\{1/6, 2/3, 1/6\}$

### 2.3 Support Vector Machines

SVMs find the optimal separating hyperplane by maximizing the margin between classes.
Non-linear classification is achieved through kernel functions. We use scikit-learn's
SVC implementation (Pedregosa et al., 2011).

"""

    # === 3. Methodology ===
    report += "## 3. Methodology\n\n"
    report += "### 3.1 Datasets\n\n"
    report += "| Dataset | Samples | Features | Classes |\n"
    report += "|---------|---------|----------|--------|\n"
    for dk in all_results:
        ds = all_results[dk]["dataset"]
        report += f"| {ds['dataset_name']} | {ds['n_samples']} | {ds['n_features']} | {ds['n_classes']} |\n"

    report += """
### 3.2 Data Preprocessing

All features were scaled to the range [0, 1] using MinMaxScaler to ensure all
dimensions have the same range, as required for meaningful distance comparisons.

### 3.3 Random Projection

For each dataset, we constructed random matrices R using both Equation (3) and
Equation (4) distributions, projecting from the original d dimensions to k=""" + str(k_target) + """.

### 3.4 SVM Classification

We trained four SVM classifiers on the projected data:
1. **Linear** kernel
2. **Polynomial (degree 2)** kernel
3. **Polynomial (degree 3)** kernel
4. **Gaussian (RBF)** kernel

All classifiers were trained on the same training set (70%) and evaluated on the
same test set (30%), using stratified splitting to preserve class proportions.

### 3.5 Evaluation Metrics

- Confusion matrix
- Accuracy
- Per-class precision, recall, and F1-score
- Macro-averaged precision, recall, and F1-score

"""

    # === 4. Results ===
    report += "## 4. Results\n\n"

    # 4.1 Distortion
    report += "### 4.1 Random Projection — Distortion Analysis\n\n"
    report += "| Dataset | Distribution | m | M | ε | Mean Ratio |\n"
    report += "|---------|-------------|---|---|---|------------|\n"
    for dk in all_results:
        ds = all_results[dk]["dataset"]
        proj = all_results[dk]["projection"]
        for eq in ["eq3", "eq4"]:
            dist = proj[f"distortion_{eq}"]
            eq_name = "Eq.3 (±1)" if eq == "eq3" else "Eq.4 (sparse)"
            report += (f"| {ds['dataset_name']} | {eq_name} | "
                       f"{dist['m']:.4f} | {dist['M']:.4f} | "
                       f"{dist['epsilon']:.4f} | {dist['mean_ratio']:.4f} |\n")

    # 4.2 SVM Results per dataset
    report += f"\n### 4.2 SVM Classification Results ({proj_label.upper()} Projection)\n\n"
    for dk in all_results:
        ds = all_results[dk]["dataset"]
        svm_res = all_results[dk][svm_key]
        kr = svm_res["kernel_results"]

        report += f"#### {ds['dataset_name']}\n\n"
        report += "| Kernel | Accuracy | Precision | Recall | F1-Score |\n"
        report += "|--------|----------|-----------|--------|----------|\n"
        for ktag, result in kr.items():
            t = result["test"]
            report += (f"| {result['kernel_name']} | {t['accuracy']:.4f} | "
                       f"{t['precision_macro']:.4f} | {t['recall_macro']:.4f} | "
                       f"{t['f1_macro']:.4f} |\n")
        report += "\n"

    # 4.3 Within-dataset (brief)
    report += "### 4.3 Within-Dataset Comparisons\n\n"
    report += ("Detailed within-dataset analyses comparing the four kernels for each "
               "dataset are provided in separate reports in the `reports/` directory.\n\n")

    # 4.4 Between-datasets
    if between_analysis is not None:
        report += "### 4.4 Between-Dataset Comparisons\n\n"
        pivot = between_analysis.get("pivot_accuracy")
        if pivot is not None:
            report += pivot.round(4).to_markdown() + "\n\n"

    # === Figures ===
    report += "## 5. Figures\n\n"
    for fname, caption in figures:
        report += f"### {caption}\n\n"
        report += f"![{caption}](latex/figures/{fname})\n\n"

    # === Discussion (every comparison is computed from the results) ===
    rows = summarize_run(all_results, svm_key)
    report += "## 6. Discussion\n\n" + discussion_markdown(rows)

    # === Conclusion ===
    low = min(rows, key=lambda r: r["acc_max"])
    high = max(rows, key=lambda r: r["acc_max"])
    report += "## 7. Conclusion\n\n"
    report += (f"With one random matrix per distribution and one split per data set, the highest test "
               f"accuracy among the four kernels on the {PROJ_NAMES.get(proj_label, proj_label)} projection "
               f"ranged from {low['acc_max']:.4f} ({low['name']}) to "
               f"{high['acc_max']:.4f} ({high['name']}). A single run cannot show whether one kernel or "
               f"one distribution is better; that needs repeated draws.\n\n")

    report += """## References

1. Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings
   into a Hilbert space. *Contemporary Mathematics*, 26, 189-206.

2. Achlioptas, D. (2003). Database-friendly random projections:
   Johnson-Lindenstrauss with binary coins. *Journal of Computer and System
   Sciences*, 66(4), 671-687.

3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
   *Journal of Machine Learning Research*, 12, 2825-2830.

4. UCI Machine Learning Repository. https://archive.ics.uci.edu/

"""

    report += get_markdown_footer()

    # Save
    os.makedirs(str(REPORTS_DIR), exist_ok=True)
    report_path = str(REPORTS_DIR / "HW3_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✓ Saved comprehensive report: {report_path}")

    return report_path


# =============================================================================
# LATEX REPORT
# =============================================================================

def generate_ieee_latex_report(all_results: dict, run_dir: str,
                               between_analysis: dict,
                               proj_label: str = "eq3") -> str:
    """
    Generate the IEEE-formatted LaTeX report hw3_report_ieee.tex.

    Returns:
        Path to the saved .tex file
    """
    os.makedirs(str(LATEX_DIR), exist_ok=True)
    svm_key = f"svm_{proj_label}"
    k_target = target_k(all_results)

    dataset_names = [all_results[k]["dataset"]["dataset_name"] for k in all_results]

    # Collect figures for LaTeX
    figures = collect_figures(all_results, run_dir, proj_label)

    tex = r"""\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{float}

\title{Random Projections and Multiclass SVM Classification}
\author{\IEEEauthorblockN{""" + AUTHOR_NAME + r"""}
\IEEEauthorblockA{""" + AUTHOR_AFFILIATION + r"""}}

\begin{document}
\maketitle

\begin{abstract}
This paper presents an implementation and analysis of random projections for
dimensionality reduction combined with multi-class Support Vector Machine (SVM)
classification. Following the Johnson--Lindenstrauss lemma and Achlioptas's
random projection framework, we reduce high-dimensional datasets to $k = """ + str(k_target) + r"""$ dimensions
using two random matrix distributions. We then train and evaluate SVM classifiers
with four kernel types on the projected data. Experiments are conducted on
""" + str(len(all_results)) + r""" dataset(s): """ + ", ".join(dataset_names) + r""".
\end{abstract}

\section{Introduction}

Dimensionality reduction is a fundamental preprocessing step in machine learning
that addresses the curse of dimensionality. The Johnson--Lindenstrauss (JL) lemma
provides a theoretical guarantee that points in high-dimensional space can be
projected into a lower-dimensional space while approximately preserving pairwise
distances. Achlioptas~\cite{achlioptas2003} showed that simple random matrices
with entries drawn from specific discrete distributions can achieve this
projection efficiently.

\section{Background}

\subsection{Johnson--Lindenstrauss Lemma}

Given $\varepsilon > 0$ and an integer $n$, for every set $P$ of $n$ points in
$\mathbb{R}^d$, there exists a mapping $f: \mathbb{R}^d \to \mathbb{R}^k$ where
$k = O(\varepsilon^{-2} \log n)$ such that for all $u, v \in P$:
\begin{equation}
(1 - \varepsilon)\|u - v\|^2 \leq \|f(u) - f(v)\|^2 \leq (1 + \varepsilon)\|u - v\|^2
\end{equation}

\subsection{Achlioptas's Random Projections}

The projection is computed as $E = \frac{1}{\sqrt{k}} A R$ where $R$ is a
$d \times k$ random matrix with entries from one of two distributions.

\subsection{Support Vector Machines}

SVMs find the optimal separating hyperplane by maximizing the margin between
classes. We use scikit-learn's SVC implementation~\cite{pedregosa2011}.

\section{Methodology}

\subsection{Datasets}

"""
    # Dataset table
    tex += r"""\begin{table}[H]
\centering
\caption{Dataset Summary}
\begin{tabular}{lccc}
\toprule
Dataset & Samples & Features & Classes \\
\midrule
"""
    for dk in all_results:
        ds = all_results[dk]["dataset"]
        tex += f"{ds['dataset_name']} & {ds['n_samples']} & {ds['n_features']} & {ds['n_classes']} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    tex += r"""
\subsection{Preprocessing}
All features were scaled to $[0, 1]$ using MinMaxScaler.

\subsection{Random Projection}
Random matrices $R$ of dimension $d \times """ + str(k_target) + r"""$ were generated using both
Equation~(3) ($\pm 1$) and Equation~(4) (sparse ternary) distributions.

\subsection{Classification}
Four SVM kernels were evaluated: linear, polynomial (degrees 2 and 3),
and Gaussian (RBF). All used the same 70/30 stratified train/test split.

\section{Results}

\subsection{Distortion Analysis}

"""
    # Distortion table
    tex += r"""\begin{table}[H]
\centering
\caption{Random Projection Distortion}
\begin{tabular}{llccc}
\toprule
Dataset & Distribution & $m$ & $M$ & $\varepsilon$ \\
\midrule
"""
    for dk in all_results:
        ds = all_results[dk]["dataset"]
        proj = all_results[dk]["projection"]
        for eq in ["eq3", "eq4"]:
            dist = proj[f"distortion_{eq}"]
            eq_name = "Eq.3" if eq == "eq3" else "Eq.4"
            tex += (f"{ds['dataset_name']} & {eq_name} & "
                    f"{dist['m']:.4f} & {dist['M']:.4f} & {dist['epsilon']:.4f} \\\\\n")
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # SVM results tables
    tex += r"""
\subsection{SVM Classification Results}

"""
    for dk in all_results:
        ds = all_results[dk]["dataset"]
        kr = all_results[dk][svm_key]["kernel_results"]
        tex += r"\begin{table}[H]" + "\n"
        tex += r"\centering" + "\n"
        tex += r"\caption{SVM Results --- " + ds["dataset_name"] + "}\n"
        tex += r"""\begin{tabular}{lcccc}
\toprule
Kernel & Accuracy & Precision & Recall & F1 \\
\midrule
"""
        for ktag, result in kr.items():
            t = result["test"]
            tex += (f"{result['kernel_name']} & {t['accuracy']:.4f} & "
                    f"{t['precision_macro']:.4f} & {t['recall_macro']:.4f} & "
                    f"{t['f1_macro']:.4f} \\\\\n")
        tex += r"""\bottomrule
\end{tabular}
\end{table}

"""

    # Select a few key figures
    fig_count = 0
    for fname, caption in figures:
        if fig_count >= 6:
            break
        safe_caption = caption.replace("—", "---").replace("±", "$\\pm$")
        tex += r"""\begin{figure}[H]
\centering
\includegraphics[width=0.9\columnwidth]{figures/""" + fname + r"""}
\caption{""" + safe_caption + r"""}
\end{figure}

"""
        fig_count += 1

    # Discussion and conclusion (every comparison is computed from the results)
    rows = summarize_run(all_results, svm_key)
    low = min(rows, key=lambda r: r["acc_max"])
    high = max(rows, key=lambda r: r["acc_max"])
    tex += "\n\\section{Discussion}\n\n" + discussion_latex(rows) + "\n"
    tex += "\\section{Conclusion}\n\n"
    tex += (f"With one random matrix per distribution and one split per data set, the highest test "
            f"accuracy among the four kernels on the {PROJ_NAMES.get(proj_label, proj_label)} projection "
            f"ranged from {low['acc_max']:.4f} ({low['name']}) to "
            f"{high['acc_max']:.4f} ({high['name']}). A single run cannot show whether one kernel or "
            f"one distribution is better; that needs repeated draws.\n\n")
    tex += r"""\begin{thebibliography}{4}
\bibitem{jl1984}
W.~B. Johnson and J.~Lindenstrauss, ``Extensions of Lipschitz mappings into a
Hilbert space,'' \textit{Contemporary Mathematics}, vol.~26, pp.~189--206, 1984.

\bibitem{achlioptas2003}
D.~Achlioptas, ``Database-friendly random projections: Johnson-Lindenstrauss
with binary coins,'' \textit{J. Computer and System Sciences}, vol.~66, no.~4,
pp.~671--687, 2003.

\bibitem{pedregosa2011}
F.~Pedregosa \textit{et al.}, ``Scikit-learn: Machine learning in Python,''
\textit{J. Machine Learning Research}, vol.~12, pp.~2825--2830, 2011.

\bibitem{uci}
UCI Machine Learning Repository, \url{https://archive.ics.uci.edu/}.
\end{thebibliography}

\end{document}
"""

    os.makedirs(str(LATEX_DIR), exist_ok=True)
    tex_path = str(LATEX_DIR / "hw3_report_ieee.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  ✓ Saved LaTeX report: {tex_path}")

    return tex_path
