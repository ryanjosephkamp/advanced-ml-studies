# kamp_hw1.py

# CS 7063 – Advanced Machine Learning Spring 2026
# Assignment #1: Parzen Classifiers – Distances in ML
# Due: 1/27/2026
# Ryan Kamp
# Date: 1/19/2026
# Revised: November 2026, for this repository

"""
Kernel classifiers on four binary Iris problems, evaluated at five train/test ratios.

The classifier gives each class's training points equal weight (alpha+_i = 1/n+ on positives,
alpha-_i = 1/n- on negatives), sets b = (alpha+ . K alpha+ - alpha- . K alpha-) / 2 from the training
kernel matrix K, and labels a point x by the sign of sum_i (alpha+_i - alpha-_i) K(x_i, x) - b, with
ties going to +1. This assigns x to the class whose mean in the kernel's feature space is nearer.

Kernels: dot product x . y; polynomial (x . y + 1)^2 and (x . y + 1)^3; Gaussian
exp(-||x - y||^2 / (2 sigma^2)) with sigma = 1/sqrt(2). Features are the raw Iris measurements in cm,
not scaled; the data comes from scikit-learn.

Problems: (1) setosa against the other two species, dot product only; (2) virginica against the rest;
(3) versicolor against the rest; (4) versicolor against virginica, with setosa removed. Problems 2 to 4
use all four kernels, so there are 13 configurations.

Splits: train_test_split with random_state=42, not stratified, at 90/10, 80/20, 70/30, 60/40, and
50/50. Every ratio uses the same shuffle, so each smaller test set lies inside the larger ones, and
each ratio is a single split. repeated_splits.py repeats the evaluation over 500 random splits per
ratio.

Outputs:
- parzen_classifier_results.csv: training and test accuracy and confusion matrices for all 65 runs.
- split_<ratio>/: that ratio's results and per-problem accuracy bar charts.
- comparison_analysis/: five comparison figures and comparison_analysis_report.md, whose text is
  computed from the results.

Usage:
    python kamp_hw1.py
"""

import numpy as np
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from datetime import datetime

""" Kernel Functions """
# Dot Product kernel
def dot_product_kernel(x: np.ndarray, y: np.ndarray) -> float:
    return np.dot(x, y)

# Polynomial kernel
def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int, c: float = 1.0) -> float:
    return (np.dot(x, y) + c) ** degree

# Gaussian kernel
def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

""" Parzen Window Classifier Implementation """
# Parzen window classifier
class ParzenWindowClassifier:
    # Initialize the classifier with the specified kernel and its parameters
    def __init__(self, kernel: Callable, **kernel_params):
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.alpha_pos = None
        self.alpha_neg = None
        self.b = None
        self.X_train = None
        self.alpha_prime = None

    # Fit the classifier to the training data
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Store training data
        self.X_train = X
        # Count positive and negative samples
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)

        # Compute alpha vectors
        self.alpha_pos = np.where(y == 1, 1 / n_pos, 0)
        self.alpha_neg = np.where(y == -1, 1 / n_neg, 0)
        
        # Compute kernel matrix
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j], **self.kernel_params)
        
        # Compute bias term b
        self.b = (self.alpha_pos.T @ (K @ self.alpha_pos) - self.alpha_neg.T @ (K @ self.alpha_neg)) / 2
        # Compute alpha_prime vector
        self.alpha_prime = self.alpha_pos - self.alpha_neg

    # Predict the class labels for the test data
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            # Compute h(x)
            h_x = sum(self.alpha_prime[i] * self.kernel(self.X_train[i], x, **self.kernel_params) for i in range(len(self.X_train))) - self.b
            # Assign class label based on sign of h(x)
            predictions.append(1 if h_x >= 0 else -1)
        return np.array(predictions)
    
# Load and preprocess the Iris dataset
def load_iris_data() -> Tuple[np.ndarray, np.ndarray]:
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Create binary labels for the classification problems
def create_binary_labels(y: np.ndarray, positive_class: int) -> np.ndarray:
    binary_y = np.where(y == positive_class, 1, -1)
    return binary_y

# Evaluate the classifier and return accuracy and confusion matrix
def evaluate_classifier(classifier: ParzenWindowClassifier, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    # Predict labels
    y_pred = classifier.predict(X)
    # Calculate accuracy and confusion matrix (rows: true -1, +1; columns: predicted -1, +1)
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred, labels=[-1, 1])
    return accuracy, conf_matrix

# Helper function to run classifiers for a given problem
def run_problem(problem_num: int, problem_name: str, X_train: np.ndarray, X_test: np.ndarray, 
                y_train: np.ndarray, y_test: np.ndarray, kernels: list, verbose: bool = True) -> list:
    """Run all kernel classifiers for a given problem and return results."""
    results = []
    if verbose:  # Print problem header
        print(f"\n{'='*80}")
        print(f"Problem {problem_num}: {problem_name}")
        print(f"{'='*80}")
    
    for kernel in kernels:
        # Handle polynomial kernel with different degrees
        if kernel == polynomial_kernel:
            for degree in [2, 3]:
                classifier = ParzenWindowClassifier(kernel=kernel, degree=degree)
                classifier.fit(X_train, y_train)
                train_acc, train_cm = evaluate_classifier(classifier, X_train, y_train)
                test_acc, test_cm = evaluate_classifier(classifier, X_test, y_test)
                kernel_name = f"Polynomial (d={degree})"
                results.append((problem_num, problem_name, kernel_name, train_acc, train_cm, test_acc, test_cm))
                if verbose:
                    print(f"  {kernel_name}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
        
        # Handle Gaussian kernel
        elif kernel == gaussian_kernel:
            classifier = ParzenWindowClassifier(kernel=kernel, sigma=1/np.sqrt(2))
            classifier.fit(X_train, y_train)
            train_acc, train_cm = evaluate_classifier(classifier, X_train, y_train)
            test_acc, test_cm = evaluate_classifier(classifier, X_test, y_test)
            kernel_name = "Gaussian"
            results.append((problem_num, problem_name, kernel_name, train_acc, train_cm, test_acc, test_cm))
            if verbose:
                print(f"  {kernel_name}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
        
        # Handle dot product kernel
        else:
            classifier = ParzenWindowClassifier(kernel=kernel)
            classifier.fit(X_train, y_train)
            train_acc, train_cm = evaluate_classifier(classifier, X_train, y_train)
            test_acc, test_cm = evaluate_classifier(classifier, X_test, y_test)
            kernel_name = "Dot Product"
            results.append((problem_num, problem_name, kernel_name, train_acc, train_cm, test_acc, test_cm))
            if verbose:
                print(f"  {kernel_name}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
    
    return results


def run_single_split_experiment(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int, 
                                 verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Run all four classification problems for a single train-test split.
    Returns a DataFrame with results and a dict with split metadata.
    """
    # Perform train-test split (consistent across all problems)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Calculate split details
    n_total = len(X) # Total samples
    n_train_full = len(X_train_full) # Training samples
    n_test_full = len(X_test_full) # Testing samples
    train_pct = (1 - test_size) * 100 # Training percentage
    test_pct = test_size * 100 # Testing percentage
    
    if verbose: # Print split information
        print(f"\n{'#'*80}")
        print(f"# TRAIN-TEST SPLIT: {train_pct:.0f}% Train / {test_pct:.0f}% Test")
        print(f"# Training samples: {n_train_full}, Testing samples: {n_test_full}")
        print(f"{'#'*80}")
    
    kernels_p1 = [dot_product_kernel]  # Kernels for Problem 1
    kernels_p234 = [dot_product_kernel, polynomial_kernel, gaussian_kernel] # Kernels for Problems 2, 3, and 4
    
    all_results = [] # Results will include accuracy and confusion matrices for all problems and kernels
    
    # Problem 1: Setosa vs. (Versicolor + Virginica)
    # Create binary labels for Problem 1
    binary_y_train = create_binary_labels(y_train_full, positive_class=0)
    binary_y_test = create_binary_labels(y_test_full, positive_class=0)

    # Run Problem 1
    results_p1 = run_problem(
        problem_num=1,
        problem_name="Setosa vs. (Versicolor + Virginica)",
        X_train=X_train_full, X_test=X_test_full,
        y_train=binary_y_train, y_test=binary_y_test,
        kernels=kernels_p1, verbose=verbose
    )
    all_results.extend(results_p1) # Add results for Problem 1
    
    # Problem 2: Virginica vs. (Setosa + Versicolor)
    # Create binary labels for Problem 2
    binary_y_train = create_binary_labels(y_train_full, positive_class=2)
    binary_y_test = create_binary_labels(y_test_full, positive_class=2)

    # Run Problem 2
    results_p2 = run_problem(
        problem_num=2,
        problem_name="Virginica vs. (Setosa + Versicolor)",
        X_train=X_train_full, X_test=X_test_full,
        y_train=binary_y_train, y_test=binary_y_test,
        kernels=kernels_p234, verbose=verbose
    )
    all_results.extend(results_p2) # Add results for Problem 2
    
    # Problem 3: Versicolor vs. (Setosa + Virginica)
    # Create binary labels for Problem 3
    binary_y_train = create_binary_labels(y_train_full, positive_class=1)
    binary_y_test = create_binary_labels(y_test_full, positive_class=1)

    # Run Problem 3
    results_p3 = run_problem(
        problem_num=3,
        problem_name="Versicolor vs. (Setosa + Virginica)",
        X_train=X_train_full, X_test=X_test_full,
        y_train=binary_y_train, y_test=binary_y_test,
        kernels=kernels_p234, verbose=verbose
    )
    all_results.extend(results_p3) # Add results for Problem 3
    
    # Problem 4: Versicolor vs. Virginica (exclude Setosa)
    # Exclude Setosa from training set
    train_mask = y_train_full != 0
    X_train_p4 = X_train_full[train_mask]
    y_train_p4 = y_train_full[train_mask]
    
    # Exclude Setosa from test set
    test_mask = y_test_full != 0
    X_test_p4 = X_test_full[test_mask]
    y_test_p4 = y_test_full[test_mask]

    # Create binary labels for Problem 4
    binary_y_train_p4 = np.where(y_train_p4 == 1, 1, -1)
    binary_y_test_p4 = np.where(y_test_p4 == 1, 1, -1)
    
    if verbose: # Print sample counts for Problem 4
        print(f"\n  [Problem 4 uses {len(X_train_p4)} training samples and {len(X_test_p4)} test samples (Setosa excluded)]")
    
    # Run Problem 4
    results_p4 = run_problem(
        problem_num=4,
        problem_name="Versicolor vs. Virginica",
        X_train=X_train_p4, X_test=X_test_p4,
        y_train=binary_y_train_p4, y_test=binary_y_test_p4,
        kernels=kernels_p234, verbose=verbose
    )
    all_results.extend(results_p4) # Add results for Problem 4
    
    # Create DataFrame from results
    df_results = pd.DataFrame(all_results,
                              columns=['Problem', 'Problem Name', 'Kernel', 'Train Accuracy', 'Train Confusion Matrix',
                                       'Test Accuracy', 'Test Confusion Matrix'])
    
    # Add train-test split columns
    def get_split_info(row):
        if row['Problem'] == 4: # Problem 4 has reduced samples (Setosa excluded)
            return len(X_train_p4), len(X_test_p4)
        else: # Other problems use full samples
            return n_train_full, n_test_full
    
    # Add split info columns
    df_results['Train Samples'] = df_results.apply(lambda row: get_split_info(row)[0], axis=1)
    df_results['Test Samples'] = df_results.apply(lambda row: get_split_info(row)[1], axis=1)
    df_results['Train %'] = df_results.apply(lambda row: f"{row['Train Samples'] / (row['Train Samples'] + row['Test Samples']) * 100:.0f}%", axis=1)
    df_results['Test %'] = df_results.apply(lambda row: f"{row['Test Samples'] / (row['Train Samples'] + row['Test Samples']) * 100:.0f}%", axis=1)
    df_results['Random State'] = random_state
    df_results['Split Label'] = f"{train_pct:.0f}/{test_pct:.0f}"
    
    # Reorder columns
    df_results = df_results[['Split Label', 'Problem', 'Problem Name', 'Kernel', 'Train Samples', 'Test Samples', 
                              'Train %', 'Test %', 'Random State', 'Train Accuracy', 'Train Confusion Matrix', 
                              'Test Accuracy', 'Test Confusion Matrix']]
    
    # Metadata for plotting
    metadata = {
        'test_size': test_size,
        'train_pct': train_pct,
        'test_pct': test_pct,
        'n_train_full': n_train_full,
        'n_test_full': n_test_full,
        'n_train_p4': len(X_train_p4),
        'n_test_p4': len(X_test_p4),
        'random_state': random_state
    }
    
    return df_results, metadata


def generate_split_plots(df_results: pd.DataFrame, metadata: dict, kernel_colors: dict):
    """Generate training and testing accuracy plots for a single split, saved to a dedicated folder."""
    train_pct = metadata['train_pct'] # Training percentage
    test_pct = metadata['test_pct'] # Testing percentage
    random_state = metadata['random_state'] # Random state used for split
    split_label = f"{train_pct:.0f}_{test_pct:.0f}" # Split label for folder naming
    
    # Create folder for this split
    folder_name = f"split_{split_label}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Generate plots for each problem
    for problem_num in df_results['Problem'].unique():
        subset = df_results[df_results['Problem'] == problem_num] # Subset of results for this problem
        problem_name = subset['Problem Name'].iloc[0] # Problem name
        n_train = subset['Train Samples'].iloc[0] # Training samples
        n_test = subset['Test Samples'].iloc[0] # Testing samples
        train_pct_str = subset['Train %'].iloc[0] # Training percentage string
        test_pct_str = subset['Test %'].iloc[0] # Testing percentage string
        
        bar_colors = [kernel_colors.get(k, '#95a5a6') for k in subset['Kernel']] # Bar colors based on kernel
        split_info = f"Train: {n_train} samples ({train_pct_str}) | Test: {n_test} samples ({test_pct_str}) | Random State: {random_state}" # Split info for titles
        
        # Testing Accuracy Plot for current individual problem
        plt.figure(figsize=(10, 6))
        plt.title(f'Problem {problem_num}: {problem_name}\nTesting Accuracy\n{split_info}', fontsize=11)
        bars = plt.bar(subset['Kernel'], subset['Test Accuracy'], color=bar_colors, edgecolor='black')
        for bar, acc in zip(bars, subset['Test Accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        plt.xlabel('Kernel')
        plt.ylabel('Testing Accuracy')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{folder_name}/test_accuracy_problem_{problem_num}.png', dpi=150)
        plt.close()
        
        # Training Accuracy Plot for current individual problem
        plt.figure(figsize=(10, 6))
        plt.title(f'Problem {problem_num}: {problem_name}\nTraining Accuracy\n{split_info}', fontsize=11)
        bars = plt.bar(subset['Kernel'], subset['Train Accuracy'], color=bar_colors, edgecolor='black')
        for bar, acc in zip(bars, subset['Train Accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        plt.xlabel('Kernel')
        plt.ylabel('Training Accuracy')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{folder_name}/train_accuracy_problem_{problem_num}.png', dpi=150)
        plt.close()
    
    # Also save the split-specific results to the folder
    df_results.to_csv(f'{folder_name}/results_split_{split_label}.csv', index=False)
    
    print(f"  Plots and results saved to folder: {folder_name}/")


def shared_problem_results(df_all_results: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Rows of the problems that every kernel ran on (Problem 1 uses the dot product only), so that
    kernel averages cover the same problems and splits, and a label for them such as "2, 3, 4"."""
    all_kernels = set(df_all_results['Kernel'])
    shared_problems = sorted(p for p in df_all_results['Problem'].unique()
                             if set(df_all_results.loc[df_all_results['Problem'] == p, 'Kernel']) == all_kernels)
    shared_label = ", ".join(str(p) for p in shared_problems)
    return df_all_results[df_all_results['Problem'].isin(shared_problems)], shared_label


def generate_comparison_analysis(df_all_results: pd.DataFrame, kernel_colors: dict, test_sizes: list, random_state: int):
    """Generate comparison analysis plots showing effect of train-test split on results."""
    
    print("\n" + "="*80)
    print("GENERATING COMPARISON ANALYSIS PLOTS")
    print("="*80)
    
    # Create comparison folder
    comparison_folder = "comparison_analysis"
    os.makedirs(comparison_folder, exist_ok=True)
    
    # Get unique splits and problems
    splits = df_all_results['Split Label'].unique()
    problems = df_all_results['Problem'].unique()
    
    # ==================================================================================
    # Plot 1: Test Accuracy vs. Train-Test Split for each Problem (one subplot per problem)
    # ==================================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Iterate over problems to create subplots
    for idx, problem_num in enumerate(problems):
        ax = axes[idx]
        problem_data = df_all_results[df_all_results['Problem'] == problem_num]
        problem_name = problem_data['Problem Name'].iloc[0]
        
        kernels = problem_data['Kernel'].unique()
        x_positions = np.arange(len(splits))
        width = 0.8 / len(kernels)
        
        # Iterate over kernels to plot bars
        for i, kernel in enumerate(kernels):
            # Get data for this kernel and sort by split label to ensure consistent ordering
            kernel_data = problem_data[problem_data['Kernel'] == kernel].sort_values('Split Label')
            test_accs = kernel_data['Test Accuracy'].values
            color = kernel_colors.get(kernel, '#95a5a6')
            ax.bar(x_positions + i * width, test_accs, width, label=kernel, color=color, edgecolor='black')
        
        ax.set_xlabel('Train/Test Split', fontsize=10)
        ax.set_ylabel('Test Accuracy', fontsize=10)
        ax.set_title(f'Problem {problem_num}: {problem_name}', fontsize=11)
        ax.set_xticks(x_positions + width * (len(kernels) - 1) / 2)
        ax.set_xticklabels(sorted(splits), rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Test Accuracy by Train-Test Split (one split per ratio, all from one shuffle)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{comparison_folder}/comparison_test_accuracy_by_problem.png', dpi=150)
    plt.close()
    print(f"  Saved: {comparison_folder}/comparison_test_accuracy_by_problem.png")
    
    # ==================================================================================
    # Plot 2: Train Accuracy vs. Train-Test Split for each Problem (one subplot per problem)
    # ==================================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Iterate over problems to create subplots
    for idx, problem_num in enumerate(problems):
        ax = axes[idx]
        problem_data = df_all_results[df_all_results['Problem'] == problem_num]
        problem_name = problem_data['Problem Name'].iloc[0]
        
        kernels = problem_data['Kernel'].unique()
        x_positions = np.arange(len(splits))
        width = 0.8 / len(kernels)
        
        # Iterate over kernels to plot bars
        for i, kernel in enumerate(kernels):
            # Get data for this kernel and sort by split label to ensure consistent ordering
            kernel_data = problem_data[problem_data['Kernel'] == kernel].sort_values('Split Label')
            train_accs = kernel_data['Train Accuracy'].values
            color = kernel_colors.get(kernel, '#95a5a6')
            ax.bar(x_positions + i * width, train_accs, width, label=kernel, color=color, edgecolor='black')
        
        ax.set_xlabel('Train/Test Split', fontsize=10)
        ax.set_ylabel('Train Accuracy', fontsize=10)
        ax.set_title(f'Problem {problem_num}: {problem_name}', fontsize=11)
        ax.set_xticks(x_positions + width * (len(kernels) - 1) / 2)
        ax.set_xticklabels(sorted(splits), rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Training Accuracy by Train-Test Split (one split per ratio, all from one shuffle)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{comparison_folder}/comparison_train_accuracy_by_problem.png', dpi=150)
    plt.close()
    print(f"  Saved: {comparison_folder}/comparison_train_accuracy_by_problem.png")
    
    # ==================================================================================
    # Plot 3: Line plots of test accuracy across splits for each kernel (one subplot per problem)
    # ==================================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Iterate over problems to create subplots 
    for idx, problem_num in enumerate(problems):
        ax = axes[idx]
        problem_data = df_all_results[df_all_results['Problem'] == problem_num]
        problem_name = problem_data['Problem Name'].iloc[0]
        
        kernels = problem_data['Kernel'].unique()
        
        # Iterate over kernels to plot lines
        for kernel in kernels:
            # Get data for this kernel and sort by split label to ensure correct line plotting
            kernel_data = problem_data[problem_data['Kernel'] == kernel].sort_values('Split Label')
            splits_sorted = kernel_data['Split Label'].values
            test_accs = kernel_data['Test Accuracy'].values
            color = kernel_colors.get(kernel, '#95a5a6')
            ax.plot(splits_sorted, test_accs, marker='o', linewidth=2, markersize=8, 
                   label=kernel, color=color)
        
        ax.set_xlabel('Train/Test Split', fontsize=10)
        ax.set_ylabel('Test Accuracy', fontsize=10)
        ax.set_title(f'Problem {problem_num}: {problem_name}', fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Test Accuracy Across Train-Test Splits (one split per ratio, all from one shuffle)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{comparison_folder}/comparison_test_accuracy_trends.png', dpi=150)
    plt.close()
    print(f"  Saved: {comparison_folder}/comparison_test_accuracy_trends.png")
    
    # ==================================================================================
    # Plot 4: Summary heatmap of test accuracy by kernel and split for each problem (one subplot per problem)
    # ==================================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Iterate over problems to create subplots
    for idx, problem_num in enumerate(problems):
        ax = axes[idx]
        problem_data = df_all_results[df_all_results['Problem'] == problem_num]
        problem_name = problem_data['Problem Name'].iloc[0]
        
        # Pivot to create heatmap data
        pivot_data = problem_data.pivot_table(index='Kernel', columns='Split Label', values='Test Accuracy')
        pivot_data = pivot_data[sorted(pivot_data.columns)]  # Sort columns
        
        # Create heatmap
        im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)
        
        # Add labels
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
        ax.set_yticklabels(pivot_data.index)
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = pivot_data.values[i, j]
                text_color = 'white' if value < 0.7 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=9)
        
        ax.set_title(f'Problem {problem_num}: {problem_name}', fontsize=11)
        ax.set_xlabel('Train/Test Split')
        ax.set_ylabel('Kernel')
    
    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Test Accuracy', fontsize=10)
    
    plt.suptitle('Test Accuracy Heatmap: Kernel × Train-Test Split', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(f'{comparison_folder}/comparison_heatmap.png', dpi=150)
    plt.close()
    print(f"  Saved: {comparison_folder}/comparison_heatmap.png")
    
    # ==================================================================================
    # Plot 5: Average accuracy for each kernel and split (two subplots – test and train), over the
    # problems every kernel ran on, so that each kernel's bars cover the same problems
    # ==================================================================================
    shared_results, shared_label = shared_problem_results(df_all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Average Test Accuracy
    ax = axes[0]
    avg_test = shared_results.groupby(['Split Label', 'Kernel'])['Test Accuracy'].mean().reset_index()
    kernels = avg_test['Kernel'].unique()
    splits_sorted = sorted(avg_test['Split Label'].unique())
    x_positions = np.arange(len(splits_sorted))
    width = 0.8 / len(kernels)
    
    # Iterate over kernels to plot bars
    for i, kernel in enumerate(kernels):
        # Get data for this kernel and sort by split label to ensure consistent ordering
        kernel_data = avg_test[avg_test['Kernel'] == kernel].set_index('Split Label').loc[splits_sorted]
        color = kernel_colors.get(kernel, '#95a5a6')
        ax.bar(x_positions + i * width, kernel_data['Test Accuracy'], width, 
               label=kernel, color=color, edgecolor='black')
    
    ax.set_xlabel('Train/Test Split', fontsize=10)
    ax.set_ylabel('Average Test Accuracy', fontsize=10)
    ax.set_title(f'Average Test Accuracy, Problems {shared_label}', fontsize=12)
    ax.set_xticks(x_positions + width * (len(kernels) - 1) / 2)
    ax.set_xticklabels(splits_sorted, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Average Train Accuracy
    ax = axes[1]
    avg_train = shared_results.groupby(['Split Label', 'Kernel'])['Train Accuracy'].mean().reset_index()
    
    # Iterate over kernels to plot bars
    for i, kernel in enumerate(kernels):
        # Get data for this kernel and sort by split label to ensure consistent ordering
        kernel_data = avg_train[avg_train['Kernel'] == kernel].set_index('Split Label').loc[splits_sorted]
        color = kernel_colors.get(kernel, '#95a5a6')
        ax.bar(x_positions + i * width, kernel_data['Train Accuracy'], width, 
               label=kernel, color=color, edgecolor='black')
    
    ax.set_xlabel('Train/Test Split', fontsize=10)
    ax.set_ylabel('Average Train Accuracy', fontsize=10)
    ax.set_title(f'Average Train Accuracy, Problems {shared_label}', fontsize=12)
    ax.set_xticks(x_positions + width * (len(kernels) - 1) / 2)
    ax.set_xticklabels(splits_sorted, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Average Accuracy by Kernel and Train-Test Split', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{comparison_folder}/comparison_average_accuracy.png', dpi=150)
    plt.close()
    print(f"  Saved: {comparison_folder}/comparison_average_accuracy.png")
    
    # ==================================================================================
    # Generate summary statistics
    # ==================================================================================
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS SUMMARY")
    print("="*80)
    
    # Summary statistics by split
    split_summary = df_all_results.groupby('Split Label').agg({
        'Train Accuracy': ['mean', 'std'],
        'Test Accuracy': ['mean', 'std']
    }).round(4)
    split_summary.columns = ['Train Mean', 'Train Std', 'Test Mean', 'Test Std']
    
    print("\nAverage Test Accuracy by Train-Test Split:")
    print(split_summary.to_string())

    # ==================================================================================
    # Generate Markdown Report
    # ==================================================================================
    generate_markdown_report(df_all_results, split_summary, comparison_folder,
                            test_sizes, random_state)


def generate_markdown_report(df_all_results: pd.DataFrame, split_summary: pd.DataFrame,
                             comparison_folder: str, test_sizes: list, random_state: int):
    """Generate a comprehensive Markdown comparison analysis report."""
    
    report_path = f"{comparison_folder}/comparison_analysis_report.md"
    
    # Get problem descriptions
    problems = df_all_results[['Problem', 'Problem Name']].drop_duplicates().sort_values('Problem')
    
    # Average by kernel over the problems that every kernel ran on
    shared_results, shared_label = shared_problem_results(df_all_results)
    kernel_avg = shared_results.groupby('Kernel').agg({
        'Train Accuracy': ['mean', 'std'],
        'Test Accuracy': ['mean', 'std']
    }).round(4)
    kernel_avg.columns = ['Train Mean', 'Train Std', 'Test Mean', 'Test Std']
    
    # Best overall kernel
    best_kernel = kernel_avg['Test Mean'].idxmax()
    best_kernel_acc = kernel_avg.loc[best_kernel, 'Test Mean']
    
    # Average test accuracy by split, in order of training share. All five splits use the same
    # random state, so they come from one shuffle and their test sets are nested: the report
    # describes the split averages but does not rank the ratios.
    by_train_share = sorted(split_summary.index, key=lambda s: int(s.split('/')[0]))
    split_means = split_summary.loc[by_train_share, 'Test Mean']
    split_steps = np.diff(split_means.values)
    if np.all(split_steps > 0):
        split_trend = "rises with each increase in training share"
    elif np.all(split_steps < 0):
        split_trend = "falls with each increase in training share"
    else:
        split_trend = "does not change steadily with training share"
    split_range = split_means.max() - split_means.min()
    split_means_text = ", ".join(f"{v:.4f}" for v in split_means)
    split_order_text = f"{by_train_share[0]} to {by_train_share[-1]}"

    # Write the report
    with open(report_path, 'w') as f:
        f.write("# comparison_analysis/comparison_analysis_report.md\n\n")
        f.write("# Parzen Window Classifier: Train-Test Split Comparison Analysis\n\n")

        # Report Header
        f.write("**CS 7063 – Advanced Machine Learning Spring 2026**  \n")
        f.write("**Assignment #1: Parzen Classifiers – Distances in ML**  \n")
        f.write("**Author:** Ryan Kamp  \n")
        f.write("**Due Date:** January 27, 2026  \n")
        generated = datetime.now()
        f.write(f"**Generated:** {generated:%B} {generated.day}, {generated.year}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report analyzes the effect of different train-test splits ")
        f.write("on the performance of Parzen window classifiers using various kernel functions on the Iris dataset.\n\n")
        f.write(f"**Key Findings:**\n")
        f.write(f"- **Best performing kernel overall:** {best_kernel} (avg test accuracy on Problems {shared_label}: {best_kernel_acc:.4f})\n")
        f.write(f"- **Split ratio:** average test accuracy for {split_order_text} is {split_means_text} ")
        f.write(f"and {split_trend}; each ratio is one split from the same shuffle (random state {random_state}), ")
        f.write("so these averages do not identify a best ratio.\n")
        f.write(f"- **Total experiments conducted:** {len(df_all_results)}\n\n")
        
        # Experimental Setup
        f.write("---\n\n")
        f.write("## Experimental Setup\n\n")
        f.write("### Dataset\n")
        f.write("- **Dataset:** Iris Dataset (150 samples, 4 features, 3 classes)\n")
        f.write("- **Features:** Sepal length, Sepal width, Petal length, Petal width\n")
        f.write("- **Classes:** Setosa (0), Versicolor (1), Virginica (2)\n\n")
        
        f.write("### Train-Test Splits Evaluated\n\n")
        f.write("| Split | Train % | Test % | Train Samples | Test Samples |\n")
        f.write("|-------|---------|--------|---------------|-------------|\n")
        for ts in test_sizes:
            train_pct = (1 - ts) * 100
            test_pct = ts * 100
            train_samples = int(150 * (1 - ts))
            test_samples = int(150 * ts)
            f.write(f"| {train_pct:.0f}/{test_pct:.0f} | {train_pct:.0f}% | {test_pct:.0f}% | {train_samples} | {test_samples} |\n")
        f.write(f"\n**Random State:** {random_state}\n\n")
        
        f.write("### Kernel Functions\n\n")
        f.write("1. **Dot Product Kernel:** $K(x, y) = x \\cdot y$\n")
        f.write("2. **Polynomial Kernel (d=2):** $K(x, y) = (x \\cdot y + 1)^2$\n")
        f.write("3. **Polynomial Kernel (d=3):** $K(x, y) = (x \\cdot y + 1)^3$\n")
        f.write("4. **Gaussian Kernel:** $K(x, y) = \\exp(-\\|x - y\\|^2 / (2\\sigma^2))$, where $\\sigma = 1/\\sqrt{2}$\n\n")
        
        f.write("### Classification Problems\n\n")
        for _, row in problems.iterrows():
            f.write(f"- **Problem {row['Problem']}:** {row['Problem Name']}\n")
        f.write("\n")
        
        # Results Summary
        f.write("---\n\n")
        f.write("## Results Summary\n\n")
        
        f.write("### Average Accuracy by Train-Test Split\n\n")
        f.write("| Split | Train Mean | Train Std | Test Mean | Test Std |\n")
        f.write("|-------|------------|-----------|-----------|----------|\n")
        for split_label in sorted(split_summary.index):
            row = split_summary.loc[split_label]
            f.write(f"| {split_label} | {row['Train Mean']:.4f} | {row['Train Std']:.4f} | {row['Test Mean']:.4f} | {row['Test Std']:.4f} |\n")
        f.write("\n")
        
        f.write("### Average Accuracy by Kernel\n\n")
        f.write(f"Averaged over Problems {shared_label}, the problems every kernel ran on.\n\n")
        f.write("| Kernel | Train Mean | Train Std | Test Mean | Test Std |\n")
        f.write("|--------|------------|-----------|-----------|----------|\n")
        for kernel in kernel_avg.index:
            row = kernel_avg.loc[kernel]
            f.write(f"| {kernel} | {row['Train Mean']:.4f} | {row['Train Std']:.4f} | {row['Test Mean']:.4f} | {row['Test Std']:.4f} |\n")
        f.write("\n")
        
        # Detailed Analysis
        f.write("---\n\n")
        f.write("## Detailed Analysis\n\n")
        
        f.write("### Train-Test Splits (one split per ratio, all from one shuffle)\n\n")
        f.write(f"Each ratio is evaluated on a single random split, and all five use random state {random_state}, ")
        f.write("so they come from one shuffle: each smaller test set lies inside the larger ones. Differences ")
        f.write("between ratios mix training-set size with which samples happen to be in the test set, and each ")
        f.write("is one draw, so this section describes the ratios without ranking them.\n\n")

        # Analysis of split effects, computed from the results
        f.write(f"1. **Average test accuracy by training share** ({split_order_text}): {split_means_text}; ")
        f.write(f"it {split_trend}.\n\n")

        f.write("2. **Spread within a split:** ")
        min_std_split = split_summary['Test Std'].idxmin()
        max_std_split = split_summary['Test Std'].idxmax()
        f.write(f"the standard deviation of test accuracy across problems and kernels ranges from ")
        f.write(f"{split_summary.loc[min_std_split, 'Test Std']:.4f} ({min_std_split}) to ")
        f.write(f"{split_summary.loc[max_std_split, 'Test Std']:.4f} ({max_std_split}). ")
        f.write("It measures spread across configurations, not variation between runs.\n\n")

        smallest, largest = by_train_share[0], by_train_share[-1]
        f.write("3. **Training and test accuracy:** mean training accuracy is ")
        f.write(f"{split_summary.loc[smallest, 'Train Mean']:.4f} for {smallest} and {split_summary.loc[largest, 'Train Mean']:.4f} for {largest}; ")
        f.write(f"mean test accuracy is {split_summary.loc[smallest, 'Test Mean']:.4f} and {split_summary.loc[largest, 'Test Mean']:.4f}.\n\n")

        f.write("### Kernel Performance Comparison\n\n")
        sorted_kernels = kernel_avg.sort_values('Test Mean', ascending=False)
        f.write(f"1. **Best Kernel:** The {sorted_kernels.index[0]} kernel achieved the highest average test accuracy ")
        f.write(f"({sorted_kernels.iloc[0]['Test Mean']:.4f}) on Problems {shared_label} across all splits.\n\n")

        f.write("2. **Kernel Ranking by Test Accuracy:**\n")
        for i, (kernel, row) in enumerate(sorted_kernels.iterrows(), 1):
            f.write(f"   {i}. {kernel}: {row['Test Mean']:.4f}\n")
        f.write("\n")

        f.write("3. **By problem (mean test accuracy over all splits):**\n")
        for problem_num in sorted(df_all_results['Problem'].unique()):
            means = df_all_results[df_all_results['Problem'] == problem_num].groupby('Kernel')['Test Accuracy'].mean()
            if len(means) > 1:
                f.write(f"   - Problem {problem_num}: highest {means.idxmax()} ({means.max():.4f}), "
                        f"lowest {means.idxmin()} ({means.min():.4f}).\n")
            else:
                f.write(f"   - Problem {problem_num}: {means.index[0]} only ({means.iloc[0]:.4f}).\n")
        f.write("\n")
        
        # Problem-specific analysis
        f.write("### Problem-Specific Analysis\n\n")
        for problem_num in sorted(df_all_results['Problem'].unique()):
            problem_data = df_all_results[df_all_results['Problem'] == problem_num]
            problem_name = problem_data['Problem Name'].iloc[0]
            
            f.write(f"#### Problem {problem_num}: {problem_name}\n\n")
            
            # Create pivot table for this problem
            pivot = problem_data.pivot_table(index='Kernel', columns='Split Label', values='Test Accuracy')
            pivot = pivot[sorted(pivot.columns)]
            
            f.write("| Kernel | " + " | ".join(sorted(pivot.columns)) + " |\n")
            f.write("|--------|" + "|".join(["-------" for _ in pivot.columns]) + "|\n")
            for kernel in pivot.index:
                values = " | ".join([f"{pivot.loc[kernel, col]:.4f}" for col in pivot.columns])
                f.write(f"| {kernel} | {values} |\n")
            f.write("\n")
        
        # Visualizations
        f.write("---\n\n")
        f.write("## Visualizations\n\n")
        f.write("The following plots are available in this folder:\n\n")
        f.write("1. **comparison_test_accuracy_by_problem.png** - Bar charts showing test accuracy across all splits for each problem\n")
        f.write("2. **comparison_train_accuracy_by_problem.png** - Bar charts showing train accuracy across all splits for each problem\n")
        f.write("3. **comparison_test_accuracy_trends.png** - Line plots showing how test accuracy changes with different splits\n")
        f.write("4. **comparison_heatmap.png** - Heatmap visualization of test accuracy (Kernel × Split) for each problem\n")
        f.write(f"5. **comparison_average_accuracy.png** - Average accuracy over Problems {shared_label} for each kernel and split\n\n")
        
        # Conclusions
        f.write("---\n\n")
        f.write("## Conclusions\n\n")
        kernel_range = kernel_avg['Test Mean'].max() - kernel_avg['Test Mean'].min()
        f.write(f"1. **Kernel:** {best_kernel} has the highest average test accuracy on Problems {shared_label} ")
        f.write(f"({best_kernel_acc:.4f}); the kernel averages span {kernel_range:.4f}.\n\n")
        f.write(f"2. **Split:** average test accuracy differs by {split_range:.4f} across the ratios ")
        f.write(f"and {split_trend}. With one split per ratio, all from the same shuffle, this comparison does not ")
        f.write("identify a best ratio; repeated_splits.py evaluates each ratio over 500 random splits.\n\n")
        
        f.write("---\n\n")
        f.write("*Report generated automatically by the Parzen Window Classifier analysis script, kamp_hw1.py.*\n")
    
    print(f"  Saved: {report_path}")


# Main function to run the experiments
def main():

    # Print assignment header
    print("#"*80)
    print("# CS 7063 – Advanced Machine Learning (Spring 2026)")
    print("# Assignment 1: Parzen Classifiers – Distances in ML")
    print("# Python Implementation with Multi-Split Analysis")
    print("# File: kamp_hw1.py")
    print("# Author: Ryan Kamp")
    print("# Date: 2026-01-19")
    print("#"*80)
    
    # Load data
    X, y = load_iris_data()
    n_total = len(X)
    random_state = 42
    
    # Define range of train-test splits to evaluate
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]  # 90/10, 80/20, 70/30, 60/40, 50/50
    
    print("\n" + "#"*80)
    print("# PARZEN WINDOW CLASSIFIER - MULTI-SPLIT ANALYSIS")
    print(f"# Total samples: {n_total}")
    print(f"# Test sizes to evaluate: {test_sizes}")
    print(f"# Random state: {random_state}")
    print("#"*80)
    
    # Define colors for different kernels
    kernel_colors = {
        'Dot Product': '#3498db',      # Blue
        'Polynomial (d=2)': '#e74c3c', # Red
        'Polynomial (d=3)': '#9b59b6', # Purple
        'Gaussian': '#2ecc71'          # Green
    }
    
    # Collect all results across all splits
    all_split_results = []
    all_metadata = []
    
    # Run experiments for each train-test split
    for test_size in test_sizes:
        df_results, metadata = run_single_split_experiment(X, y, test_size, random_state, verbose=True)
        all_split_results.append(df_results)
        all_metadata.append(metadata)
        
        # Generate plots for this split
        generate_split_plots(df_results, metadata, kernel_colors)
    
    # Combine all results into a single DataFrame
    df_all_results = pd.concat(all_split_results, ignore_index=True)
    
    # Save all results to CSV
    df_all_results.to_csv('parzen_classifier_results.csv', index=False)
    print(f"\n{'='*80}")
    print("All results saved to parzen_classifier_results.csv")
    
    # Generate comparison analysis plots and report
    generate_comparison_analysis(df_all_results, kernel_colors, test_sizes, random_state)
    
    # Print final results summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY - ALL SPLITS")
    print("="*80)
    print(f"\nTrain-Test Splits Evaluated: {[f'{(1-ts)*100:.0f}/{ts*100:.0f}' for ts in test_sizes]}")
    print(f"Random State: {random_state}")
    print(f"Total Experiments: {len(df_all_results)}")
    print("\nComplete Results Table:")
    print(df_all_results.to_string(index=False))

    
# Run the main function
if __name__ == "__main__":
    main()
