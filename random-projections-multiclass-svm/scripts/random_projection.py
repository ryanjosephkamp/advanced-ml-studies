#!/usr/bin/env python3
"""
random_projection.py - Random Projections for Dimensionality Reduction

This module implements Part I of the assignment: Achlioptas-style random projections
with two distribution types (Equation 3: ±1, Equation 4: sparse ternary with sqrt(3) scaling). 
It includes data scaling, projection, pairwise distance computation, and distortion analysis by the
Johnson–Lindenstrauss lemma.

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
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


# =============================================================================
# DATA SCALING
# =============================================================================

def scale_data(X: np.ndarray) -> tuple:
    """
    Scale features to [0, 1] using MinMaxScaler so all dimensions have the
    same range (as required by the assignment).
    Scaling the features in this way ensures that the random projection is not biased 
    by the scale of the original features, and it allows the Johnson–Lindenstrauss lemma 
    to hold more effectively by ensuring that all dimensions contribute equally to the distance calculations.

    Args:
        X: Data matrix of shape (n, d)

    Returns:
        (X_scaled, scaler) tuple
            X_scaled: Scaled data matrix of shape (n, d)
            scaler: Fitted MinMaxScaler object (useful for inverse transformations if needed)
    """
    scaler = MinMaxScaler() # Initialize the MinMaxScaler to scale features to the range [0, 1]
    X_scaled = scaler.fit_transform(X) # Fit the scaler to the data and transform it, resulting in a scaled data matrix where each feature is scaled to the [0, 1] range
    return X_scaled, scaler # Return the scaled data matrix and the fitted scaler object for potential future use (e.g., inverse scaling)


# =============================================================================
# RANDOM MATRIX CONSTRUCTION
# =============================================================================

def build_random_matrix_eq3(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build a d×k random matrix R using Equation (3): symmetric ±1 distribution.

    r_{ij} = +1 with probability 1/2
             -1 with probability 1/2

    Construction:
        1. Generate uniform random M1 of shape (d, k).
        2. R[M1 <= 0.5] = -1, R[M1 > 0.5] = +1.

    Args:
        d: Original dimensionality (number of features).
        k: Target dimensionality.
        rng: NumPy random generator.

    Returns:
        R matrix of shape (d, k).
    """
    M1 = rng.random((d, k)) # Generate uniform random matrix of shape (d, k)
    R = np.ones((d, k), dtype=np.float64) # Initialize R with ones
    R[M1 <= 0.5] = -1.0 # Set entries to -1 where M1 <= 0.5, leaving the rest as +1
    return R


def build_random_matrix_eq4(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build a d×k random matrix R using Equation (4): sparse ternary distribution.

    r_{ij} = sqrt(3) × { +1  with probability 1/6
                          0  with probability 2/3
                         -1  with probability 1/6 }

    Construction:
        1. Generate uniform random M1 of shape (d, k).
        2. R[M1 <= 1/6] = -1.
        3. R[M1 > 5/6] = +1.
        4. R[1/6 < M1 <= 5/6] = 0.
        5. Multiply entire R by sqrt(3).

    Args:
        d: Original dimensionality.
        k: Target dimensionality.
        rng: NumPy random generator.

    Returns:
        R matrix of shape (d, k).
    """
    M1 = rng.random((d, k)) # Generate uniform random matrix of shape (d, k)
    R = np.zeros((d, k), dtype=np.float64) # Initialize R with zeros (the default value for the middle 2/3 of the distribution)
    R[M1 <= 1.0 / 6.0] = -1.0 # Set entries to -1 where M1 <= 1/6
    R[M1 > 5.0 / 6.0] = 1.0 # Set entries to +1 where M1 > 5/6, leaving the rest as 0
    # Middle values stay 0
    R = R * np.sqrt(3.0) # Scale the entire matrix by sqrt(3)
    # The scaling by sqrt(3) is necessary to ensure that the variance of the distribution is 1, 
    # which is a key requirement for the Johnson–Lindenstrauss lemma to hold effectively.
    return R


# =============================================================================
# PROJECTION
# =============================================================================

def project_data(A: np.ndarray, R: np.ndarray, k: int) -> np.ndarray:
    """
    Project data matrix A using random matrix R.

    E = (1 / sqrt(k)) * A @ R

    Args:
        A: Scaled data matrix of shape (n, d).
        R: Random matrix of shape (d, k).
        k: Target dimensionality.

    Returns:
        Projected matrix E of shape (n, k).
    """
    # Perform the matrix multiplication A @ R and scale by 1/sqrt(k) to get the projected data matrix E of shape (n, k):
    E = (1.0 / np.sqrt(k)) * (A @ R) 
    return E


# =============================================================================
# PAIRWISE DISTANCES
# =============================================================================

def compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix.

    Args:
        X: Data matrix of shape (n, d) or (n, k).

    Returns:
        Distance matrix D of shape (n, n).
    """
    # Compute pairwise Euclidean distances and convert to a square matrix of shape (n, n) where D[i, j] is the distance between samples i and j:
    D = squareform(pdist(X, metric="euclidean")) 
    return D


# =============================================================================
# DISTORTION ANALYSIS
# =============================================================================

def compute_distortion(D1: np.ndarray, D2: np.ndarray) -> dict:
    """
    Compute distortion bounds m and M and empirical epsilon.

    Distortion is defined as the tightest bounds m and M such that for all pairs of distinct points (i, j):
    m * D1(i, j) <= D2(i, j) <= M * D1(i, j) for all i != j, where D1 is the original distance matrix and D2 is the projected distance matrix.
    
    The empirical epsilon is then calculated as max(1 - m, M - 1), which represents the maximum relative distortion of distances due to the projection. 
    
    An epsilon close to 0 indicates that the projection preserves distances well, while a larger epsilon indicates more distortion.

    Distortion analysis involves comparing the pairwise distances in the original space (D1) with those in the projected space (D2)
    to assess how well the random projection preserves the geometry of the data.

    By finding the tightest bounds m and M, we can quantify the worst-case distortion of distances due to the projection, 
    and the empirical epsilon gives us a single value that summarizes this distortion in terms of how much distances can shrink (1 - m) 
    or stretch (M - 1) relative to the original distances.

    Conceptually, this analysis is crucial for understanding the effectiveness of the random projection in preserving the structure of the data.
    
    The Johnson–Lindenstrauss lemma states that with high probability, a random projection can preserve distances within a factor of (1 ± ε) 
    for a sufficiently large target dimensionality k.

    Distortion analysis allows us to empirically verify this property for our specific random projection and dataset, 
    and to compare the performance of different random matrix constructions (e.g., Eq.3 vs Eq.4) in terms of how well they preserve distances.

    Find tightest m, M such that:
        m * D1(i,j) <= D2(i,j) <= M * D1(i,j)  for all i != j

    Then epsilon = max(1 - m, M - 1).

    Args:
        D1: Original pairwise distance matrix (n, n).
        D2: Projected pairwise distance matrix (n, n).

    Returns:
        Dictionary with keys: m, M, epsilon, eps_from_m, eps_from_M, ratios, mean_ratio, std_ratio, median_ratio, n_pairs.
    """
    # Only consider upper triangle (unique pairs) where D1 > 0:
    # This ensures we are only comparing distances between distinct points and avoids division by zero when computing ratios.
    mask = np.triu(D1, k=1) > 0 # Create a boolean mask for the upper triangle of D1 (excluding the diagonal) where distances are greater than 0, indicating valid pairs of distinct points
    d1_vals = D1[mask] # Extract the valid distance values from D1 using the mask
    d2_vals = D2[mask] # Extract the corresponding distance values from D2 using the same mask, ensuring that we are comparing the same pairs of points in both matrices

    # Compute ratios D2/D1:
    # These ratios represent how much the distances have changed due to the projection. 
    # A ratio of 1 means perfect preservation, a ratio less than 1 means the distance has shrunk, 
    # and a ratio greater than 1 means the distance has stretched.
    ratios = d2_vals / d1_vals

    m = float(ratios.min()) # The minimum ratio gives us the tightest lower bound m, indicating the maximum amount by which distances can shrink due to the projection
    M = float(ratios.max()) # The maximum ratio gives us the tightest upper bound M, indicating the maximum amount by which distances can stretch due to the projection

    # epsilon: the single value such that m >= 1-eps and M <= 1+eps
    eps_from_m = 1.0 - m
    eps_from_M = M - 1.0
    epsilon = max(eps_from_m, eps_from_M) 

    # Return a dictionary containing all the relevant distortion metrics:
    # This includes the tightest bounds m and M, the empirical epsilon, the individual contributions to epsilon from m and M, the array of distance ratios, 
    # and summary statistics of the ratios.
    return {
        "m": m, # Tightest lower bound on distance preservation (maximum shrinkage)
        "M": M, # Tightest upper bound on distance preservation (maximum stretch)
        "epsilon": epsilon, # Empirical distortion measure summarizing the maximum relative distortion of distances due to the projection
        "eps_from_m": eps_from_m, # Contribution to epsilon from the lower bound m (how much distances can shrink)
        "eps_from_M": eps_from_M, # Contribution to epsilon from the upper bound M (how much distances can stretch)
        "ratios": ratios, # Array of distance ratios D2/D1 for all valid pairs, which can be used for further analysis or plotting
        "mean_ratio": float(ratios.mean()), # Mean of the distance ratios, indicating the average distortion of distances due to the projection
        "std_ratio": float(ratios.std()), # Standard deviation of the distance ratios, indicating the variability of distortion across different pairs of points
        "median_ratio": float(np.median(ratios)), # Median of the distance ratios, providing a robust measure of central tendency for the distortion of distances
        "n_pairs": len(ratios), # Number of valid pairs of points considered in the distortion analysis (i.e., pairs with D1 > 0)
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_2d_projection(E: np.ndarray, y: np.ndarray, class_names: list,
                       title: str, save_path: str) -> None:
    """
    Generate a 2-D scatter plot of projected data colored by class.
    
    This plot visualizes the projected data in 2D space, with points colored according to their class labels.
    It helps to visually assess how well the random projection has preserved the structure of the data,
    such as whether points from the same class are still clustered together and whether different classes are still separated after projection.

    The plot is saved to the specified path, and it includes a title, axis labels, a legend, and a grid for better readability.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("Set2", len(class_names))

    for ci, cn in enumerate(class_names):
        mask = y == ci
        ax.scatter(E[mask, 0], E[mask, 1], c=[colors[ci]], label=cn,
                   alpha=0.7, edgecolors="k", linewidth=0.3, s=50)

    ax.set_xlabel("Projected Dimension 1", fontsize=12)
    ax.set_ylabel("Projected Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_distance_ratios(ratios: np.ndarray, m: float, M: float,
                         title: str, save_path: str) -> None:
    """
    Generate a histogram of D2/D1 distance ratios with m and M marked.

    This histogram visualizes the distribution of the distance ratios (D2/D1) for all pairs of points, showing how distances have changed due to the projection.
    The vertical lines indicate the ideal ratio (1.0) and the distortion bounds (m and M), helping to assess the quality of the projection.

    A good projection will have most ratios clustered around 1.0, with m close to 1 and M close to 1, indicating that distances are well preserved.

    The plot is saved to the specified path, and it includes a title, axis labels, a legend, and a grid for better readability.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=50, alpha=0.7, color="steelblue", edgecolor="black",
            linewidth=0.3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Ratio = 1.0")
    ax.axvline(m, color="red", linestyle="--", linewidth=1.5, label=f"m = {m:.4f}")
    ax.axvline(M, color="darkred", linestyle="--", linewidth=1.5, label=f"M = {M:.4f}")
    ax.set_xlabel("Distance Ratio (D2 / D1)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_d1_vs_d2(D1: np.ndarray, D2: np.ndarray, m: float, M: float,
                  title: str, save_path: str) -> None:
    """
    Generate a scatter plot of D1 vs D2 with distortion bounds.

    This scatter plot visualizes the relationship between the original distances (D1) and the projected distances (D2) for all pairs of points.
    The y=x line represents perfect preservation of distances, while the lines y=m*x and y=M*x represent the distortion bounds.

    The plot is saved to the specified path, and it includes a title, axis labels, a legend, and a grid for better readability.
    """
    mask = np.triu(D1, k=1) > 0 # Create a boolean mask for the upper triangle of D1 (excluding the diagonal) where distances are greater than 0, indicating valid pairs of distinct points
    d1_vals = D1[mask] # Extract the valid distance values from D1 using the mask
    d2_vals = D2[mask] # Extract the corresponding distance values from D2 using the same mask, ensuring that we are comparing the same pairs of points in both matrices

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(d1_vals, d2_vals, alpha=0.15, s=5, c="steelblue")

    # Plot y=x line (perfect preservation)
    max_val = max(d1_vals.max(), d2_vals.max()) * 1.05
    line_x = np.array([0, max_val])
    ax.plot(line_x, line_x, "k--", linewidth=1.5, label="y = x (perfect)")
    ax.plot(line_x, m * line_x, "r--", linewidth=1.2,
            label=f"y = {m:.3f}x (lower bound)")
    ax.plot(line_x, M * line_x, "r-.", linewidth=1.2,
            label=f"y = {M:.3f}x (upper bound)")

    ax.set_xlabel("Original Distance (D1)", fontsize=12)
    ax.set_ylabel("Projected Distance (D2)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def projection_rng(seed: int, dataset_key: str, legacy_seeding: bool = False) -> np.random.Generator:
    """
    Random generator for one dataset's projection matrices.

    By default each dataset gets its own stream, keyed by the seed and a stable hash (CRC-32) of
    the dataset key, so the draws are independent across datasets and don't depend on the order
    in which datasets are run.

    With legacy_seeding=True every dataset starts from default_rng(seed), as in the February 13,
    2026 run: the Eq.3 matrices of all datasets then share their first rows. Use it to reproduce
    the original numbers (python scripts/kamp_hw3.py --legacy-seeding ...).
    """
    if legacy_seeding:
        return np.random.default_rng(seed)
    return np.random.default_rng([seed, zlib.crc32(dataset_key.encode("utf-8"))])


def run_random_projection(dataset_dict: dict, output_dir: str,
                          k: int = 2, seed: int = 42,
                          legacy_seeding: bool = False) -> dict:
    """
    Run the full Part I pipeline for one dataset: scale, project (both
    distributions), compute distances, measure distortion, generate plots.

    Args:
        dataset_dict: Dataset dictionary from data_loader.
        output_dir: Directory to save results.
        k: Target dimensionality (default: 2).
        seed: Random seed for reproducibility.
        legacy_seeding: Reproduce the February 13, 2026 seeding (same stream for every dataset).

    Returns:
        Dictionary with all projection results.
    """
    ds_key = dataset_dict["dataset_key"] # e.g., "iris", "breast_cancer"
    ds_name = dataset_dict["dataset_name"] # e.g., "Iris", "Breast Cancer Wisconsin"
    X = dataset_dict["X"] # Original data matrix of shape (n, d) with raw feature values
    y = dataset_dict["y"] # Class labels for each sample, used for coloring in plots
    class_names = dataset_dict["class_names"] # List of class names corresponding to the unique values in y, used for legend in plots
    n = dataset_dict["n_samples"] # Number of samples in the dataset, used for distance calculations and reporting
    d = dataset_dict["n_features"] # Original dimensionality (number of features) in the dataset, used for constructing random matrices and reporting

    os.makedirs(output_dir, exist_ok=True)
    rng = projection_rng(seed, ds_key, legacy_seeding)

    print(f"\n{'═' * 60}")
    print(f"  RANDOM PROJECTION — {ds_name}")
    print(f"  n={n}, d={d}, target k={k}, seed={seed}"
          f"{' (legacy seeding)' if legacy_seeding else ''}")
    print(f"{'═' * 60}")

    # --- Step 1: Scale data ---
    print("\n  [Step 1] Scaling data (MinMaxScaler to [0, 1])...")
    A, scaler = scale_data(X)
    print(f"    Scaled data shape: {A.shape}")

    # --- Step 2: Build random matrices ---
    print("\n  [Step 2] Constructing random matrices...")
    R_eq3 = build_random_matrix_eq3(d, k, rng)
    R_eq4 = build_random_matrix_eq4(d, k, rng)
    print(f"    R (Eq.3) shape: {R_eq3.shape}  |  R (Eq.4) shape: {R_eq4.shape}")
    print(f"    R (Eq.3) unique values: {sorted(np.unique(R_eq3))}")
    print(f"    R (Eq.4) unique values: {sorted(np.unique(R_eq4).round(4))}")

    # --- Step 3: Project data ---
    print("\n  [Step 3] Projecting data into k={} dimensions...".format(k))
    E_eq3 = project_data(A, R_eq3, k)
    E_eq4 = project_data(A, R_eq4, k)
    print(f"    E (Eq.3) shape: {E_eq3.shape}  |  E (Eq.4) shape: {E_eq4.shape}")

    # --- Step 4: Pairwise distances in original space ---
    print("\n  [Step 4] Computing pairwise distances in ORIGINAL space...")
    D1 = compute_pairwise_distances(A)
    n_pairs = n * (n - 1) // 2
    print(f"    D1 shape: {D1.shape}  |  Number of pairs: {n_pairs}")

    # --- Step 5: Pairwise distances in projected space ---
    print("\n  [Step 5] Computing pairwise distances in PROJECTED space...")
    D2_eq3 = compute_pairwise_distances(E_eq3)
    D2_eq4 = compute_pairwise_distances(E_eq4)
    print(f"    D2 (Eq.3) shape: {D2_eq3.shape}")
    print(f"    D2 (Eq.4) shape: {D2_eq4.shape}")

    # --- Step 6: Distortion analysis ---
    print("\n  [Step 6] Computing distortion bounds...")
    dist_eq3 = compute_distortion(D1, D2_eq3)
    dist_eq4 = compute_distortion(D1, D2_eq4)

    # Print distortion summary
    dist_table = [
        ["Eq.3 (±1)", f"{dist_eq3['m']:.6f}", f"{dist_eq3['M']:.6f}",
         f"{dist_eq3['epsilon']:.6f}", f"{dist_eq3['mean_ratio']:.6f}",
         f"{dist_eq3['std_ratio']:.6f}"],
        ["Eq.4 (sparse)", f"{dist_eq4['m']:.6f}", f"{dist_eq4['M']:.6f}",
         f"{dist_eq4['epsilon']:.6f}", f"{dist_eq4['mean_ratio']:.6f}",
         f"{dist_eq4['std_ratio']:.6f}"],
    ]
    print("\n  Distortion Summary:")
    print(tabulate(dist_table,
                   headers=["Distribution", "m", "M", "ε", "Mean Ratio", "Std Ratio"],
                   tablefmt="grid"))

    # === Save CSVs ===
    # Distortion summary
    dist_summary_df = pd.DataFrame([
        {"distribution": "Eq3_symmetric", "m": dist_eq3["m"], "M": dist_eq3["M"],
         "epsilon": dist_eq3["epsilon"], "mean_ratio": dist_eq3["mean_ratio"],
         "std_ratio": dist_eq3["std_ratio"], "median_ratio": dist_eq3["median_ratio"],
         "n_pairs": dist_eq3["n_pairs"]},
        {"distribution": "Eq4_sparse", "m": dist_eq4["m"], "M": dist_eq4["M"],
         "epsilon": dist_eq4["epsilon"], "mean_ratio": dist_eq4["mean_ratio"],
         "std_ratio": dist_eq4["std_ratio"], "median_ratio": dist_eq4["median_ratio"],
         "n_pairs": dist_eq4["n_pairs"]},
    ])
    dist_summary_df.to_csv(os.path.join(output_dir, f"{ds_key}_distortion_summary.csv"),
                           index=False)

    # Projected data CSVs
    for label, E_data in [("eq3", E_eq3), ("eq4", E_eq4)]:
        proj_df = pd.DataFrame(E_data, columns=[f"dim_{i+1}" for i in range(k)])
        proj_df["class"] = y
        proj_df["class_name"] = [class_names[int(c)] for c in y]
        proj_df.to_csv(os.path.join(output_dir, f"{ds_key}_projected_data_{label}.csv"),
                       index=False)

    # === Generate plots ===
    print("\n  Generating plots...")

    # 2-D scatter plots
    plot_2d_projection(
        E_eq3, y, class_names,
        f"{ds_name} — Random Projection (Eq.3: ±1)",
        os.path.join(output_dir, f"{ds_key}_projection_eq3.png"),
    )
    plot_2d_projection(
        E_eq4, y, class_names,
        f"{ds_name} — Random Projection (Eq.4: Sparse Ternary)",
        os.path.join(output_dir, f"{ds_key}_projection_eq4.png"),
    )

    # Distance ratio histograms
    plot_distance_ratios(
        dist_eq3["ratios"], dist_eq3["m"], dist_eq3["M"],
        f"{ds_name} — Distance Ratios (Eq.3)",
        os.path.join(output_dir, f"{ds_key}_dist_ratios_eq3.png"),
    )
    plot_distance_ratios(
        dist_eq4["ratios"], dist_eq4["m"], dist_eq4["M"],
        f"{ds_name} — Distance Ratios (Eq.4)",
        os.path.join(output_dir, f"{ds_key}_dist_ratios_eq4.png"),
    )

    # D1 vs D2 scatter
    plot_d1_vs_d2(
        D1, D2_eq3, dist_eq3["m"], dist_eq3["M"],
        f"{ds_name} — Original vs Projected Distances (Eq.3)",
        os.path.join(output_dir, f"{ds_key}_d1_vs_d2_eq3.png"),
    )
    plot_d1_vs_d2(
        D1, D2_eq4, dist_eq4["m"], dist_eq4["M"],
        f"{ds_name} — Original vs Projected Distances (Eq.4)",
        os.path.join(output_dir, f"{ds_key}_d1_vs_d2_eq4.png"),
    )

    print(f"  ✓ Random projection complete for {ds_name}")

    # Return a comprehensive dictionary containing all relevant results and objects for potential further analysis or use in subsequent parts of the assignment:
    return {
        "dataset_key": ds_key, # e.g., "iris", "breast_cancer"
        "dataset_name": ds_name, # e.g., "Iris", "Breast Cancer Wisconsin"
        "A": A, # Scaled data matrix of shape (n, d) used for projection and distance calculations (instead of the original X, which is not scaled)
        "E_eq3": E_eq3, # Projected data matrix using Equation (3) random matrix, of shape (n, k)
        "E_eq4": E_eq4, # Projected data matrix using Equation (4) random matrix, of shape (n, k)
        "R_eq3": R_eq3, # Random matrix used for Equation (3) projection, of shape (d, k)
        "R_eq4": R_eq4, # Random matrix used for Equation (4) projection, of shape (d, k)
        "D1": D1, # Pairwise distance matrix in the original scaled space, of shape (n, n)
        "D2_eq3": D2_eq3, # Pairwise distance matrix in the projected space using Equation (3), of shape (n, n)
        "D2_eq4": D2_eq4, # Pairwise distance matrix in the projected space using Equation (4), of shape (n, n)
        "distortion_eq3": dist_eq3, # Distortion analysis results for Equation (3) projection, containing m, M, epsilon, and ratio statistics
        "distortion_eq4": dist_eq4, # Distortion analysis results for Equation (4) projection, containing m, M, epsilon, and ratio statistics
        "scaler": scaler, # Fitted MinMaxScaler object used for scaling the original data, which can be useful for inverse transformations if needed
        "k": k, # Target dimensionality used for the projection
        "d": d, # Original dimensionality (number of features) in the dataset
        "n": n, # Number of samples in the dataset
        "y": y, # Class labels for each sample, used for coloring in plots and potential further analysis
        "class_names": class_names, # List of class names corresponding to the unique values in y, used for legend in plots and potential further analysis
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    from data_loader import load_dataset

    HW3_DIR = Path(__file__).parent.parent # Assuming this script is in HW3/scripts/, we go up one level to HW3/ to access the data_loader and results directory
    # Load the Iris dataset and run the random projection pipeline, saving results to the specified directory:
    ds = load_dataset("iris")
    results = run_random_projection(ds, str(HW3_DIR / "results" / "test_projection"))
    print(f"\nDistortion (Eq3): eps = {results['distortion_eq3']['epsilon']:.4f}")
    print(f"Distortion (Eq4): eps = {results['distortion_eq4']['epsilon']:.4f}")
