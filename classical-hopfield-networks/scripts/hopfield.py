#!/usr/bin/env python3
"""
hopfield.py - Hopfield Network Implementation

This script implements a Hopfield network for pattern recognition/recall.
It includes functions for computing weight matrices, updating queries,
and computing network energy.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 2 – Hopfield Networks
Date: February 2, 2026
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm


def compute_weight_matrix(v: np.ndarray) -> np.ndarray:
    """
    Compute the weight matrix for a single pattern vector.
    
    The weight matrix is computed using the Hebbian learning rule:
        W[i,j] = (2*v[i] - 1) * (2*v[j] - 1) for i != j
        W[i,j] = 0 for i == j (no self-connections)
    
    This transforms binary {0,1} values to bipolar {-1,+1} before computing.
    
    Args:
        v: A 1D binary vector of shape (n_features,), e.g., (144,) for 12x12 images
        
    Returns:
        Weight matrix of shape (n_features, n_features), e.g., (144, 144)
    """
    n = len(v)
    
    # Convert binary {0,1} to bipolar {-1,+1}. Cast to a signed type first:
    # with unsigned input (the .mat loader returns uint8), 2*0 - 1 wraps to 255.
    v_bipolar = 2 * np.asarray(v, dtype=np.int64) - 1
    
    # Compute outer product: W[i,j] = v_bipolar[i] * v_bipolar[j]
    W = np.outer(v_bipolar, v_bipolar)
    
    # Set diagonal to zero (no self-connections)
    np.fill_diagonal(W, 0)
    
    return W


def compute_all_weight_matrices(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Compute weight matrices for all patterns in the training data.
    
    For each pattern k, computes a weight matrix W_k of shape (144, 144).
    
    Args:
        data: Training data as DataFrame or 2D array with shape (n_patterns, n_features)
              e.g., (18, 144) for 18 patterns of 12x12 images
              
    Returns:
        3D NumPy array of shape (n_patterns, n_features, n_features)
        e.g., (18, 144, 144) - access individual matrices via weights[k]
    """
    # Convert to numpy array if DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    n_patterns, n_features = data.shape
    
    # Initialize the weights array
    weights = np.zeros((n_patterns, n_features, n_features), dtype=np.float64)
    
    # Compute weight matrix for each pattern
    for k in range(n_patterns):
        v_k = data[k]
        weights[k] = compute_weight_matrix(v_k)
    
    return weights


def compute_combined_weight_matrix(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Compute the combined weight matrix by summing all individual weight matrices.
    
    This is the weight matrix used for pattern recall - it encodes all training patterns.
    
    Args:
        data: Training data as DataFrame or 2D array with shape (n_patterns, n_features)
              
    Returns:
        Combined weight matrix of shape (n_features, n_features), e.g., (144, 144)
    """
    all_weights = compute_all_weight_matrices(data)
    return np.sum(all_weights, axis=0)


def verify_weight_matrices(weights: np.ndarray, data: Union[pd.DataFrame, np.ndarray], 
                           verbose: bool = True) -> bool:
    """
    Verify that weight matrices were computed correctly.
    
    Checks:
    1. Correct shape
    2. Diagonal is all zeros
    3. Matrix is symmetric
    4. Sample values match expected computation
    
    Args:
        weights: The computed weight matrices of shape (n_patterns, n_features, n_features)
        data: Original training data
        verbose: Whether to print verification results
        
    Returns:
        True if all checks pass, False otherwise
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    n_patterns, n_features = data.shape
    
    all_passed = True
    
    if verbose:
        print("\n" + "="*60)
        print("WEIGHT MATRICES VERIFICATION")
        print("="*60)
    
    # Check 1: Shape
    expected_shape = (n_patterns, n_features, n_features)
    shape_ok = weights.shape == expected_shape
    if verbose:
        print(f"\n[1] Shape check:")
        print(f"    Expected: {expected_shape}")
        print(f"    Actual:   {weights.shape}")
        print(f"    Status:   {'✓ PASSED' if shape_ok else '✗ FAILED'}")
    all_passed = all_passed and shape_ok
    
    # Check 2: Diagonal is zero for all matrices
    diag_ok = True
    for k in range(n_patterns):
        if not np.allclose(np.diag(weights[k]), 0):
            diag_ok = False
            break
    if verbose:
        print(f"\n[2] Diagonal zero check (no self-connections):")
        print(f"    Status:   {'✓ PASSED' if diag_ok else '✗ FAILED'}")
    all_passed = all_passed and diag_ok
    
    # Check 3: Symmetry (W[i,j] == W[j,i])
    sym_ok = True
    for k in range(n_patterns):
        if not np.allclose(weights[k], weights[k].T):
            sym_ok = False
            break
    if verbose:
        print(f"\n[3] Symmetry check (W[i,j] == W[j,i]):")
        print(f"    Status:   {'✓ PASSED' if sym_ok else '✗ FAILED'}")
    all_passed = all_passed and sym_ok
    
    # Check 4: Sample value verification
    # Pick pattern 0, indices (0, 1) and verify manually
    k, i, j = 0, 0, 1
    v_k = data[k]
    expected_val = (2 * v_k[i] - 1) * (2 * v_k[j] - 1)
    actual_val = weights[k, i, j]
    sample_ok = np.isclose(expected_val, actual_val)
    if verbose:
        print(f"\n[4] Sample value check (pattern 0, indices [0,1]):")
        print(f"    v[0] = {v_k[i]}, v[1] = {v_k[j]}")
        print(f"    Expected: (2*{v_k[i]}-1) * (2*{v_k[j]}-1) = {expected_val}")
        print(f"    Actual:   {actual_val}")
        print(f"    Status:   {'✓ PASSED' if sample_ok else '✗ FAILED'}")
    all_passed = all_passed and sample_ok
    
    # Check 5: Value range (should be -1 or +1 for off-diagonal)
    values_ok = True
    unique_vals = set()
    for k in range(n_patterns):
        # Get off-diagonal elements
        mask = ~np.eye(n_features, dtype=bool)
        off_diag = weights[k][mask]
        unique_vals.update(np.unique(off_diag).tolist())
    
    expected_vals = {-1.0, 1.0}
    values_ok = unique_vals == expected_vals or unique_vals.issubset(expected_vals)
    if verbose:
        print(f"\n[5] Value range check (off-diagonal should be -1 or +1):")
        print(f"    Unique values found: {unique_vals}")
        print(f"    Status:   {'✓ PASSED' if values_ok else '✗ FAILED'}")
    all_passed = all_passed and values_ok
    
    # Summary
    if verbose:
        print("\n" + "="*60)
        print(f"VERIFICATION RESULT: {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗'}")
        print("="*60)
        
        # Additional info
        print(f"\nWeight matrices summary:")
        print(f"  • Number of patterns:     {n_patterns}")
        print(f"  • Features per pattern:   {n_features}")
        print(f"  • Weight matrix size:     {n_features}×{n_features}")
        print(f"  • Total weights array:    {weights.shape}")
        print(f"  • Memory usage:           {weights.nbytes / 1024:.2f} KB")
    
    return all_passed


# =============================================================================
# ENERGY COMPUTATION
# =============================================================================

def compute_energy(W: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the energy of the Hopfield network for a given state.
    
    Energy formula: E = -1/2 * sum_i sum_j w_ij * q_i * q_j

    q is the binary {0, 1} state, as in the assignment's energy equation and
    its update rule (update_neuron). With W symmetric and a zero diagonal,
    setting q_i to 1 when W[:, i] . q >= 0 (and to 0 otherwise) never raises
    this energy, so it is non-increasing under the asynchronous update.
    The bipolar energy -1/2 s^T W s (s = 2q - 1) is the corresponding function
    for the standard update on {-1, +1} states; under the 0/1 update used here
    it can rise.

    Args:
        W: Combined weight matrix of shape (n_features, n_features)
        q: Current state vector of shape (n_features,) in binary {0,1}

    Returns:
        Energy value (scalar)
    """
    q = np.asarray(q, dtype=np.float64)

    # E = -1/2 * q^T * W * q
    energy = -0.5 * np.dot(q, np.dot(W, q))

    return energy


# =============================================================================
# ASYNCHRONOUS UPDATE
# =============================================================================

@dataclass
class UpdateResult:
    """Container for results from a single pattern recall."""
    query_original: np.ndarray      # Original query pattern (1D, 144)
    query_final: np.ndarray         # Final recalled pattern (1D, 144)
    query_original_2d: np.ndarray   # Original as 12x12
    query_final_2d: np.ndarray      # Final as 12x12
    converged: bool                 # Whether the network converged
    n_iterations: int               # Number of iterations to converge
    n_updates: int                  # Total number of neuron updates
    energy_history: List[float]     # Energy after each iteration
    hamming_distance: int           # Hamming distance between original and final
    pattern_index: int              # Index of the query pattern (-1 if unknown)


def update_neuron(W: np.ndarray, q: np.ndarray, i: int) -> int:
    """
    Update a single neuron using the Hopfield update rule.
    
    q'_i = 1 if W[:,i] · q >= 0, else 0
    
    This follows the assignment specification exactly where W is computed using 
    bipolar encoding but q remains in binary {0,1} form.
    
    Args:
        W: Combined weight matrix (computed from bipolar patterns)
        q: Current state vector (binary {0,1})
        i: Index of neuron to update
        
    Returns:
        New value for neuron i (0 or 1)
    """
    # Compute the weighted sum (local field) as per assignment: W[:,i] · q
    h_i = np.dot(W[:, i], q)
    
    # Apply threshold: output 1 if h_i >= 0, else 0
    return 1 if h_i >= 0 else 0


def async_update_iteration(W: np.ndarray, q: np.ndarray, 
                            rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    """
    Perform one iteration of asynchronous updates (update all neurons once in random order).
    
    Args:
        W: Combined weight matrix
        q: Current state vector (binary {0,1})
        rng: Random number generator for reproducibility
        
    Returns:
        Tuple of (updated state vector, number of neurons that changed)
    """
    n = len(q)
    q_new = q.copy()
    n_changed = 0
    
    # Random permutation of neuron indices
    update_order = rng.permutation(n)
    
    for i in update_order:
        old_val = q_new[i]
        new_val = update_neuron(W, q_new, i)
        if new_val != old_val:
            n_changed += 1
        q_new[i] = new_val
    
    return q_new, n_changed


def recall_pattern(W: np.ndarray, query: np.ndarray, 
                   max_iterations: int = 100,
                   random_seed: int = 42,
                   pattern_index: int = -1,
                   verbose: bool = False) -> UpdateResult:
    """
    Recall a pattern using asynchronous Hopfield network updates.
    
    Updates neurons randomly until convergence (no changes) or max iterations reached.
    
    Args:
        W: Combined weight matrix of shape (n_features, n_features)
        query: Query pattern to recall, shape (n_features,) in binary {0,1}
        max_iterations: Maximum number of full iterations through all neurons
        random_seed: Random seed for reproducible neuron update order
        pattern_index: Index of this pattern (for tracking purposes)
        verbose: Whether to print progress
        
    Returns:
        UpdateResult containing the recall results and statistics
    """
    rng = np.random.default_rng(random_seed)
    
    # Initialize (copy to avoid modifying original)
    q = query.copy().astype(np.float64)
    n_features = len(q)
    
    # Tracking variables: energy history, updates, convergence
    energy_history = [compute_energy(W, q)]
    total_updates = 0
    converged = False
    
    for iteration in range(max_iterations):
        # Perform one asynchronous update iteration
        q_new, n_changed = async_update_iteration(W, q, rng)
        total_updates += n_features
        
        # Compute energy after this iteration
        energy = compute_energy(W, q_new)
        energy_history.append(energy)
        
        if verbose:
            print(f"    Iteration {iteration + 1}: {n_changed} neurons changed, "
                  f"energy = {energy:.2f}")
        
        # Check for convergence
        if n_changed == 0:
            converged = True
            break
        
        # Update state for next iteration
        q = q_new
    
    # Compute final results
    hamming_dist = int(np.sum(query != q))
    
    return UpdateResult(
        query_original=query, # Original query pattern
        query_final=q.astype(np.uint8), # Final recalled pattern
        query_original_2d=query.reshape(12, 12), # Original as 12x12
        query_final_2d=q.reshape(12, 12).astype(np.uint8), # Final as 12x12
        converged=converged, # Whether converged
        n_iterations=iteration + 1 if 'iteration' in dir() else 0, # Iterations taken
        n_updates=total_updates, # Total neuron updates
        energy_history=energy_history, # Energy history
        hamming_distance=hamming_dist, # Hamming distance to original
        pattern_index=pattern_index # Index of the pattern
    )


def recall_all_patterns(W: np.ndarray, queries: Union[pd.DataFrame, np.ndarray],
                        max_iterations: int = 100,
                        random_seed: int = 42,
                        dataset_name: str = "Dataset",
                        verbose: bool = True) -> List[UpdateResult]:
    """
    Recall all patterns in a dataset using the Hopfield network.
    
    Args:
        W: Combined weight matrix
        queries: Query patterns as DataFrame or array, shape (n_patterns, n_features)
        max_iterations: Maximum iterations per pattern
        random_seed: Base random seed (incremented for each pattern)
        dataset_name: Name for progress display
        verbose: Whether to show progress
        
    Returns:
        List of UpdateResult objects, one per query pattern
    """
    # Convert to numpy array if DataFrame
    if isinstance(queries, pd.DataFrame):
        queries = queries.values
    
    n_patterns = queries.shape[0]
    results = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RECALLING PATTERNS: {dataset_name}")
        print(f"{'='*60}")
        print(f"Patterns to recall: {n_patterns}")
        print(f"Max iterations per pattern: {max_iterations}")
        print()
    
    # Progress bar setup
    iterator = tqdm(range(n_patterns), desc=f"Recalling {dataset_name}", 
                    disable=not verbose)
    
    for idx in iterator:
        # Recall one pattern
        query = queries[idx]
        result = recall_pattern(
            W, query, 
            max_iterations=max_iterations,
            random_seed=random_seed + idx,  # Different seed for each pattern
            pattern_index=idx,
            verbose=False
        )
        results.append(result)
        
        # Update progress bar with convergence info
        if verbose:
            status = "✓" if result.converged else "⚠"
            iterator.set_postfix({
                'conv': f"{status}",
                'iters': result.n_iterations,
                'ham': result.hamming_distance
            })
    
    if verbose:
        # Summary statistics
        n_converged = sum(1 for r in results if r.converged)
        avg_iterations = np.mean([r.n_iterations for r in results])
        avg_hamming = np.mean([r.hamming_distance for r in results])
        total_hamming = sum(r.hamming_distance for r in results)
        
        print(f"\n{dataset_name} Recall Summary:")
        print(f"  • Converged:           {n_converged}/{n_patterns} patterns")
        print(f"  • Avg iterations:      {avg_iterations:.2f}")
        print(f"  • Avg Hamming dist:    {avg_hamming:.2f}")
        print(f"  • Total Hamming dist:  {total_hamming}")
    
    return results


# =============================================================================
# ERROR/DISTANCE METRICS
# =============================================================================

def compute_hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Hamming distance between two binary vectors."""
    return int(np.sum(a != b))


def compute_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def compute_pattern_error(original: np.ndarray, recalled: np.ndarray) -> Dict[str, float]:
    """
    Compute various error metrics between original and recalled patterns.
    
    Args:
        original: Original pattern (binary, 1D)
        recalled: Recalled pattern (binary, 1D)
        
    Returns:
        Dictionary with error metrics
    """
    hamming = compute_hamming_distance(original, recalled)
    euclidean = compute_euclidean_distance(original, recalled)
    n_features = len(original)
    
    return {
        "hamming_distance": hamming,
        "hamming_percent": (hamming / n_features) * 100,
        "euclidean_distance": euclidean,
        "accuracy": ((n_features - hamming) / n_features) * 100,
        "match": hamming == 0
    }


# =============================================================================
# TRAINING (WEIGHT MATRIX ACCUMULATION)
# =============================================================================

def train_hopfield_network(train_data: Union[pd.DataFrame, np.ndarray],
                           verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a Hopfield network by computing and accumulating weight matrices.
    
    Args:
        train_data: Training patterns, shape (n_patterns, n_features)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (combined_weight_matrix, individual_weight_matrices)
    """
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data.values
    
    n_patterns, n_features = train_data.shape
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING HOPFIELD NETWORK")
        print("="*60)
        print(f"Training patterns:    {n_patterns}")
        print(f"Features per pattern: {n_features}")
        print()
    
    # Compute individual weight matrices with progress bar
    if verbose:
        print("Computing individual weight matrices...")
    
    individual_weights = np.zeros((n_patterns, n_features, n_features), dtype=np.float64)
    
    iterator = tqdm(range(n_patterns), desc="Training", disable=not verbose)
    for k in iterator:
        individual_weights[k] = compute_weight_matrix(train_data[k])
    
    # Check the individual matrices (shape, zero diagonal, symmetry, values of +/-1)
    if not verify_weight_matrices(individual_weights, train_data, verbose=verbose):
        raise ValueError("Weight matrix verification failed; check the input data type.")
    
    # Compute combined weight matrix
    if verbose:
        print("\nAccumulating combined weight matrix...")
    
    combined_weights = np.sum(individual_weights, axis=0)
    
    if verbose:
        print(f"\n✓ Training complete!")
        print(f"  • Combined weight matrix shape: {combined_weights.shape}")
        print(f"  • Weight matrix memory: {combined_weights.nbytes / 1024:.2f} KB")
        
        # Check properties
        print(f"  • Diagonal sum (should be 0): {np.sum(np.diag(combined_weights)):.0f}")
        print(f"  • Is symmetric: {np.allclose(combined_weights, combined_weights.T)}")
        print(f"  • Weight range: [{combined_weights.min():.0f}, {combined_weights.max():.0f}]")
    
    return combined_weights, individual_weights


# =============================================================================
# COMPARISON WITH ORIGINAL TRAINING PATTERNS
# =============================================================================

def find_closest_training_pattern(recalled: np.ndarray, 
                                   training_patterns: np.ndarray) -> Tuple[int, int]:
    """
    Find the closest training pattern to a recalled pattern.
    
    Args:
        recalled: Recalled pattern (1D binary)
        training_patterns: All training patterns (2D array)
        
    Returns:
        Tuple of (closest_pattern_index, hamming_distance)
    """
    min_dist = float('inf')
    closest_idx = -1
    
    for i, pattern in enumerate(training_patterns):
        dist = compute_hamming_distance(recalled, pattern)
        # Update closest if needed
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    return closest_idx, int(min_dist)


def evaluate_recall_results(results: List[UpdateResult], 
                            training_patterns: np.ndarray,
                            original_patterns: np.ndarray,
                            dataset_name: str = "Dataset",
                            verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation of recall results.
    
    Args:
        results: List of UpdateResult from recall
        training_patterns: Original training patterns used to train the network
        original_patterns: The query patterns (may be same as training or noisy versions)
        dataset_name: Name for display
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation metrics
    """
    n_patterns = len(results)
    
    # Compute metrics
    converged = [r.converged for r in results] # Convergence status (list of bools)
    iterations = [r.n_iterations for r in results] # Number of iterations (list of ints)
    hamming_to_original = [r.hamming_distance for r in results] # Hamming distance to original (list of ints)
    
    # For each recalled pattern, find distance to closest training pattern
    closest_training = []
    hamming_to_training = []
    exact_matches = 0
    
    for r in results:
        idx, dist = find_closest_training_pattern(r.query_final, training_patterns)
        closest_training.append(idx)
        hamming_to_training.append(dist)
        if dist == 0:
            exact_matches += 1
    
    # Compile evaluation
    evaluation = {
        "dataset_name": dataset_name,
        "n_patterns": n_patterns,
        "n_converged": sum(converged),
        "convergence_rate": sum(converged) / n_patterns * 100,
        "avg_iterations": np.mean(iterations),
        "max_iterations": max(iterations),
        "min_iterations": min(iterations),
        "avg_hamming_to_original": np.mean(hamming_to_original),
        "total_hamming_to_original": sum(hamming_to_original),
        "avg_hamming_to_training": np.mean(hamming_to_training),
        "total_hamming_to_training": sum(hamming_to_training),
        "exact_matches_to_training": exact_matches,
        "exact_match_rate": exact_matches / n_patterns * 100,
        "closest_training_indices": closest_training,
        "hamming_distances_to_original": hamming_to_original,
        "hamming_distances_to_training": hamming_to_training,
        "results": results
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION: {dataset_name}")
        print(f"{'='*60}")
        print(f"\nConvergence:")
        print(f"  • Converged:        {evaluation['n_converged']}/{n_patterns} "
              f"({evaluation['convergence_rate']:.1f}%)")
        print(f"  • Avg iterations:   {evaluation['avg_iterations']:.2f}")
        print(f"  • Iteration range:  [{evaluation['min_iterations']}, {evaluation['max_iterations']}]")
        
        print(f"\nReconstruction Quality:")
        print(f"  • Avg Hamming to query:     {evaluation['avg_hamming_to_original']:.2f}")
        print(f"  • Avg Hamming to training:  {evaluation['avg_hamming_to_training']:.2f}")
        print(f"  • Exact matches to training: {exact_matches}/{n_patterns} "
              f"({evaluation['exact_match_rate']:.1f}%)")
    
    return evaluation


if __name__ == "__main__":
    # Test the implementation
    import sys
    from pathlib import Path
    
    # Add scripts directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Print console header
    try:
        from author_info import print_console_header
        print_console_header("hopfield.py")
    except ImportError:
        pass
    
    from get_data import load_all_datasets
    
    # Load training data
    print("Loading training data...")
    train_df, test_df, valid_df, train_data, test_data, valid_data, info = load_all_datasets(
        verbose=False
    )
    
    # Compute weight matrices
    print("\nComputing weight matrices for training data...")
    train_weights = compute_all_weight_matrices(train_df)
    
    # Verify weight matrices
    verify_weight_matrices(train_weights, train_df, verbose=True)
