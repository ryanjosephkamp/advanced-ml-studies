#!/usr/bin/env python3
"""
similarity.py - Signal Similarity / Distance Measures

Computes pairwise distance matrices using Pearson correlation,
Euclidean distance, and Itakura–Saito distance between IMF feature vectors.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 4 – ML for Signal Analysis using EMD
Date: March 8, 2026
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

SIMILARITY_MEASURES = ['pearson', 'euclidean', 'itakura_saito']


def compute_distance_matrix(features, measure):
    """
    Compute pairwise distance matrix between feature vectors:
        - Pearson correlation distance: d = 1 - rho (range [0, 2])
        - Euclidean distance: standard L2 distance
        - Itakura–Saito distance: symmetrized version for non-negative features

    For the Itakura–Saito distance, zero-valued features are replaced with a per-column floor equal to the
    minimum non-zero value in that column. The divergence is defined for power-like (positive) vectors; on
    entropies and frequencies it is computable but has no spectral meaning.

    Args:
        features: np.ndarray of shape (n_segments, n_features)
        measure: 'pearson', 'euclidean', or 'itakura_saito'

    Returns:
        np.ndarray of shape (n_segments, n_segments) — symmetric distance matrix
    """
    if measure not in SIMILARITY_MEASURES:
        raise ValueError(f"Unknown measure: {measure}. Choose from {SIMILARITY_MEASURES}")

    if measure == 'pearson':
        return _pearson_distance(features)
    elif measure == 'euclidean':
        return _euclidean_distance(features)
    elif measure == 'itakura_saito':
        return _itakura_saito_distance_matrix(features)


# =============================================================================
# IMPLEMENTATIONS
# =============================================================================

def _pearson_distance(features):
    """
    Pearson correlation distance: d = 1 - rho.
    Range: [0, 2], where 0 = identical, 2 = anti-correlated.
    """
    n = features.shape[0]
    if n < 2:
        return np.zeros((n, n))

    # Handle constant rows (zero std) by adding tiny noise
    stds = np.std(features, axis=1)
    if np.any(stds < 1e-12):
        features = features.copy()
        features[stds < 1e-12] += np.random.RandomState(42).randn(
            np.sum(stds < 1e-12), features.shape[1]) * 1e-10

    corr_matrix = np.corrcoef(features)
    dist_matrix = 1.0 - corr_matrix
    # Clip to valid range [0, 2]
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix


def _euclidean_distance(features):
    """Standard Euclidean distance."""
    return squareform(pdist(features, metric='euclidean'))


def _itakura_saito_distance_matrix(features):
    """
    Symmetrized Itakura–Saito distance matrix.
    Requires strictly positive inputs.

    Zero-valued features (for example the entropy of a constant IMF) are
    replaced with a per-column floor equal to the minimum non-zero value in that
    column. With unpadded IMFs this rarely applies.
    """
    features_pos = features.copy()

    # Per-column floor: minimum non-zero value in each feature dimension
    for col in range(features_pos.shape[1]):
        col_vals = features_pos[:, col]
        nonzero_mask = col_vals > 0
        if np.any(nonzero_mask):
            col_floor = np.min(col_vals[nonzero_mask])
            col_vals[~nonzero_mask] = col_floor
        else:
            # Entire column is zero — set to 1.0 (ratio = 1, contribution = 0)
            col_vals[:] = 1.0
        features_pos[:, col] = col_vals

    return itakura_saito_cross(features_pos, features_pos)


def itakura_saito_cross(A, B, chunk=256):
    """
    Symmetrized Itakura–Saito distances between the rows of A and the rows of B.

    In the symmetrized form the log terms cancel:
        (Div_IS(x||y) + Div_IS(y||x)) / 2 = sum(x/y + y/x - 2) / 2
    which is computed here in blocks of rows, without a Python loop over pairs.

    Args:
        A: (n_a, n_features) strictly positive
        B: (n_b, n_features) strictly positive
    Returns:
        (n_a, n_b) distance matrix
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    out = np.empty((A.shape[0], B.shape[0]))
    for i in range(0, A.shape[0], chunk):
        ratio = A[i:i + chunk, None, :] / B[None, :, :]
        out[i:i + chunk] = 0.5 * (ratio + 1.0 / ratio - 2.0).sum(axis=2)
    return out


def _itakura_saito_divergence(x, y):
    """
    Div_IS(x || y) = sum(x/y - log(x/y) - 1)
    """
    ratio = x / y
    return np.sum(ratio - np.log(ratio) - 1)


def _itakura_saito_symmetric(x, y):
    """
    Symmetrized IS distance: (Div_IS(x||y) + Div_IS(y||x)) / 2
    """
    return (_itakura_saito_divergence(x, y) + _itakura_saito_divergence(y, x)) / 2.0
