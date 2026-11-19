#!/usr/bin/env python3
"""
emd_processor.py - Empirical Mode Decomposition Processing

Applies EMD to signal segments and keeps a fixed number of intrinsic mode
functions (IMFs) per segment for downstream feature extraction.

PyEMD's `emd()` returns the IMFs with the residue (the slow trend left after the
last IMF) appended as an extra row. The residue is not an IMF, so it is taken
apart with `get_imfs_and_residue()` and kept separately. Feature vectors use the
first `n_imfs` IMFs of every segment, so position k always means "IMF k"; there is
no zero padding. Segments with fewer than `n_imfs` IMFs are left out and counted.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 4 – ML for Signal Analysis using EMD
Date: March 8, 2026
"""

import numpy as np
from tqdm import tqdm
from PyEMD import EMD


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def decompose(signal, emd=None):
    """
    Decompose one signal into IMFs and a residue.

    Args:
        signal: 1-D array
        emd: a PyEMD EMD instance (created if None)

    Returns:
        (imfs, residue): imfs is (n_imfs, n_samples), residue is (n_samples,);
        imfs.sum(axis=0) + residue reproduces the signal.
    """
    if emd is None:
        emd = EMD()
    emd.emd(signal)
    imfs, residue = emd.get_imfs_and_residue()
    return np.asarray(imfs), np.asarray(residue)


def apply_emd_to_signals(signals, n_imfs=4, verbose=True):
    """
    Apply EMD to each signal segment and keep its first `n_imfs` IMFs:
        - Constant signals are skipped.
        - The residue is separated from the IMFs (it is not used as a feature).
        - Segments with fewer than `n_imfs` IMFs are skipped and counted, so
          every feature vector has the same meaning at every position.

    Args:
        signals: np.ndarray of shape (n_segments, n_samples)
        n_imfs: Number of IMFs kept per segment
        verbose: Print progress

    Returns:
        results: list of dicts, one per kept segment, with keys
            'imfs' (n_imfs, n_samples), 'imfs_all' (all IMFs), 'residual' (residue),
            'n_imfs_total' (IMF count before selection), 'original_index'
        n_imfs: the number of IMFs kept per segment
        skipped_indices: indices of segments that were skipped
    """
    n_segments = signals.shape[0]
    emd = EMD()

    results = []
    imf_counts = []
    failed_indices = []
    too_few_indices = []

    for i in tqdm(range(n_segments), desc="Applying EMD to signals", disable=not verbose):
        signal = signals[i]
        if np.std(signal) < 1e-12:
            failed_indices.append(i)
            continue
        try:
            imfs, residue = decompose(signal, emd)
        except Exception as e:
            if verbose and len(failed_indices) < 5:
                print(f"    Warning: EMD failed on segment {i}: {e}")
            failed_indices.append(i)
            continue

        imf_counts.append(imfs.shape[0])
        if imfs.shape[0] < n_imfs:
            too_few_indices.append(i)
            continue

        results.append({
            'imfs': imfs[:n_imfs],          # the first n_imfs IMFs, used for features
            'imfs_all': imfs,               # every IMF, for plots
            'residual': residue,            # the residue (trend), kept separately
            'n_imfs_total': imfs.shape[0],
            'original_index': i,
        })

    if not results:
        raise RuntimeError("EMD produced no usable segments. Check data quality.")

    if verbose:
        counts = np.array(imf_counts)
        print(f"  EMD completed: {len(results)}/{n_segments} segments kept")
        print(f"  IMF count range (residue excluded): {counts.min()} – {counts.max()} "
              f"(median: {int(np.median(counts))})")
        if failed_indices:
            print(f"  Segments where EMD failed: {len(failed_indices)}")
        print(f"  Segments with fewer than {n_imfs} IMFs (skipped): {len(too_few_indices)}")
        print(f"  Using the first {n_imfs} IMFs of every segment")

    return results, n_imfs, sorted(failed_indices + too_few_indices)
