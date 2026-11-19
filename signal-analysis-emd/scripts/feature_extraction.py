#!/usr/bin/env python3
"""
feature_extraction.py - IMF Feature Extraction

Extracts features (variance, histogram entropy, mean instantaneous frequency)
from IMFs to create fixed-length feature vectors for each signal segment. Every
segment arrives with the same number of IMFs (see emd_processor.py), so position
k of a feature vector always describes IMF k.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 4 – ML for Signal Analysis using EMD
Date: March 8, 2026
"""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import entropy as scipy_entropy


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

FEATURE_TYPES = ['variance', 'entropy', 'inst_freq']


def extract_imf_features(emd_results, feature_type, n_imfs=None, sfreq=1.0):
    """
    Extract features (variance, entropy, instantaneous frequency) from IMFs for all segments.

    Args:
        emd_results: list of dicts from apply_emd_to_signals().
                     Each dict has 'imfs' of shape (n_imfs, n_samples).
        feature_type: 'variance', 'entropy', or 'inst_freq'
        n_imfs: Number of IMFs to use (None = infer from data)
        sfreq: Sampling frequency (needed for inst_freq)

    Returns:
        np.ndarray of shape (n_segments, n_imfs) — feature matrix
    """
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"Unknown feature type: {feature_type}. Choose from {FEATURE_TYPES}")

    if n_imfs is None and len(emd_results) > 0:
        n_imfs = emd_results[0]['imfs'].shape[0]

    features = []
    for item in emd_results:
        imfs = item['imfs']  # (n_imfs, n_samples)
        if imfs.shape[0] != n_imfs:
            raise ValueError(f"Expected {n_imfs} IMFs per segment, got {imfs.shape[0]}")

        if feature_type == 'variance':
            feat = _compute_variance(imfs)
        elif feature_type == 'entropy':
            feat = _compute_entropy(imfs)
        elif feature_type == 'inst_freq':
            feat = _compute_inst_freq(imfs, sfreq)

        features.append(feat)

    return np.array(features)


# =============================================================================
# FEATURE IMPLEMENTATIONS
# =============================================================================

def _compute_variance(imfs):
    """Compute variance of each IMF."""
    return np.var(imfs, axis=1)


def _compute_entropy(imfs):
    """
    Compute the amplitude-histogram (Shannon) entropy of each IMF:
        - For each IMF, a 50-bin histogram of its sample values estimates the distribution of amplitudes.
        - The Shannon entropy of that distribution (in nats) measures how spread out the amplitudes are.
          It describes the shape of the amplitude distribution, not the spectrum, and does not change
          when the IMF is scaled.
        - If an IMF has very low variance (effectively constant), its entropy is set to 0.

    Args:
        imfs: np.ndarray of shape (n_imfs, n_samples)

    Returns:
        np.ndarray of shape (n_imfs,) — histogram entropy for each IMF
    """
    result = []
    for imf in imfs:
        if np.std(imf) < 1e-12:
            result.append(0.0)
            continue
        # Compute normalized histogram → probability distribution
        hist, _ = np.histogram(imf, bins=50, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            result.append(0.0)
            continue
        hist = hist / hist.sum()
        result.append(float(scipy_entropy(hist)))
    return np.array(result)


def _compute_inst_freq(imfs, sfreq=1.0):
    """
    Compute mean instantaneous frequency of each IMF via Hilbert transform:
        - For each IMF, we compute the analytic signal using the Hilbert transform.
        - We then calculate the instantaneous phase and unwrap it to avoid discontinuities.
        - The instantaneous frequency is obtained by differentiating the unwrapped phase and scaling by the sampling frequency.
        - If an IMF has very low variance (effectively constant), we assign it an instantaneous frequency of 0 to avoid issues with the Hilbert transform.
    
    Args:
        imfs: np.ndarray of shape (n_imfs, n_samples)
        sfreq: Sampling frequency (Hz)
   
     Returns:
        np.ndarray of shape (n_imfs,) — mean instantaneous frequency for each IMF
    """
    result = []
    for imf in imfs:
        # Handle constant IMFs by assigning zero frequency to avoid issues with Hilbert transform
        if np.std(imf) < 1e-12:
            result.append(0.0)
            continue
        # Compute analytic signal and instantaneous frequency via Hilbert transform
        try:
            # Compute the analytic signal using the Hilbert transform, which provides a complex representation of the IMF. 
            # The real part is the original signal, and the imaginary part is the Hilbert transform of the signal.
            analytic = hilbert(imf) 
            # Calculate the instantaneous phase by taking the angle of the analytic signal and unwrapping it to remove discontinuities.
            inst_phase = np.unwrap(np.angle(analytic))
            # The instantaneous frequency is obtained by differentiating the unwrapped phase and scaling by the sampling frequency.
            inst_freq = np.diff(inst_phase) / (2 * np.pi) * sfreq
            # We take the mean of the absolute instantaneous frequency values to get a single representative frequency for each IMF.
            result.append(float(np.mean(np.abs(inst_freq))))
        except Exception:
            # In case of any numerical issues during Hilbert transform or frequency computation, assign zero frequency
            result.append(0.0)
    return np.array(result)
