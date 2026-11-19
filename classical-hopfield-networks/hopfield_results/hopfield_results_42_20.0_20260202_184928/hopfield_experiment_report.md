# Hopfield Network Experiment Report

---

**Author:** Ryan Kamp  
**Affiliation:** Department of Computer Science, University of Cincinnati  
**Location:** Cincinnati, OH, USA  
**GitHub:** https://github.com/ryanjosephkamp

**Course:** CS7063 Advanced Machine Learning  
**Assignment:** Assignment 2 – Hopfield Networks  
**Date:** November 11, 2026

---


**Generated:** 2026-11-11 23:45:58
**Random Seed:** 42
**Validation Noise Percentage:** 20.0%

## Setup

The network stores 18 binary character patterns (12×12 = 144 units) and is queried with three sets of 18 patterns, where query k is a copy of stored pattern k:

- **Training:** 0.00 of 144 pixels differ from the stored pattern on average (0.00%), range 0 to 0.
- **Test (10% noise):** 13.00 of 144 pixels differ from the stored pattern on average (9.03%), range 13 to 13.
- **Validation (20.0% noise):** 28.78 of 144 pixels differ from the stored pattern on average (19.98%), range 21 to 39.

## Network Configuration

| Parameter | Value |
|-----------|-------|
| Number of units | 144 (12×12) |
| Number of stored patterns | 18 |
| Weights | Sum over stored patterns of (2x_i − 1)(2x_j − 1); zero diagonal |
| Weight range | -14 to 18 |
| Unit states | 0 and 1 |
| Update rule | q_i ← 1 if W[:, i] · q ≥ 0, else 0; one unit at a time, random order |
| Stopping rule | A full sweep changes no unit, or 100 sweeps |

## Results Summary

| Metric | Training | Test (10% noise) | Validation (20.0% noise) |
|--------|----------|----------|----------|
| Recalled own pattern exactly | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) |
| Final state equals some stored pattern | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) |
| Converged | 18/18 (100.0%) | 18/18 (100.0%) | 18/18 (100.0%) |
| Mean sweeps (including the last, unchanged sweep) | 2.22 | 2.22 | 2.11 |
| Mean Hamming, final state to query (q vs. q') | 26.72 | 35.72 | 44.06 |
| Mean Hamming, final state to own pattern | 26.72 | 26.72 | 26.72 |
| Mean Hamming, final state to nearest stored pattern | 8.00 | 8.00 | 8.00 |

## Detailed Analysis

### Training

**Convergence:** 18/18 queries converged (100.0%)

**Sweeps:** Average 2.22, range [2, 3]

**Exact recall of own pattern:** 0/18

**Per-query Hamming distances from the final state:**

| Query | To Query | To Own Pattern | To Nearest Stored | Nearest Stored Idx |
|-------|----------|----------------|-------------------|--------------------|
| 0 | 43 | 43 | 8 | 11 |
| 1 | 51 | 51 | 8 | 11 |
| 2 | 53 | 53 | 8 | 11 |
| 3 | 55 | 55 | 8 | 11 |
| 4 | 27 | 27 | 8 | 11 |
| 5 | 20 | 20 | 8 | 11 |
| 6 | 23 | 23 | 8 | 11 |
| 7 | 37 | 37 | 8 | 11 |
| 8 | 16 | 16 | 8 | 11 |
| 9 | 15 | 15 | 8 | 11 |
| 10 | 51 | 51 | 8 | 11 |
| 11 | 8 | 8 | 8 | 11 |
| 12 | 21 | 21 | 8 | 11 |
| 13 | 14 | 14 | 8 | 11 |
| 14 | 13 | 13 | 8 | 11 |
| 15 | 10 | 10 | 8 | 11 |
| 16 | 8 | 8 | 8 | 11 |
| 17 | 16 | 16 | 8 | 11 |

### Test (10% noise)

**Convergence:** 18/18 queries converged (100.0%)

**Sweeps:** Average 2.22, range [2, 3]

**Exact recall of own pattern:** 0/18

**Per-query Hamming distances from the final state:**

| Query | To Query | To Own Pattern | To Nearest Stored | Nearest Stored Idx |
|-------|----------|----------------|-------------------|--------------------|
| 0 | 50 | 43 | 8 | 11 |
| 1 | 58 | 51 | 8 | 11 |
| 2 | 58 | 53 | 8 | 11 |
| 3 | 56 | 55 | 8 | 11 |
| 4 | 34 | 27 | 8 | 11 |
| 5 | 29 | 20 | 8 | 11 |
| 6 | 32 | 23 | 8 | 11 |
| 7 | 46 | 37 | 8 | 11 |
| 8 | 25 | 16 | 8 | 11 |
| 9 | 24 | 15 | 8 | 11 |
| 10 | 58 | 51 | 8 | 11 |
| 11 | 21 | 8 | 8 | 11 |
| 12 | 34 | 21 | 8 | 11 |
| 13 | 27 | 14 | 8 | 11 |
| 14 | 22 | 13 | 8 | 11 |
| 15 | 21 | 10 | 8 | 11 |
| 16 | 21 | 8 | 8 | 11 |
| 17 | 27 | 16 | 8 | 11 |

### Validation (20.0% noise)

**Convergence:** 18/18 queries converged (100.0%)

**Sweeps:** Average 2.11, range [2, 3]

**Exact recall of own pattern:** 0/18

**Per-query Hamming distances from the final state:**

| Query | To Query | To Own Pattern | To Nearest Stored | Nearest Stored Idx |
|-------|----------|----------------|-------------------|--------------------|
| 0 | 54 | 43 | 8 | 11 |
| 1 | 56 | 51 | 8 | 11 |
| 2 | 59 | 53 | 8 | 11 |
| 3 | 70 | 55 | 8 | 11 |
| 4 | 42 | 27 | 8 | 11 |
| 5 | 43 | 20 | 8 | 11 |
| 6 | 38 | 23 | 8 | 11 |
| 7 | 51 | 37 | 8 | 11 |
| 8 | 40 | 16 | 8 | 11 |
| 9 | 41 | 15 | 8 | 11 |
| 10 | 49 | 51 | 8 | 11 |
| 11 | 34 | 8 | 8 | 11 |
| 12 | 42 | 21 | 8 | 11 |
| 13 | 35 | 14 | 8 | 11 |
| 14 | 42 | 13 | 8 | 11 |
| 15 | 37 | 10 | 8 | 11 |
| 16 | 27 | 8 | 8 | 11 |
| 17 | 33 | 16 | 8 | 11 |

## Comparative Analysis

### 1. Recall of the Stored Patterns

Presented unchanged, 0 of 18 stored patterns were recalled exactly. A stored pattern is recalled from itself only if the update leaves it unchanged (a fixed point), so none of the 18 is a fixed point of this network.

Exact recall from the noisy queries: 0 of 18 from the test (10% noise) queries; 0 of 18 from the validation (20.0% noise) queries.

### 2. Final States

The 54 queries ended in 1 distinct final state.

| Final state | Queries ending there | Pixels on | Nearest stored pattern | Hamming to it |
|-------------|----------------------|-----------|------------------------|---------------|
| 1 | 54 | 133 | 11 | 8 |

### 3. Noise Comparison

| Query set | Mean pixels flipped | Recalled own pattern | Mean Hamming to own pattern | Mean Hamming to nearest stored |
|-----------|---------------------|----------------------|-----------------------------|--------------------------------|
| Training | 0.00 | 0/18 (0.0%) | 26.72 | 8.00 |
| Test (10% noise) | 13.00 | 0/18 (0.0%) | 26.72 | 8.00 |
| Validation (20.0% noise) | 28.78 | 0/18 (0.0%) | 26.72 | 8.00 |

- Exact recall of the own pattern: 0/18 on all three sets.
- Mean Hamming distance from the final state to the own pattern: 26.72 on all three sets.
- Every query ends in the same state, so neither distance depends on the noise in the query.

### 4. Energy

The energy is computed on the 0/1 state q, as the assignment defines it:

$$E = -\frac{1}{2} \sum_i \sum_j w_{ij} q_i q_j$$

It is recorded once before the first sweep and once after every sweep. With symmetric weights and a zero diagonal, a single-unit update under the rule above cannot raise E.

- Recorded sweeps in which E rose: 0 of 118, over all 54 queries.
- Final energies: -75,880.

### 5. Convergence

- 54 of 54 queries reached a state that a full sweep left unchanged, within the limit of 100 sweeps.
- Mean sweeps, counting the last, unchanged sweep: 2.22 (training), 2.22 (test), and 2.11 (validation); the most any query took was 3.

## Summary of Results

1. **Exact recall of the own pattern:** 0 of 18 (training), 0 of 18 (test), 0 of 18 (validation).
2. **Final states:** all 54 queries end in one state (133 pixels on, 8 bits from stored pattern 11).
3. **Energy:** rose in 0 of 118 recorded sweeps.
4. **Convergence:** 54 of 54 queries converged within 100 sweeps.

## Visualizations

The following visualizations have been generated:

- `plots/comparison_metrics.png` - Performance comparison across datasets
- `plots/hamming_distribution.png` - Hamming distance per pattern
- `plots/training_patterns_grid.png` - Grid of training q vs q'
- `plots/test_patterns_grid.png` - Grid of test q vs q'
- `plots/validation_patterns_grid.png` - Grid of validation q vs q'
- `plots/*_all_energy_curves.png` - Energy curves of all queries in each set
- `energy_plots/` - Individual energy curves for each test and validation query
- `pattern_comparisons/` - Individual pattern comparison images

---
*Report generated with seed=42, validation noise=20.0%*