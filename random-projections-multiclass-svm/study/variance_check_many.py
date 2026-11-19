"""Many-draw check of the second-moment formula Var q = (2 + (s - 3) * kappa) / k.

study/results/variance_check.csv estimates each pair's variance of q from the study's 100 draws,
which is noisy at small k (q is heavy-tailed there, and every pair in a draw shares one matrix).
This script repeats the check with many more draws of the same matrices: draw r uses
base_matrices(r, dataset), so draws 0 to 99 are the study's own. It uses the same 2,000 pairs per
data set as run_distortion.py (MinMax scaling on all rows, pair subset from default_rng(12345)), and
the unbiased variance across draws (divisor draws - 1).

Output: study/results/variance_check_many.csv, one row per (dataset, k, family), with the same
columns as variance_check.csv. CPU only; about a minute per 10,000 draws.

Example:
  python study/variance_check_many.py --draws 20000
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASETS, KMAX, RESULTS_DIR, base_matrices, families_for, k_values,  # noqa: E402
                    load_dataset, projection_matrix, sparsity)

N_CHECK_PAIRS = 2000  # as in run_distortion.py


def pair_subset(A: np.ndarray):
    """The difference vectors of run_distortion.py's fixed pair subset."""
    n = A.shape[0]
    keep = pdist(A, "sqeuclidean") > 0
    iu = np.triu_indices(n, 1)
    pair_idx = np.flatnonzero(keep)
    sub = np.random.default_rng(12345).choice(pair_idx, min(N_CHECK_PAIRS, len(pair_idx)), replace=False)
    return A[iu[0][sub]] - A[iu[1][sub]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=20_000)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    t0 = time.perf_counter()
    rows = []
    for dataset in args.datasets:
        X, _, _ = load_dataset(dataset)
        d = X.shape[1]
        delta = pair_subset(MinMaxScaler().fit_transform(X))
        norm2 = (delta ** 2).sum(1)
        kappa = (delta ** 4).sum(1) / norm2 ** 2
        ks = k_values(d)
        fams = families_for(d)
        s1 = {f: np.zeros((len(delta), len(ks))) for f in fams}
        s2 = {f: np.zeros((len(delta), len(ks))) for f in fams}
        cols = np.array(ks) - 1
        kk = np.array(ks, float)
        for r in range(args.draws):
            G, U = base_matrices(r, dataset, d)
            for fam in fams:
                R = projection_matrix(fam, G, U, KMAX)  # the first k columns give the k-dimensional matrix
                q = np.cumsum((delta @ R) ** 2, axis=1)[:, cols] / (kk * norm2[:, None])
                s1[fam] += q
                s2[fam] += q * q
        n = args.draws
        for fam in fams:
            mean_q = s1[fam] / n
            emp = (s2[fam] - n * mean_q ** 2) / (n - 1)
            s = sparsity(fam, d)
            for j, k in enumerate(ks):
                pred = (2 + (s - 3) * kappa) / k
                rows.append({"dataset": dataset, "k": k, "family": fam, "draws": n, "pairs": len(delta),
                             "mean_q": mean_q[:, j].mean(), "mean_emp_var": emp[:, j].mean(),
                             "mean_pred_var": pred.mean(), "emp_over_pred": emp[:, j].mean() / pred.mean(),
                             "corr_emp_pred": float(np.corrcoef(emp[:, j], pred)[0, 1]) if pred.std() > 1e-12 else np.nan,
                             "kappa_mean": kappa.mean(), "kappa_min": kappa.min(), "kappa_max": kappa.max()})
        print(f"{dataset}: done at {time.perf_counter() - t0:.0f} s", flush=True)
    out = os.path.join(RESULTS_DIR, "variance_check_many.csv")
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")
    print("written to", out)


if __name__ == "__main__":
    main()
