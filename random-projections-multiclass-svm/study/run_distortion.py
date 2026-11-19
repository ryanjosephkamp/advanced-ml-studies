"""Distance distortion of the same projection matrices the classification runs use.

For each (dataset, k, family, draw), on the MinMax-scaled data set (all rows), with
q = |(u - v) R|^2 / (k |u - v|^2) the squared distance ratio over all pairs of rows at positive distance:
  - mean, variance, and quantiles of q; shares of pairs with |q - 1| > eps, and lower and upper
    tail shares, for comparison with the per-pair Johnson-Lindenstrauss tail bounds;
  - the report's distance-ratio summary (m, M, epsilon = max(1 - m, M - 1), mean ratio);
  - the Spearman correlation between original and projected distances, on the same pairs;
  - 5-nearest-neighbor label agreement in the projected space (and once in the original space),
    from the full distance matrix, leaving out only each row's distance to itself;
  - all-zero columns and rows of R.
Rows go to study/results/raw/distortion_<dataset>.csv (resumable).

Per-pair second-moment check: on a fixed random subset of up to 2,000 pairs per data set, the
variance of q across the 100 draws is compared with (2 + (s - 3) * kappa) / k, where
kappa = sum(delta^4) / (sum(delta^2))^2 for the pair's difference vector delta (2 / k for the
Gaussian). Written to study/results/raw/variance_check_<dataset>.csv.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (N_DRAWS, RAW_DIR, base_matrices, families_for, k_values, load_dataset,  # noqa: E402
                    project, projection_matrix, sparsity)

EPS = [0.1, 0.25, 0.5]
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
N_CHECK_PAIRS = 2000
NN = 5


def knn_agreement(D: np.ndarray, y: np.ndarray, n_neighbors: int = NN) -> float:
    """Mean share of each point's nearest neighbors (by distance matrix D) that share its label."""
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    nbrs = np.argpartition(D, n_neighbors, axis=1)[:, :n_neighbors]
    return float((y[nbrs] == y[:, None]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--max-minutes", type=float, default=9.0)
    args = ap.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    t0 = time.perf_counter()
    for dataset in args.datasets:
        X, y, _ = load_dataset(dataset)
        n, d = X.shape
        A = MinMaxScaler().fit_transform(X)
        d1 = pdist(A, "sqeuclidean")
        keep = d1 > 0
        d1k = d1[keep]
        r1 = rankdata(d1k)
        r1c = (r1 - r1.mean()) / r1.std()
        D1sq = squareform(np.sqrt(d1))
        knn_orig = knn_agreement(D1sq, y)
        # fixed pair subset for the per-pair variance check
        iu = np.triu_indices(n, 1)
        pair_idx = np.flatnonzero(keep)
        sub = np.random.default_rng(12345).choice(pair_idx, min(N_CHECK_PAIRS, len(pair_idx)), replace=False)
        delta = A[iu[0][sub]] - A[iu[1][sub]]
        kappa = (delta ** 4).sum(1) / (delta ** 2).sum(1) ** 2
        out = os.path.join(RAW_DIR, f"distortion_{dataset}.csv")
        done = set()
        if os.path.exists(out):
            prev = pd.read_csv(out, usecols=["k", "family", "draw"])
            done = set(zip(prev["k"], prev["family"], prev["draw"]))
        fams = families_for(d)
        sums = {}
        for r in range(args.draws):
            if (time.perf_counter() - t0) / 60 > args.max_minutes:
                print(f"stopping at the {args.max_minutes}-minute limit; rerun to resume", flush=True)
                return
            G, U = base_matrices(r, dataset, d)
            rows = []
            for fam in fams:
                for k in k_values(d):
                    R = projection_matrix(fam, G, U, k)
                    E = project(A, R)
                    # per-pair variance accumulators (always updated, so the check covers every draw)
                    qs = ((delta @ R) ** 2).sum(1) / (k * (delta ** 2).sum(1))
                    acc = sums.setdefault((k, fam), [np.zeros(len(sub)), np.zeros(len(sub)), 0])
                    acc[0] += qs; acc[1] += qs ** 2; acc[2] += 1
                    if (k, fam, r) in done:
                        continue
                    d2 = pdist(E, "sqeuclidean")
                    q = d2[keep] / d1k
                    ratio = np.sqrt(q)
                    row = {"dataset": dataset, "n": n, "d": d, "k": k, "family": fam, "draw": r,
                           "n_pairs": int(keep.sum()),
                           "zero_columns": int((np.abs(R).sum(0) == 0).sum()),
                           "dropped_features": int((np.abs(R).sum(1) == 0).sum()),
                           "q_mean": q.mean(), "q_var": q.var(), "q_absdev_max": np.abs(q - 1).max()}
                    for p, v in zip(QUANTILES, np.quantile(q, QUANTILES)):
                        row[f"q_p{int(round(p * 100)):02d}"] = v
                    for e in EPS:
                        tag = str(e).replace(".", "")
                        row[f"share_absdev_gt_{tag}"] = float((np.abs(q - 1) > e).mean())
                        row[f"share_low_{tag}"] = float((q <= 1 - e).mean())
                        row[f"share_high_{tag}"] = float((q >= 1 + e).mean())
                    row["m"] = ratio.min(); row["M"] = ratio.max()
                    row["eps_report"] = max(1 - row["m"], row["M"] - 1)
                    row["mean_dist_ratio"] = ratio.mean()
                    r2 = rankdata(d2[keep])
                    row["spearman"] = float(np.mean(r1c * (r2 - r2.mean()) / r2.std())) if r2.std() > 0 else np.nan
                    row["knn5_agreement"] = knn_agreement(squareform(np.sqrt(d2)), y)
                    row["knn5_agreement_original"] = knn_orig
                    rows.append(row)
            if rows:
                pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        vrows = []
        for (k, fam), (s1, s2, cnt) in sums.items():
            mean_q = s1 / cnt
            emp = s2 / cnt - mean_q ** 2
            s = sparsity(fam, d)
            pred = (2 + (s - 3) * kappa) / k
            vrows.append({"dataset": dataset, "k": k, "family": fam, "draws": cnt, "pairs": len(sub),
                          "mean_q": mean_q.mean(), "mean_emp_var": emp.mean(), "mean_pred_var": pred.mean(),
                          "emp_over_pred": emp.mean() / pred.mean(),
                          "corr_emp_pred": float(np.corrcoef(emp, pred)[0, 1]) if pred.std() > 1e-12 else np.nan,
                          "kappa_mean": kappa.mean(), "kappa_min": kappa.min(), "kappa_max": kappa.max()})
        pd.DataFrame(vrows).to_csv(os.path.join(RAW_DIR, f"variance_check_{dataset}.csv"), index=False)
        print(f"{dataset}: done at {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
