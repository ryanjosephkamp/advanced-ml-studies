"""Replicate the February 13, 2026 pipeline over 100 draws.

Same steps as scripts/kamp_hw3.py with --legacy-seeding: MinMax scaling on all rows, k = 2,
Eq. 3 and Eq. 4 matrices from np.random.default_rng(seed) (one stream per seed, Eq. 3 drawn first),
one stratified 70/30 split with random_state = 42, and default scikit-learn SVMs (C = 1,
gamma = "scale", polynomial coef0 = 0). Iris is read from iris.data, the file the run used.
Seeds 0 to 99 include 42, the report's draw.

Writes study/results/raw/replication.csv: one row per (dataset, seed, distribution, kernel).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "scripts"))
from common import DATASETS, RAW_DIR, load_dataset  # noqa: E402
from random_projection import (build_random_matrix_eq3, build_random_matrix_eq4,  # noqa: E402
                               project_data, scale_data)

KERNELS = [("linear", {"kernel": "linear"}), ("poly2", {"kernel": "poly", "degree": 2}),
           ("poly3", {"kernel": "poly", "degree": 3}), ("rbf", {"kernel": "rbf"})]
SPLIT_SEED = 42


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        X, y, _ = load_dataset(dataset, iris_file="iris.data")
        A, _ = scale_data(X)
        n, d = A.shape
        tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=SPLIT_SEED, stratify=y)
        d1 = pdist(A)
        keep = d1 > 0
        for seed in range(100):
            rng = np.random.default_rng(seed)
            mats = {"eq3": build_random_matrix_eq3(d, 2, rng), "eq4": build_random_matrix_eq4(d, 2, rng)}
            for eq, R in mats.items():
                E = project_data(A, R, 2)
                ratio = pdist(E)[keep] / d1[keep]
                base = {"dataset": dataset, "seed": seed, "distribution": eq,
                        "m": ratio.min(), "M": ratio.max(), "eps_report": max(1 - ratio.min(), ratio.max() - 1),
                        "mean_dist_ratio": ratio.mean(), "mean_sq_ratio": (ratio ** 2).mean(),
                        "zero_columns": int((np.abs(R).sum(0) == 0).sum()),
                        "dropped_features": int((np.abs(R).sum(1) == 0).sum())}
                for name, kw in KERNELS:
                    pred = SVC(random_state=42, **kw).fit(E[tr], y[tr]).predict(E[te])
                    rows.append({**base, "kernel": name, "acc": float((pred == y[te]).mean()), "n_test": len(te)})
        print(dataset, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RAW_DIR, "replication.csv"), index=False)


if __name__ == "__main__":
    main()
