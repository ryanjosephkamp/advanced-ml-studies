"""Classification after random projection: tuned SVMs, 100 paired draws per setting.

For each (dataset, k, family, draw): split 70/30 (stratified, seed = draw); fit MinMax scaling on
the training part; project with the family's matrix (or fit PCA to k on the training part, or keep
all features); standardize the projected training features; tune four SVM kernels by 3-fold
stratified cross-validation inside the training part; refit; score the held-out 30%.
The headline model ("cv") is the kernel with the best inner cross-validation accuracy.

One row per (dataset, k, family, draw) is appended to study/results/raw/svm_<dataset>.csv.
Rows already present are skipped, so a stopped job resumes where it left off.

Examples:
  python study/run_svm.py --datasets iris seeds
  python study/run_svm.py --datasets breast_cancer --ks 1 --families gaussian rademacher achlioptas3 verysparse onepercol pca
  python study/run_svm.py --datasets breast_cancer --families full
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (N_DRAWS, RAW_DIR, TEST_SIZE, base_matrices, families_for, k_values,  # noqa: E402
                    load_dataset, project, projection_matrix)

KERNELS = ["linear", "poly2", "poly3", "rbf"]
# libsvm has no iteration limit by default, and a few polynomial fits on heavily overlapping 1-D
# data (k = 1) never converge. 1e8 iterations take at least about a second, longer than any
# complete cell took before the cap was added, so the cap only stops fits that would not finish.
MAX_ITER = 100_000_000
GRIDS = {
    "linear": {"kernel": ["linear"], "C": [0.01, 0.1, 1, 10, 100]},
    "poly2": {"kernel": ["poly"], "degree": [2], "coef0": [1.0], "C": [0.1, 1, 10], "gamma": [0.5, 1, 2]},
    "poly3": {"kernel": ["poly"], "degree": [3], "coef0": [1.0], "C": [0.1, 1, 10], "gamma": [0.5, 1, 2]},
    "rbf": {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": [0.1, 0.3, 1, 3, 10]},
}


COLUMNS = (["dataset", "k", "family", "draw", "n_train", "n_test", "zero_columns", "dropped_features",
            "pca_var_explained"]
           + [f"{kern}_{m}" for kern in KERNELS for m in ["acc", "bal_acc", "f1_macro", "cv_acc", "C", "gamma_rel"]]
           + ["cv_kernel", "cv_acc", "cv_bal_acc", "cv_f1_macro", "majority_acc", "capped_fits", "seconds"])


def grid_for(kernel: str, n_features: int) -> dict:
    """Parameter grid with gamma relative to 1/n_features (scikit-learn's "scale" on standardized data)."""
    g = dict(GRIDS[kernel])
    if "gamma" in g:
        g["gamma"] = [v / n_features for v in g["gamma"]]
    return g


def run_cell(X, y, dataset, k, family, draw):
    """Train and score every kernel for one cell; return a flat dict (one CSV row)."""
    d = X.shape[1]
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=draw, stratify=y)
    mm = MinMaxScaler().fit(X[idx_tr])
    Xtr, Xte = mm.transform(X[idx_tr]), mm.transform(X[idx_te])
    ytr, yte = y[idx_tr], y[idx_te]
    row = {"dataset": dataset, "k": k, "family": family, "draw": draw, "n_train": len(ytr), "n_test": len(yte)}
    if family == "full":
        row["k"] = d
    elif family == "pca":
        p = PCA(n_components=k).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        row["pca_var_explained"] = float(p.explained_variance_ratio_.sum())
    else:
        G, U = base_matrices(draw, dataset, d)
        R = projection_matrix(family, G, U, k)
        row["zero_columns"] = int((np.abs(R).sum(0) == 0).sum())
        row["dropped_features"] = int((np.abs(R).sum(1) == 0).sum())
        Xtr, Xte = project(Xtr, R), project(Xte, R)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=draw)
    best_cv, best_kernel = -1.0, None
    capped = 0
    for kern in KERNELS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            gs = GridSearchCV(SVC(max_iter=MAX_ITER), grid_for(kern, Xtr.shape[1]), cv=cv, scoring="accuracy",
                              n_jobs=1).fit(Xtr, ytr)
        capped += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        pred = gs.predict(Xte)
        row[f"{kern}_acc"] = accuracy_score(yte, pred)
        row[f"{kern}_bal_acc"] = balanced_accuracy_score(yte, pred)
        row[f"{kern}_f1_macro"] = f1_score(yte, pred, average="macro", zero_division=0)
        row[f"{kern}_cv_acc"] = gs.best_score_
        row[f"{kern}_C"] = gs.best_params_["C"]
        row[f"{kern}_gamma_rel"] = gs.best_params_["gamma"] * Xtr.shape[1] if "gamma" in gs.best_params_ else np.nan
        if gs.best_score_ > best_cv + 1e-12:  # ties go to the earlier (simpler) kernel
            best_cv, best_kernel = gs.best_score_, kern
    row["cv_kernel"] = best_kernel
    for m in ["acc", "bal_acc", "f1_macro"]:
        row[f"cv_{m}"] = row[f"{best_kernel}_{m}"]
    row["majority_acc"] = float(np.bincount(yte).max() / len(yte))
    row["capped_fits"] = capped  # fits (of 133 per cell) stopped at MAX_ITER
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=None, help="target sizes (default: all k < d)")
    ap.add_argument("--families", nargs="+", default=None,
                    help="projection families plus 'pca' and 'full' (default: all of them)")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--max-minutes", type=float, default=9.0, help="stop cleanly after this long; rerun to resume")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(RAW_DIR, exist_ok=True)
    t0 = time.perf_counter()
    for dataset in args.datasets:
        X, y, _ = load_dataset(dataset)
        d = X.shape[1]
        fams = args.families or (families_for(d) + ["pca", "full"])
        ks = args.ks or k_values(d)
        out = os.path.join(RAW_DIR, f"svm_{dataset}.csv")
        done = set()
        if os.path.exists(out):
            prev = pd.read_csv(out, usecols=["k", "family", "draw"])
            done = set(zip(prev["k"], prev["family"], prev["draw"]))
        cells = []
        for fam in fams:
            if fam == "verysparse" and fam not in families_for(d):
                continue
            for k in ([d] if fam == "full" else ks):
                cells += [(k, fam, r) for r in range(args.draws) if (k, fam, r) not in done]
        print(f"{dataset}: {len(cells)} cells to run ({len(done)} already done)", flush=True)
        for i, (k, fam, r) in enumerate(cells):
            if (time.perf_counter() - t0) / 60 > args.max_minutes:
                print(f"stopping at the {args.max_minutes}-minute limit; rerun to resume", flush=True)
                return
            t = time.perf_counter()
            row = run_cell(X, y, dataset, k if fam != "full" else d, fam, r)
            row["seconds"] = time.perf_counter() - t
            pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(out, mode="a", header=not os.path.exists(out),
                                                                index=False)
            if (i + 1) % 100 == 0:
                print(f"  {dataset} {i + 1}/{len(cells)} cells, {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
