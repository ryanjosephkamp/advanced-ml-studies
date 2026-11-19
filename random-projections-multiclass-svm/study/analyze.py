"""Aggregate the raw study output into the summary tables in study/results/.

Intervals are 95% percentile bootstrap intervals of the mean over the 100 draws (10,000 resamples,
fixed seed). Draws are independent (each has its own split and matrices), and within a draw every
family, baseline, and kernel shares the split, so differences are paired by draw.

Primary comparisons: the cross-validated model's test accuracy (balanced accuracy for Glass),
for Eq. 3, Eq. 4, very sparse, and one per column against the Gaussian, and Eq. 4 against Eq. 3,
at every (dataset, k). p-values come from paired t-tests and are Holm-adjusted across all primary
comparisons. "Equivalent" means the 95% interval lies inside +/-2 percentage points.

Outputs:
  accuracy_summary.csv     mean, SD, and interval per (dataset, k, family, model, metric)
  paired_differences.csv   paired differences with intervals, shares of draws, and p-values
  distortion_summary.csv   distortion, rank, and neighbor measures per (dataset, k, family), with theory
  variance_check.csv       per-pair variance of q across draws against (2 + (s - 3) kappa) / k
  replication_summary.csv  the original pipeline over 100 draws, with the report's draw ranked
  raw_svm.csv.gz, raw_distortion.csv.gz, raw_replication.csv.gz   per-draw rows (rounded)

Reads the per-draw working files in study/results/raw/ when present, otherwise the committed
.csv.gz copies (so the summaries can be rebuilt without rerunning the study). The variance check
needs the working files; without them the committed variance_check.csv is kept. If a set of working
files does not cover the full grid (a run stopped at its time limit), analyze.py warns and leaves that
set's committed files as they are (its raw_*.csv.gz copy, variance_check.csv, or, for the replication,
replication_summary.csv and raw_replication.csv.gz); the accuracy, paired-difference, and distortion
summaries are still written from the rows present (the paired differences only once some comparison
has two or more paired draws; until then, paired_differences.csv is left as it is). Each output takes
its bootstrap resamples from its own number of draws, so a short run of one script does not change
another output's intervals.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASETS, N_DRAWS, RAW_DIR, RESULTS_DIR, families_for, k_values,  # noqa: E402
                    load_dataset)

B = 10_000
MODELS = ["cv", "linear", "poly2", "poly3", "rbf"]
METRICS = ["acc", "bal_acc"]
COMPARISONS = [("rademacher", "gaussian"), ("achlioptas3", "gaussian"), ("verysparse", "gaussian"),
               ("onepercol", "gaussian"), ("achlioptas3", "rademacher")]
EQUIV = 0.02


def boot_ci(x: np.ndarray, idx: np.ndarray):
    m = x[idx].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def resample_index(n: int) -> np.ndarray:
    """B bootstrap resamples of n draws, from the same seed for every output (so with 100 draws, every
    output uses the same resamples)."""
    return np.random.default_rng(20261111).integers(0, n, size=(B, n))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load_raw(pattern: str, packed: str):
    """Per-draw rows from the working files in raw/, or from the committed .csv.gz copy if raw/ is absent.
    Also returns whether the rows came from the working files."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, pattern)))
    if files:
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True), True
    path = os.path.join(RESULTS_DIR, packed)
    return (pd.read_csv(path) if os.path.exists(path) else None), False


def full_grid():
    """The rows a finished study writes: (dataset, k, family, draw) for run_svm.py and run_distortion.py,
    (dataset, k, family) for the variance check, and (dataset, seed, distribution, kernel) for replicate.py."""
    svm, dist, var = set(), set(), set()
    for ds in DATASETS:
        d = load_dataset(ds)[0].shape[1]
        for r in range(N_DRAWS):
            svm.add((ds, d, "full", r))
            for k in k_values(d):
                svm.add((ds, k, "pca", r))
                for fam in families_for(d):
                    svm.add((ds, k, fam, r))
                    dist.add((ds, k, fam, r))
                    var.add((ds, k, fam))
    rep = {(ds, seed, eq, kern) for ds in DATASETS for seed in range(100) for eq in ("eq3", "eq4")
           for kern in ("linear", "poly2", "poly3", "rbf")}
    return {"svm": svm, "distortion": dist, "variance": var, "replication": rep}


def complete(df: pd.DataFrame, cols: list, grid: set, what: str, kept: str) -> bool:
    """True if the working files hold every row of the grid once; otherwise print a warning."""
    have = set(df[cols].itertuples(index=False, name=None))
    if have == grid and len(df) == len(grid):
        return True
    print(f"warning: the working files in {RAW_DIR} hold {len(have & grid)} of the {len(grid)} {what} rows of a "
          f"finished run ({len(df)} rows in all). Left as they are: {kept}. The other outputs use the rows present. "
          f"Finish the run first: run_svm.py and run_distortion.py print \"rerun to resume\" until they are done, "
          f"and replicate.py writes its rows when it ends.",
          flush=True)
    return False


def primary_metric(dataset: str) -> str:
    return "bal_acc" if dataset == "glass" else "acc"


def main():
    svm, svm_working = load_raw("svm_*.csv", "raw_svm.csv.gz")
    grid = full_grid() if os.path.isdir(RAW_DIR) else None
    pack_svm = not svm_working or complete(svm, ["dataset", "k", "family", "draw"], grid["svm"],
                                           "classification", "raw_svm.csv.gz")
    n_draws = svm["draw"].nunique()
    idx = resample_index(n_draws)

    # ---- accuracy summary ----
    rows = []
    for (ds, k, fam), g in svm.groupby(["dataset", "k", "family"]):
        g = g.sort_values("draw")
        for model in MODELS:
            for metric in METRICS:
                x = g[f"{model}_{metric}"].to_numpy()
                lo, hi = boot_ci(x, idx) if len(x) == n_draws else (np.nan, np.nan)
                rows.append({"dataset": ds, "k": k, "family": fam, "model": model, "metric": metric,
                             "n_draws": len(x), "mean": x.mean(), "sd": x.std(ddof=1), "ci_low": lo,
                             "ci_high": hi, "median": np.median(x)})
        rows.append({"dataset": ds, "k": k, "family": fam, "model": "majority", "metric": "acc",
                     "n_draws": len(g), "mean": g["majority_acc"].mean(), "sd": g["majority_acc"].std(ddof=1),
                     "ci_low": np.nan, "ci_high": np.nan, "median": g["majority_acc"].median()})
    acc = pd.DataFrame(rows)
    acc.to_csv(os.path.join(RESULTS_DIR, "accuracy_summary.csv"), index=False, float_format="%.6g")

    # ---- paired differences ----
    piv = {}
    for (ds, k, fam), g in svm.groupby(["dataset", "k", "family"]):
        piv[(ds, k, fam)] = g.set_index("draw").sort_index()
    full = {ds: g.set_index("draw").sort_index() for ds, g in svm[svm.family == "full"].groupby("dataset")}
    prow = []
    for ds in sorted(svm.dataset.unique()):
        ks = sorted(svm[(svm.dataset == ds) & (svm.family != "full")]["k"].unique())
        for k in ks:
            pairs = list(COMPARISONS) + [(f, "full") for f in
                                         ["gaussian", "rademacher", "achlioptas3", "verysparse", "onepercol", "pca"]]
            pairs += [("gaussian", "pca")]
            for a, b in pairs:
                A = piv.get((ds, k, a))
                Bf = full.get(ds) if b == "full" else piv.get((ds, k, b))
                if A is None or Bf is None:
                    continue
                common_draws = A.index.intersection(Bf.index)
                for model in MODELS:
                    for metric in METRICS:
                        diff = (A.loc[common_draws, f"{model}_{metric}"] - Bf.loc[common_draws, f"{model}_{metric}"]).to_numpy()
                        if len(diff) < 2:
                            continue
                        lo, hi = boot_ci(diff, idx) if len(diff) == n_draws else (np.nan, np.nan)
                        p = stats.ttest_1samp(diff, 0.0).pvalue if diff.std() > 0 else 1.0
                        prow.append({"dataset": ds, "k": k, "a": a, "b": b, "model": model, "metric": metric,
                                     "n_draws": len(diff), "mean_diff": diff.mean(), "ci_low": lo, "ci_high": hi,
                                     "share_a_higher": (diff > 0).mean(), "share_equal": (diff == 0).mean(),
                                     "share_a_lower": (diff < 0).mean(), "p_t": p,
                                     "equivalent_2pts": bool(lo > -EQUIV and hi < EQUIV),
                                     "primary": bool(model == "cv" and metric == primary_metric(ds)
                                                     and (a, b) in COMPARISONS)})
    pdf = pd.DataFrame(prow)
    if pdf.empty:
        print("warning: no comparison has two or more paired draws in the rows present, so "
              "paired_differences.csv is left as it is.", flush=True)
    else:
        pdf["p_holm_primary"] = np.nan
        prim = pdf["primary"]
        pdf.loc[prim, "p_holm_primary"] = holm(pdf.loc[prim, "p_t"].to_numpy())
        pdf.to_csv(os.path.join(RESULTS_DIR, "paired_differences.csv"), index=False, float_format="%.6g")

    # ---- distortion ----
    dist, dist_working = load_raw("distortion_*.csv", "raw_distortion.csv.gz")
    if dist is not None:
        nd = dist["draw"].nunique()
        didx = resample_index(nd)
        num = [c for c in dist.columns if c not in ("dataset", "family", "draw", "n", "d", "k", "n_pairs")]
        g = dist.groupby(["dataset", "k", "family"])
        ds_sum = g[num].mean().add_suffix("_mean").reset_index()
        ds_sum["share_draws_zero_column"] = g["zero_columns"].apply(lambda s: (s > 0).mean()).to_numpy()
        ds_sum["share_draws_dropped_feature"] = g["dropped_features"].apply(lambda s: (s > 0).mean()).to_numpy()
        for c in ["spearman", "knn5_agreement", "q_var"]:
            ds_sum[f"{c}_ci_low"] = g[c].apply(lambda s: boot_ci(s.to_numpy(), didx)[0] if len(s) == nd else np.nan).to_numpy()
            ds_sum[f"{c}_ci_high"] = g[c].apply(lambda s: boot_ci(s.to_numpy(), didx)[1] if len(s) == nd else np.nan).to_numpy()
        ds_sum["n_draws"] = g.size().to_numpy()
        ks = ds_sum["k"].to_numpy(float)
        for e in [0.1, 0.25, 0.5]:
            tag = str(e).replace(".", "")
            # Gaussian: q ~ chi2_k / k exactly, for every pair
            ds_sum[f"gauss_exact_low_{tag}"] = stats.chi2.cdf(ks * (1 - e), ks)
            ds_sum[f"gauss_exact_high_{tag}"] = stats.chi2.sf(ks * (1 + e), ks)
            # Per-pair Chernoff bounds (Dasgupta and Gupta, 2003): exp(k/2 * (1 - b + ln b)), b = 1 -/+ e
            ds_sum[f"bound_low_{tag}"] = np.exp(ks / 2 * (e + np.log(1 - e)))
            ds_sum[f"bound_high_{tag}"] = np.exp(ks / 2 * (-e + np.log(1 + e)))
        ds_sum.to_csv(os.path.join(RESULTS_DIR, "distortion_summary.csv"), index=False, float_format="%.6g")
        if not dist_working or complete(dist, ["dataset", "k", "family", "draw"], grid["distortion"],
                                        "distortion", "raw_distortion.csv.gz"):
            dist.round(6).to_csv(os.path.join(RESULTS_DIR, "raw_distortion.csv.gz"), index=False, compression={"method": "gzip", "mtime": 0})
    vfiles = sorted(glob.glob(os.path.join(RAW_DIR, "variance_check_*.csv")))
    if vfiles:
        vc = pd.concat([pd.read_csv(f) for f in vfiles])
        if complete(vc, ["dataset", "k", "family"], grid["variance"], "variance check", "variance_check.csv"):
            vc.to_csv(os.path.join(RESULTS_DIR, "variance_check.csv"), index=False, float_format="%.6g")

    # ---- replication of the original pipeline ----
    rep, rep_working = load_raw("replication.csv", "raw_replication.csv.gz")
    if rep is not None and (not rep_working or complete(rep, ["dataset", "seed", "distribution", "kernel"],
                                                         grid["replication"], "replication",
                                                         "replication_summary.csv and raw_replication.csv.gz")):
        ridx = resample_index(rep["seed"].nunique())
        best = rep.groupby(["dataset", "seed", "distribution"])["acc"].agg(["max", "min"]).reset_index()
        rrows = []
        for ds, g in best.groupby("dataset"):
            w = g.pivot(index="seed", columns="distribution", values="max")
            wmin = g.pivot(index="seed", columns="distribution", values="min")
            for eq in ["eq3", "eq4"]:
                x = w[eq].to_numpy(); v = w.loc[42, eq]
                lo, hi = boot_ci(x, ridx)
                rrows.append({"dataset": ds, "quantity": f"best_kernel_acc_{eq}", "mean": x.mean(), "ci_low": lo,
                              "ci_high": hi, "median": np.median(x), "p05": np.percentile(x, 5),
                              "p95": np.percentile(x, 95), "min": x.min(), "max": x.max(), "report_seed42": v,
                              "draws_lower": int((x < v).sum()), "draws_equal_excl_42": int((x == v).sum() - 1),
                              "draws_higher": int((x > v).sum()),
                              "four_way_ties": int((w[eq] == wmin[eq]).sum())})
            diff = (w["eq4"] - w["eq3"]).to_numpy()
            lo, hi = boot_ci(diff, ridx)
            rrows.append({"dataset": ds, "quantity": "best_eq4_minus_best_eq3", "mean": diff.mean(), "ci_low": lo,
                          "ci_high": hi, "median": np.median(diff), "p05": np.percentile(diff, 5),
                          "p95": np.percentile(diff, 95), "min": diff.min(), "max": diff.max(),
                          "report_seed42": diff[list(w.index).index(42)], "draws_lower": int((diff < 0).sum()),
                          "draws_equal_excl_42": int((diff == 0).sum()), "draws_higher": int((diff > 0).sum())})
            for eq in ["eq3", "eq4"]:
                r = rep[(rep.dataset == ds) & (rep.distribution == eq) & (rep.kernel == "linear")]
                rrows.append({"dataset": ds, "quantity": f"mean_sq_ratio_{eq}", "mean": r.mean_sq_ratio.mean(),
                              "median": r.mean_sq_ratio.median(), "min": r.mean_sq_ratio.min(),
                              "max": r.mean_sq_ratio.max()})
                rrows.append({"dataset": ds, "quantity": f"mean_dist_ratio_{eq}", "mean": r.mean_dist_ratio.mean(),
                              "median": r.mean_dist_ratio.median(), "min": r.mean_dist_ratio.min(),
                              "max": r.mean_dist_ratio.max(), "p05": np.percentile(r.mean_dist_ratio, 5),
                              "p95": np.percentile(r.mean_dist_ratio, 95)})
                rrows.append({"dataset": ds, "quantity": f"eps_report_{eq}", "mean": r.eps_report.mean(),
                              "median": r.eps_report.median(), "min": r.eps_report.min(), "max": r.eps_report.max()})
        pd.DataFrame(rrows).to_csv(os.path.join(RESULTS_DIR, "replication_summary.csv"), index=False,
                                   float_format="%.6g")
        rep.round(6).to_csv(os.path.join(RESULTS_DIR, "raw_replication.csv.gz"), index=False, compression={"method": "gzip", "mtime": 0})

    if pack_svm:
        svm.round(6).to_csv(os.path.join(RESULTS_DIR, "raw_svm.csv.gz"), index=False, compression={"method": "gzip", "mtime": 0})
    print("analysis written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
