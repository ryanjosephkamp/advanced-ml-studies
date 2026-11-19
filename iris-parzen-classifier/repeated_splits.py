# repeated_splits.py

# CS 7063 – Advanced Machine Learning Spring 2026
# Assignment #1: Parzen Classifiers – Distances in ML
# Ryan Kamp
# Repeated-split evaluation, November 2026

"""
Repeated random-split evaluation of the classifiers in kamp_hw1.py.

kamp_hw1.py evaluates each train/test ratio on one split (random_state=42). Because the five
splits come from the same shuffle, their test sets are nested, and each ratio is a single draw.
This script repeats the same procedure (train_test_split, not stratified) with 500 seeds per
ratio, seeds 0 to 499, for every problem and kernel, and reports:

- the mean test accuracy over the 500 splits, with a 95% confidence interval for that mean
  (mean ± 1.96 SD / sqrt(500); it reflects the choice of splits only, not new flowers). With all
  ratios pooled, the five splits drawn with one seed share a shuffle, so the interval uses the SD
  of the 500 per-seed means (each the mean over the five ratios) and n = 500;
- the 2.5th to 97.5th percentiles of single-split test accuracy;
- the mean training accuracy and the mean majority-class rate of the test sets (the larger
  class's share of each test set, so at least what predicting one fixed label scores on it);
- the accuracy of the original split (seed 42, which is also one of seeds 0 to 499) for
  comparison;
- for the Gaussian kernel, how often it is at least as accurate as the best other kernel on the
  same split.

As an added comparison, the script repeats the evaluation on the same splits with standardized
features: each feature is centered and divided by its standard deviation (population, divide by
n), both computed on the problem's own training rows (for Problem 4, versicolor and virginica
only), and the test rows are scaled with the same values. The study itself uses the raw
measurements in cm.

The classifier is the same rule as ParzenWindowClassifier in kamp_hw1.py, written with matrix
operations so that 32,500 fits run in seconds. Before the evaluation, the script checks that it
gives the same predictions as ParzenWindowClassifier on the 65 original runs, and the same test
accuracies as parzen_classifier_results.csv.

Outputs, in repeated_splits/:
- repeated_split_results.csv: one row per problem, kernel, and ratio, plus one row per problem
  and kernel with all ratios pooled ("All").
- repeated_split_by_kernel.png: mean and 2.5th to 97.5th percentiles of single-split test
  accuracy per kernel at 80/20.
- repeated_split_by_ratio.png: mean test accuracy over Problems 2 to 4 by training share, with
  the 2.5th to 97.5th percentiles of the three-problem mean on single splits.
- standardized_feature_results.csv: raw against standardized features on the same splits, one
  row per problem, kernel, and ratio, plus "All".

Usage:
    python repeated_splits.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "repeated_splits"

SEEDS = range(500)                        # random_state values for train_test_split
TEST_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5]    # 90/10, 80/20, 70/30, 60/40, 50/50
ORIGINAL_SEED = 42                        # the split used by kamp_hw1.py
SIGMA = 1 / np.sqrt(2)                    # Gaussian kernel width, as in kamp_hw1.py

KERNELS = ["Dot Product", "Polynomial (d=2)", "Polynomial (d=3)", "Gaussian"]
# Problem number: (name, positive class, class excluded or None)
PROBLEMS = {
    1: ("Setosa vs. (Versicolor + Virginica)", 0, None),
    2: ("Virginica vs. (Setosa + Versicolor)", 2, None),
    3: ("Versicolor vs. (Setosa + Virginica)", 1, None),
    4: ("Versicolor vs. Virginica", 1, 0),
}
CONFIGS = [(1, "Dot Product")] + [(p, k) for p in (2, 3, 4) for k in KERNELS]

# Okabe-Ito colors (colorblind-safe) and distinct markers per kernel
STYLE = {
    "Dot Product": dict(color="#0072B2", marker="o", label="Dot product"),
    "Polynomial (d=2)": dict(color="#E69F00", marker="s", label="Polynomial, d = 2"),
    "Polynomial (d=3)": dict(color="#CC79A7", marker="^", label="Polynomial, d = 3"),
    "Gaussian": dict(color="#009E73", marker="D", label="Gaussian, σ = 1/√2"),
}


def split_label(test_size: float) -> str:
    return f"{round((1 - test_size) * 100)}/{round(test_size * 100)}"


def kernel_matrix(A: np.ndarray, B: np.ndarray, kernel: str) -> np.ndarray:
    """Kernel values between the rows of A and the rows of B."""
    G = A @ B.T
    if kernel == "Dot Product":
        return G
    if kernel == "Polynomial (d=2)":
        return (G + 1) ** 2
    if kernel == "Polynomial (d=3)":
        return (G + 1) ** 3
    if kernel == "Gaussian":
        sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * G
        return np.exp(-np.maximum(sq, 0) / (2 * SIGMA ** 2))
    raise ValueError(kernel)


def fit_predict(X_train, y_train, X_eval, kernel):
    """The kamp_hw1.py rule: sign(sum_i alpha'_i K(x_i, x) - b), ties to +1."""
    alpha_pos = (y_train == 1) / np.sum(y_train == 1)
    alpha_neg = (y_train == -1) / np.sum(y_train == -1)
    K = kernel_matrix(X_train, X_train, kernel)
    b = (alpha_pos @ K @ alpha_pos - alpha_neg @ K @ alpha_neg) / 2
    scores = kernel_matrix(X_eval, X_train, kernel) @ (alpha_pos - alpha_neg) - b
    return np.where(scores >= 0, 1, -1)


def problem_data(X, y, problem):
    """Rows and +1/-1 labels for one problem."""
    _, positive, excluded = PROBLEMS[problem]
    keep = np.ones(len(y), bool) if excluded is None else y != excluded
    return X[keep], np.where(y[keep] == positive, 1, -1)


def evaluate_split(X, y, test_size, seed, standardize=False):
    """Train and test accuracy and test majority rate for every configuration on one split.

    With standardize=True, features are scaled with the mean and SD (divide by n) of the
    problem's own training rows, and the test rows with the same values.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    out = {}
    for problem, kernel in CONFIGS:
        Xp_tr, yp_tr = problem_data(X_tr, y_tr, problem)
        Xp_te, yp_te = problem_data(X_te, y_te, problem)
        if standardize:
            mean, sd = Xp_tr.mean(axis=0), Xp_tr.std(axis=0)
            Xp_tr, Xp_te = (Xp_tr - mean) / sd, (Xp_te - mean) / sd
        train_acc = np.mean(fit_predict(Xp_tr, yp_tr, Xp_tr, kernel) == yp_tr)
        test_acc = np.mean(fit_predict(Xp_tr, yp_tr, Xp_te, kernel) == yp_te)
        majority = max(np.mean(yp_te == 1), np.mean(yp_te == -1))
        out[(problem, kernel)] = (train_acc, test_acc, majority, len(yp_tr), len(yp_te))
    return out


def check_against_original(X, y):
    """Same predictions as ParzenWindowClassifier, and same accuracies as the results CSV."""
    import kamp_hw1 as hw
    original = {
        "Dot Product": (hw.dot_product_kernel, {}),
        "Polynomial (d=2)": (hw.polynomial_kernel, {"degree": 2}),
        "Polynomial (d=3)": (hw.polynomial_kernel, {"degree": 3}),
        "Gaussian": (hw.gaussian_kernel, {"sigma": SIGMA}),
    }
    logged = pd.read_csv(ROOT / "parzen_classifier_results.csv").set_index(
        ["Split Label", "Problem", "Kernel"])["Test Accuracy"]
    for test_size in TEST_SIZES:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=ORIGINAL_SEED)
        for problem, kernel in CONFIGS:
            Xp_tr, yp_tr = problem_data(X_tr, y_tr, problem)
            Xp_te, yp_te = problem_data(X_te, y_te, problem)
            fn, params = original[kernel]
            clf = hw.ParzenWindowClassifier(kernel=fn, **params)
            clf.fit(Xp_tr, yp_tr)
            for X_eval in (Xp_tr, Xp_te):
                assert np.array_equal(clf.predict(X_eval), fit_predict(Xp_tr, yp_tr, X_eval, kernel))
            test_acc = np.mean(fit_predict(Xp_tr, yp_tr, Xp_te, kernel) == yp_te)
            assert abs(test_acc - logged[(split_label(test_size), problem, kernel)]) < 1e-12
    print(f"Check passed: same predictions as ParzenWindowClassifier on all {len(CONFIGS) * len(TEST_SIZES)} original runs.")


def summarize(values, seed_means=None):
    """Mean, SD, and percentiles of single-split accuracies, and a 95% CI for the mean.

    The CI is mean ± 1.96 SD / sqrt(n) over the splits, or, when seed_means is given (pooled rows),
    over the per-seed means, because the five splits drawn with one seed share a shuffle.
    """
    values = np.asarray(values, float)
    ci_values = values if seed_means is None else np.asarray(seed_means, float)
    mean, sd = values.mean(), values.std(ddof=1)
    half = 1.96 * ci_values.std(ddof=1) / np.sqrt(len(ci_values))
    return dict(test_mean=mean, test_sd=sd, mean_ci95_low=mean - half, mean_ci95_high=mean + half,
                range95_low=np.percentile(values, 2.5), range95_high=np.percentile(values, 97.5))


def run():
    X, y = datasets.load_iris(return_X_y=True)
    check_against_original(X, y)

    # Accuracies for every split: results[test_size][(problem, kernel)] -> arrays over seeds
    results = {}
    for test_size in TEST_SIZES:
        per_seed = [evaluate_split(X, y, test_size, seed) for seed in SEEDS]
        results[test_size] = {
            cfg: np.array([s[cfg] for s in per_seed]) for cfg in CONFIGS
        }
    original = {ts: evaluate_split(X, y, ts, ORIGINAL_SEED) for ts in TEST_SIZES}

    rows = []
    for problem, kernel in CONFIGS:
        pooled_test, pooled_train, pooled_major = [], [], []
        for test_size in TEST_SIZES:
            arr = results[test_size][(problem, kernel)]
            train_acc, test_acc, majority, n_train, n_test = arr.T
            pooled_test.append(test_acc)
            pooled_train.append(train_acc)
            pooled_major.append(majority)
            rows.append(dict(problem=problem, problem_name=PROBLEMS[problem][0], kernel=kernel,
                             split=split_label(test_size), test_size=test_size,
                             train_n=int(n_train[0]) if problem != 4 else np.nan,
                             test_n=int(n_test[0]) if problem != 4 else np.nan,
                             n_splits=len(SEEDS), seed_first=min(SEEDS), seed_last=max(SEEDS),
                             **summarize(test_acc), train_mean=train_acc.mean(),
                             majority_mean=majority.mean(),
                             original_seed42_test=original[test_size][(problem, kernel)][1]))
        test_all = np.concatenate(pooled_test)
        seed_means = np.mean(pooled_test, axis=0)  # one mean over the five ratios per seed
        rows.append(dict(problem=problem, problem_name=PROBLEMS[problem][0], kernel=kernel, split="All",
                         test_size=np.nan, train_n=np.nan, test_n=np.nan,
                         n_splits=len(SEEDS) * len(TEST_SIZES), seed_first=min(SEEDS), seed_last=max(SEEDS),
                         **summarize(test_all, seed_means), train_mean=np.concatenate(pooled_train).mean(),
                         majority_mean=np.concatenate(pooled_major).mean(),
                         original_seed42_test=np.mean([original[ts][(problem, kernel)][1] for ts in TEST_SIZES])))

    df = pd.DataFrame(rows)
    df[["train_n", "test_n"]] = df[["train_n", "test_n"]].astype("Int64")

    # Gaussian against the best other kernel on the same split (Problems 2 to 4)
    df["share_ge_best_other"] = np.nan
    df["share_gt_best_other"] = np.nan
    df["mean_margin_vs_best_other"] = np.nan
    for problem in (2, 3, 4):
        for test_size in TEST_SIZES + [None]:
            sizes = TEST_SIZES if test_size is None else [test_size]
            gauss = np.concatenate([results[ts][(problem, "Gaussian")][:, 1] for ts in sizes])
            others = np.max([np.concatenate([results[ts][(problem, k)][:, 1] for ts in sizes])
                             for k in KERNELS[:3]], axis=0)
            label = "All" if test_size is None else split_label(test_size)
            m = (df.problem == problem) & (df.kernel == "Gaussian") & (df.split == label)
            df.loc[m, "share_ge_best_other"] = np.mean(gauss >= others)
            df.loc[m, "share_gt_best_other"] = np.mean(gauss > others)
            df.loc[m, "mean_margin_vs_best_other"] = np.mean(gauss - others)

    OUT_DIR.mkdir(exist_ok=True)
    df.round(6).to_csv(OUT_DIR / "repeated_split_results.csv", index=False)
    print(f"Saved: {OUT_DIR / 'repeated_split_results.csv'}")

    plot_by_kernel(df)
    plot_by_ratio(results, original)

    pooled = df[df.split == "All"].set_index(["problem", "kernel"])
    print("\nMean test accuracy over all 2,500 splits (95% CI of the mean) "
          "[2.5th to 97.5th percentiles of single splits]:")
    for (problem, kernel), r in pooled.iterrows():
        print(f"  Problem {problem} {kernel:17s} {r.test_mean:.3f} ({r.mean_ci95_low:.3f} to {r.mean_ci95_high:.3f}) "
              f"[{r.range95_low:.3f} to {r.range95_high:.3f}]; original split mean {r.original_seed42_test:.3f}")

    compare_standardized(X, y, results)


def compare_standardized(X, y, results):
    """Raw against standardized features on the same splits (an added comparison; the study uses raw features)."""
    standardized = {}
    for test_size in TEST_SIZES:
        per_seed = [evaluate_split(X, y, test_size, seed, standardize=True) for seed in SEEDS]
        standardized[test_size] = {cfg: np.array([s[cfg] for s in per_seed]) for cfg in CONFIGS}

    rows = []
    for problem, kernel in CONFIGS:
        for test_size in TEST_SIZES + [None]:
            sizes = TEST_SIZES if test_size is None else [test_size]
            raw = np.concatenate([results[ts][(problem, kernel)] for ts in sizes])
            std = np.concatenate([standardized[ts][(problem, kernel)] for ts in sizes])
            seed_means = None if test_size is not None else np.mean(
                [standardized[ts][(problem, kernel)][:, 1] for ts in TEST_SIZES], axis=0)
            summary = summarize(std[:, 1], seed_means)
            change = std[:, 1] - raw[:, 1]  # same split, standardized minus raw
            rows.append(dict(problem=problem, problem_name=PROBLEMS[problem][0], kernel=kernel,
                             split="All" if test_size is None else split_label(test_size),
                             n_splits=len(raw),
                             raw_test_mean=raw[:, 1].mean(), raw_train_mean=raw[:, 0].mean(),
                             standardized_test_mean=summary["test_mean"],
                             standardized_mean_ci95_low=summary["mean_ci95_low"],
                             standardized_mean_ci95_high=summary["mean_ci95_high"],
                             standardized_range95_low=summary["range95_low"],
                             standardized_range95_high=summary["range95_high"],
                             standardized_train_mean=std[:, 0].mean(),
                             change_mean=change.mean(),
                             share_standardized_higher=np.mean(change > 0),
                             share_standardized_lower=np.mean(change < 0)))

    df = pd.DataFrame(rows)
    df.round(6).to_csv(OUT_DIR / "standardized_feature_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'standardized_feature_results.csv'}")

    print("\nMean test accuracy over all 2,500 splits, raw -> standardized features (mean change on the same splits):")
    for _, r in df[df.split == "All"].iterrows():
        print(f"  Problem {r.problem} {r.kernel:17s} {r.raw_test_mean:.3f} -> {r.standardized_test_mean:.3f} "
              f"({r.change_mean:+.3f}); train {r.raw_train_mean:.3f} -> {r.standardized_train_mean:.3f}")


def _style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=14)


def plot_by_kernel(df):
    """Mean and 2.5th to 97.5th percentiles of single-split test accuracy per kernel at 80/20, one
    panel per problem."""
    sub = df[df.split == "80/20"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True, facecolor="white")
    ypos = {k: i for i, k in enumerate(reversed(KERNELS))}
    for ax, problem in zip(axes.flat, (1, 2, 3, 4)):
        _style_axes(ax)
        rows = sub[sub.problem == problem]
        for _, r in rows.iterrows():
            st = STYLE[r.kernel]
            yv = ypos[r.kernel]
            ax.plot([r.range95_low, r.range95_high], [yv, yv], color=st["color"], linewidth=3, solid_capstyle="butt")
            ax.plot(r.test_mean, yv, marker=st["marker"], color=st["color"], markersize=11, linestyle="none")
            ax.plot(r.original_seed42_test, yv, marker=st["marker"], markersize=11, linestyle="none",
                    markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.5)
        ax.axvline(rows.majority_mean.iloc[0], color="#555555", linestyle="--", linewidth=1.5)
        short = {1: "1: setosa vs. rest", 2: "2: virginica vs. rest",
                 3: "3: versicolor vs. rest", 4: "4: versicolor vs. virginica"}[problem]
        ax.set_title(f"Problem {short}", fontsize=15, loc="left")
        ax.set_xlim(0.25, 1.02)
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([STYLE[k]["label"] for k in ypos])
    for ax in axes[1]:
        ax.set_xlabel("Test accuracy", fontsize=15)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=10, label="Mean of 500 random splits"),
        Line2D([], [], color="black", linewidth=3, label="2.5th to 97.5th percentiles of single splits"),
        Line2D([], [], marker="o", linestyle="none", markersize=10, markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=1.5, label="Original split (seed 42)"),
        Line2D([], [], color="#555555", linestyle="--", linewidth=1.5, label="Majority-class rate"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Test accuracy at 80/20, 500 random splits", fontsize=16, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    fig.savefig(OUT_DIR / "repeated_split_by_kernel.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'repeated_split_by_kernel.png'}")


def plot_by_ratio(results, original):
    """Mean test accuracy over Problems 2 to 4 by training share, with the 2.5th to 97.5th percentiles
    of the three-problem mean on single splits."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor="white")
    _style_axes(ax)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    shares = np.array([round((1 - ts) * 100) for ts in TEST_SIZES])
    offsets = {k: off for k, off in zip(KERNELS, (-2.1, -0.7, 0.7, 2.1))}  # side by side at each share
    for kernel in KERNELS:
        st = STYLE[kernel]
        x = shares + offsets[kernel]
        # Per split, the mean of the three problems' test accuracies
        per_split = [np.mean([results[ts][(p, kernel)][:, 1] for p in (2, 3, 4)], axis=0) for ts in TEST_SIZES]
        means = np.array([v.mean() for v in per_split])
        lo = np.array([np.percentile(v, 2.5) for v in per_split])
        hi = np.array([np.percentile(v, 97.5) for v in per_split])
        orig = [np.mean([original[ts][(p, kernel)][1] for p in (2, 3, 4)]) for ts in TEST_SIZES]
        ax.vlines(x, lo, hi, color=st["color"], linewidth=3)
        ax.plot(x, means, color=st["color"], marker=st["marker"], markersize=10, linewidth=1.5, label=st["label"])
        ax.plot(x, orig, linestyle="none", marker=st["marker"], markersize=10,
                markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.5)
    ax.set_xticks(shares)
    ax.set_xticklabels([f"{s}%" for s in shares])
    ax.set_xlabel("Training share of the 150 flowers", fontsize=15)
    ax.set_ylabel("Test accuracy, mean of Problems 2 to 4", fontsize=15)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Test accuracy by training share, 500 random splits each", fontsize=15, loc="left")
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color="black", linewidth=3,
               label="2.5th to 97.5th percentiles of the\nthree-problem mean on single splits"),
        Line2D([], [], marker="o", linestyle="none", markersize=10, markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=1.5, label="Original split (seed 42)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.17, 1, 1])
    fig.savefig(OUT_DIR / "repeated_split_by_ratio.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'repeated_split_by_ratio.png'}")


if __name__ == "__main__":
    run()
