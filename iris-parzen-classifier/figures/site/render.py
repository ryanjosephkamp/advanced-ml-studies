"""The course book's copies of this study's two repeated-split figures, for its day and night pages.

Each figure is drawn by repeated_splits.py's own function: plot_by_kernel from the saved summary
(repeated_splits/repeated_split_results.csv), and plot_by_ratio from the per-split accuracies, which the script keeps
in memory, so they are recomputed here with its own functions and checked against the saved summary first. The
book's inks replace the study's colors: the Gaussian kernel, the one the chapter follows, takes the chapter's ink.
The drawing and the checks are in tools/site_figures.py.

Usage, from the repository's root or this project's folder:
    python iris-parzen-classifier/figures/site/render.py [--check]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # 32,500 tiny fits run fastest on one thread
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "tools"))
import repeated_splits as S  # noqa: E402
import site_figures  # noqa: E402

SUMMARY = "repeated_splits/repeated_split_results.csv"

# The study's colors (repeated_splits.py's STYLE and plot functions) and the book's ink for each
INKS = {
    S.STYLE["Gaussian"]["color"]: "violet",          # the chapter's ink
    S.STYLE["Dot Product"]["color"]: "ink2",
    S.STYLE["Polynomial (d=2)"]["color"]: "gold",
    S.STYLE["Polynomial (d=3)"]["color"]: "crimson",
    "black": "ink",                                   # open markers' edges, legend keys, and text
    "#555555": "ink",                                 # the majority-class rate (dashed)
    "white": "paper",                                 # open markers' fill
}
# Black marks the original split's open markers and #555555 the dashed majority-class line: different marks,
# so one ink serves both.
MERGE_OK = [("black", "#555555")]


def per_split():
    """repeated_splits.run()'s per-split accuracies, computed the same way, checked against the saved summary."""
    from sklearn import datasets
    X, y = datasets.load_iris(return_X_y=True)
    results = {ts: {} for ts in S.TEST_SIZES}
    for ts in S.TEST_SIZES:
        per_seed = [S.evaluate_split(X, y, ts, seed) for seed in S.SEEDS]
        results[ts] = {cfg: np.array([s[cfg] for s in per_seed]) for cfg in S.CONFIGS}
    original = {ts: S.evaluate_split(X, y, ts, S.ORIGINAL_SEED) for ts in S.TEST_SIZES}
    saved = pd.read_csv(PROJECT / SUMMARY)
    for ts in S.TEST_SIZES:
        for (p, k) in S.CONFIGS:
            r = saved[(saved.problem == p) & (saved.kernel == k) & (saved.split == S.split_label(ts))].iloc[0]
            got = S.summarize(results[ts][(p, k)][:, 1])
            for col in ("test_mean", "range95_low", "range95_high"):
                assert abs(got[col] - r[col]) < 1e-6, (ts, p, k, col)
            assert abs(original[ts][(p, k)][1] - r.original_seed42_test) < 1e-6
    return results, original


def figures():
    df = pd.read_csv(PROJECT / SUMMARY)
    results, original = per_split()
    common = dict(inks=INKS, merge_ok=MERGE_OK)
    return [
        dict(name="repeated_split_by_kernel", committed="repeated_splits/repeated_split_by_kernel.png",
             source="repeated_splits.py: plot_by_kernel", data=[SUMMARY],
             draw=lambda: S.plot_by_kernel(df), **common),
        dict(name="repeated_split_by_ratio", committed="repeated_splits/repeated_split_by_ratio.png",
             source="repeated_splits.py: plot_by_ratio (per-split accuracies recomputed with evaluate_split, "
                    "checked against the saved summary)", data=[SUMMARY],
             draw=lambda: S.plot_by_ratio(results, original), **common),
    ]


if __name__ == "__main__":
    ok = site_figures.render(PROJECT, figures(), check="--check" in sys.argv)
    sys.exit(0 if ok else 1)
