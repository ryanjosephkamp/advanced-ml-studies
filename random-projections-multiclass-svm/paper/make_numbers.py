# make_numbers.py
# Writes the numbers and tables used by main.tex from the committed result files, so that no number
# in the paper is typed by hand. The text asks for a number with \N{key}; the keys are described in
# resolve() below.
#
# Inputs (relative to the project folder):
#   results/original_run_20260213_172002/between_datasets_eq{3,4}_comparison.csv and
#       distortion_comparison_eq3.csv (and distortion_comparison_eq4.csv when present)
#   study/results/{accuracy_summary,paired_differences,distortion_summary,replication_summary,
#                  variance_check_many}.csv and raw_svm.csv.gz
#
# Outputs (in paper/): numbers.tex, table_data.tex, table_original.tex, table_replication.tex,
# table_comparisons.tex, table_k2.tex
#
# Usage (from the project folder or from paper/):
#     python paper/make_numbers.py

import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RES = ROOT / "study" / "results"
ORIG = ROOT / "results" / "original_run_20260213_172002"

DS = ["iris", "breast_cancer", "wine", "seeds", "glass"]
DS_LABEL = {"iris": "Iris", "breast_cancer": "Breast Cancer", "wine": "Wine", "seeds": "Seeds", "glass": "Glass"}
CLASSES = {"iris": 3, "breast_cancer": 2, "wine": 3, "seeds": 3, "glass": 6}   # UCI data set pages
FAM = {"gauss": "gaussian", "eq3": "rademacher", "eq4": "achlioptas3", "vs": "verysparse", "opc": "onepercol", "pca": "pca", "full": "full"}
FAM_LABEL = {"gaussian": "Gaussian", "rademacher": "Eq. 3", "achlioptas3": "Eq. 4", "verysparse": "Very sparse", "onepercol": "One per column", "pca": "PCA", "full": "All features"}
SHORT = {"gauss": "Gauss.", "eq3": "Eq. 3", "eq4": "Eq. 4", "vs": "V. sparse", "opc": "1/col.", "pca": "PCA"}
KERN = {"linear": "Linear", "poly2": "Poly 2", "poly3": "Poly 3", "rbf": "RBF"}
ZERO_TOL = 1e-6   # see the "excl0" count in resolve()


def rnd(x, places):
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(round(float(x), 8))).quantize(q, rounding=ROUND_HALF_UP))


def pct1(x):
    return rnd(100 * x, 1)


def sgn1(x):
    s = rnd(100 * x, 1)
    return s.replace("-", "$-$") if s.startswith("-") else s


def metric(ds):
    return "bal_acc" if ds == "glass" else "acc"


class Data:
    def __init__(self):
        self.acc = pd.read_csv(RES / "accuracy_summary.csv")
        self.pd = pd.read_csv(RES / "paired_differences.csv")
        self.dz = pd.read_csv(RES / "distortion_summary.csv")
        self.rep = pd.read_csv(RES / "replication_summary.csv")
        self.vm = pd.read_csv(RES / "variance_check_many.csv")
        self.raw = pd.read_csv(RES / "raw_svm.csv.gz")
        self.o3 = pd.read_csv(ORIG / "between_datasets_eq3_comparison.csv")
        self.o4 = pd.read_csv(ORIG / "between_datasets_eq4_comparison.csv")
        self.cv = self.acc[self.acc.model == "cv"]
        self.prim = self.pd[(self.pd.primary) & (self.pd.model == "cv")]


def acc_row(D, ds, k, fam, model="cv"):
    r = D.acc[(D.acc.dataset == ds) & (D.acc.k == int(k)) & (D.acc.family == fam) & (D.acc.model == model) & (D.acc.metric == metric(ds))]
    assert len(r) == 1, (ds, k, fam, model)
    return r.iloc[0]


def pd_row(D, ds, k, a, b):
    r = D.pd[(D.pd.dataset == ds) & (D.pd.k == int(k)) & (D.pd.a == a) & (D.pd.b == b) & (D.pd.model == "cv") & (D.pd.metric == metric(ds))]
    assert len(r) == 1, (ds, k, a, b)
    return r.iloc[0]


def pooled_perdraw(D, a, b, drop_iris=False):
    """Per draw: the mean over a data set's k of (a - b), then the mean over data sets."""
    r = D.raw[D.raw.family.isin([a, b])].copy()
    r["m"] = np.where(r.dataset == "glass", r.cv_bal_acc, r.cv_acc)
    piv = r.pivot_table(index=["dataset", "k", "draw"], columns="family", values="m").reset_index()
    piv["d"] = piv[a] - piv[b]
    per = piv.groupby(["dataset", "draw"]).d.mean().reset_index()
    if drop_iris:
        per = per[per.dataset != "iris"]
    return per.groupby("draw").d.mean()


def pooled(D, a, b, drop_iris=False):
    """Per draw, the mean over a data set's k of (a - b) test accuracy, then the mean over data sets; t-test over draws."""
    r = D.raw[D.raw.family.isin([a, b])].copy()
    r["m"] = np.where(r.dataset == "glass", r.cv_bal_acc, r.cv_acc)
    piv = r.pivot_table(index=["dataset", "k", "draw"], columns="family", values="m").reset_index()
    piv["d"] = piv[a] - piv[b]
    per = piv.groupby(["dataset", "draw"]).d.mean().reset_index()
    if drop_iris:
        per = per[per.dataset != "iris"]
    perdraw = per.groupby("draw").d.mean()
    t = stats.ttest_1samp(perdraw, 0.0)
    return perdraw.mean(), t.pvalue


def resolve(key, D):
    """Return the text for one key. Key forms, fields separated by colons:

    o:<ds>:<eq3|eq4>:<linear|poly2|poly3|rbf|best>   original run, test accuracy (%)
    o:<ds>:delta                                      best Eq. 4 minus best Eq. 3 (points)
    a:<ds>:<k>:<family>[:lo|hi]                       study accuracy, cv model, % (balanced for Glass)
    af:<ds>[:lo|hi]                                   all features
    maj:<ds>                                          majority-class test share, %
    pd:<ds>:<k>:<a>:<b>[:lo|hi|p]                     paired difference, points (p: Holm-adjusted)
    rep:<ds>:<quantity>[:field]                       replication summary
    dz:<ds>:<k>:<family>:<field>                      distortion summary
    vm:<stat>                                         20,000-draw variance check
    c:<name>                                          counts over the primary comparisons
    pool:<a>:<b>[:noiris][:p]                         pooled per-draw difference
    z:<name>                                          other derived values
    """
    p = key.split(":")
    k0 = p[0]
    if k0 == "o":
        ds = p[1]
        def orig(dist, kern):
            d = D.o3 if dist == "eq3" else D.o4
            r = d[(d.dataset_key == ds) & (d.kernel_tag == kern)]
            return float(r.accuracy.iloc[0])
        if p[2] == "delta":
            return sgn1(max(orig("eq4", t) for t in KERN) - max(orig("eq3", t) for t in KERN))
        if p[3] == "best":
            return pct1(max(orig(p[2], t) for t in KERN))
        return pct1(orig(p[2], p[3]))
    if k0 == "a":
        r = acc_row(D, p[1], p[2], FAM.get(p[3], p[3]))
        f = {"mean": "mean", "lo": "ci_low", "hi": "ci_high"}[p[4] if len(p) > 4 else "mean"]
        return pct1(r[f])
    if k0 == "af":
        ds = p[1]
        r = D.acc[(D.acc.dataset == ds) & (D.acc.family == "full") & (D.acc.model == "cv") & (D.acc.metric == metric(ds))].iloc[0]
        f = {"mean": "mean", "lo": "ci_low", "hi": "ci_high"}[p[2] if len(p) > 2 else "mean"]
        return pct1(r[f])
    if k0 == "maj":
        r = D.acc[(D.acc.dataset == p[1]) & (D.acc.model == "majority") & (D.acc.metric == "acc")].iloc[0]
        return pct1(r["mean"])
    if k0 == "pd":
        r = pd_row(D, p[1], p[2], FAM.get(p[3], p[3]), FAM.get(p[4], p[4]))
        f = p[5] if len(p) > 5 else "mean"
        if f == "p":
            return rnd(r.p_holm_primary, 3)
        return sgn1(r[{"mean": "mean_diff", "lo": "ci_low", "hi": "ci_high"}[f]])
    if k0 == "pda":
        return resolve("pd:" + key[4:], D).replace("$-$", "")
    if k0 == "mag":
        # magnitude of any signed key, for sentences that already give the direction ("trails by")
        return resolve(key[4:], D).replace("$-$", "")
    if k0 == "r0":
        # any key rounded to a whole number
        return rnd(float(resolve(key[3:], D).replace("$-$", "-").replace(",", "")), 0)
    if k0 == "rep3":
        # original pipeline, per seed: data sets on which the best Eq. 4 kernel beats the best Eq. 3 kernel
        rr = pd.read_csv(RES / "raw_replication.csv.gz")
        best = rr.groupby(["dataset", "seed", "distribution"]).acc.max().unstack("distribution")
        ahead = ((best["eq4"] - best["eq3"]) > 1e-9).groupby("seed").sum()
        return {"ge3": str(int((ahead >= 3).sum())), "seeds": str(int(len(ahead)))}[p[1]]
    if k0 == "vsr":
        # very sparse over Gaussian, measured variance of q, mean over k (20,000-draw check)
        v = D.vm[D.vm.dataset == p[1]]
        g = v[v.family == "gaussian"].set_index("k").mean_emp_var
        s = v[v.family == "verysparse"].set_index("k").mean_emp_var
        return rnd((s / g).dropna().mean(), 2)
    if k0 == "wxh":
        # Wilcoxon signed-rank test per primary comparison, Holm-adjusted over all of them
        return rnd(wilcoxon_holm(D)[(p[1], int(p[2]), FAM[p[3]], FAM[p[4]])], 2)
    if k0 == "rep":
        r = D.rep[(D.rep.dataset == p[1]) & (D.rep.quantity == p[2])].iloc[0]
        f = p[3] if len(p) > 3 else "mean"
        v = r[f]
        if f in ("draws_lower", "draws_higher", "draws_equal_excl_42", "four_way_ties"):
            return str(int(v))
        return sgn1(v) if p[2].startswith("best_eq4_minus") else pct1(v)
    if k0 == "dz":
        r = D.dz[(D.dz.dataset == p[1]) & (D.dz.k == int(p[2])) & (D.dz.family == FAM.get(p[3], p[3]))].iloc[0]
        f = p[4]
        v = r[f]
        if f.startswith(("share", "bound", "gauss_exact")):
            return pct1(v)
        if f in ("dropped_features_mean",):
            return rnd(v, 2)
        return rnd(v, 2)
    if k0 == "vm":
        v = D.vm
        k3 = v[v.k >= 3]
        return {"ratio_min": rnd(v.emp_over_pred.min(), 3), "ratio_max": rnd(v.emp_over_pred.max(), 3), "ratio_median": rnd(v.emp_over_pred.median(), 3),
                "ratio_k3_min": rnd(k3.emp_over_pred.min(), 3), "ratio_k3_max": rnd(k3.emp_over_pred.max(), 3),
                "max_dev_pct": rnd(100 * (v.emp_over_pred - 1).abs().max(), 1), "k3_dev_pct": rnd(100 * (k3.emp_over_pred - 1).abs().max(), 1),
                "mean_q_min": rnd(v.mean_q.min(), 3), "mean_q_max": rnd(v.mean_q.max(), 3), "cells": str(len(v)), "draws": f"{int(v.draws.iloc[0]):,}"}[p[1]]
    if k0 == "vm100":
        v = pd.read_csv(RES / "variance_check.csv")
        return {"ratio_min": rnd(v.emp_over_pred.min(), 2), "ratio_max": rnd(v.emp_over_pred.max(), 2)}[p[1]]
    if k0 == "c":
        P = D.prim
        def cnt(a, b):
            return P[(P.a == a) & (P.b == b)]
        pairs = {"eq3": ("rademacher", "gaussian"), "eq4": ("achlioptas3", "gaussian"), "vs": ("verysparse", "gaussian"), "opc": ("onepercol", "gaussian"), "e43": ("achlioptas3", "rademacher")}
        n = p[1]
        if n == "n_primary":
            return str(len(P))
        if n == "n_holm":
            return str(int((P.p_holm_primary < 0.05).sum()))
        name, what = n.split("_", 1)
        c = cnt(*pairs[name])
        if what == "cells":
            return str(len(c))
        if what == "equiv":
            return str(int(c.equivalent_2pts.sum()))
        if what == "excl0":
            # An interval end within 1e-6 of 0 counts as 0. A difference of zero test rows can sit a few
            # 1e-8 from 0 when study/analyze.py rebuilds the summaries from the six-decimal scores in
            # raw_svm.csv.gz. Every other end is at least 0.00017 from 0 (accuracy as a fraction; the
            # closest is Breast Cancer, k = 12, very sparse minus Gaussian).
            return str(int(((c.ci_low > ZERO_TOL) | (c.ci_high < -ZERO_TOL)).sum()))
        if what == "holm":
            return str(int((c.p_holm_primary < 0.05).sum()))
        if what == "neg":
            return str(int((c.mean_diff < 0).sum()))
        if what == "min":
            return sgn1(c.mean_diff.min())
        if what == "max":
            return sgn1(c.mean_diff.max())
    if k0 == "poolci":
        # percentile bootstrap interval of the pooled per-draw difference (20,000 resamples, seed 7)
        a, b = FAM[p[1]], FAM[p[2]]
        v = pooled_perdraw(D, a, b, "noiris" in p).to_numpy()
        rng = np.random.default_rng(7)
        m = v[rng.integers(0, len(v), (20000, len(v)))].mean(axis=1)
        return sgn2(np.percentile(m, 2.5 if p[-1] == "lo" else 97.5))
    if k0 == "pool":
        a, b = FAM[p[1]], FAM[p[2]]
        drop = "noiris" in p
        m, pv = pooled(D, a, b, drop)
        return rnd(pv, 3 if pv >= 0.001 else 4) if p[-1] == "p" else sgn2(m)
    if k0 == "z":
        n = p[1]
        if n == "equiv_total":
            return str(int(D.prim.equivalent_2pts.sum()))
        if n == "n_vs_gauss":
            # primary comparisons against the Gaussian: Eq. 3, Eq. 4, very sparse, and one per column
            return str(int((D.prim.b == "gaussian").sum()))
        if n == "equiv_vs_gauss":
            # of those, the comparisons whose 95% interval lies inside +-2 points
            return str(int(D.prim[D.prim.b == "gaussian"].equivalent_2pts.sum()))
        if n == "max_ci_half":
            return rnd(100 * ((D.prim.ci_high - D.prim.ci_low) / 2).max(), 1)
        if n == "max_ci_side":
            # percentile intervals are asymmetric: the largest distance from a mean to either end of its interval
            return rnd(100 * np.maximum(D.prim.mean_diff - D.prim.ci_low, D.prim.ci_high - D.prim.mean_diff).max(), 1)
        if n == "max_ci_side_nonsig":
            # the same, over the primary intervals that do not survive Holm's correction
            c = D.prim[D.prim.p_holm_primary >= 0.05]
            return rnd(100 * np.maximum(c.mean_diff - c.ci_low, c.ci_high - c.mean_diff).max(), 1)
        if n == "theory_both_iris":
            # an Eq. 4 matrix zeroes both petal rows of Iris at k = 2: each of the four entries is zero with probability 2/3
            return pct1((2 / 3) ** 4)
        if n == "n_fits_m":
            # 133 fits per classification row: 3 inner folds x (5 + 9 + 9 + 20) grid points, plus 4 refits
            return rnd(133 * len(D.raw) / 1e6, 1)
        if n == "test_optimism":
            r = D.raw
            return rnd(100 * (r[["linear_acc", "poly2_acc", "poly3_acc", "rbf_acc"]].max(axis=1).mean() - r.cv_acc.mean()), 1)
        if n == "best_test_acc":
            r = D.raw
            return rnd(100 * r[["linear_acc", "poly2_acc", "poly3_acc", "rbf_acc"]].max(axis=1).mean(), 1)
        if n == "cv_test_acc":
            return rnd(100 * D.raw.cv_acc.mean(), 1)
        if n == "n_rows":
            return f"{len(D.raw):,}"
        if n == "n_test":
            ds = p[2]
            return str(int(D.raw[D.raw.dataset == ds].n_test.iloc[0]))
        if n == "n_total":
            ds = p[2]
            r = D.raw[D.raw.dataset == ds].iloc[0]
            return str(int(r.n_train + r.n_test))
        if n == "d":
            ds = p[2]
            r = D.acc[(D.acc.dataset == ds) & (D.acc.family == "full")].iloc[0]
            return str(int(r.k))
        if n == "capped":
            return str(int(D.raw.capped_fits.sum()))
        if n == "eq4_iris_zero_share":
            r = D.dz[(D.dz.dataset == "iris") & (D.dz.k == 2) & (D.dz.family == "achlioptas3")].iloc[0]
            return pct1(r.share_draws_zero_column)
        if n == "theory_zero_iris":
            return pct1(1 - (1 - (2 / 3) ** 4) ** 2)
        if n == "k0":
            ds = p[2]
            r = D.raw[D.raw.dataset == ds].iloc[0]
            return f"{24 * np.log(r.n_train + r.n_test):.0f}"
        if n in ("eps_min", "eps_max"):
            # The committed distortion_comparison_eq3.csv holds both distributions; the current
            # scripts/between_datasets_analysis.py writes one file per distribution. Read both when present,
            # and stop unless Eq. 3 and Eq. 4 have a finite epsilon for every data set, since the text gives
            # the range over both (a blank epsilon is no value, not one to skip; with neither file, every pair is missing).
            files = [ORIG / f"distortion_comparison_{e}.csv" for e in ("eq3", "eq4")]
            e = pd.concat([pd.read_csv(f) for f in files if f.exists()] or [pd.DataFrame(columns=["dataset_key", "distribution", "epsilon"])], ignore_index=True)
            e["epsilon"] = pd.to_numeric(e.epsilon, errors="coerce")
            e = e[np.isfinite(e.epsilon)]
            missing = {(ds, q) for ds in DS for q in ("EQ3", "EQ4")} - set(zip(e.dataset_key, e.distribution))
            if missing:
                raise ValueError(f"distortion_comparison_eq*.csv has no finite epsilon for {sorted(missing)}")
            return rnd(e.epsilon.min() if n == "eps_min" else e.epsilon.max(), 3)
        if n == "opc_holm_min":
            c = D.prim[(D.prim.a == "onepercol") & (D.prim.b == "gaussian") & (D.prim.p_holm_primary < 0.05)]
            return rnd(100 * c.mean_diff.abs().min(), 1)
        if n == "opc_holm_max":
            c = D.prim[(D.prim.a == "onepercol") & (D.prim.b == "gaussian") & (D.prim.p_holm_primary < 0.05)]
            return rnd(100 * c.mean_diff.abs().max(), 1)
        if n in ("opc_zero_min", "opc_zero_max"):
            c = D.dz[(D.dz.k == 2) & (D.dz.family == "onepercol")].share_draws_zero_column
            return pct1(c.min() if n == "opc_zero_min" else c.max())
        if n == "kern_spread_max":
            a = D.acc[(D.acc.family == "gaussian") & (D.acc.metric == "acc") & (D.acc.model.isin(list(KERN))) & (D.acc.dataset != "glass")]
            g = a.groupby(["dataset", "k"])["mean"].agg(lambda s: s.max() - s.min())
            import math
            return f"{math.ceil(round(100 * g.max(), 6) * 10) / 10:.1f}"     # an upper bound: "within"
        if n in ("glass_trail_min", "glass_trail_max"):
            a = D.acc[(D.acc.family == "gaussian") & (D.acc.metric == "acc") & (D.acc.dataset == "glass") & (D.acc.model.isin(list(KERN)))]
            piv = a.pivot_table(index="k", columns="model", values="mean")
            tr = piv.drop(columns="linear").max(axis=1) - piv["linear"]
            return rnd(100 * (tr.min() if n == "glass_trail_min" else tr.max()), 1)
        if n == "pick":
            ds, kern = p[2], p[3]
            r = D.raw[D.raw.dataset == ds]
            return rnd(100 * (r.cv_kernel == kern).mean(), 1)
        if n.startswith("iris_k2_"):
            r = D.raw[(D.raw.dataset == "iris") & (D.raw.k == 2) & (D.raw.family.isin(["achlioptas3", "gaussian"]))]
            piv = r.pivot_table(index="draw", columns="family", values="cv_acc")
            zc = D.raw[(D.raw.dataset == "iris") & (D.raw.k == 2) & (D.raw.family == "achlioptas3")].set_index("draw").zero_columns
            diff = (piv["achlioptas3"] - piv["gaussian"]) * 100
            z = zc.loc[diff.index] > 0
            both = zc.loc[diff.index] >= 2
            return {"iris_k2_zero_n": str(int(z.sum())), "iris_k2_zero_mean": sgn1(diff[z].mean() / 100), "iris_k2_full_n": str(int((~z).sum())),
                    "iris_k2_full_mean": sgn1(diff[~z].mean() / 100), "iris_k2_both_n": str(int(both.sum())), "iris_k2_both_mean": sgn1(diff[both].mean() / 100),
                    "iris_k2_full_p": rnd(stats.ttest_1samp(diff[~z], 0).pvalue, 2)}[n]
        if n == "equiv_near":
            e = D.prim[D.prim.equivalent_2pts]
            return str(int((np.maximum(e.ci_low.abs(), e.ci_high.abs()) >= 0.018).sum()))
        if n == "svs":
            d = int(resolve(f"z:d:{p[2]}", D))
            return f"{np.sqrt(d):.3g}"
        if n == "row_pts":
            return rnd(100 / int(resolve(f"z:n_test:{p[2]}", D)), 1)
    raise KeyError(key)


_WX = {}
_DENOM = {}


def score_denominator(D, ds, draw):
    """A whole number M such that every test score of this draw is a whole number over M: accuracy is a
    count over the test rows, and Glass's balanced accuracy is the mean over its classes of a count over
    that class's test rows (the same stratified 70/30 split as study/run_svm.py)."""
    if ds != "glass":
        return int(D.raw[D.raw.dataset == ds].n_test.iloc[0])
    if draw not in _DENOM:
        from math import lcm
        from sklearn.model_selection import train_test_split
        y = pd.read_csv(ROOT / "data" / "glass" / "glass_cached.csv")["target"].to_numpy(int)
        _, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=int(draw), stratify=y)
        counts = [int(c) for c in np.bincount(y[te]) if c > 0]
        _DENOM[draw] = len(counts) * lcm(*counts)
    return _DENOM[draw]


def wilcoxon_holm(D):
    if not _WX:
        r = D.raw.copy()
        r["m"] = np.where(r.dataset == "glass", r.cv_bal_acc, r.cv_acc)
        piv = r.pivot_table(index=["dataset", "k", "draw"], columns="family", values="m")
        rows = list(D.prim[["dataset", "k", "a", "b"]].itertuples(index=False, name=None))
        raw_p = []
        for ds, k, a, b in rows:
            d = (piv.loc[(ds, k)][a] - piv.loc[(ds, k)][b]).dropna()
            # The scores are stored to six decimals, so two equal differences can differ in the last digit
            # and get different ranks. Snap each difference to its exact value, a whole number over M.
            M = np.array([score_denominator(D, ds, draw) for draw in d.index], float)
            d = np.round(d.to_numpy() * M) / M
            raw_p.append(stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0)
        order = np.argsort(raw_p)
        m, run = len(raw_p), 0.0
        adj = [0.0] * m
        for i, j in enumerate(order):
            run = max(run, min(1.0, (m - i) * raw_p[j]))
            adj[j] = run
        _WX.update({(ds, int(k), a, b): v for (ds, k, a, b), v in zip(rows, adj)})
    return _WX


def sgn2(x):
    s = rnd(x * 100, 2)
    return s.replace("-", "$-$") if s.startswith("-") else s


KEY = re.compile(r"\\N\{([^}]*)\}")


def compute():
    D = Data()
    keys = []
    for f in sorted((PAPER / "sections").glob("*.tex")):
        keys += KEY.findall(f.read_text())
    N = {}
    for kname in dict.fromkeys(keys):
        N[kname] = resolve(kname, D)
    return N, D


def write_numbers(N):
    lines = ["% Generated by make_numbers.py from the committed results. Do not edit.",
             r"\providecommand{\N}[1]{\csname num@#1\endcsname}"]
    for k, v in N.items():
        lines.append(r"\expandafter\gdef\csname num@%s\endcsname{%s}" % (k, v))
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")


def write_tables(D):
    # Data sets
    out = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Data set & Rows & Features $d$ & Classes & Majority test share (\%) \\", r"\midrule"]
    for ds in DS:
        out.append(f"{DS_LABEL[ds]} & {resolve(f'z:n_total:{ds}', D)} & {resolve(f'z:d:{ds}', D)} & {CLASSES[ds]} & {resolve(f'maj:{ds}', D)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_data.tex").write_text("\n".join(out) + "\n")

    # Original run: test accuracy at k = 2
    out = [r"\begin{tabular}{@{}l" + "c" * 8 + r"@{}}", r"\toprule", r" & \multicolumn{4}{c}{Eq. 3} & \multicolumn{4}{c}{Eq. 4} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r"Data set & Lin. & P2 & P3 & RBF & Lin. & P2 & P3 & RBF \\", r"\midrule"]
    for ds in DS:
        cells = [resolve(f"o:{ds}:{e}:{k}", D) for e in ("eq3", "eq4") for k in ("linear", "poly2", "poly3", "rbf")]
        out.append(f"{DS_LABEL[ds]} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_original.tex").write_text("\n".join(out) + "\n")

    # Replication
    out = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
           r"Data set & Eq. 3, best kernel & Eq. 4, best kernel & Report draw (Eq. 3 / Eq. 4) & Eq. 4 $-$ Eq. 3 (points) \\", r"\midrule"]
    for ds in DS:
        e3 = f"{resolve(f'rep:{ds}:best_kernel_acc_eq3', D)} [{resolve(f'rep:{ds}:best_kernel_acc_eq3:ci_low', D)}, {resolve(f'rep:{ds}:best_kernel_acc_eq3:ci_high', D)}]"
        e4 = f"{resolve(f'rep:{ds}:best_kernel_acc_eq4', D)} [{resolve(f'rep:{ds}:best_kernel_acc_eq4:ci_low', D)}, {resolve(f'rep:{ds}:best_kernel_acc_eq4:ci_high', D)}]"
        rep = f"{resolve(f'rep:{ds}:best_kernel_acc_eq3:report_seed42', D)} / {resolve(f'rep:{ds}:best_kernel_acc_eq4:report_seed42', D)}"
        dd = f"{resolve(f'rep:{ds}:best_eq4_minus_best_eq3', D)} [{resolve(f'rep:{ds}:best_eq4_minus_best_eq3:ci_low', D)}, {resolve(f'rep:{ds}:best_eq4_minus_best_eq3:ci_high', D)}]"
        out.append(f"{DS_LABEL[ds]} & {e3} & {e4} & {rep} & {dd} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_replication.tex").write_text("\n".join(out) + "\n")

    # Primary comparisons summary
    names = [("eq3", "Eq. 3 $-$ Gaussian"), ("eq4", "Eq. 4 $-$ Gaussian"), ("vs", "Very sparse $-$ Gaussian"), ("opc", "One per column $-$ Gaussian"), ("e43", "Eq. 4 $-$ Eq. 3")]
    out = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule", r"Comparison & Cells & Mean difference (points) & Within $\pm 2$ & Interval excludes 0 & After Holm \\", r"\midrule"]
    for key, lab in names:
        g = lambda w: resolve(f"c:{key}_{w}", D)
        out.append(f"{lab} & {g('cells')} & {g('min')} to {g('max')} & {g('equiv')} & {g('excl0')} & {g('holm')} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_comparisons.tex").write_text("\n".join(out) + "\n")

    # Accuracy at k = 2
    fams = ["gauss", "eq3", "eq4", "vs", "opc", "pca"]
    out = [r"\begin{tabular}{@{}l" + "c" * 7 + r"@{}}", r"\toprule", "Data set & " + " & ".join(SHORT[f] for f in fams) + r" & All feat. \\", r"\midrule"]
    for ds in DS:
        cells = []
        for f in fams:
            if ds == "glass" and f == "vs":
                cells.append("--")
            else:
                cells.append(resolve(f"a:{ds}:2:{f}", D))
        out.append(f"{DS_LABEL[ds]} & " + " & ".join(cells) + f" & {resolve(f'af:{ds}', D)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_k2.tex").write_text("\n".join(out) + "\n")


def main():
    N, D = compute()
    write_numbers(N)
    write_tables(D)
    print(f"wrote {len(N)} numbers and five tables to {PAPER}")


if __name__ == "__main__":
    main()
