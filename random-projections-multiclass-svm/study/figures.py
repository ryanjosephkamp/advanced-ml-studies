"""Figures for the random-projection study, drawn from the summary tables in study/results/.

PNG at 200 dpi on a white background, one column of panels so they read at phone width. Figures 1 and
2 also come in a wide version for the two-column paper (*_wide.png, 300 dpi): the five panels in one row.
Colors follow the projection family in every figure (validated as colorblind-safe for adjacent
series); markers repeat the identity, and every interval shown is a 95% bootstrap interval over
100 draws.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, FAMILY_LABELS, FIG_DIR, LABELS, RESULTS_DIR, K_GRID  # noqa: E402

COLORS = {"gaussian": "#2a78d6", "rademacher": "#eb6834", "achlioptas3": "#1baf7a",
          "verysparse": "#eda100", "onepercol": "#e87ba4"}
MARKERS = {"gaussian": "o", "rademacher": "s", "achlioptas3": "^", "verysparse": "D", "onepercol": "v"}
KERNEL_COLORS = {"linear": "#008300", "poly2": "#6250d6", "poly3": "#e34948", "rbf": "#eda100"}
KERNEL_LABELS = {"linear": "Linear", "poly2": "Polynomial, degree 2", "poly3": "Polynomial, degree 3",
                 "rbf": "RBF", "cv": "Chosen by cross-validation"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMS = list(COLORS)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "lines.markersize": 7, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def metric_for(ds):
    return "bal_acc" if ds == "glass" else "acc"


def metric_label(ds):
    return "Balanced accuracy (%)" if ds == "glass" else "Accuracy (%)"


def k_axis(ax, ks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlim(min(ks) / 1.3, max(ks) * 1.3)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def family_legend(fig, extra=(), ncol=2, y=1.0):
    handles = [Line2D([], [], color=COLORS[f], marker=MARKERS[f], label=FAMILY_LABELS[f]) for f in FAMS]
    handles += list(extra)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


def fig_accuracy_vs_k(acc):
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(6.2, 3.3 * len(DATASETS)), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        m = metric_for(ds)
        a = acc[(acc.dataset == ds) & (acc.model == "cv") & (acc.metric == m)]
        ks = sorted(a[a.family != "full"]["k"].unique())
        full = a[a.family == "full"].iloc[0]
        ax.axhspan(100 * full.ci_low, 100 * full.ci_high, color="#d7d6d1", zorder=0)
        ax.axhline(100 * full["mean"], color=INK2, lw=1.5, zorder=1)
        maj = acc[(acc.dataset == ds) & (acc.model == "majority") & (acc.family == "full")]["mean"].iloc[0]
        if m == "acc":
            ax.axhline(100 * maj, color=INK2, lw=1.2, ls=":", zorder=1)
        else:
            ax.axhline(100 / N_CLASSES[ds], color=INK2, lw=1.2, ls=":", zorder=1)  # chance balanced accuracy
        p = a[a.family == "pca"].sort_values("k")
        ax.plot(p.k, 100 * p["mean"], color=INK, ls="--", lw=1.6, marker="x", ms=7, zorder=3)
        for f in FAMS:
            s = a[a.family == f].sort_values("k")
            if s.empty:
                continue
            ax.fill_between(s.k, 100 * s.ci_low, 100 * s.ci_high, color=COLORS[f], alpha=0.15, lw=0, zorder=2)
            ax.plot(s.k, 100 * s["mean"], color=COLORS[f], marker=MARKERS[f], zorder=4)
        k_axis(ax, ks)
        d = int(full.k)
        ax.set_title(f"{LABELS[ds]} (d = {d})")
        ax.set_ylabel(metric_label(ds))
    axes[-1].set_xlabel("Target size k")
    extra = [Line2D([], [], color=INK, ls="--", marker="x", label="PCA to k (reference)"),
             Line2D([], [], color=INK2, lw=6, alpha=0.4, label="All features (band: 95% interval)"),
             Line2D([], [], color=INK2, ls=":", label="Majority class (chance for Glass)")]
    family_legend(fig, extra)
    save(fig, "fig1_accuracy_vs_k.png")


def fig_paired(pdf):
    comps = ["rademacher", "achlioptas3", "verysparse", "onepercol"]
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(6.2, 3.1 * len(DATASETS)), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        m = metric_for(ds)
        x = pdf[(pdf.dataset == ds) & (pdf.model == "cv") & (pdf.metric == m) & (pdf.b == "gaussian")]
        ks = sorted(x.k.unique())
        ax.axhspan(-2, 2, color="#ecebe7", zorder=0)
        ax.axhline(0, color=INK2, lw=1.2, zorder=1)
        offs = {c: o for c, o in zip(comps, [-0.09, -0.03, 0.03, 0.09])}
        for c in comps:
            s = x[x.a == c].sort_values("k")
            if s.empty:
                continue
            xs = s.k * 2 ** offs[c]
            ax.errorbar(xs, 100 * s.mean_diff, yerr=[100 * (s.mean_diff - s.ci_low), 100 * (s.ci_high - s.mean_diff)],
                        color=COLORS[c], marker=MARKERS[c], lw=1.6, capsize=3, zorder=3)
        k_axis(ax, ks)
        ax.set_title(LABELS[ds] + (" (balanced accuracy)" if ds == "glass" else ""))
        ax.set_ylabel("Difference vs.\nGaussian (points)")
    axes[-1].set_xlabel("Target size k")
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], label=FAMILY_LABELS[c] + " − Gaussian") for c in comps]
    handles.append(Line2D([], [], color="#d9d8d3", lw=8, label="±2 points (equivalence margin)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    save(fig, "fig2_paired_difference_vs_gaussian.png")


def fig_variance(vc):
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(6.2, 3.0 * len(DATASETS)), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        v = vc[vc.dataset == ds]
        ks = sorted(v.k.unique())
        for f in FAMS:
            s = v[v.family == f].sort_values("k")
            if s.empty:
                continue
            ax.plot(s.k, s.k * s.mean_pred_var, color=COLORS[f], ls="--", lw=1.5, zorder=2)
            ax.plot(s.k, s.k * s.mean_emp_var, color=COLORS[f], marker=MARKERS[f], ls="none", zorder=3)
        k_axis(ax, ks)
        ax.set_ylim(0, None)
        ax.set_title(f"{LABELS[ds]} (mean κ = {v.kappa_mean.iloc[0]:.2f})")
        ax.set_ylabel("k × Var(q)")
    axes[-1].set_xlabel("Target size k")
    extra = [Line2D([], [], color=INK2, ls="--", label="Theory: 2 + (s − 3)κ (Gaussian and Eq. 4: 2)"),
             Line2D([], [], color=INK2, marker="o", ls="none", label="Measured over 100 draws")]
    family_legend(fig, extra)
    save(fig, "fig3_distortion_variance_vs_theory.png")


def fig_tails(ds_sum, ds="breast_cancer"):
    d = ds_sum[ds_sum.dataset == ds]
    ks = sorted(d.k.unique())
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.0), constrained_layout=True)
    for ax, side, lab in [(axes[0], "low", f"{LABELS[ds]}: q ≤ 0.5 (distance shrinks 29% or more)"),
                          (axes[1], "high", f"{LABELS[ds]}: q ≥ 1.5 (distance grows 22% or more)")]:
        g = d[d.family == "gaussian"].sort_values("k")
        ax.plot(g.k, 100 * g[f"gauss_exact_{side}_05"], color=INK, lw=1.5, label="Gaussian, exact (χ²ₖ/k)")
        ax.plot(g.k, 100 * np.minimum(1, g[f"bound_{side}_05"]), color=INK2, ls="--", lw=1.5,
                label="Gaussian per-pair bound (Dasgupta–Gupta)")
        for f in FAMS:
            s = d[d.family == f].sort_values("k")
            if s.empty:
                continue
            ax.plot(s.k, 100 * s[f"share_{side}_05_mean"], color=COLORS[f], marker=MARKERS[f], lw=1.6)
        k_axis(ax, ks)
        ax.set_title(lab, fontsize=13)
        ax.set_ylabel("Pairs (%)")
    axes[-1].set_xlabel("Target size k")
    extra = [Line2D([], [], color=INK, lw=1.5, label="Gaussian, exact (χ²ₖ/k)"),
             Line2D([], [], color=INK2, ls="--", lw=1.5, label="Gaussian per-pair bound\n(Dasgupta–Gupta)")]
    family_legend(fig, extra)
    save(fig, "fig4_distortion_tails_vs_bound.png")


def fig_structure(acc, ds_sum):
    from scipy.stats import spearmanr
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(6.2, 2.9 * len(DATASETS)), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        m = metric_for(ds)
        a = acc[(acc.dataset == ds) & (acc.model == "cv") & (acc.metric == m) & acc.family.isin(FAMS)]
        j = a.merge(ds_sum[ds_sum.dataset == ds][["k", "family", "knn5_agreement_mean", "knn5_agreement_original_mean"]],
                    on=["k", "family"])
        for f in FAMS:
            s = j[j.family == f]
            ax.scatter(100 * s.knn5_agreement_mean, 100 * s["mean"], s=46, color=COLORS[f], marker=MARKERS[f],
                       edgecolor="white", linewidth=0.8, zorder=3)
        ax.axvline(100 * j.knn5_agreement_original_mean.iloc[0], color=INK2, ls=":", lw=1.2)
        rho = spearmanr(j.knn5_agreement_mean, j["mean"]).statistic
        ax.set_title(f"{LABELS[ds]} (rank correlation {rho:.2f})")
        ax.set_ylabel(metric_label(ds))
    axes[-1].set_xlabel("5-nearest-neighbor label agreement after projection (%)")
    extra = [Line2D([], [], color=INK2, ls=":", label="Agreement with all features")]
    family_legend(fig, extra)
    save(fig, "fig5_structure_vs_accuracy.png")


def fig_degeneracy(ds_sum):
    fams = ["achlioptas3", "verysparse", "onepercol"]
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(6.2, 2.8 * len(DATASETS)), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        d = ds_sum[ds_sum.dataset == ds]
        ks = sorted(d.k.unique())
        for f in fams:
            s = d[d.family == f].sort_values("k")
            if s.empty:
                continue
            ax.plot(s.k, 100 * s.share_draws_zero_column, color=COLORS[f], marker=MARKERS[f])
            ax.plot(s.k, 100 * s.dropped_features_mean / DIMS[ds], color=COLORS[f], ls="--", lw=1.4)
        k_axis(ax, ks)
        ax.set_ylim(-3, 103)
        ax.set_title(f"{LABELS[ds]} (d = {DIMS[ds]})")
        ax.set_ylabel("Percent")
    axes[-1].set_xlabel("Target size k")
    handles = [Line2D([], [], color=COLORS[f], marker=MARKERS[f], label=FAMILY_LABELS[f]) for f in fams]
    handles += [Line2D([], [], color=INK2, marker="o", label="Draws with at least one all-zero column"),
                Line2D([], [], color=INK2, ls="--", label="Features dropped (all-zero row), mean share")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    save(fig, "fig6_sparse_degeneracy.png")


def fig_replication(rep):
    best = rep.groupby(["dataset", "seed", "distribution"])["acc"].max().reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 6.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    yt = []
    for i, ds in enumerate(DATASETS):
        for j, (eq, col, mk) in enumerate([("eq3", COLORS["rademacher"], "s"), ("eq4", COLORS["achlioptas3"], "^")]):
            y0 = i + (-0.17 if j == 0 else 0.17)
            v = best[(best.dataset == ds) & (best.distribution == eq)]
            ax.scatter(100 * v.acc, y0 + rng.uniform(-0.09, 0.09, len(v)), s=16, color=col, marker=mk, alpha=0.55,
                       linewidth=0, zorder=2)
            ax.plot([100 * v.acc.median()] * 2, [y0 - 0.13, y0 + 0.13], color=INK, lw=2.2, zorder=3)
            r42 = v[v.seed == 42].acc.iloc[0]
            ax.scatter([100 * r42], [y0], s=110, marker="D", facecolor="white", edgecolor=INK, linewidth=1.8, zorder=4)
        yt.append(i)
    ax.set_yticks(yt)
    ax.set_yticklabels([LABELS[d] for d in DATASETS])
    ax.invert_yaxis()
    ax.set_xlabel("Best-kernel test accuracy at k = 2 (%)")
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=COLORS["rademacher"], marker="s", ls="none", label="Eq. 3, one point per draw"),
               Line2D([], [], color=COLORS["achlioptas3"], marker="^", ls="none", label="Eq. 4, one point per draw"),
               Line2D([], [], color=INK, lw=2.2, label="Median of 100 draws"),
               Line2D([], [], color=INK, marker="D", markerfacecolor="white", ls="none", ms=9,
                      label="The report's draw (seed 42)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, "fig7_original_pipeline_100_draws.png")


def fig_kernels(acc, family="gaussian"):
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(6.2, 3.0 * len(DATASETS)), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        m = metric_for(ds)
        a = acc[(acc.dataset == ds) & (acc.family == family) & (acc.metric == m)]
        ks = sorted(a.k.unique())
        for kern, col in KERNEL_COLORS.items():
            s = a[a.model == kern].sort_values("k")
            ax.fill_between(s.k, 100 * s.ci_low, 100 * s.ci_high, color=col, alpha=0.12, lw=0)
            ax.plot(s.k, 100 * s["mean"], color=col, marker="o", ms=6)
        s = a[a.model == "cv"].sort_values("k")
        ax.plot(s.k, 100 * s["mean"], color=INK, ls="--", lw=1.5)
        k_axis(ax, ks)
        ax.set_title(LABELS[ds])
        ax.set_ylabel(metric_label(ds))
    axes[-1].set_xlabel("Target size k (Gaussian projection)")
    handles = [Line2D([], [], color=c, marker="o", label=KERNEL_LABELS[k]) for k, c in KERNEL_COLORS.items()]
    handles.append(Line2D([], [], color=INK, ls="--", label=KERNEL_LABELS["cv"]))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, "fig8_tuned_kernels_vs_k.png")


# Wide versions of Figures 1 and 2 for the two-column paper: one row of panels at the full text width,
# drawn at print size (7.16 in, IEEE two-column text width) so that the type is 6.5 to 7.5 pt.
WIDE_IN = 7.16
WIDE_RC = {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
           "ytick.labelsize": 6.5, "legend.fontsize": 7, "lines.linewidth": 1.1, "lines.markersize": 3.2,
           "grid.linewidth": 0.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
           "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
           "legend.handlelength": 2.2, "legend.columnspacing": 1.6}


def wide_row(height):
    """One panel per data set in a row; each panel's width follows its range of k, so k has one scale."""
    span = [np.log2(max(ks) / min(ks)) + 2 * np.log2(1.3) for ks in
            ([k for k in K_GRID if k < DIMS[ds]] for ds in DATASETS)]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(WIDE_IN, height), constrained_layout=True,
                             gridspec_kw={"width_ratios": span})
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02)
    return fig, axes


def k_axis_wide(ax, ks):
    """k_axis, with the label at 16 left off (at this width it would run into 12); the tick stays."""
    k_axis(ax, ks)
    ax.set_xticklabels(["" if k == 16 and 12 in ks else str(k) for k in ks])


def save_wide(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_accuracy_vs_k_wide(acc):
    with plt.rc_context(WIDE_RC):
        fig, axes = wide_row(2.35)
        for ax, ds in zip(axes, DATASETS):
            m = metric_for(ds)
            a = acc[(acc.dataset == ds) & (acc.model == "cv") & (acc.metric == m)]
            ks = sorted(a[a.family != "full"]["k"].unique())
            full = a[a.family == "full"].iloc[0]
            ax.axhspan(100 * full.ci_low, 100 * full.ci_high, color="#d7d6d1", zorder=0)
            ax.axhline(100 * full["mean"], color=INK2, lw=1.0, zorder=1)
            maj = acc[(acc.dataset == ds) & (acc.model == "majority") & (acc.family == "full")]["mean"].iloc[0]
            ax.axhline(100 * maj if m == "acc" else 100 / N_CLASSES[ds], color=INK2, lw=0.9, ls=":", zorder=1)
            p = a[a.family == "pca"].sort_values("k")
            ax.plot(p.k, 100 * p["mean"], color=INK, ls="--", lw=1.0, marker="x", ms=3.5, zorder=3)
            for f in FAMS:
                s = a[a.family == f].sort_values("k")
                if s.empty:
                    continue
                ax.fill_between(s.k, 100 * s.ci_low, 100 * s.ci_high, color=COLORS[f], alpha=0.15, lw=0, zorder=2)
                ax.plot(s.k, 100 * s["mean"], color=COLORS[f], marker=MARKERS[f], zorder=4)
            k_axis_wide(ax, ks)
            ax.set_title(f"{LABELS[ds]} (d = {int(full.k)})")
            ax.set_ylabel("Balanced accuracy (%)" if ds == "glass" else ("Accuracy (%)" if ds == DATASETS[0] else ""))
        fig.supxlabel("Target size k", fontsize=7)
        handles = [Line2D([], [], color=COLORS[f], marker=MARKERS[f], label=FAMILY_LABELS[f]) for f in FAMS]
        handles += [Line2D([], [], color=INK, ls="--", marker="x", ms=3.5, lw=1.0, label="PCA to k (reference)"),
                    Line2D([], [], color=INK2, lw=4, alpha=0.4, label="All features (band: 95% interval)"),
                    Line2D([], [], color=INK2, ls=":", lw=0.9, label="Majority class (chance for Glass)")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
        save_wide(fig, "fig1_accuracy_vs_k_wide.png")


def fig_paired_wide(pdf):
    comps = ["rademacher", "achlioptas3", "verysparse", "onepercol"]
    offs = dict(zip(comps, [-0.15, -0.05, 0.05, 0.15]))    # log2 offsets, to tell the intervals at one k apart (caps can still touch)
    with plt.rc_context(WIDE_RC):
        fig, axes = wide_row(2.2)
        for ax, ds in zip(axes, DATASETS):
            m = metric_for(ds)
            x = pdf[(pdf.dataset == ds) & (pdf.model == "cv") & (pdf.metric == m) & (pdf.b == "gaussian")]
            ks = sorted(x.k.unique())
            ax.axhspan(-2, 2, color="#ecebe7", zorder=0)
            ax.axhline(0, color=INK2, lw=0.9, zorder=1)
            for c in comps:
                s = x[x.a == c].sort_values("k")
                if s.empty:
                    continue
                ax.errorbar(s.k * 2 ** offs[c], 100 * s.mean_diff,
                            yerr=[100 * (s.mean_diff - s.ci_low), 100 * (s.ci_high - s.mean_diff)],
                            color=COLORS[c], marker=MARKERS[c], lw=0.9, elinewidth=0.8, capsize=1.5, zorder=3)
            k_axis_wide(ax, ks)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10], integer=True))
            ax.set_title(LABELS[ds] + (" (balanced)" if ds == "glass" else ""))
            if ds == DATASETS[0]:
                ax.set_ylabel("Difference vs. Gaussian (points)")
        fig.supxlabel("Target size k", fontsize=7)
        handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], label=FAMILY_LABELS[c] + " − Gaussian")
                   for c in comps]
        handles.append(Line2D([], [], color="#d9d8d3", lw=5, label="±2 points (equivalence margin)"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
        save_wide(fig, "fig2_paired_difference_vs_gaussian_wide.png")


DIMS = {"iris": 4, "breast_cancer": 30, "wine": 13, "seeds": 7, "glass": 9}
N_CLASSES = {"iris": 3, "breast_cancer": 2, "wine": 3, "seeds": 3, "glass": 6}


def main():
    acc = pd.read_csv(os.path.join(RESULTS_DIR, "accuracy_summary.csv"))
    pdf = pd.read_csv(os.path.join(RESULTS_DIR, "paired_differences.csv"))
    fig_accuracy_vs_k(acc)
    fig_paired(pdf)
    fig_kernels(acc)
    fig_accuracy_vs_k_wide(acc)
    fig_paired_wide(pdf)
    vpath = os.path.join(RESULTS_DIR, "variance_check.csv")
    dpath = os.path.join(RESULTS_DIR, "distortion_summary.csv")
    if os.path.exists(vpath):
        fig_variance(pd.read_csv(vpath))
    if os.path.exists(dpath):
        ds_sum = pd.read_csv(dpath)
        fig_tails(ds_sum)
        fig_structure(acc, ds_sum)
        fig_degeneracy(ds_sum)
    rpath = os.path.join(RESULTS_DIR, "raw_replication.csv.gz")
    if os.path.exists(rpath):
        fig_replication(pd.read_csv(rpath))
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
