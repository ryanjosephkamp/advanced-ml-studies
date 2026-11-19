"""Intervals, summary tables, and figures for the capacity study.

Run from the repository root after run_study.py:

    python study/analyze.py

Reads study/results/*.csv and writes study/results/summary_*.csv and study/figures/*.png.

Intervals are 95% percentile intervals from 2,000 cluster-bootstrap resamples:
- synthetic grid: resample the 50 pattern sets (the same sets serve every load, since the load-P
  network stores the first P patterns of each set), so curves and capacities keep their
  within-set correlation;
- character subsets: resample the 200 subsets at each size;
- the full character set: resample the 200 draws.
Capacity P50 is the load at which exact recall first falls below 50%, interpolated linearly in
log P between the two loads around the crossing (loads 2 to 9 in steps of 1, then 12 to 40). "<2" means
recall was already below 50% at P = 2; ">40" means it never fell below 50% up to P = 40. A value between
2 and 3 says only that the crossing lies between those two loads: when recall is 1 at P = 2 and 0 at
P = 3, the interpolation returns sqrt(6) = 2.45 whatever the rule. Compare recall at P = 2 and 3 instead.
Character subsets get the same P50 over sizes 2 to 18 (">18" if never below 50%), with intervals from
the same subset resamples as their recall intervals (summary_character_subsets_p50.csv).
"""
import csv
import gzip
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "study" / "results"
FIG = ROOT / "study" / "figures"
B = 2000
BOOT_SEED = 20261110

LOADS = np.array([2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 24, 30, 40])
PROTO_LEVELS = [round(0.1 * k, 1) for k in range(10)]
CONDS = [("proto", lv) for lv in PROTO_LEVELS] + [("biased", 0.65), ("biased", 0.8)]
RULES = ["hebb", "hebb01", "storkey", "pinv", "cov", "modern4", "modern16", "modern64", "nn"]
NOISES = [10, 20]
N_SETS = 50

LABEL = {"hebb": "Hebbian", "hebb01": "Hebbian, 0/1 update", "storkey": "Storkey",
         "pinv": "Pseudoinverse", "cov": "Covariance", "modern4": "Modern, β = 4",
         "modern16": "Modern, β = 16", "modern64": "Modern, β = 64", "nn": "Nearest pattern"}
# Okabe-Ito colors, plus gray; line styles and markers also differ by rule
COLOR = {"hebb": "#000000", "hebb01": "#6E6E6E", "storkey": "#E69F00", "pinv": "#0072B2",
         "cov": "#009E73", "modern4": "#CC79A7", "modern16": "#CC79A7", "modern64": "#D55E00",
         "nn": "#56B4E9"}
STYLE = {"hebb": "-", "hebb01": "--", "storkey": "-", "pinv": "-", "cov": "-", "modern4": ":",
         "modern16": "--", "modern64": "-", "nn": "-."}
MARK = {"hebb": "o", "hebb01": "o", "storkey": "s", "pinv": "D", "cov": "^", "modern4": "v",
        "modern16": "v", "modern64": "v", "nn": "x"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5, "legend.fontsize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.2,
})


def read(name):
    gz = RES / f"{name}.csv.gz"
    if gz.exists():
        with gzip.open(gz, "rt", newline="") as f:
            return list(csv.DictReader(f))
    with open(RES / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def boot_idx(n, *key):
    rng = np.random.default_rng(np.random.SeedSequence(BOOT_SEED, spawn_key=key))
    return rng.integers(0, n, (B, n))


def ci(samples, axis=-1):
    return np.percentile(samples, 2.5, axis=axis), np.percentile(samples, 97.5, axis=axis)


def p50(curves, loads):
    """curves: (..., L) recall rates. Returns (value, flag) arrays; flag 0 ok, -1 below, +1 above."""
    curves = np.asarray(curves, dtype=float)
    below = curves < 0.5
    has = below.any(-1)
    first = np.argmax(below, -1)
    logp = np.log(loads)
    val = np.full(curves.shape[:-1], np.nan)
    flag = np.zeros(curves.shape[:-1], dtype=int)
    flag[~has] = 1
    flag[has & (first == 0)] = -1
    ok = has & (first > 0)
    i = first[ok]
    c = curves[ok]
    r0 = np.take_along_axis(c, (i - 1)[:, None], -1)[:, 0]
    r1 = np.take_along_axis(c, i[:, None], -1)[:, 0]
    frac = (r0 - 0.5) / np.where(r0 - r1 == 0, 1, r0 - r1)
    val[ok] = np.exp(logp[i - 1] + frac * (logp[i] - logp[i - 1]))
    val[flag == 1] = np.inf
    val[flag == -1] = 0.0
    return val, flag


def fmt_p50(v, top=40):
    if np.isinf(v):
        return f">{top}"
    if v == 0:
        return "<2"
    return f"{v:.1f}"


# ------------------------------------------------------------------------------------------------
# Synthetic grid
# ------------------------------------------------------------------------------------------------

def load_synthetic():
    ci_map = {(f, float(lv)): k for k, (f, lv) in enumerate(CONDS)}
    li = {int(p): k for k, p in enumerate(LOADS)}
    ri = {r: k for k, r in enumerate(RULES)}
    E = np.zeros((len(CONDS), len(RULES), len(NOISES), len(LOADS), N_SETS))
    OV = np.zeros_like(E)
    for row in read("synthetic_recall"):
        c = ci_map[(row["family"], float(row["level"]))]
        idx = (c, ri[row["rule"]], NOISES.index(int(row["noise"])), li[int(row["P"])], int(row["set"]))
        E[idx] = int(row["exact"]) / int(row["P"])
        OV[idx] = float(row["overlap"])
    F = np.zeros((len(CONDS), len(RULES), len(LOADS), N_SETS))
    for row in read("synthetic_fixed"):
        c = ci_map[(row["family"], float(row["level"]))]
        F[c, ri[row["rule"]], li[int(row["P"])], int(row["set"])] = int(row["fixed"]) / int(row["P"])
    PO = np.zeros((len(CONDS), len(LOADS), N_SETS))
    PD = np.zeros_like(PO)
    for row in read("synthetic_patterns"):
        c = ci_map[(row["family"], float(row["level"]))]
        PO[c, li[int(row["P"])], int(row["set"])] = float(row["mean_overlap"])
        PD[c, li[int(row["P"])], int(row["set"])] = float(row["density"])
    return E, OV, F, PO, PD


def summarize_synthetic(E, OV, F, PO, PD):
    idx = boot_idx(N_SETS, 0)
    C, R, NN, L, S = E.shape
    curve = np.zeros((C, R, NN, L, 3))
    fixed = np.zeros((C, R, L, 3))
    cap = {}
    for c in range(C):
        for r in range(R):
            fb = F[c, r][:, idx].mean(-1)                       # (L, B)
            lo, hi = ci(fb)
            fixed[c, r] = np.stack([F[c, r].mean(-1), lo, hi], -1)
            for n in range(NN):
                rb = E[c, r, n][:, idx].mean(-1)               # (L, B)
                lo, hi = ci(rb)
                point = E[c, r, n].mean(-1)
                curve[c, r, n] = np.stack([point, lo, hi], -1)
                v, flag = p50(point[None, :], LOADS)
                vb, _ = p50(rb.T, LOADS)
                cap[(c, r, n)] = (v[0], np.percentile(vb, 2.5, method="lower"),
                                  np.percentile(vb, 97.5, method="higher"))
    with open(RES / "summary_synthetic_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "level", "rule", "noise", "P", "recall", "recall_lo", "recall_hi",
                    "fixed", "fixed_lo", "fixed_hi", "final_overlap", "pattern_overlap", "pattern_density"])
        for c, (fam, lv) in enumerate(CONDS):
            for r, rule in enumerate(RULES):
                for n, noise in enumerate(NOISES):
                    for l, P in enumerate(LOADS):
                        w.writerow([fam, lv, rule, noise, P, *(f"{x:.4f}" for x in curve[c, r, n, l]),
                                    *(f"{x:.4f}" for x in fixed[c, r, l]), f"{OV[c, r, n, l].mean():.4f}",
                                    f"{PO[c, l].mean():.4f}", f"{PD[c, l].mean():.4f}"])
    with open(RES / "summary_capacity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "level", "rule", "noise", "p50", "p50_lo", "p50_hi", "pattern_overlap_P40"])
        for c, (fam, lv) in enumerate(CONDS):
            for r, rule in enumerate(RULES):
                for n, noise in enumerate(NOISES):
                    w.writerow([fam, lv, rule, noise, *(fmt_p50(x) for x in cap[(c, r, n)]),
                                f"{PO[c, -1].mean():.4f}"])
    return curve, fixed, cap


def covariance_check(E):
    """Paired difference covariance - Hebbian at rho = 0, by load and noise, with bootstrap CIs."""
    idx = boot_idx(N_SETS, 1)
    c = CONDS.index(("proto", 0.0))
    rh, rc = RULES.index("hebb"), RULES.index("cov")
    rows = []
    ok = True
    for n, noise in enumerate(NOISES):
        d = E[c, rc, n] - E[c, rh, n]                          # (L, S)
        db = d[:, idx].mean(-1)
        lo, hi = ci(db)
        for l, P in enumerate(LOADS):
            inside = lo[l] <= 0 <= hi[l]
            ok &= bool(inside)
            rows.append([noise, P, f"{E[c, rh, n, l].mean():.4f}", f"{E[c, rc, n, l].mean():.4f}",
                         f"{d[l].mean():.4f}", f"{lo[l]:.4f}", f"{hi[l]:.4f}", int(inside)])
    with open(RES / "summary_covariance_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise", "P", "hebb_recall", "cov_recall", "difference", "diff_lo", "diff_hi", "ci_includes_zero"])
        w.writerows(rows)
    return ok, max(abs(float(r[4])) for r in rows)


# ------------------------------------------------------------------------------------------------
# Characters
# ------------------------------------------------------------------------------------------------

def summarize_characters():
    full = read("characters_full")
    out = []
    dist = {}
    for qs in ["test", "valid42", "fresh10", "fresh20"]:
        for r, rule in enumerate(RULES):
            x = np.array([int(row["exact"]) for row in full if row["queryset"] == qs and row["rule"] == rule])
            dist[(qs, rule)] = x
            mb = x[boot_idx(len(x), 2, r, ["test", "valid42", "fresh10", "fresh20"].index(qs))].mean(-1)
            lo, hi = ci(mb)
            out.append([qs, rule, len(x), f"{x.mean():.3f}", f"{lo:.3f}", f"{hi:.3f}", x.min(), x.max(),
                        int(np.percentile(x, 5)), int(np.percentile(x, 95)), f"{(x == 18).mean():.3f}"])
    with open(RES / "summary_characters_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["queryset", "rule", "draws", "mean_exact_of_18", "mean_lo", "mean_hi", "min", "max",
                    "p5", "p95", "share_all_18"])
        w.writerows(out)

    sub = read("characters_subsets")
    Ps = list(range(2, 19))
    rec = {}
    boot = {}
    with open(RES / "summary_character_subsets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P", "rule", "noise", "recall", "recall_lo", "recall_hi", "fixed", "fixed_lo", "fixed_hi",
                    "subset_overlap", "subset_density"])
        grouped = {}
        for row in sub:
            grouped.setdefault((int(row["P"]), row["rule"], int(row["noise"])), []).append(row)
        for P in Ps:
            for r, rule in enumerate(RULES):
                for noise in NOISES:
                    rows = sorted(grouped[(P, rule, noise)], key=lambda z: int(z["subset"]))
                    x = np.array([int(z["exact"]) / P for z in rows])
                    fx = np.array([int(z["fixed"]) / P for z in rows])
                    ib = boot_idx(len(x), 3, P, r, noise)
                    boot[(P, rule, noise)] = x[ib].mean(-1)
                    lo, hi = ci(boot[(P, rule, noise)])
                    flo, fhi = ci(fx[ib].mean(-1))
                    so = np.mean([float(z["subset_overlap"]) for z in rows])
                    sd = np.mean([float(z["subset_density"]) for z in rows])
                    rec[(P, rule, noise)] = (x.mean(), lo, hi)
                    w.writerow([P, rule, noise, f"{x.mean():.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{fx.mean():.4f}",
                                f"{flo:.4f}", f"{fhi:.4f}", f"{so:.4f}", f"{sd:.4f}"])

    # Capacity P50 over subset sizes, with intervals from the same per-size subset resamples
    loads = np.array(Ps)
    with open(RES / "summary_character_subsets_p50.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "noise", "p50", "p50_lo", "p50_hi"])
        for rule in RULES:
            for noise in NOISES:
                point = np.array([rec[(P, rule, noise)][0] for P in Ps])
                v, _ = p50(point[None, :], loads)
                vb, _ = p50(np.stack([boot[(P, rule, noise)] for P in Ps], -1), loads)
                w.writerow([rule, noise, fmt_p50(v[0], 18), fmt_p50(np.percentile(vb, 2.5, method="lower"), 18),
                            fmt_p50(np.percentile(vb, 97.5, method="higher"), 18)])
    return dist, rec


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------

def log_load_axis(ax, loads=(2, 4, 6, 9, 12, 18, 24, 40)):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(loads)))
    ax.xaxis.set_major_formatter(FixedFormatter([str(p) for p in loads]))
    ax.xaxis.set_minor_locator(NullLocator())


def fig1(curve, noise_i, path):
    base = plt.get_cmap("plasma")

    def cmap(t):  # plasma without its palest yellow end, so every line reads on white
        return base(0.85 * t)
    rules = RULES
    fig, axes = plt.subplots(5, 2, figsize=(7.5, 15.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for a, rule in zip(axes, rules):
        r = RULES.index(rule)
        for k, lv in enumerate(PROTO_LEVELS):
            c = CONDS.index(("proto", lv))
            col = cmap(k / (len(PROTO_LEVELS) - 1))
            pt, lo, hi = curve[c, r, noise_i].T
            a.fill_between(LOADS, lo, hi, color=col, alpha=0.18, lw=0)
            a.plot(LOADS, pt, color=col, lw=1.8, label=f"ρ = {lv:.1f}")
        a.set_title(LABEL[rule])
        log_load_axis(a, (2, 6, 12, 24, 40))
        a.set_ylim(-0.03, 1.03)
        a.grid(alpha=0.25)
    axes[-1].axis("off")
    axes[-3].tick_params(labelbottom=True)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title="Pattern overlap", loc="center", ncol=2, frameon=False)
    for a in axes[::2]:
        a.set_ylabel("Exact recall")
    for a in axes[-3:-1]:
        a.set_xlabel("Patterns stored, P")
    axes[-2].set_xlabel("Patterns stored, P")
    fig.suptitle(f"Exact recall from {NOISES[noise_i]}% noise, prototype family", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cap_plot_value(v):
    return 52.0 if np.isinf(v) else (1.6 if v == 0 else v)


def fig2(cap, path):
    rules = ["hebb", "hebb01", "storkey", "cov", "pinv", "modern16", "modern64", "nn"]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 11.5), sharex=True)
    for a, n in zip(axes, range(len(NOISES))):
        # Rules that never fell below 50% up to P = 40 are drawn in the shaded band, one row each,
        # so overlapping lines stay visible; their position within the band carries no value.
        a.axhspan(44, 76, color="#F2F2F2", lw=0, zorder=0)
        for k, rule in enumerate(rules):
            r = RULES.index(rule)
            top = 46 * 1.055 ** k

            def y(v, top=top):
                return top if np.isinf(v) else (1.6 if v == 0 else v)
            xs, ys, lo, hi = [], [], [], []
            for lv in PROTO_LEVELS:
                c = CONDS.index(("proto", lv))
                v, l, h = cap[(c, r, n)]
                xs.append(lv); ys.append(y(v)); lo.append(y(l)); hi.append(y(h))
            a.fill_between(xs, lo, hi, color=COLOR[rule], alpha=0.15, lw=0)
            a.plot(xs, ys, STYLE[rule], color=COLOR[rule], marker=MARK[rule], ms=6, label=LABEL[rule])
        a.set_yscale("log")
        a.yaxis.set_major_locator(FixedLocator([2, 4, 8, 16, 32, 58]))
        a.yaxis.set_major_formatter(FixedFormatter(["2", "4", "8", "16", "32", ">40"]))
        a.yaxis.set_minor_locator(NullLocator())
        a.set_ylim(1.8, 78)
        a.set_ylabel("Capacity P50 (patterns)")
        a.set_title(f"{NOISES[n]}% of bits flipped in the query")
        a.grid(alpha=0.25)
    axes[-1].set_xlabel("Expected overlap between stored patterns, ρ")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Load at which exact recall falls to 50%", y=0.995)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig3(cap, PO, path):
    rules = ["hebb", "hebb01", "storkey", "cov", "pinv", "nn"]
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 11.5), sharex=True, sharey=True)
    for a, rule in zip(axes.ravel(), rules):
        r = RULES.index(rule)
        xs, ys, lo, hi = [], [], [], []
        for lv in PROTO_LEVELS:
            c = CONDS.index(("proto", lv))
            v, l, h = cap[(c, r, 0)]
            xs.append(PO[c, -1].mean()); ys.append(cap_plot_value(v))
            lo.append(cap_plot_value(l)); hi.append(cap_plot_value(h))
        a.fill_between(xs, lo, hi, color="#0072B2", alpha=0.18, lw=0)
        a.plot(xs, ys, "-", color="#0072B2", marker="o", ms=5, label="Shared prototype, density 0.5")
        for d, mk in [(0.65, "s"), (0.8, "D")]:
            c = CONDS.index(("biased", d))
            v, l, h = cap[(c, r, 0)]
            x = PO[c, -1].mean()
            y = cap_plot_value(v)
            a.errorbar([x], [y], yerr=[[y - cap_plot_value(l)], [cap_plot_value(h) - y]], fmt=mk,
                       color="#D55E00", ms=8, capsize=4, lw=1.8, label=f"Independent bits, density {d:.2f}")
        a.set_title(LABEL[rule])
        a.set_yscale("log")
        a.yaxis.set_major_locator(FixedLocator([1.6, 2, 4, 8, 16, 32, 52]))
        a.yaxis.set_major_formatter(FixedFormatter(["<2", "2", "4", "8", "16", "32", ">40"]))
        a.yaxis.set_minor_locator(NullLocator())
        a.set_ylim(1.4, 60)
        a.grid(alpha=0.25)
    for a in axes[:, 0]:
        a.set_ylabel("Capacity P50")
    for a in axes[-1]:
        a.set_xlabel("Mean overlap of stored patterns")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Density vs. shared structure at similar overlap (10% noise)", y=0.995)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig4(curve, rec, path):
    rules = ["hebb", "storkey", "cov", "pinv", "modern16", "nn"]
    Ps = np.arange(2, 19)
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 11.5), sharex=True, sharey=True)
    for a, rule in zip(axes.ravel(), rules):
        r = RULES.index(rule)
        for lv, ls in [(0.4, "--"), (0.5, ":")]:
            c = CONDS.index(("proto", lv))
            pt, lo, hi = curve[c, r, 0].T
            m = LOADS <= 18
            a.fill_between(LOADS[m], lo[m], hi[m], color="#999999", alpha=0.18, lw=0)
            a.plot(LOADS[m], pt[m], ls, color="#555555", lw=2, label=f"Synthetic, ρ = {lv}")
        c = CONDS.index(("biased", 0.8))
        pt, lo, hi = curve[c, r, 0].T
        m = LOADS <= 18
        a.plot(LOADS[m], pt[m], "-.", color="#D55E00", lw=2, label="Synthetic, density 0.80")
        y = np.array([rec[(P, rule, 10)] for P in Ps])
        a.fill_between(Ps, y[:, 1], y[:, 2], color="#0072B2", alpha=0.25, lw=0)
        a.plot(Ps, y[:, 0], "-", color="#0072B2", lw=2.6, label="Character subsets")
        a.set_title(LABEL[rule])
        log_load_axis(a, (2, 4, 6, 9, 12, 18))
        a.set_ylim(-0.03, 1.03)
        a.grid(alpha=0.25)
    for a in axes[:, 0]:
        a.set_ylabel("Exact recall")
    for a in axes[-1]:
        a.set_xlabel("Patterns stored, P")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Course characters vs. synthetic patterns (10% noise)", y=0.995)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig5(dist, path):
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.5), sharex=True)
    for a, qs, title in [(axes[0], "valid42", "Committed 20% validation file: 200 update orders"),
                         (axes[1], "fresh20", "Fresh 20% noise: 200 draws of noise and order")]:
        x = dist[(qs, "pinv")]
        counts = np.bincount(x, minlength=19)
        a.bar(np.arange(19), counts / len(x), color="#0072B2", width=0.8)
        a.axvline(x.mean(), color="#D55E00", lw=2.4, ls="--", label=f"Mean {Decimal(repr(round(float(x.mean()), 6))).quantize(Decimal('0.01'), ROUND_HALF_UP)} of 18")
        if qs == "valid42":
            a.annotate("Original report:\n17 of 18, one run", xy=(17, 0.004), xytext=(16.6, 0.16),
                       ha="center", fontsize=12, arrowprops=dict(arrowstyle="->", color="#333333"))
        a.set_title(title)
        a.set_ylabel("Share of draws")
        a.legend(loc="upper left", frameon=False)
        a.grid(alpha=0.25, axis="y")
    axes[1].set_xlabel("Patterns recalled exactly, of 18 (pseudoinverse rule)")
    axes[1].set_xticks(range(6, 19, 2))
    axes[1].set_xlim(5.4, 18.6)
    fig.suptitle("One run is one draw from this spread", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig6(curve, fixed, path):
    c = CONDS.index(("proto", 0.0))
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 9.5), sharex=True, sharey=True)
    for a, n in zip(axes, range(len(NOISES))):
        for rule in ["hebb", "hebb01"]:
            r = RULES.index(rule)
            pt, lo, hi = curve[c, r, n].T
            a.fill_between(LOADS, lo, hi, color=COLOR[rule], alpha=0.2, lw=0)
            a.plot(LOADS, pt, STYLE[rule], color=COLOR[rule], marker=MARK[rule], ms=6,
                   label=f"{LABEL[rule]}: exact recall")
            fp, flo, fhi = fixed[c, r].T
            a.plot(LOADS, fp, ":", color=COLOR[rule], lw=2, label=f"{LABEL[rule]}: fixed points")
        log_load_axis(a)
        a.set_ylim(-0.03, 1.03)
        a.set_ylabel("Share of stored patterns")
        a.set_title(f"Random balanced patterns (ρ = 0), {NOISES[n]}% noise")
        a.grid(alpha=0.25)
    axes[-1].set_xlabel("Patterns stored, P")
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=11)
    fig.suptitle("The assignment's 0/1-state update vs. the standard update", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    E, OV, F, PO, PD = load_synthetic()
    curve, fixed, cap = summarize_synthetic(E, OV, F, PO, PD)
    cov_ok, cov_maxdiff = covariance_check(E)
    dist, rec = summarize_characters()
    fig1(curve, 0, FIG / "fig1_recall_vs_load.png")
    fig1(curve, 1, FIG / "fig1b_recall_vs_load_noise20.png")
    fig2(cap, FIG / "fig2_capacity_vs_overlap.png")
    fig3(cap, PO, FIG / "fig3_density_vs_structure.png")
    fig4(curve, rec, FIG / "fig4_characters_vs_synthetic.png")
    fig5(dist, FIG / "fig5_pseudoinverse_order_spread.png")
    fig6(curve, fixed, FIG / "fig6_assignment_update.png")
    print(f"Covariance rule vs. Hebbian at rho = 0: every paired-difference interval includes 0: {cov_ok}; "
          f"largest mean difference {cov_maxdiff:.4f} (details in summary_covariance_check.csv)")
    print(f"Summaries in {RES}, figures in {FIG}.")


if __name__ == "__main__":
    main()
