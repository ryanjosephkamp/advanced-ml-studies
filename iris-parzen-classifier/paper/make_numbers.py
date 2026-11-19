# make_numbers.py
# Writes the numbers and tables used by main.tex from the committed result files, so that no
# number in the paper is typed by hand.
#
# Inputs (relative to the project folder):
#   parzen_classifier_results.csv                    the 65 original single-split runs
#   repeated_splits/repeated_split_results.csv       500 random splits per ratio
#   repeated_splits/standardized_feature_results.csv raw against standardized features
#
# Outputs (in paper/):
#   numbers.tex             one macro per number, used in the text as \N{key}
#   table_kernels.tex       Table I: test accuracy by problem and kernel
#   table_standardized.tex  Table II: raw against standardized features
#
# Usage (from the project folder or from paper/):
#     python paper/make_numbers.py

import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent

KERNELS = {"dot": "Dot Product", "d2": "Polynomial (d=2)", "d3": "Polynomial (d=3)", "gauss": "Gaussian"}
KERNEL_LABEL = {"dot": "Dot product", "d2": "Polynomial, $d=2$", "d3": "Polynomial, $d=3$", "gauss": "Gaussian"}
RATIOS = {"90/10": "90", "80/20": "80", "70/30": "70", "60/40": "60", "50/50": "50"}
CONFIGS = [(1, "dot")] + [(p, k) for p in (2, 3, 4) for k in KERNELS]


def f3(x):
    # Round half up once, from a value in the result files (six decimals) or a value computed from them.
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def ceil3(x):
    # An upper bound to three decimals, for statements of the form "at most". Scaling to thousandths
    # and rounding to six decimals first removes floating-point noise, so a value on a multiple of
    # 0.001 is not pushed to the next one. It also moves a value within about 5e-10 of a multiple of
    # 0.001 onto it (the exact rule is the rounding of the scaled value), so the bound can fall short of
    # x by about that much. Every input the paper uses is at least 2e-5 from a thousandth.
    return f"{math.ceil(round(x * 1000, 6)) / 1000:.3f}"


def above3(x):
    # The smallest three-decimal value strictly above x, for statements of the form "under". Scaling
    # to thousandths and rounding to six decimals first removes floating-point noise, so a value on a
    # multiple of 0.001 gets the next one. It also moves a value within about 5e-10 of a multiple of
    # 0.001 onto it (the exact rule is the rounding of the scaled value), so a value that close below
    # one gets the multiple after it: above x, but not the smallest.
    return f"{(math.floor(round(x * 1000, 6)) + 1) / 1000:.3f}"


def f2(x):
    return f"{x:.2f}"


def signed3(x):
    s = f"{x:+.3f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def pct1(x):
    return f"{100 * x:.1f}"


def pct1_cell(x):
    # A share in a table: a nonzero share that rounds to 0.0 prints as "<0.1", so it does not read as none.
    s = pct1(x)
    return "$<$0.1" if x > 0 and s == "0.0" else s


def load():
    orig = pd.read_csv(ROOT / "parzen_classifier_results.csv")
    rs = pd.read_csv(ROOT / "repeated_splits" / "repeated_split_results.csv")
    sf = pd.read_csv(ROOT / "repeated_splits" / "standardized_feature_results.csv")
    return orig, rs, sf


def row(df, problem, kernel, split="All"):
    r = df[(df["problem"] == problem) & (df["kernel"] == KERNELS[kernel]) & (df["split"] == split)]
    assert len(r) == 1, (problem, kernel, split, len(r))
    return r.iloc[0]


def iris_distances():
    # Distances between the mean measurements of neighboring species (cm), from scikit-learn's copy.
    from sklearn.datasets import load_iris
    d = load_iris()
    m = [d.data[d.target == c].mean(axis=0) for c in range(3)]
    return np.linalg.norm(m[0] - m[1]), np.linalg.norm(m[1] - m[2])


def compute():
    orig, rs, sf = load()
    N = {}

    # Pooled test accuracy (all five ratios, 2,500 splits per row)
    for p, k in CONFIGS:
        r = row(rs, p, k)
        t = f"{k}_p{p}"
        N[f"{t}_mean"] = f3(r["test_mean"])
        N[f"{t}_cilo"] = f3(r["mean_ci95_low"])
        N[f"{t}_cihi"] = f3(r["mean_ci95_high"])
        N[f"{t}_rlo"] = f3(r["range95_low"])
        N[f"{t}_rhi"] = f3(r["range95_high"])
        N[f"{t}_train"] = f3(r["train_mean"])
        N[f"{t}_orig"] = f3(r["original_seed42_test"])
    for p in (1, 2, 3, 4):
        N[f"maj_p{p}"] = f3(row(rs, p, "dot")["majority_mean"])

    # Problems 2 to 4 together: the mean of the three pooled means
    for k in KERNELS:
        N[f"{k}_p234"] = f3(np.mean([row(rs, p, k)["test_mean"] for p in (2, 3, 4)]))
        N[f"{k}_p234_orig"] = f3(np.mean([row(rs, p, k)["original_seed42_test"] for p in (2, 3, 4)]))

    # Gaussian against the best other kernel on the same split
    for p in (2, 3, 4):
        r = row(rs, p, "gauss")
        N[f"gauss_ge_p{p}"] = pct1(r["share_ge_best_other"])
        N[f"gauss_gt_p{p}"] = pct1(r["share_gt_best_other"])
        N[f"gauss_margin_p{p}"] = f3(r["mean_margin_vs_best_other"])
    g = [row(rs, p, "gauss") for p in (2, 3, 4)]
    N["gauss_ci_halfwidth_max"] = f3(max((r["mean_ci95_high"] - r["mean_ci95_low"]) / 2 for r in g))
    # An upper bound ("within") on the Gaussian half-widths, from the interval's endpoints (the pooled
    # interval comes from the per-seed means, not from the SD column); the largest is just over 0.003.
    N["gauss_ci_halfwidth_ub"] = ceil3(max((r["mean_ci95_high"] - r["mean_ci95_low"]) / 2 for r in g))
    N["gauss_rlo_min"] = f3(min(r["range95_low"] for r in g))
    N["gauss_ge_min"] = pct1(min(r["share_ge_best_other"] for r in g))
    N["gauss_ge_max"] = pct1(max(r["share_ge_best_other"] for r in g))
    N["p3_spread"] = f3(row(rs, 3, "gauss")["test_mean"] - row(rs, 3, "d3")["test_mean"])

    # Train and test agree over random splits, except degree 3 on Problem 3
    gaps = [abs(row(rs, p, k)["train_mean"] - row(rs, p, k)["test_mean"])
            for p, k in CONFIGS if (p, k) != (3, "d3")]
    N["train_test_gap_max"] = ceil3(max(gaps))

    # Training share
    per = {s: rs[rs["split"] == s] for s in RATIOS}
    for s, tag in RATIOS.items():
        d = per[s]
        assert len(d) == 13
        N[f"all13_r{tag}"] = f3(d["test_mean"].mean())
        N[f"orig13_r{tag}"] = f3(d["original_seed42_test"].mean())
        gs = [row(rs, p, "gauss", s)["test_mean"] for p in (2, 3, 4)]
        N[f"gauss234_r{tag}"] = f3(np.mean(gs))
        # The same two means under the names the text uses.
        N[f"all13d_r{tag}"] = N[f"all13_r{tag}"]
        N[f"gauss234d_r{tag}"] = N[f"gauss234_r{tag}"]
    all13 = [per[s]["test_mean"].mean() for s in RATIOS]
    orig13 = [per[s]["original_seed42_test"].mean() for s in RATIOS]
    N["all13_span"] = f3(max(all13) - min(all13))
    N["all13_span_ub"] = above3(max(all13) - min(all13))   # "under": strictly above the largest difference
    spans = rs[rs["split"].isin(list(RATIOS))].groupby(["problem", "kernel"])["test_mean"].agg(lambda s: s.max() - s.min())
    N["cfg_share_span_max"] = f3(spans.max())   # largest change across shares for one configuration
    N["orig13_min"] = f3(min(orig13))
    N["orig13_max"] = f3(max(orig13))
    N["orig13_span"] = f2(max(orig13) - min(orig13))

    # Where the original split sits among the 500 random splits
    ratio_rows = rs[rs["split"] != "All"]
    N["orig_above_n"] = str(int((ratio_rows["original_seed42_test"] > ratio_rows["test_mean"]).sum()))
    N["orig_below_n"] = str(int((ratio_rows["original_seed42_test"] < ratio_rows["test_mean"]).sum()))
    N["orig_total_n"] = str(len(ratio_rows))
    N["orig_mean65"] = f3(ratio_rows["original_seed42_test"].mean())
    N["rand_mean65"] = f3(ratio_rows["test_mean"].mean())

    # Counts
    N["n_splits_all"] = f"{int(row(rs, 2, 'dot')['n_splits']):,}"
    N["n_splits_ratio"] = str(int(rs[rs['split'] == '90/10'].iloc[0]['n_splits']))
    N["n_fits"] = f"{int(ratio_rows['n_splits'].sum()):,}"
    N["n_runs"] = str(len(orig))
    N["seed_last"] = str(int(rs["seed_last"].max()))

    # Original split: sizes, and Problem 1's training errors
    for s, tag in RATIOS.items():
        r = orig[(orig["Split Label"] == s) & (orig["Problem"] == 1)].iloc[0]
        N[f"train_n_r{tag}"] = str(int(r["Train Samples"]))
        N[f"test_n_r{tag}"] = str(int(r["Test Samples"]))
    p1 = orig[orig["Problem"] == 1]
    N["p1_orig_train_min"] = f3(p1["Train Accuracy"].min())
    N["p1_orig_train_max"] = f3(p1["Train Accuracy"].max())
    N["p1_orig_test_min"] = f3(p1["Test Accuracy"].min())
    N["p4_samples"] = str(int(orig[(orig["Problem"] == 4) & (orig["Split Label"] == "90/10")].iloc[0][["Train Samples", "Test Samples"]].sum()))

    def train_errors(cm):
        a = np.array([int(v) for v in cm.replace("[", " ").replace("]", " ").split()]).reshape(2, 2)
        return int(a.sum() - np.trace(a))
    errs = {r["Split Label"]: train_errors(r["Train Confusion Matrix"]) for _, r in p1.iterrows()}
    assert errs["90/10"] == 2 and all(v == 1 for s, v in errs.items() if s != "90/10"), errs
    N["p1_err_r90"] = str(errs["90/10"])
    N["p1_err_other"] = "1"

    # Standardized features
    for p, k in CONFIGS:
        r = row(sf, p, k)
        t = f"s_{k}_p{p}"
        assert abs(r["raw_test_mean"] - row(rs, p, k)["test_mean"]) < 1e-9
        N[f"{t}_raw"] = f3(r["raw_test_mean"])
        N[f"{t}_std"] = f3(r["standardized_test_mean"])
        N[f"{t}_train"] = f3(r["standardized_train_mean"])
        N[f"{t}_chg"] = signed3(r["change_mean"])
        N[f"{t}_hi"] = pct1(r["share_standardized_higher"])
        N[f"{t}_lo"] = pct1(r["share_standardized_lower"])
    poly24 = [abs(row(sf, p, k)["change_mean"]) for p in (2, 4) for k in ("dot", "d2", "d3")]
    N["s_p24_maxabs"] = ceil3(max(poly24))       # an upper bound: "at most"
    N["s_gap_p3_raw"] = f2(row(rs, 3, "gauss")["test_mean"] - row(rs, 3, "d2")["test_mean"])
    N["s_gap_p3_std"] = f2(row(sf, 3, "gauss")["standardized_test_mean"] - row(sf, 3, "d2")["standardized_test_mean"])
    # The Gaussian still has the highest standardized mean on each problem
    for p in (2, 3, 4):
        best_other = max(row(sf, p, k)["standardized_test_mean"] for k in ("dot", "d2", "d3"))
        assert row(sf, p, "gauss")["standardized_test_mean"] > best_other
        N[f"s_best_other_p{p}"] = f3(best_other)
    # Problem 3's polynomial kernels: train and test still agree after standardizing
    N["s_p3_poly_gap_max"] = f3(max(abs(row(sf, 3, k)["standardized_train_mean"] - row(sf, 3, k)["standardized_test_mean"]) for k in ("d2", "d3")))

    N["sigma"] = f2(2 ** -0.5)

    # Iris geometry (scikit-learn's copy of the data)
    a, b = iris_distances()
    N["dist_set_ver"] = f"{a:.1f}"
    N["dist_ver_vir"] = f"{b:.1f}"

    # Claims the text makes, checked here so a change in the results stops the build
    assert all(float(row(rs, 3, k)["test_mean"]) < float(N["maj_p3"]) for k in ("dot", "d2", "d3"))
    for p in (2, 3, 4):
        assert all(row(rs, p, "gauss")["test_mean"] > row(rs, p, k)["test_mean"] for k in ("dot", "d2", "d3"))
    assert all(float(N[f"all13_r{t}"]) > 0.8 for t in RATIOS.values())
    return N, rs, sf


def tex_name(key):
    return "num@" + key


def write_numbers(N):
    lines = ["% Generated by make_numbers.py from the committed results. Do not edit.",
             r"\providecommand{\N}[1]{\csname num@#1\endcsname}"]
    for k, v in N.items():
        lines.append(r"\expandafter\gdef\csname num@%s\endcsname{%s}" % (k, v))
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")


def ci(r, p, k):
    return f"{f3(r['mean_ci95_low'])}--{f3(r['mean_ci95_high'])}"


def write_table_kernels(rs):
    out = [r"\begin{tabular}{@{}clcccc@{}}", r"\toprule",
           r"Prob. & Kernel & Mean & 95\% CI of mean & 2.5th to 97.5th percentiles of splits & Original \\", r"\midrule"]
    for p, k in CONFIGS:
        r = row(rs, p, k)
        first = k == "dot" and p == 1 or (p != 1 and k == "dot")
        label = str(p) if first else ""
        rng = f"{f3(r['range95_low'])}--{f3(r['range95_high'])}"
        cells = [f3(r["test_mean"]), ci(r, p, k), rng, f3(r["original_seed42_test"])]
        if k == "gauss":
            cells[0] = r"\textbf{" + cells[0] + "}"
        out.append(f"{label} & {KERNEL_LABEL[k]} & " + " & ".join(cells) + r" \\")
        if k == "dot" and p == 1 or k == "gauss" and p < 4:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_kernels.tex").write_text("\n".join(out) + "\n")


def write_table_standardized(sf):
    out = [r"\begin{tabular}{@{}clcccc@{}}", r"\toprule",
           r"Prob. & Kernel & Raw & Standardized & Change & Higher / lower \\", r"\midrule"]
    for p, k in CONFIGS:
        r = row(sf, p, k)
        label = str(p) if k == "dot" else ""
        cells = [f3(r["raw_test_mean"]), f3(r["standardized_test_mean"]), signed3(r["change_mean"]),
                 f"{pct1_cell(r['share_standardized_higher'])}\\% / {pct1_cell(r['share_standardized_lower'])}\\%"]
        out.append(f"{label} & {KERNEL_LABEL[k]} & " + " & ".join(cells) + r" \\")
        if k == "dot" and p == 1 or k == "gauss" and p < 4:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_standardized.tex").write_text("\n".join(out) + "\n")


def main():
    N, rs, sf = compute()
    write_numbers(N)
    write_table_kernels(rs)
    write_table_standardized(sf)
    print(f"wrote {len(N)} numbers and two tables to {PAPER}")


if __name__ == "__main__":
    main()
