# make_numbers.py
# Writes the numbers and tables used by main.tex from the committed data and result files, so that
# no number in the paper is typed by hand. The text asks for a number with \N{key}; the keys are
# described in resolve() below.
#
# Inputs (relative to the project folder):
#   data/characters.csv, data/test.csv, data/valid_datasets/valid_dataset_42_20.0/valid_42_20.0.csv
#   hopfield_results/*/experiment_results.csv and hopfield_experiment_report.md
#   study/results/summary_*.csv, characters_fixed.csv, characters_subsets.csv
#
# Outputs (in paper/): numbers.tex (one macro per key used), table_baseline.tex, table_capacity.tex,
# table_characters.tex
#
# Usage (from the project folder or from paper/):
#     python paper/make_numbers.py

import itertools
import math
import re
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RES = ROOT / "study" / "results"

LABEL = {"hebb": "Hebbian", "hebb01": "Hebbian, 0/1 update", "storkey": "Storkey", "pinv": "Pseudoinverse",
         "cov": "Covariance", "modern4": "Modern, $\\beta=4$", "modern16": "Modern, $\\beta=16$",
         "modern64": "Modern, $\\beta=64$", "nn": "Nearest pattern"}


def rnd(x, places):
    q = Decimal(1).scaleb(-places)
    return str(Decimal(f"{x:.{places + 1}f}" if False else repr(round(float(x), 6))).quantize(q, rounding=ROUND_HALF_UP))


def f3(x):
    return rnd(float(x), 3)


def f2(x):
    return rnd(float(x), 2)


def f1(x):
    return rnd(float(x), 1)


def num(x):
    """Format a summary-file value: a number as written, or an inequality such as >40."""
    s = str(x)
    if s.startswith(">") or s.startswith("<"):
        return "$" + s[0] + "$" + s[1:]
    return s


def p50_cell(v):
    s = str(v)
    if s[0] in "<>":
        return num(s)
    return "2--3" if float(s) < 3.0 else s


class Data:
    def __init__(self):
        self.chars = np.loadtxt(ROOT / "data/characters.csv", delimiter=",", skiprows=1, dtype=int)
        self.test = np.loadtxt(ROOT / "data/test.csv", delimiter=",", skiprows=1, dtype=int)
        self.valid = np.loadtxt(ROOT / "data/valid_datasets/valid_dataset_42_20.0/valid_42_20.0.csv", delimiter=",", skiprows=1, dtype=int)
        # Rerunning kamp_hw2.py adds a second timestamped folder; refuse rather than pick one silently.
        folders = sorted(f for f in (ROOT / "hopfield_results").glob("hopfield_results_*") if f.is_dir())
        if len(folders) != 1:
            raise SystemExit(f"expected one hopfield_results/hopfield_results_* folder, found {len(folders)}: "
                             + ", ".join(f.name for f in folders))
        folder = folders[0]
        self.exp = pd.read_csv(folder / "experiment_results.csv")
        self.report = (folder / "hopfield_experiment_report.md").read_text()
        self.cap = pd.read_csv(RES / "summary_capacity.csv", dtype={"p50": str, "p50_lo": str, "p50_hi": str})
        self.curves = pd.read_csv(RES / "summary_synthetic_curves.csv")
        self.full = pd.read_csv(RES / "summary_characters_full.csv")
        self.sub = pd.read_csv(RES / "summary_character_subsets.csv")
        self.subp50 = pd.read_csv(RES / "summary_character_subsets_p50.csv", dtype=str)
        self.cov = pd.read_csv(RES / "summary_covariance_check.csv")
        self.fixed = pd.read_csv(RES / "characters_fixed.csv")
        self.subsets_raw = pd.read_csv(RES / "characters_subsets.csv")
        self.full_raw = pd.read_csv(RES / "characters_full.csv")
        X = 2 * self.chars - 1
        self.pm = X
        n = X.shape[1]
        iu = np.triu_indices(len(X), 1)
        self.overlap = (X @ X.T / n)[iu]
        self.ham = ((1 - X @ X.T / n) / 2 * n)[iu].round().astype(int)
        self.pearson = np.corrcoef(self.chars)[iu]
        self.density = self.chars.mean(axis=1)


def cond(fam, level):
    return fam, float(level)


def resolve(key, D):
    """Return the text for one key. Key forms, with fields separated by colons:

    d:<name>                              data statistics (see the branches below)
    b:<name>[:<set>]                      the corrected baseline run (set: Training, Test, Validation)
    c:<family>:<level>:<rule>:<noise>[:lo|hi]   capacity P50 (summary_capacity.csv)
    r:<family>:<level>:<rule>:<noise>:<P>[:lo|hi]   exact recall at load P (summary_synthetic_curves.csv)
    po:<family>:<level>                   measured mean overlap at P = 40
    f:<queryset>:<rule>:<field>           the 18 characters, whole set (summary_characters_full.csv)
    x:<rule>                              fixed points among the 18 (characters_fixed.csv)
    s:<P>:<rule>:<noise>[:lo|hi]          character-subset recall
    sp:<rule>:<noise>[:lo|hi]             character-subset P50
    v:<name>                              covariance check at rho = 0
    o:<rule>:<field>:<noise>:<stat>       Storkey over 200 presentation orders (subset size 18)
    """
    p = key.split(":")
    k = p[0]
    if k == "d":
        n = p[1]
        e = D.exp
        if n == "n_patterns":
            return str(len(D.chars))
        if n == "n_neurons":
            return str(D.chars.shape[1])
        if n in ("pix_train", "pix_test", "pix_valid"):
            return f"{int({'pix_train': D.chars, 'pix_test': D.test, 'pix_valid': D.valid}[n].sum()):,}"
        if n == "pix_total":
            return f"{D.chars.size:,}"
        if n in ("dens_train", "dens_test", "dens_valid"):
            return f2(100 * {'dens_train': D.chars, 'dens_test': D.test, 'dens_valid': D.valid}[n].mean())
        if n == "dens_min":
            return f1(100 * D.density.min())
        if n == "dens_max":
            return f1(100 * D.density.max())
        if n == "dens_max_pattern":
            return str(int(D.density.argmax()))
        if n == "dens_max_pixels":
            return str(int(D.chars[D.density.argmax()].sum()))
        if n == "overlap_mean":
            return f3(D.overlap.mean())
        if n == "overlap_sd":
            return f3(D.overlap.std(ddof=0))
        if n == "overlap_min":
            return f3(D.overlap.min())
        if n == "overlap_max":
            return f3(D.overlap.max())
        if n == "pearson_mean":
            return f3(D.pearson.mean())
        if n == "overlap_indep":
            return f3((2 * D.chars.mean() - 1) ** 2)
        if n == "all_on":
            return str(int((D.chars.sum(axis=0) == len(D.chars)).sum()))
        if n == "ham_mean":
            return f1(D.ham.mean())
        if n == "ham_min":
            return str(int(D.ham.min()))
        if n == "ham_max":
            return str(int(D.ham.max()))
        if n in ("closest_a", "closest_b"):
            iu = np.triu_indices(len(D.chars), 1)
            i = int(D.ham.argmin())
            return str(int(iu[0 if n == "closest_a" else 1][i]))
        if n == "test_flips":
            f = (D.test != D.chars).sum(axis=1)
            assert len(set(f)) == 1
            return str(int(f[0]))
        if n == "test_flip_pct":
            return f2(100 * (D.test != D.chars).mean())
        if n in ("valid_flip_mean", "valid_flip_min", "valid_flip_max", "valid_flip_pct"):
            f = (D.valid != D.chars).sum(axis=1)
            return {"valid_flip_mean": f2(f.mean()), "valid_flip_min": str(int(f.min())), "valid_flip_max": str(int(f.max())),
                    "valid_flip_pct": f2(100 * f.mean() / D.chars.shape[1])}[n]
        if n in ("test_nearest_own", "valid_nearest_own"):
            q = D.test if n == "test_nearest_own" else D.valid
            d = (q[:, None, :] != D.chars[None, :, :]).sum(axis=2)
            return str(int((d.argmin(axis=1) == np.arange(len(q))).sum()))
        if n == "pattern_pixels":
            return str(int(D.chars[int(p[2])].sum()))
    if k == "b":
        n, e = p[1], D.exp
        s = e[e.dataset.str.startswith(p[2])] if len(p) > 2 else e
        if n == "n_queries":
            return str(len(e))
        if n == "converged":
            return f"{int(s.converged.sum())}"
        if n == "recalled":
            # recalled: the final state is the query's own stored pattern
            return f"{int(((s.hamming_to_training == 0) & (s.closest_training_idx == s.pattern_idx)).sum())}"
        if n == "sweeps_mean":
            return f2(s.iterations.mean())
        if n == "sweeps_max":
            return str(int(e.iterations.max()))
        if n == "sweeps_total":
            return str(int(e.iterations.sum()))
        if n == "to_query_mean":
            return f2(s.hamming_to_query.mean())
        if n == "to_query_min":
            return str(int(s.hamming_to_query.min()))
        if n == "to_query_max":
            return str(int(s.hamming_to_query.max()))
        if n == "nearest_ham":
            assert e.hamming_to_training.nunique() == 1
            return str(int(e.hamming_to_training.iloc[0]))
        if n == "nearest_idx":
            assert e.closest_training_idx.nunique() == 1
            return str(int(e.closest_training_idx.iloc[0]))
        if n == "nearest_all":
            # every stored pattern at the nearest distance: the training queries are the clean
            # patterns, so a training row's distance to its query is the final state's distance
            # to that pattern (closest_training_idx keeps only the first of a tie)
            assert e.hamming_to_training.nunique() == 1
            t = e[e.dataset.str.startswith("Training")]
            idx = sorted(int(i) for i in t[t.hamming_to_query == e.hamming_to_training.iloc[0]].pattern_idx)
            return " and ".join(map(str, idx)) if len(idx) < 3 else ", ".join(map(str, idx[:-1])) + ", and " + str(idx[-1])
        if n == "energy":
            assert e.final_energy.nunique() == 1
            return f"{int(e.final_energy.iloc[0]):,}".replace("-", "$-$")
        if n == "final_pixels":
            return re.search(r"\| 1 \| 54 \| (\d+) \|", D.report).group(1)
        if n == "energy_rises":
            return re.search(r"rose in (\d+) of (\d+) recorded sweeps", D.report).group(1)
        if n == "energy_sweeps":
            return re.search(r"rose in (\d+) of (\d+) recorded sweeps", D.report).group(2)
        if n == "weight_min":
            return re.search(r"Weight range \| (-?\d+) to (-?\d+)", D.report).group(1).replace("-", "$-$")
        if n == "weight_max":
            return re.search(r"Weight range \| (-?\d+) to (-?\d+)", D.report).group(2)
    if k == "c":
        r = D.cap[(D.cap.family == p[1]) & (np.isclose(D.cap.level, float(p[2]))) & (D.cap.rule == p[3]) & (D.cap.noise == int(p[4]))]
        assert len(r) == 1, key
        r = r.iloc[0]
        field = p[5] if len(p) > 5 else "p50"
        return num(r[{"p50": "p50", "lo": "p50_lo", "hi": "p50_hi"}[field]])
    if k == "r":
        r = D.curves[(D.curves.family == p[1]) & (np.isclose(D.curves.level, float(p[2]))) & (D.curves.rule == p[3]) &
                     (D.curves.noise == int(p[4])) & (D.curves.P == int(p[5]))]
        assert len(r) == 1, key
        r = r.iloc[0]
        field = p[6] if len(p) > 6 else "recall"
        return f3(r[{"recall": "recall", "lo": "recall_lo", "hi": "recall_hi"}[field]])
    if k == "po":
        r = D.cap[(D.cap.family == p[1]) & (np.isclose(D.cap.level, float(p[2])))].iloc[0]
        return rnd(r.pattern_overlap_P40, 4) if len(p) > 3 and p[3] == "4" else f3(r.pattern_overlap_P40)
    if k == "f":
        r = D.full[(D.full.queryset == p[1]) & (D.full.rule == p[2])].iloc[0]
        field = {"mean": "mean_exact_of_18", "lo": "mean_lo", "hi": "mean_hi", "min": "min", "max": "max", "share": "share_all_18"}[p[3]]
        v = r[field]
        if p[3] in ("min", "max"):
            return str(int(v))
        return f2(v) if p[3] in ("mean", "lo", "hi") else f3(v)
    if k == "x":
        return str(int(D.fixed[D.fixed.rule == p[1]].fixed.iloc[0]))
    if k == "s":
        r = D.sub[(D.sub.P == int(p[1])) & (D.sub.rule == p[2]) & (D.sub.noise == int(p[3]))].iloc[0]
        field = p[4] if len(p) > 4 else "recall"
        return f3(r[{"recall": "recall", "lo": "recall_lo", "hi": "recall_hi"}[field]])
    if k == "sp":
        r = D.subp50[(D.subp50.rule == p[1]) & (D.subp50.noise == p[2])].iloc[0]
        field = p[3] if len(p) > 3 else "p50"
        return num(r[{"p50": "p50", "lo": "p50_lo", "hi": "p50_hi"}[field]])
    if k == "v":
        c = D.cov
        if p[1] == "max_abs_diff":
            return f3(c.difference.abs().max())
        if p[1] == "n_total":
            return str(len(c))
        if p[1] == "n_zero":
            return str(int(((c.diff_lo == 0) & (c.diff_hi == 0)).sum()))
        if p[1] == "n_informative":
            return str(int(((c.diff_lo != 0) | (c.diff_hi != 0)).sum()))
        if p[1] == "n_excl_zero":
            inf = c[(c.diff_lo != 0) | (c.diff_hi != 0)]
            return str(int(((inf.diff_lo > 0) | (inf.diff_hi < 0)).sum()))
        if p[1].startswith("excl_"):
            inf = c[(c.diff_lo > 0) | (c.diff_hi < 0)].iloc[0]
            return {"excl_P": str(int(inf.P)), "excl_noise": str(int(inf.noise)), "excl_diff": f3(inf.difference).replace("-", "$-$"),
                    "excl_lo": f3(inf.diff_lo).replace("-", "$-$"), "excl_hi": f3(inf.diff_hi).replace("-", "$-$")}[p[1]]
    if k == "g":
        # derived figures
        if p[1] == "hebb01_loss_pct":
            a = float(D.cap[(D.cap.family == "proto") & (D.cap.level == 0) & (D.cap.rule == "hebb") & (D.cap.noise == 10)].p50.iloc[0])
            b = float(D.cap[(D.cap.family == "proto") & (D.cap.level == 0) & (D.cap.rule == "hebb01") & (D.cap.noise == 10)].p50.iloc[0])
            return f"{round(100 * (a - b) / a):.0f}"
        if p[1] == "pinv_ge17_n":
            r = D.full_raw[(D.full_raw.queryset == "valid42") & (D.full_raw.rule == "pinv")]
            return str(int((r.exact >= 17).sum()))
        if p[1] == "pinv_ge17_pct":
            r = D.full_raw[(D.full_raw.queryset == "valid42") & (D.full_raw.rule == "pinv")]
            return f1(100 * (r.exact >= 17).mean())
        if p[1] == "pinv_all18_pct":
            r = D.full[(D.full.queryset == "valid42") & (D.full.rule == "pinv")].iloc[0]
            return f1(100 * r.share_all_18)
    if k == "o":
        r = D.subsets_raw[(D.subsets_raw.P == 18) & (D.subsets_raw.rule == p[1]) & (D.subsets_raw.noise == int(p[3]))]
        col = {"fixed": "fixed", "exact": "exact"}[p[2]]
        if p[2] == "fixed":
            r = D.subsets_raw[(D.subsets_raw.P == 18) & (D.subsets_raw.rule == p[1]) & (D.subsets_raw.noise == int(p[3]))]
        v = r[col]
        return {"min": str(int(v.min())), "max": str(int(v.max())), "mean": f2(v.mean())}[p[4]]
    if k == "pfx":
        # pseudoinverse networks in the synthetic grid where some stored pattern is not a fixed point
        f = pd.read_csv(RES / "synthetic_fixed.csv.gz")
        f = f[f.rule == "pinv"]
        return {"bad": str(int((f.fixed < f.P).sum())), "total": f"{len(f):,}",
                "minlevel": str(f[f.fixed < f.P].level.min())}[p[1]]
    if k == "cap":
        # recalls in the synthetic grid that hit the 50-sweep cap
        u = pd.read_csv(RES / "synthetic_recall.csv.gz", usecols=["family", "level", "P", "rule", "noise", "unconverged"])
        u = u[u.unconverged > 0]
        if p[1] == "n":
            return f"{int(u.unconverged.sum()):,}"
        if p[1] == "cells":
            return str(len(u.groupby(["family", "level", "P", "rule", "noise"])))
    raise KeyError(key)


KEY = re.compile(r"\\N\{([^}]*)\}")


def keys_in(text):
    return KEY.findall(text)


def compute():
    D = Data()
    keys = []
    for f in sorted((PAPER / "sections").glob("*.tex")):
        keys += keys_in(f.read_text())
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
    # Baseline table
    sets = ["Training", "Test", "Validation"]
    heads = {"Training": "Training", "Test": "Test file", "Validation": "20\\% file"}
    rows = [
        ("Queries converged", [f"{resolve(f'b:converged:{s}', D)} of 18" for s in sets]),
        ("Recalled own pattern exactly", [f"{resolve(f'b:recalled:{s}', D)} of 18" for s in sets]),
        ("Mean sweeps", [resolve(f"b:sweeps_mean:{s}", D) for s in sets]),
        ("Final state to query (bits), mean", [resolve(f"b:to_query_mean:{s}", D) for s in sets]),
        ("Final state to query (bits), range", [resolve(f"b:to_query_min:{s}", D) + "--" + resolve(f"b:to_query_max:{s}", D) for s in sets]),
        ("Final state to nearest stored (bits)", [resolve("b:nearest_ham", D)] * 3),
    ]
    # not the table's: Section III states in words that no final state equals a stored pattern, so a
    # run in which one did must stop the build (the recalled row itself is computed, above)
    assert int(D.exp.hamming_to_training.min()) > 0
    # nor is this: Section VIII states that no classical recall in the synthetic grid was still changing at
    # the 50-sweep cap (only the classical rules can add to cap:n: run_study.py records the modern network,
    # which stops after at most 10 updates, and nearest-pattern lookup as converged)
    assert resolve("cap:n", D) == "0", "some classical synthetic recalls were still changing at the 50-sweep cap; update Section VIII"
    out = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", "Measure & " + " & ".join(heads[s] for s in sets) + r" \\", r"\midrule"]
    out += [f"{a} & " + " & ".join(b) + r" \\" for a, b in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_baseline.tex").write_text("\n".join(out) + "\n")

    # Capacity table, 10% noise
    rules = ["hebb", "hebb01", "storkey", "cov", "pinv", "modern16", "modern64", "nn"]
    conds = [("proto", f"{0.1 * i:.1f}", f"$\\rho={0.1 * i:.1f}$") for i in range(10)] + [("biased", "0.65", "density 0.65"), ("biased", "0.8", "density 0.80")]
    short = {"hebb": "Hebbian", "hebb01": "Hebb.\\ 0/1", "storkey": "Storkey", "cov": "Covariance", "pinv": "Pseudoinv.",
             "modern16": "Modern 16", "modern64": "Modern 64", "nn": "Nearest"}
    hdr = ["Condition"] + [short[r] for r in rules]
    out = [r"\begin{tabular}{@{}l" + "c" * len(rules) + r"@{}}", r"\toprule", " & ".join(hdr) + r" \\", r"\midrule"]
    for fam, lv, lab in conds:
        cells = []
        for r in rules:
            v = resolve(f"c:{fam}:{lv}:{r}:10", D)
            if v.startswith("$"):
                cells.append(v)
            elif float(v) < 3.0:
                cells.append("2--3")
            else:
                lo, hi = resolve(f"c:{fam}:{lv}:{r}:10:lo", D), resolve(f"c:{fam}:{lv}:{r}:10:hi", D)
                cells.append(f"{v} ({lo}--{hi})")
        if lab == "density 0.65":
            out.append(r"\midrule")
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_capacity.tex").write_text("\n".join(out) + "\n")

    # Characters table: whole set, exact recall of 18
    qs = [("test", "Test file"), ("valid42", "20\\% file"), ("fresh10", "Fresh 10\\%"), ("fresh20", "Fresh 20\\%")]
    rules = ["hebb", "hebb01", "storkey", "cov", "pinv", "modern4", "modern16", "modern64", "nn"]
    out = [r"\begin{tabular}{@{}l" + "c" * len(qs) + r"c@{}}", r"\toprule", "Rule & " + " & ".join(q[1] for q in qs) + r" & Fixed \\", r"\midrule"]
    for r in rules:
        cells = []
        for q, _ in qs:
            mn, mx = resolve(f"f:{q}:{r}:min", D), resolve(f"f:{q}:{r}:max", D)
            mean = resolve(f"f:{q}:{r}:mean", D)
            cells.append(mean.rstrip("0").rstrip(".") if mn == mx else f"{mean} ({mn}--{mx})")
        out.append(f"{LABEL[r]} & " + " & ".join(cells) + f" & {resolve(f'x:{r}', D)}" + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_characters.tex").write_text("\n".join(out) + "\n")


def main():
    N, D = compute()
    write_numbers(N)
    write_tables(D)
    print(f"wrote {len(N)} numbers and three tables to {PAPER}")


if __name__ == "__main__":
    main()
