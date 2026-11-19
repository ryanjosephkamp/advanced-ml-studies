# make_numbers.py
# Writes the tables used by main.tex, and the numbers the text asks for with \N{key}, from the committed
# result files, so that those numbers are not typed by hand (the text still types a few counts and
# constants; see the reproducibility paragraph). The keys are described in resolve() below.
#
# Inputs (relative to the project folder): results/study/*.csv, results/study/diagnostics.json,
#   results/run_2026-11-11_01-25-31/comparative_analysis/comparison_table.csv
#
# Outputs (in paper/): numbers.tex, table_data.tex, table_baseline.tex, table_lda.tex, table_eegmat.tex
#
# Usage (from the project folder or from paper/):
#     python paper/make_numbers.py

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
ST = ROOT / "results" / "study"
RUN = ROOT / "results" / "run_2026-11-11_01-25-31" / "comparative_analysis"

VAR = ["eeg_mmi", "bci_civ_2a", "eeg_ma"]
VAR_LABEL = {"eeg_mmi": "EEG MMI", "bci_civ_2a": "BCI IV 2a", "bci_civ_2a_late": "BCI IV 2a (0.5--2.5 s)", "eeg_ma": "EEGMAT"}
CHANCE = {"eeg_mmi": 0.5, "bci_civ_2a": 0.25, "eeg_ma": 0.5}
FS_LABEL = {"F1_avg": "Band power, averaged", "F1_ch": "Band power, per channel", "F2_avg": "IMF features, averaged", "F2_ch": "IMF features, per channel",
            "F3_avg": "Hilbert band power, averaged", "F3_ch": "Hilbert band power, per channel"}


def rnd(x, places):
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(round(float(x), 9))).quantize(q, rounding=ROUND_HALF_UP))


def f3(x):
    return rnd(x, 3)


def s3(x):
    s = f"{float(x):+.3f}" if False else None
    v = Decimal(repr(round(float(x), 9))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    t = f"{v:+}"
    return t.replace("-", "$-$")


def mn3(x):
    return rnd(x, 3).replace("-", "$-$")


class Data:
    def __init__(self):
        self.sm = pd.read_csv(ST / "summary.csv")
        self.pt = pd.read_csv(ST / "paired_tests.csv")
        self.ss = pd.read_csv(ST / "split_summary.csv")
        self.sn = pd.read_csv(ST / "split_null.csv")
        self.cl = pd.read_csv(ST / "clustering.csv")
        self.stt = pd.read_csv(ST / "subject_tests.csv")
        self.es = pd.read_csv(ST / "eegmat_subsets.csv")
        self.sp = pd.read_csv(ST / "spectra.csv")
        self.bl = pd.read_csv(RUN / "comparison_table.csv")
        self.ml = pd.read_csv(ST / "march_2026_comparison_table.csv")
        self.dg = json.load(open(ST / "diagnostics.json"))["variants"]
        self.dgall = json.load(open(ST / "diagnostics.json"))


def sm_row(D, v, fs, clf, ev, norm):
    r = D.sm[(D.sm.variant == v) & (D.sm.feature_set == fs) & (D.sm.classifier == clf) & (D.sm.evaluation == ev) & (D.sm.norm == norm)]
    assert len(r) == 1, (v, fs, clf, ev, norm)
    return r.iloc[0]


def resolve(key, D):
    """Return the text for one key. Key forms, fields separated by colons:

    dg:<variant>:<field>                   data diagnostics (windows, subjects, dropped, artifact flags,
                                           windows_without_4_imfs_p_avg|p_ch)
    sm:<variant>:<fs>:<clf>:<eval>:<norm>[:lo|hi|bh]   study summary, balanced accuracy (bh: subjects above chance, 10,000 shuffles)
    pt:<variant>:<eval>:<norm>:<clf>:<a>:<b>[:lo|hi|ph]  paired test, difference
    pd:<variant>:<eval>:<norm>:<clf>:<a>:<b>:<higher|lower>  subjects whose balanced accuracy with a is higher (lower) than with b
    ps:<variant>:<F1|F2|F3>[:<norm>]       subjects higher per channel than averaged (held out, LDA; norm subject by default)
    ss:<variant>:<fs>:<clf>:<field>        original 70/30 window split, over seeds
    sn:<variant>:<fs>:<clf>:<lo|hi>        window-split null, middle 95%
    cl:<variant>:<fs>:<method>:<distance>:<field>   clustering
    bl:<ds>:<feature>:<sim>:<field>        corrected baseline (field: f1, bacc, ...)
    blr:<ds>:<field>:<min|max>             range over the corrected baseline's classification rows
    blm:<ds>:<field>:<min|max>             March values (weighted F1)
    sp:<variant>:<channel>:<freq>:<field>  spectra
    es:<subset>:<norm>[:lo|hi|diff|n]      EEGMAT subsets
    c:<name>                               derived counts
    """
    p = key.split(":")
    k = p[0]
    if k == "dg":
        v = D.dg[p[1]]
        f = p[2]
        if f in ("n_windows", "n_subjects", "dropped_flat_windows", "artifact_flagged_windows"):
            return f"{int(v[f]):,}"
        if f == "sfreq":
            return str(int(v["sfreq"]))
        if f.startswith("class"):
            return f"{int(v['class_counts'][f.split('_')[1]]):,}"
        if f == "dropped_files":
            return str(len(v["dropped_files_sampling_rate"]))
        if f.startswith("windows_without_4_imfs"):
            return f"{int(v[f]):,}"
    if k == "sm":
        r = sm_row(D, p[1], p[2], p[3], p[4], p[5])
        f = p[6] if len(p) > 6 else "mean"
        if f == "bh":
            t = D.stt[(D.stt.variant == p[1]) & (D.stt.feature_set == p[2]) & (D.stt.classifier == p[3]) & (D.stt.evaluation == p[4]) & (D.stt.norm == p[5])]
            return str(int(t.n_subjects_above_chance_bh.iloc[0]))
        if f == "n":
            return str(int(r.n_subjects))
        return f3(r[{"mean": "subject_bacc_mean", "lo": "subject_bacc_ci_low", "hi": "subject_bacc_ci_high"}[f]])
    if k == "pt":
        r = D.pt[(D.pt.variant == p[1]) & (D.pt.evaluation == p[2]) & (D.pt.norm == p[3]) & (D.pt.classifier == p[4]) & (D.pt.a == p[5]) & (D.pt.b == p[6])]
        assert len(r) == 1, key
        r = r.iloc[0]
        f = p[7] if len(p) > 7 else "mean"
        if f == "ph":
            v = r.p_holm
            if v >= 0.9995:
                return "1"
            if v >= 0.1:
                return rnd(v, 2)
            if v >= 0.001:
                return rnd(v, 3)
            m, e = f"{v:.1e}".split("e")
            return f"${m}\\times 10^{{{int(e)}}}$"
        return s3(r[{"mean": "mean_diff", "lo": "diff_ci_low", "hi": "diff_ci_high"}[f]])
    if k == "ss":
        r = D.ss[(D.ss.variant == p[1]) & (D.ss.feature_set == p[2]) & (D.ss.classifier == p[3])]
        assert len(r) == 1, key
        r = r.iloc[0]
        return f3(r[{"mean": "balanced_accuracy_mean", "min": "balanced_accuracy_min", "max": "balanced_accuracy_max", "seed42": "balanced_accuracy_seed42"}[p[4]]])
    if k == "sn":
        r = D.sn[(D.sn.variant == p[1]) & (D.sn.feature_set == p[2]) & (D.sn.classifier == p[3])].balanced_accuracy
        return f3(np.percentile(r, 2.5 if p[4] == "lo" else 97.5))
    if k == "cl":
        r = D.cl[(D.cl.variant == p[1]) & (D.cl.feature_set == p[2]) & (D.cl.method == p[3])]
        if p[3] != "kmeans":
            r = r[r.distance == p[4]]
        r = r.iloc[:1] if p[3] == "kmeans" else r
        assert len(r) == 1, key
        return mn3(r.iloc[0][p[5]])
    if k == "bl":
        ds, feat, sim, f = p[1], p[2], p[3], p[4]
        r = D.bl[(D.bl.dataset_key == ds) & (D.bl.feature == feat) & (D.bl.similarity == sim) & (D.bl.task == ("clustering" if f in ("silhouette", "ari", "nmi") else "classification"))]
        assert len(r) == 1, key
        return mn3(r.iloc[0][{"f1": "f1_score", "bacc": "balanced_accuracy"}.get(f, f)])
    if k == "blr":
        ds, f, w = p[1], p[2], p[3]
        r = D.bl[(D.bl.dataset_key == ds) & (D.bl.task == "classification")]
        col = {"f1": "f1_score", "bacc": "balanced_accuracy"}[f]
        return mn3(r[col].min() if w == "min" else r[col].max())
    if k == "blm":
        ds, f, w = p[1], p[2], p[3]
        r = D.ml[(D.ml.dataset_key == ds) & (D.ml.task == "classification")]
        col = {"f1": "f1_score"}[f]
        return mn3(r[col].min() if w == "min" else r[col].max())
    if k == "mls":
        r = D.ml[(D.ml.dataset_key == p[1]) & (D.ml.feature == p[2]) & (D.ml.similarity == p[3]) & (D.ml.task == "clustering")]
        assert len(r) == 1, key
        return mn3(r.iloc[0].silhouette)
    if k == "blc":
        ds, w = p[1], p[2]
        r = D.bl[D.bl.task == "clustering"] if ds == "all" else D.bl[(D.bl.dataset_key == ds) & (D.bl.task == "clustering")]
        return mn3(r.ari.min() if w == "ari_min" else r.ari.max())
    if k == "sp":
        r = D.sp[(D.sp.variant == p[1]) & (D.sp.channel == p[2]) & (np.isclose(D.sp.freq_hz, float(p[3])))]
        assert len(r) == 1, key
        r = r.iloc[0]
        return rnd(r[{"mean": "mean_db", "lo": "ci_low_db", "hi": "ci_high_db"}[p[4]]], 2).replace("-", "$-$")
    if k == "es":
        r = D.es[(D.es.subset == p[1]) & (D.es.norm == p[2])]
        assert len(r) == 1, key
        r = r.iloc[0]
        f = p[3] if len(p) > 3 else "mean"
        if f == "n":
            return str(int(r.n_features))
        if f == "diff":
            return mn3(r.diff_from_full_mean)
        if f in ("difflo", "diffhi"):
            return mn3(r["diff_from_full_ci_low" if f == "difflo" else "diff_from_full_ci_high"])
        return f3(r[{"mean": "subject_bacc_mean", "lo": "subject_bacc_ci_low", "hi": "subject_bacc_ci_high"}[f]])
    if k == "mag":
        # magnitude of any signed key, for sentences that already give the direction ("a loss of")
        return resolve(key[4:], D).replace("$-$", "").lstrip("+")
    if k == "snp":
        # window-split null pooled over its three configurations (null_* columns of split_summary.csv)
        r = D.ss[D.ss.variant == p[1]].iloc[0]
        m = "f1_weighted" if len(p) > 3 and p[3] == "f1" else "balanced_accuracy"
        return f3(r[f"null_{m}_{'p2_5' if p[2] == 'lo' else 'p97_5'}"])
    if k == "ar":
        # always predicting rest on EEGMAT: accuracy is the rest share p, weighted F1 is 2p^2 / (1 + p)
        if p[1] == "corrected":
            # support of each class in the corrected baseline's test set (70/30 split, seed 42)
            run = sorted((ROOT / "results").glob("run_*"))[-1]
            rep = (run / "eeg_ma" / "inst_freq_euclidean_classification" / "classification_report.txt").read_text()
            rest = int(re.search(r"resting\s+\S+\s+\S+\s+\S+\s+(\d+)", rep).group(1))
            arith = int(re.search(r"mental_arithmetic\s+\S+\s+\S+\s+\S+\s+(\d+)", rep).group(1))
            share = rest / (rest + arith)
        else:
            # March windows: the corrected counts plus the dropped tail windows, one per recording and class
            v = D.dg["eeg_ma"]
            tails = int(v["dropped_flat_windows"]) / 2
            rest = int(v["class_counts"]["0"]) + tails
            arith = int(v["class_counts"]["1"]) + tails
            share = rest / (rest + arith)
        return f3(share if p[2] == "acc" else 2 * share ** 2 / (1 + share))
    if k == "ps":
        # subjects whose per-subject accuracy is higher with per-channel features than averaged (held out, LDA;
        # standardized within each subject unless a norm is given)
        v, fs = p[1], p[2]
        norm = p[3] if len(p) > 3 else "subject"
        ps = pd.read_csv(ST / "per_subject.csv")
        base = ps[(ps.variant == v) & (ps.classifier == "lda") & (ps.evaluation == "heldout") & (ps.norm == norm)]
        a = base[base.feature_set == fs + "_ch"].set_index("subject").balanced_accuracy
        b = base[base.feature_set == fs + "_avg"].set_index("subject").balanced_accuracy
        return str(int((a > b.loc[a.index]).sum()))
    if k == "pd":
        # direction count for a paired comparison: subjects whose accuracy with feature set a is higher (or lower) than with b
        v, ev, norm, clf, fa, fb, w = p[1:8]
        ps = pd.read_csv(ST / "per_subject.csv")
        base = ps[(ps.variant == v) & (ps.classifier == clf) & (ps.evaluation == ev) & (ps.norm == norm)]
        a = base[base.feature_set == fa].set_index("subject").balanced_accuracy
        b = base[base.feature_set == fb].set_index("subject").balanced_accuracy.loc[a.index]
        return str(int((a > b).sum() if w == "higher" else (a < b).sum()))
    if k == "c":
        n = p[1]
        t = D.pt[(D.pt.a.str.startswith(("F2", "F3"))) & (D.pt.b.str.startswith("F1")) & (D.pt.a.str[3:] == D.pt.b.str[3:])]
        if n == "emd_total":
            return str(len(t))
        if n == "emd_above":
            return str(int((t.diff_ci_low > 0).sum()))
        if n == "emd_below":
            return str(int((t.diff_ci_high < 0).sum()))
        if n == "emd_holm":
            return str(int((t.p_holm < 0.05).sum()))
        if n == "emd_above_max":
            return mn3(t[t.diff_ci_low > 0].mean_diff.max())
        if n == "lda_ch_heldout_total":
            u = t[(t.classifier == "lda") & (t.evaluation == "heldout") & (t.a.str.endswith("_ch"))]
            return str(len(u))
        if n == "lda_ch_heldout_emd_better":
            u = t[(t.classifier == "lda") & (t.evaluation == "heldout") & (t.a.str.endswith("_ch"))]
            return str(int((u.diff_ci_low > 0).sum()))
        if n == "lda_ch_heldout_bp_better":
            u = t[(t.classifier == "lda") & (t.evaluation == "heldout") & (t.a.str.endswith("_ch"))]
            return str(int((u.diff_ci_high < 0).sum()))
        if n in ("hilb_min", "hilb_max", "avgcost_min", "avgcost_max", "imfbp_min", "imfbp_max",
                 "avgcost_none_min", "avgcost_none_max", "imfbp_none_min", "imfbp_none_max"):
            # held out, LDA, the three main data sets; standardized within each subject unless the name says "none"
            h = D.pt[(D.pt.classifier == "lda") & (D.pt.evaluation == "heldout") & (D.pt.variant.isin(VAR))]
            norm = "none" if "_none_" in n else "subject"
            if n.startswith("hilb"):
                x = h[(h.a == "F3_ch") & (h.b == "F1_ch")].mean_diff
            elif n.startswith("imfbp"):
                x = h[(h.a == "F2_ch") & (h.b == "F1_ch") & (h.norm == norm)].mean_diff
            else:
                x = h[(h.a.str.endswith("_ch")) & (h.b.str.endswith("_avg")) & (h.a.str[:2] == h.b.str[:2]) & (h.norm == norm)].mean_diff
            return mn3(x.min() if n.endswith("min") else x.max())
        if n == "eegmat_rest_pct":
            # share of EEGMAT windows at rest, in percent
            cc = D.dg["eeg_ma"]["class_counts"]
            return rnd(100 * int(cc["0"]) / (int(cc["0"]) + int(cc["1"])), 0)
        if n == "f0_rows":
            s = D.sm[(D.sm.feature_set.str.startswith("F0")) & (D.sm.variant != "bci_civ_2a_late") & (D.sm.variant.isin(["eeg_mmi", "bci_civ_2a", "eeg_ma"]))]
            return str(len(s))
        if n.startswith("f0_range"):
            v = p[2]
            s = D.sm[(D.sm.feature_set.str.startswith("F0")) & (D.sm.variant == v)]
            return f3(s.subject_bacc_mean.min()) + "--" + f3(s.subject_bacc_mean.max())
        if n == "ari_labels_min":
            return mn3(D.cl.ari_labels.min())
        if n == "ari_labels_max":
            return mn3(D.cl.ari_labels.max())
        if n == "ari_subj_bci_max":
            return f3(D.cl[(D.cl.variant == "bci_civ_2a")].ari_subjects.max())
        if n == "avg_link_total":
            return str(int((D.cl.method == "average_linkage").sum()))
        if n == "avg_link_95":
            c = D.cl[D.cl.method == "average_linkage"]
            return str(int((c.largest_cluster_share >= 0.95).sum()))
        if n == "perm_subject_shuffles":
            return f"{int(D.stt.n_perm.iloc[0]):,}"
        if n == "split_seeds":
            return str(int(D.ss.n_seeds.iloc[0]))
        if n == "total_seconds":
            return rnd(D.dgall["full_run"]["all_stages_seconds"], 1)
        if n in ("silmax_small", "silmax_n"):
            c = D.cl[(D.cl.method == "average_linkage") & (D.cl.largest_cluster_share >= 0.95)]
            r = c.loc[c.silhouette.idxmax()]
            return f"{int(round(r.n_windows * (1 - r.largest_cluster_share))):,}" if n == "silmax_small" else f"{int(r.n_windows):,}"
        if n == "ari_bci_kmeans_cls_max":
            return f3(D.cl[(D.cl.variant == "bci_civ_2a") & (D.cl.method == "kmeans")].ari_labels.max())
        if n in ("mmi_win_min", "mmi_win_max"):
            ps = pd.read_csv(ST / "per_subject.csv")
            w = ps[(ps.variant == "eeg_mmi") & (ps.evaluation == "heldout")].groupby("subject").n_windows.first()
            return str(int(w.min() if n.endswith("min") else w.max()))
        if n == "total_minutes":
            return rnd(D.dgall["full_run"]["all_stages_seconds"] / 60, 0)
        if n in ("sil95_min", "sil95_max"):
            c = D.cl[(D.cl.method == "average_linkage") & (D.cl.largest_cluster_share >= 0.95)].silhouette
            return f3(c.min() if n.endswith("min") else c.max())
        if n in ("f0_n", "f0_excl", "f0_excl_max", "f0_expected"):
            # F0 configurations on motor imagery, three subject-based evaluations: intervals that exclude chance
            s = D.sm[(D.sm.feature_set.str.startswith("F0")) & (D.sm.variant.isin(["eeg_mmi", "bci_civ_2a"])) & (D.sm.evaluation.isin(["within", "heldout"]))]
            ch = s.variant.map({"eeg_mmi": 0.5, "bci_civ_2a": 0.25})
            ex = (s.subject_bacc_ci_low > ch) | (s.subject_bacc_ci_high < ch)
            return {"f0_n": str(len(s)), "f0_excl": str(int(ex.sum())), "f0_expected": rnd(0.05 * len(s), 1),
                    "f0_excl_max": f3((s[ex].subject_bacc_mean - ch[ex]).abs().max())}[n]
    raise KeyError(key)


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
    ch = {"eeg_mmi": "C3, Cz, C4", "bci_civ_2a": "C3, Cz, C4", "eeg_ma": "Fp1, Fp2, F3, F4, Fz"}
    hz = {"eeg_mmi": "160", "bci_civ_2a": "250", "eeg_ma": "500"}
    cls = {"eeg_mmi": "2 (left, right)", "bci_civ_2a": "4", "eeg_ma": "2 (rest, arithmetic)"}
    out = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule", r"Data set & Subjects & Windows & Classes & Channels & Rate (Hz) \\", r"\midrule"]
    for v in VAR:
        out.append(f"{VAR_LABEL[v]} & {resolve(f'dg:{v}:n_subjects', D)} & {resolve(f'dg:{v}:n_windows', D)} & {cls[v]} & {ch[v]} & {hz[v]} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_data.tex").write_text("\n".join(out) + "\n")

    # Baseline: weighted F1 March and corrected, balanced accuracy range
    out = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Data set & W. F1, March & W. F1, corrected & Bal. acc., corrected \\", r"\midrule"]
    for v in VAR:
        a = f"{resolve(f'blm:{v}:f1:min', D)}--{resolve(f'blm:{v}:f1:max', D)}"
        b = f"{resolve(f'blr:{v}:f1:min', D)}--{resolve(f'blr:{v}:f1:max', D)}"
        c = f"{resolve(f'blr:{v}:bacc:min', D)}--{resolve(f'blr:{v}:bacc:max', D)}"
        out.append(f"{VAR_LABEL[v]} & {a} & {b} & {c} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_baseline.tex").write_text("\n".join(out) + "\n")

    # LDA: within, held out as is, held out standardized
    out = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule", r"Data set & Features & Within subject & Held out, as is & Held out, standardized \\", r"\midrule"]
    for v in VAR:
        first = True
        for fs in ["F1_avg", "F1_ch", "F2_ch", "F3_ch"]:
            cells = []
            for ev, norm in [("within", "none"), ("heldout", "none"), ("heldout", "subject")]:
                cells.append(f"{resolve(f'sm:{v}:{fs}:lda:{ev}:{norm}', D)} [{resolve(f'sm:{v}:{fs}:lda:{ev}:{norm}:lo', D)}, {resolve(f'sm:{v}:{fs}:lda:{ev}:{norm}:hi', D)}]")
            lab = f"{VAR_LABEL[v]} ({resolve(f'sm:{v}:{fs}:lda:within:none:n', D)})" if first else ""
            first = False
            out.append(f"{lab} & {FS_LABEL[fs]} & " + " & ".join(cells) + r" \\")
        if v != VAR[-1]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_lda.tex").write_text("\n".join(out) + "\n")

    # EEGMAT subsets
    subs = [("all channels, all bands", "All channels and bands"), ("without Fp1 and Fp2", "Without Fp1, Fp2"), ("Fp1 and Fp2 only", "Fp1, Fp2 only"),
            ("alpha band only", "Alpha band only"), ("without 1-4 and 30-45 Hz", "Without 1--4, 30--45 Hz")]
    out = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Features & No. & Standardized & As is \\", r"\midrule"]
    for s, lab in subs:
        cells = [f"{resolve(f'es:{s}:{n}', D)} [{resolve(f'es:{s}:{n}:lo', D)}, {resolve(f'es:{s}:{n}:hi', D)}]" for n in ("subject", "none")]
        if s == "all channels, all bands":
            cells = [f"{resolve(f'sm:eeg_ma:F1_ch:lda:heldout:{n}', D)} [{resolve(f'sm:eeg_ma:F1_ch:lda:heldout:{n}:lo', D)}, {resolve(f'sm:eeg_ma:F1_ch:lda:heldout:{n}:hi', D)}]" for n in ("subject", "none")]
        out.append(f"{lab} & {resolve(f'es:{s}:none:n', D)} & {cells[0]} & {cells[1]} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "table_eegmat.tex").write_text("\n".join(out) + "\n")


def main():
    N, D = compute()
    write_numbers(N)
    write_tables(D)
    print(f"wrote {len(N)} numbers and four tables to {PAPER}")


if __name__ == "__main__":
    main()
