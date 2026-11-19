"""Capacity study: exact recall against load and pattern overlap, for nine recall rules.

Run from the repository root:

    python study/run_study.py [--raw PATH.csv.gz]

It writes aggregate CSVs to study/results/ (see analyze.py for intervals and figures); the two
largest, synthetic_recall and synthetic_fixed, are gzip-compressed. In each recall row, `exact`
counts queries recalled exactly out of P (or 18). With --raw, it also writes one row per recall to
a gzip CSV.

To add loads to the synthetic grid without rerunning the rest, run only those loads into another
folder and merge; every seed is keyed by load, so the result is the same as a full run:

    python study/run_study.py --loads 3,5,7,8 --synthetic-only --out /tmp/extra
    python study/run_study.py --merge-from /tmp/extra

Every pattern set, noise draw, and update order comes from numpy.random.SeedSequence(SEED,
spawn_key=...), keyed as described below, so any single recall can be reproduced on its own. NumPy
is limited to one thread.

Synthetic grid
    Families: prototype (overlap rho = 0, 0.1, ..., 0.9, density 0.5) and biased (independent
    bits at density 0.65 and 0.80). Loads P = 2 to 9 in steps of 1, then 12, 15, 18, 24, 30, 40. 50 pattern sets per family and level; the
    load-P network stores the first P patterns of a 40-pattern set. Queries flip 14 or 29 of the
    144 bits (10% and 20%). Every rule sees the same noisy queries, and every rule that updates one
    neuron at a time sees the same update orders.
The 18 course characters
    The whole set, with the test file (13 flipped bits per pattern), the committed 20% validation
    file, and fresh 10% and 20% noise, each over 200 draws of update order (for the rules that have
    one) and of noise (for fresh queries).
    Random subsets of 2 to 18 characters, 200 per size, with fresh 10% and 20% noise.

Seed keys: patterns (0, family, level, set); synthetic noise (1, family, level, P, set, query,
noise); synthetic update order (2, ...same); character draws (3 and 4, query set, draw, query);
subsets (5, P, subset), subset noise (6, P, subset, query, noise), subset order (7, ...same).
"""
import os

# One BLAS thread, even if the environment asks for more: the pseudoinverse results at high overlap
# change with the thread count, and the committed results were made on one thread.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import gzip
import io
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hopfield_rules as R  # noqa: E402

SEED = 20260202
N = R.N
FAMILIES = [("proto", 0, [round(0.1 * k, 1) for k in range(10)]), ("biased", 1, [0.65, 0.80])]
LOADS = [2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 24, 30, 40]
N_SETS = 50
NOISES = [(10, 14), (20, 29)]  # (percent of bits, bits flipped)
BETAS = [4, 16, 64]
RULES = ["hebb", "hebb01", "storkey", "pinv", "cov"] + [f"modern{b}" for b in BETAS] + ["nn"]
N_CHAR_DRAWS = 200
N_SUBSETS = 200
MAX_MINUTES = 25.0

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "study" / "results"
GZIPPED = {"synthetic_recall", "synthetic_fixed"}
SYNTHETIC = ("synthetic_recall", "synthetic_fixed", "synthetic_patterns")
FAM_INDEX = {fam: i for fam, i, _ in FAMILIES}
LEVEL_INDEX = {(fam, lv): k for fam, _, lvs in FAMILIES for k, lv in enumerate(lvs)}


def final_name(name):
    return f"{name}.csv.gz" if name in GZIPPED else f"{name}.csv"


class CsvOut:
    """CSV writer; gzip output has no timestamp or file name in its header, so reruns match byte for byte."""

    def __init__(self, path, gz):
        self.raw = open(path, "wb")
        if gz:
            self.gz = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0, compresslevel=9)
            self.text = io.TextIOWrapper(self.gz, encoding="utf-8", newline="")
        else:
            self.gz = None
            self.text = io.TextIOWrapper(self.raw, encoding="utf-8", newline="")
        self.writer = csv.writer(self.text)

    def close(self):
        self.text.flush()
        if self.gz is not None:
            self.text.detach()
            self.gz.close()
        else:
            self.text.detach()
        self.raw.close()


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        return header, list(r)


def sort_key(name, header):
    """Row order of a full run: family, level, set, load, then noise and rule."""
    col = {c: i for i, c in enumerate(header)}
    rule_index = {r: i for i, r in enumerate(RULES)}

    def key(row):
        fam = row[col["family"]]
        k = [FAM_INDEX[fam], LEVEL_INDEX[(fam, float(row[col["level"]]))], int(row[col["set"]]),
             LOADS.index(int(row[col["P"]]))]
        if "noise" in col:
            k.append([n for n, _ in NOISES].index(int(row[col["noise"]])))
        if "rule" in col:
            k.append(rule_index[row[col["rule"]]])
        return k
    return key


def merge_from(extra):
    """Merge synthetic rows from a partial run (other loads) into study/results."""
    for name in SYNTHETIC:
        cur = next(p for p in (OUT / f"{name}.csv.gz", OUT / f"{name}.csv") if p.exists())
        h1, rows = read_rows(cur)
        h2, extra_rows = read_rows(Path(extra) / final_name(name))
        assert h1 == h2, f"{name}: column mismatch"
        key = sort_key(name, h1)
        have = {tuple(key(r)) for r in rows}
        added = [r for r in extra_rows if tuple(key(r)) not in have]
        merged = sorted(rows + added, key=key)
        tmp = OUT / f".{name}.tmp"
        out = CsvOut(tmp, name in GZIPPED)
        out.writer.writerow(h1)
        out.writer.writerows(merged)
        out.close()
        tmp.replace(OUT / final_name(name))
        if cur != OUT / final_name(name):
            cur.unlink()
        print(f"{name}: {len(rows)} rows kept, {len(added)} added, {len(merged)} total")


def rng_for(*key):
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=tuple(int(k) for k in key)))


def flip(xi, n_flips, rng):
    q = xi.astype(np.float64).copy()
    q[rng.choice(N, n_flips, replace=False)] *= -1
    return q


def build(S, storkey_w=None):
    """Weights and biases for every rule on pattern set S (P x N, +/-1)."""
    nets = {}
    for name in ("hebb", "hebb01", "pinv", "cov"):
        nets[name] = R.ITERATIVE_RULES[name](S)
    nets["storkey"] = (storkey_w, np.zeros(N)) if storkey_w is not None else R.storkey(S)
    return nets


def run_query(nets, S, X, k, q, order_key):
    """Recall query q (target S[k]) under every rule. Returns {rule: (exact, overlap, sweeps, conv)}."""
    target = S[k].astype(np.float64)
    out = {}
    for name in ("hebb", "hebb01", "storkey", "pinv", "cov"):
        W, b = nets[name]
        s, sweeps, conv = R.recall(W, b, q, rng_for(*order_key))
        out[name] = (bool(np.array_equal(s, target)), float(s @ target) / N, sweeps, conv)
    x = q / np.sqrt(N)
    for beta in BETAS:
        s = R.modern_update(X, x, beta)
        out[f"modern{beta}"] = (bool(np.array_equal(s, target)), float(s @ target) / N, 1, True)
    s = R.nearest(S, q)
    out["nn"] = (bool(np.array_equal(s, target)), float(s @ target) / N, 0, True)
    return out


def fixed_counts(nets, S, X):
    counts = {}
    for name in ("hebb", "hebb01", "storkey", "pinv", "cov"):
        W, b = nets[name]
        counts[name] = sum(R.is_fixed_point(W, b, xi) for xi in S.astype(np.float64))
    for beta in BETAS:
        counts[f"modern{beta}"] = sum(
            np.array_equal(R.modern_update(X, xi / np.sqrt(N), beta), xi) for xi in S.astype(np.float64))
    counts["nn"] = sum(np.array_equal(R.nearest(S, xi), xi) for xi in S.astype(np.float64))
    return counts


class Progress:
    def __init__(self, total_units):
        self.total = total_units
        self.done = 0
        self.t0 = time.time()

    def step(self, units, label):
        self.done += units
        el = time.time() - self.t0
        proj = el / max(self.done, 1) * self.total
        print(f"[{el / 60:5.1f} min] {label}: {100 * self.done / self.total:5.1f}% done, "
              f"projected total {proj / 60:.1f} min", flush=True)
        if self.done >= 0.05 * self.total and proj > MAX_MINUTES * 60:
            print(f"Projected run time {proj / 60:.1f} min is over {MAX_MINUTES:.0f} min; stopping.", flush=True)
            sys.exit(3)


def load_characters():
    def read(path):
        a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
        return (2 * a - 1).astype(np.int8)
    return (read(ROOT / "data" / "characters.csv"), read(ROOT / "data" / "test.csv"),
            read(ROOT / "data" / "valid_datasets" / "valid_dataset_42_20.0" / "valid_42_20.0.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", help="optional gzip CSV for one row per recall")
    ap.add_argument("--loads", help="comma-separated subset of the synthetic loads to run (default: all)")
    ap.add_argument("--synthetic-only", action="store_true", help="skip the course characters")
    ap.add_argument("--out", help="output folder (default: study/results)")
    ap.add_argument("--merge-from", help="merge a partial run's synthetic CSVs into study/results, then stop")
    args = ap.parse_args()
    if args.merge_from:
        merge_from(args.merge_from)
        return
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    loads = [int(p) for p in args.loads.split(",")] if args.loads else list(LOADS)
    assert set(loads) <= set(LOADS), f"loads must come from {LOADS}"
    loads = [p for p in LOADS if p in loads]

    synth_queries = sum(len(lv) for _, _, lv in FAMILIES) * N_SETS * sum(loads) * len(NOISES)
    char_queries = 0 if args.synthetic_only else \
        4 * N_CHAR_DRAWS * 18 + N_SUBSETS * sum(range(2, 19)) * len(NOISES)
    prog = Progress(synth_queries + char_queries)

    names = SYNTHETIC if args.synthetic_only else SYNTHETIC + (
        "characters_full", "characters_by_query", "characters_fixed", "characters_subsets")
    tmp = {name: out_dir / f".{name}.tmp" for name in names}
    files = {k: CsvOut(v, k in GZIPPED) for k, v in tmp.items()}
    w = {k: f.writer for k, f in files.items()}
    w["synthetic_recall"].writerow(["family", "level", "P", "set", "rule", "noise", "exact",
                                    "overlap", "sweeps", "unconverged"])
    w["synthetic_fixed"].writerow(["family", "level", "P", "set", "rule", "fixed"])
    w["synthetic_patterns"].writerow(["family", "level", "P", "set", "density", "mean_overlap", "min_hamming"])
    raw = gzip.open(args.raw, "wt", newline="") if args.raw else None
    rw = csv.writer(raw) if raw else None
    if rw:
        rw.writerow(["source", "family", "level", "P", "set", "query", "rule", "noise", "exact", "overlap", "sweeps"])

    # ---------------- synthetic grid ----------------
    for fam, fam_id, levels in FAMILIES:
        for li, level in enumerate(levels):
            for st in range(N_SETS):
                prng = rng_for(0, fam_id, li, st)
                full = R.prototype_family(max(LOADS), level, prng) if fam == "proto" else \
                    R.biased_family(max(LOADS), level, prng)
                stork = R.storkey_snapshots(full, set(loads))
                for P in loads:
                    S = full[:P]
                    d, mo, mh = R.pattern_stats(S)
                    w["synthetic_patterns"].writerow([fam, level, P, st, f"{d:.4f}", f"{mo:.4f}", mh])
                    nets = build(S, stork[P])
                    X = S.astype(np.float64) / np.sqrt(N)
                    for rule, c in fixed_counts(nets, S, X).items():
                        w["synthetic_fixed"].writerow([fam, level, P, st, rule, c])
                    for ni, (frac, nflip) in enumerate(NOISES):
                        agg = {r: [0, 0.0, 0, 0] for r in RULES}
                        for k in range(P):
                            q = flip(S[k], nflip, rng_for(1, fam_id, li, P, st, k, ni))
                            res = run_query(nets, S, X, k, q, (2, fam_id, li, P, st, k, ni))
                            for r, (ex, ov, sw, conv) in res.items():
                                a = agg[r]
                                a[0] += ex; a[1] += ov; a[2] += sw; a[3] += (not conv)
                                if rw:
                                    rw.writerow(["synthetic", fam, level, P, st, k, r, frac, int(ex), f"{ov:.4f}", sw])
                        for r in RULES:
                            ex, ov, sw, unc = agg[r]
                            w["synthetic_recall"].writerow([fam, level, P, st, r, frac, ex,
                                                            f"{ov / P:.4f}", f"{sw / P:.2f}", unc])
            prog.step(N_SETS * sum(loads) * len(NOISES), f"{fam} {level}")

    if args.synthetic_only:
        finish(files, tmp, out_dir, prog, raw)
        return

    # ---------------- the 18 characters ----------------
    tr, te, va = load_characters()
    nets = build(tr)
    X = tr.astype(np.float64) / np.sqrt(N)
    w["characters_fixed"].writerow(["rule", "fixed", "n"])
    for rule, c in fixed_counts(nets, tr, X).items():
        w["characters_fixed"].writerow([rule, c, 18])
    w["characters_full"].writerow(["queryset", "draw", "rule", "n", "exact", "overlap"])
    w["characters_by_query"].writerow(["queryset", "rule", "query", "successes", "draws"])
    qsets = [("test", 0, te, None), ("valid42", 1, va, None), ("fresh10", 2, None, 14), ("fresh20", 3, None, 29)]
    for qs, qid, fixed_q, nflip in qsets:
        per_query = {r: np.zeros(18, dtype=int) for r in RULES}
        for dr in range(N_CHAR_DRAWS):
            agg = {r: [0, 0.0] for r in RULES}
            for k in range(18):
                q = fixed_q[k].astype(np.float64) if fixed_q is not None else flip(tr[k], nflip, rng_for(4, qid, dr, k))
                res = run_query(nets, tr, X, k, q, (3, qid, dr, k))
                for r, (ex, ov, sw, conv) in res.items():
                    agg[r][0] += ex; agg[r][1] += ov
                    per_query[r][k] += ex
                    if rw:
                        rw.writerow(["characters", qs, "", 18, dr, k, r, "", int(ex), f"{ov:.4f}", sw])
            for r in RULES:
                w["characters_full"].writerow([qs, dr, r, 18, agg[r][0], f"{agg[r][1] / 18:.4f}"])
        for r in RULES:
            for k in range(18):
                w["characters_by_query"].writerow([qs, r, k, int(per_query[r][k]), N_CHAR_DRAWS])
        prog.step(N_CHAR_DRAWS * 18, f"characters {qs}")

    w["characters_subsets"].writerow(["P", "subset", "rule", "noise", "exact", "overlap", "fixed",
                                      "subset_overlap", "subset_density"])
    for P in range(2, 19):
        for u in range(N_SUBSETS):
            idx = rng_for(5, P, u).choice(18, P, replace=False)
            S = tr[idx]
            d, mo, _ = R.pattern_stats(S)
            nets = build(S)
            X = S.astype(np.float64) / np.sqrt(N)
            fixed = fixed_counts(nets, S, X)
            for ni, (frac, nflip) in enumerate(NOISES):
                agg = {r: [0, 0.0] for r in RULES}
                for k in range(P):
                    q = flip(S[k], nflip, rng_for(6, P, u, k, ni))
                    res = run_query(nets, S, X, k, q, (7, P, u, k, ni))
                    for r, (ex, ov, sw, conv) in res.items():
                        agg[r][0] += ex; agg[r][1] += ov
                for r in RULES:
                    w["characters_subsets"].writerow([P, u, r, frac, agg[r][0], f"{agg[r][1] / P:.4f}",
                                                      fixed[r], f"{mo:.4f}", f"{d:.4f}"])
        prog.step(N_SUBSETS * P * len(NOISES), f"character subsets P={P}")

    finish(files, tmp, out_dir, prog, raw)


def finish(files, tmp, out_dir, prog, raw):
    for f in files.values():
        f.close()
    if raw:
        raw.close()
    for name, path in tmp.items():
        path.replace(out_dir / final_name(name))
    print(f"Done in {(time.time() - prog.t0) / 60:.1f} min. CSVs in {out_dir}.")


if __name__ == "__main__":
    main()
