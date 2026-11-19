#!/usr/bin/env python3
"""
run_study.py - EMD feature study on three EEG data sets.

Question: do features built from empirical mode decomposition (EMD) separate EEG
classes once the channels are kept separate, and do they add anything over plain
band power, within subjects and on subjects held out of training?

Data: EEG Motor Movement/Imagery (runs 4, 8, 12; left vs. right fist imagery;
C3, Cz, C4), BCI Competition IV 2a (training sessions; four imagery classes;
C3, Cz, C4; windows starting 0 s and 0.5 s after the cue), and EEG Mental
Arithmetic (rest vs. serial subtraction; Fp1, Fp2, F3, F4, Fz). Windows are 2 s.
See features.py for the preprocessing (P-avg, P-ch) and feature sets (F0 to F3)
and evaluate.py for the classifiers, evaluations, and statistics.

Run from the repository root:

    python study/run_study.py --raw-dir /path/to/raw --cache-dir /path/to/cache

--raw-dir holds one folder per data set: eeg_mmi/ (S001/S001R04.edf, ...),
bci_iv_2a/ (A01T.gdf, ...), and eegmat/ (Subject00_1.edf, ...). The raw
recordings are not part of the repository. Per-window features are cached in
--cache-dir (about 15 MB); aggregated results and figures are written to --out
(default results/study/).

Options:
    --stage features|evaluate|figures|all   run one stage or all (default all)
    --variants ...                          a subset of eeg_mmi, bci_civ_2a,
                                            bci_civ_2a_late, eeg_ma
    --max-subjects N                        use the first N subjects of each data
                                            set (a timing slice; output goes to
                                            <out>_slice<N>)
    --workers N                             processes for EMD (default 2)
    --seed S                                base seed (default 2026)

Every random step (window splits, label permutations, bootstrap resamples,
K-Means starts) is seeded, so a rerun reproduces the results exactly. The
evaluate stage needs only the cached features; the figures stage needs only the
result files in --out.
"""

import os
import sys
import json
import time
import argparse
import platform
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import features as feat      # noqa: E402  (also puts scripts/ on sys.path)
import evaluate as ev        # noqa: E402

PROJECT_DIR = os.path.dirname(HERE)
VARIANT_ORDER = ['eeg_mmi', 'bci_civ_2a', 'bci_civ_2a_late', 'eeg_ma']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-dir', required=False, default=None,
                   help='Folder with one subfolder of raw files per data set')
    p.add_argument('--cache-dir', required=True, help='Folder for cached per-window features')
    p.add_argument('--out', default=os.path.join(PROJECT_DIR, 'results', 'study'),
                   help='Folder for aggregated results and figures')
    p.add_argument('--stage', default='all', choices=['features', 'evaluate', 'figures', 'all'])
    p.add_argument('--variants', nargs='+', default=VARIANT_ORDER, choices=VARIANT_ORDER)
    p.add_argument('--max-subjects', type=int, default=None)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--baseline-run', default=None,
                   help='Corrected baseline run folder (results/run_*) for the null figure')
    return p.parse_args()


def _rng(seed, variant, stage):
    return np.random.default_rng([seed, VARIANT_ORDER.index(variant), stage])


def _distribution(a):
    vals, counts = np.unique(np.asarray(a).ravel(), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def spectra_rows(variant, data, rng):
    """Class-difference spectra (dB) per channel: mean and 95% bootstrap interval
    over subjects. Motor imagery: right minus left hand. EEG MA: arithmetic minus rest."""
    S = data['spectra']                       # (subjects, classes, channels + 1, freqs)
    names = [str(c) for c in data['channel_names']] + ['average']
    diff_db = 10.0 * (S[:, 1] - S[:, 0])      # class 1 minus class 0
    contrast = 'arithmetic_minus_rest' if variant == 'eeg_ma' else 'right_minus_left'
    rows = []
    ok = ~np.any(np.isnan(diff_db), axis=(1, 2))
    d = diff_db[ok]
    boot = rng.integers(0, len(d), size=(ev.N_BOOT, len(d)))
    means = d[boot].mean(axis=1)              # (boot, channels, freqs)
    lo = np.percentile(means, 2.5, axis=0)
    hi = np.percentile(means, 97.5, axis=0)
    mu = d.mean(axis=0)
    for c, name in enumerate(names):
        for k, f in enumerate(data['spec_freqs']):
            rows.append({'variant': variant, 'contrast': contrast, 'channel': name,
                         'freq_hz': float(f), 'mean_db': float(mu[c, k]),
                         'ci_low_db': float(lo[c, k]), 'ci_high_db': float(hi[c, k]),
                         'n_subjects': int(ok.sum())})
    return rows


def write_imf_example(data, out_dir):
    """Decompose the first EEG MMI window (P-avg) for the IMF example figure."""
    from emd_processor import decompose
    x = np.asarray(data['example_signal'], dtype=float)
    imfs, residue = decompose(x)
    df = pd.DataFrame({'time_s': np.arange(len(x)) / float(data['sfreq']), 'signal': x})
    for k in range(imfs.shape[0]):
        df[f'imf_{k + 1}'] = imfs[k]
    df['residue'] = residue
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'imf_example.csv'), index=False, float_format='%.6g')


def cache_file(args, out_cache, variant):
    """The cached feature file of one variant, as features.build_variant names it."""
    tag = '' if args.max_subjects is None else f'_n{args.max_subjects}'
    return os.path.join(out_cache, f'{variant}{tag}_features.npz')


def cache_records(args, out_cache):
    """The cache file of each variant, relative to --cache-dir."""
    return {v: {'file': os.path.relpath(cache_file(args, out_cache, v), args.cache_dir)}
            for v in args.variants}


def run_features(args, out_cache):
    timing = {}
    for v in args.variants:
        t0 = time.time()
        feat.build_variant(v, args.raw_dir, out_cache, workers=args.workers,
                           max_subjects=args.max_subjects)
        timing[v] = round(time.time() - t0, 1)
    return timing


def run_evaluate(args, out_cache, out_dir):
    split_rows, null_rows, summary, per_subject, clusters, spec, diag = [], [], [], [], [], [], {}
    timing = {}
    for v in args.variants:
        data = feat.build_variant(v, args.raw_dir, out_cache, workers=args.workers,
                                  max_subjects=args.max_subjects, verbose=False)
        t0 = time.time()
        r, n = ev.evaluate_split(v, data, _rng(args.seed, v, 1))
        split_rows += r
        null_rows += n
        t1 = time.time()
        s, p = ev.evaluate_within(v, data, _rng(args.seed, v, 2))
        summary += s
        per_subject += p
        t2 = time.time()
        s, p = ev.evaluate_heldout(v, data, _rng(args.seed, v, 3))
        summary += s
        per_subject += p
        t3 = time.time()
        clusters += ev.clustering(v, data)
        t4 = time.time()
        spec += spectra_rows(v, data, _rng(args.seed, v, 4))
        if v == 'eeg_mmi':
            write_imf_example(data, out_dir)
        timing[v] = {'split_s': round(t1 - t0, 1), 'within_s': round(t2 - t1, 1),
                     'heldout_s': round(t3 - t2, 1), 'clustering_s': round(t4 - t3, 1)}
        y = data['labels']
        diag[v] = {
            'n_windows': int(len(y)),
            'n_subjects': int(len(np.unique(data['subjects']))),
            'class_counts': {str(int(c)): int(np.sum(y == c)) for c in np.unique(y)},
            'channels': [str(c) for c in data['channel_names']],
            'sfreq': float(data['sfreq']),
            'dropped_flat_windows': int(data['dropped_flat']),
            'dropped_files_sampling_rate': [str(f) for f in data['dropped_files_sfreq']],
            'artifact_flagged_windows': int(np.sum(data['artifact'])),
            'imf_count_p_avg': _distribution(data['n_imfs_avg']),
            'imf_count_p_ch': _distribution(data['n_imfs_ch']),
            'f0_rows_p_avg': _distribution(data['f0_rows']),
            'windows_without_4_imfs_p_avg': int(np.sum(np.isnan(data['F2_avg']).any(axis=1))),
            'windows_without_4_imfs_p_ch': int(np.sum(np.isnan(data['F2_ch']).any(axis=1))),
            'emd_seconds_p_avg_p_ch': [round(float(x), 1) for x in data['emd_seconds']],
            'feature_build_seconds': round(float(data['build_seconds']), 1),
            'evaluation_seconds': timing[v],
        }
        print(f"  {v}: evaluated in {t4 - t0:.1f} s")

    os.makedirs(out_dir, exist_ok=True)
    split = pd.DataFrame(split_rows)
    split.to_csv(os.path.join(out_dir, 'split_seeds.csv'), index=False)
    pd.DataFrame(null_rows).to_csv(os.path.join(out_dir, 'split_null.csv'), index=False)

    # Split summary: mean and range over seeds 0-9, the seed-42 value, and the null band
    rows = []
    nulls = pd.DataFrame(null_rows)
    for (v, fs, clf), g in split.groupby(['variant', 'feature_set', 'classifier'], sort=False):
        g10 = g[g.seed != 42]
        g42 = g[g.seed == 42].iloc[0]
        row = {'variant': v, 'feature_set': fs, 'classifier': clf, 'n_windows': int(g42.n_windows),
               'n_test': int(g42.n_test), 'n_seeds': len(g10)}
        for m in ('balanced_accuracy', 'kappa', 'f1_weighted', 'f1_macro', 'accuracy'):
            row[f'{m}_mean'] = float(g10[m].mean())
            row[f'{m}_min'] = float(g10[m].min())
            row[f'{m}_max'] = float(g10[m].max())
            row[f'{m}_seed42'] = float(g42[m])
        nv = nulls[nulls.variant == v]
        for m in ('balanced_accuracy', 'f1_weighted'):
            pooled = nv[m].values
            row[f'null_{m}_p2_5'] = float(np.percentile(pooled, 2.5))
            row[f'null_{m}_p97_5'] = float(np.percentile(pooled, 97.5))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'split_summary.csv'), index=False)

    pd.DataFrame(summary).to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    pd.DataFrame(per_subject).to_csv(os.path.join(out_dir, 'per_subject.csv'), index=False)
    pd.DataFrame(ev.paired_tests(per_subject, np.random.default_rng([args.seed, 99]))).to_csv(
        os.path.join(out_dir, 'paired_tests.csv'), index=False)
    pd.DataFrame(clusters).to_csv(os.path.join(out_dir, 'clustering.csv'), index=False)
    pd.DataFrame(spec).to_csv(os.path.join(out_dir, 'spectra.csv'), index=False)
    return diag


def environment():
    import importlib.metadata as md
    pk = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'mne', 'EMD-signal', 'matplotlib', 'pandas'):
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    return {'python': platform.python_version(), 'packages': pk}


def main():
    args = parse_args()
    out_dir = args.out if args.max_subjects is None else f'{args.out}_slice{args.max_subjects}'
    out_cache = os.path.join(args.cache_dir, 'study')
    t_all = time.time()
    meta_path = os.path.join(out_dir, 'diagnostics.json')
    meta = {}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    # The evaluate stage writes evaluate_environment, --stage all writes full_run, and
    # the stages that read or build the cache replace feature_cache with the paths of
    # this run's variants. Keys of an older layout are removed (see below); every other
    # key this run does not write stays from earlier runs.
    env = environment()

    if args.stage in ('features', 'all'):
        if args.raw_dir is None:
            sys.exit('--raw-dir is required to build features')
        meta['feature_stage_seconds'] = run_features(args, out_cache)
    if args.stage in ('evaluate', 'all'):
        t0 = time.time()
        meta['variants'] = run_evaluate(args, out_cache, out_dir)
        meta['evaluate_stage_seconds'] = round(time.time() - t0, 1)
        meta['settings'] = {'seed': args.seed, 'workers': args.workers,
                            'max_subjects': args.max_subjects,
                            'split_seeds': ev.SPLIT_SEEDS, 'n_perm_null': ev.N_PERM_NULL,
                            'n_perm_subject': ev.N_PERM_SUBJECT, 'n_boot': ev.N_BOOT,
                            'k_neighbors': ev.K_NEIGHBORS, 'bands_hz': feat.BANDS,
                            'n_imfs_f2': feat.N_IMFS_F2, 'f0_slots': feat.F0_SLOTS,
                            'flat_run_seconds': 0.1}
        for old in ('environment', 'imf_example'):   # records of an older layout
            meta.pop(old, None)
        meta['evaluate_environment'] = env
    if args.stage in ('features', 'evaluate', 'all'):    # the stages that read or build the cache
        meta['feature_cache'] = cache_records(args, out_cache)
    if args.stage in ('figures', 'all'):
        import figures
        t0 = time.time()
        figures.make_all(out_dir, PROJECT_DIR, baseline_run=args.baseline_run)
        meta['figures_stage_seconds'] = round(time.time() - t0, 1)
    if args.stage == 'all':
        meta.pop('all_stages_seconds', None)     # the older place of full_run's timing
        meta['full_run'] = {'all_stages_seconds': round(time.time() - t_all, 1),
                            'environment': env}
    os.makedirs(out_dir, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=1)
    print(f"Done in {time.time() - t_all:.1f} s; results in {os.path.relpath(out_dir, PROJECT_DIR)}")


if __name__ == '__main__':
    main()
