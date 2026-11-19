#!/usr/bin/env python3
"""
extra_checks.py - Two follow-up checks on the study, from the cached features.

1. Per-subject permutation tests with 10,000 shuffles (subject_tests.csv).
   run_study.py tests each subject's balanced accuracy against 200 shuffles of
   that subject's labels, so the smallest possible p is 1/201. Under
   Benjamini-Hochberg at q = 0.05, k of m subjects pass only if the k-th smallest
   p is at most 0.05k/m, so any nonzero count of passing subjects is at least 11
   of the 106 EEG MMI subjects and at least 4 of the 36 EEG MA subjects. The
   counts in summary.csv therefore jump from 0 to 11 or more on EEG MMI. This
   check rebuilds the out-of-fold predictions of the LDA and Euclidean KNN
   configurations with the same folds as evaluate.py,
   confirms that every per-subject balanced accuracy matches per_subject.csv, and
   repeats each subject's test with 10,000 shuffles (smallest p 1/10,001).
   For held-out subjects, the model never saw the subject, so the shuffles need
   no retraining, but the test is exact only if the subject's windows are
   exchangeable. Windows that share a recording are not, and on EEG MA, each
   class is one recording, so a pass there shows at most that the subject's two
   recordings differ. Within subjects, the test is also approximate: the
   predictions come from models trained on the same subject's other windows,
   and the shuffles do not retrain them.

2. EEG MA channel and band subsets (eegmat_subsets.csv). Per-channel band power
   with LDA on held-out subjects (leave one subject out), as in the headline
   result, with every channel and band, and again without the frontal-polar
   channels Fp1 and Fp2 (where eye movements and muscle activity are largest),
   with Fp1 and Fp2 alone, with the alpha band (8-13 Hz) alone, and without the
   1-4 Hz and 30-45 Hz bands. Features as they are and standardized within each
   subject. Mean per-subject balanced accuracy with a 95% bootstrap interval, and
   the paired difference from the full feature set.

Run from the repository root after run_study.py has cached the features:

    python study/extra_checks.py --cache-dir /path/to/cache

About 80 seconds on a laptop CPU. Seeded, so a rerun gives the same numbers.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import features as feat      # noqa: E402  (also puts scripts/ on sys.path)
import evaluate as ev        # noqa: E402

PROJECT_DIR = os.path.dirname(HERE)
VARIANT_ORDER = ['eeg_mmi', 'bci_civ_2a', 'bci_civ_2a_late', 'eeg_ma']
N_PERM = 10000
CONFIGS = ([(fs, 'lda') for fs in ('F1_avg', 'F1_ch', 'F2_avg', 'F2_ch', 'F3_avg', 'F3_ch')]
           + [(fs, 'knn_euclidean') for fs in ev.FEATURE_SETS])
EVALUATIONS = [('within', 'none'), ('heldout', 'none'), ('heldout', 'subject')]

BAND_INDEX = {name: i for i, name in enumerate(feat.BAND_NAMES)}
SUBSETS = [  # (name, channels or None for all, bands or None for all)
    ('all channels, all bands', None, None),
    ('without Fp1 and Fp2', ['EEG F3', 'EEG F4', 'EEG Fz'], None),
    ('Fp1 and Fp2 only', ['EEG Fp1', 'EEG Fp2'], None),
    ('alpha band only', None, ['alpha_mu']),
    ('alpha band only, without Fp1 and Fp2', ['EEG F3', 'EEG F4', 'EEG Fz'], ['alpha_mu']),
    ('without 1-4 and 30-45 Hz', None, ['theta', 'alpha_mu', 'beta']),
    ('without 1-4 and 30-45 Hz, without Fp1 and Fp2',
     ['EEG F3', 'EEG F4', 'EEG Fz'], ['theta', 'alpha_mu', 'beta']),
]


def within_folds(y, subj):
    """The folds of evaluate.evaluate_within."""
    folds = []
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        counts = np.bincount(y[idx])
        counts = counts[counts > 0]
        n_splits = int(min(5, counts.min()))
        if n_splits < 2 or len(counts) < len(np.unique(y)):
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for tr, te in skf.split(idx, y[idx]):
            folds.append((idx[tr], idx[te]))
    return folds


def heldout_folds(subj):
    """The folds of evaluate.evaluate_heldout."""
    idx = np.arange(len(subj))
    if len(np.unique(subj)) > 40:
        return list(GroupKFold(n_splits=10).split(idx, groups=subj))
    return list(LeaveOneGroupOut().split(idx, groups=subj))


def perm_test(y, pred, classes, rng, n_perm=N_PERM):
    """Balanced accuracy and its permutation p-value (labels shuffled against
    fixed predictions), as evaluate.perm_pvalue_bacc but vectorized."""
    n_c = np.array([np.sum(y == c) for c in classes])
    hits = [pred == c for c in classes]

    def bacc(Y):
        return np.mean([((Y == c) & hits[k]).sum(axis=1) / n_c[k]
                        for k, c in enumerate(classes)], axis=0)

    obs = float(bacc(y[None, :])[0])
    null = bacc(rng.permuted(np.tile(y, (n_perm, 1)), axis=1))
    return obs, float((1 + np.sum(null >= obs - 1e-12)) / (1 + n_perm))


def subject_tests(variant, data, per_subject, summary, rng):
    rows = []
    for fs, clf in CONFIGS:
        ok = ev._valid(fs, data)
        y_ok, subj_ok = data['labels'][ok], data['subjects'][ok]
        for evaluation, norm in EVALUATIONS:
            folds = (within_folds(y_ok, subj_ok) if evaluation == 'within'
                     else heldout_folds(subj_ok))
            y, subj, pred = ev._out_of_fold(fs, clf, data, ok, folds, norm)
            classes = np.unique(y)
            ref = per_subject[(per_subject.variant == variant) & (per_subject.feature_set == fs)
                              & (per_subject.classifier == clf)
                              & (per_subject.evaluation == evaluation)
                              & (per_subject.norm == norm)].set_index('subject')
            accs, ps, diffs = [], [], []
            for s in np.unique(subj):
                m = (subj == s) & (pred >= 0)
                if m.sum() == 0 or len(np.unique(y[m])) < len(classes):
                    continue
                b, p = perm_test(y[m], pred[m], classes, rng)
                accs.append(b)
                ps.append(p)
                diffs.append(abs(b - ref.loc[s, 'balanced_accuracy']))
            if len(accs) != len(ref):
                raise RuntimeError(f'{variant} {fs} {clf} {evaluation} {norm}: '
                                   f'{len(accs)} subjects, per_subject.csv has {len(ref)}')
            old = summary[(summary.variant == variant) & (summary.feature_set == fs)
                          & (summary.classifier == clf) & (summary.evaluation == evaluation)
                          & (summary.norm == norm)].iloc[0]
            rows.append({
                'variant': variant, 'feature_set': fs, 'classifier': clf,
                'evaluation': evaluation, 'norm': norm, 'n_subjects': len(accs),
                'subject_bacc_mean': float(np.mean(accs)),
                'n_perm': N_PERM, 'smallest_possible_p': 1.0 / (N_PERM + 1),
                'n_subjects_p_below_0_05': int(np.sum(np.asarray(ps) < 0.05)),
                'n_subjects_above_chance_bh': int(ev.bh_reject(ps).sum()),
                'n_subjects_above_chance_bh_200_shuffles': int(old.n_subjects_above_chance_bh),
                'max_abs_diff_vs_per_subject_csv': float(max(diffs)),
            })
    return rows


def subset_columns(names, chans, bands):
    """Columns of F1_ch (channel-major, five bands per channel) for a subset."""
    nb = len(feat.BANDS)
    ch_idx = range(len(names)) if chans is None else [names.index(c) for c in chans]
    b_idx = range(nb) if bands is None else [BAND_INDEX[b] for b in bands]
    return [c * nb + b for c in ch_idx for b in b_idx]


def balanced_accuracy(y, pred, classes):
    return float(np.mean([np.mean(pred[y == c] == c) for c in classes]))


def eegmat_subsets(data, summary, rng):
    names = [str(c) for c in data['channel_names']]
    ok = ev._valid('F1_ch', data)
    folds = heldout_folds(data['subjects'][ok])
    rows = []
    for norm in ('none', 'subject'):
        per = {}
        for name, chans, bands in SUBSETS:
            d = dict(data)
            d['F1_ch'] = data['F1_ch'][:, subset_columns(names, chans, bands)]
            y, subj, pred = ev._out_of_fold('F1_ch', 'lda', d, ok, folds, norm)
            classes = np.unique(y)
            per[name] = pd.Series({s: balanced_accuracy(y[subj == s], pred[subj == s], classes)
                                   for s in np.unique(subj)})
        ref = summary[(summary.variant == 'eeg_ma') & (summary.feature_set == 'F1_ch')
                      & (summary.classifier == 'lda') & (summary.evaluation == 'heldout')
                      & (summary.norm == norm)].iloc[0]
        base = per[SUBSETS[0][0]]
        if abs(base.mean() - ref.subject_bacc_mean) > 1e-12:
            raise RuntimeError(f'the full feature set does not reproduce summary.csv ({norm})')
        for name, chans, bands in SUBSETS:
            mean, lo, hi = ev.bootstrap_mean_ci(per[name].values, rng)
            dm, dlo, dhi = ev.bootstrap_mean_ci((per[name] - base).values, rng)
            rows.append({'variant': 'eeg_ma', 'feature_set': 'F1_ch', 'classifier': 'lda',
                         'evaluation': 'heldout', 'norm': norm, 'subset': name,
                         'channels': ' '.join(c.replace('EEG ', '') for c in (chans or names)),
                         'bands': ' '.join(bands or feat.BAND_NAMES),
                         'n_features': len(subset_columns(names, chans, bands)),
                         'n_subjects': len(per[name]),
                         'subject_bacc_mean': mean, 'subject_bacc_ci_low': lo,
                         'subject_bacc_ci_high': hi,
                         'diff_from_full_mean': dm, 'diff_from_full_ci_low': dlo,
                         'diff_from_full_ci_high': dhi})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache-dir', required=True,
                   help='The --cache-dir given to run_study.py')
    p.add_argument('--out', default=os.path.join(PROJECT_DIR, 'results', 'study'),
                   help='Folder with the study results; outputs are written here')
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    feature_dir = os.path.join(args.cache_dir, 'study')
    per_subject = pd.read_csv(os.path.join(args.out, 'per_subject.csv'))
    summary = pd.read_csv(os.path.join(args.out, 'summary.csv'))
    t0 = time.time()
    rows = []
    for v in VARIANT_ORDER:
        if not os.path.exists(os.path.join(feature_dir, f'{v}_features.npz')):
            sys.exit(f'No cached features for {v} in {feature_dir}; run run_study.py first')
        data = feat.build_variant(v, None, feature_dir, verbose=False)
        t = time.time()
        rows += subject_tests(v, data, per_subject, summary,
                              np.random.default_rng([args.seed, 5, VARIANT_ORDER.index(v)]))
        print(f'  {v}: subject tests in {time.time() - t:.1f} s')
        if v == 'eeg_ma':
            subsets = eegmat_subsets(data, summary, np.random.default_rng([args.seed, 6]))
    tests = pd.DataFrame(rows)
    tests.to_csv(os.path.join(args.out, 'subject_tests.csv'), index=False)
    pd.DataFrame(subsets).to_csv(os.path.join(args.out, 'eegmat_subsets.csv'), index=False)
    print(f'Largest difference from per_subject.csv: '
          f'{tests.max_abs_diff_vs_per_subject_csv.max():.1e}')
    print(f'Done in {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
