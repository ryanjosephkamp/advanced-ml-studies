"""
evaluate.py - Classification, permutation tests, and clustering for the study.

Feature sets and classifiers (31 configurations per data set variant):
    F0_var, F0_ent, F0_if (P-avg)   KNN with Pearson, Euclidean, and Itakura-Saito
    F1_avg, F1_ch, F3_avg, F3_ch    KNN with Euclidean and Pearson on standardized
                                    log10 power, KNN with Itakura-Saito on power,
                                    and shrinkage LDA on standardized log10 power
    F2_avg, F2_ch                   KNN with Euclidean and Pearson, and shrinkage LDA,
                                    on standardized features
KNN uses k = 5 with a majority vote (ties go to the lowest class index, as in
scikit-learn). Fold-level standardization statistics come from the training
windows only. Before log10 or Itakura-Saito, a zero is replaced by its column's
smallest positive value (1.0 if the column has none) over all valid windows of
the data set variant, test windows included, without labels (_floor_zeros).

Evaluations:
    split     stratified 70/30 split of windows, seeds 0-9 and 42 (the original
              protocol; windows of one subject fall on both sides)
    within    stratified 5-fold cross-validation inside each subject
    heldout   subjects held out: one at a time (BCI CIV 2a, EEG MA) or 10 subject
              folds (EEG MMI); features used as they are ('none') or standardized
              within each subject using all of that subject's windows, without
              labels ('subject')

Statistics:
    - per-subject balanced accuracy from out-of-fold predictions; the mean over
      subjects with a 95% bootstrap interval (2,000 resamples of subjects);
    - a per-subject permutation p-value (200 shuffles of that subject's labels
      against its predictions) with Benjamini-Hochberg control at q = 0.05;
    - a 200-shuffle label-permutation null for the original split (seed 42);
    - paired Wilcoxon signed-rank tests across subjects (Holm-corrected).

Clustering: average-linkage clustering on each distance and K-Means on the
feature vectors, k = number of classes, scored against the labels and against
the subject IDs (ARI, NMI) and by the silhouette.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import wilcoxon
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GroupKFold, LeaveOneGroupOut)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score,
                             accuracy_score, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_score)

from similarity import itakura_saito_cross   # scripts/ is on sys.path via features.py

K_NEIGHBORS = 5
SPLIT_SEEDS = list(range(10)) + [42]
N_PERM_NULL = 200
N_PERM_SUBJECT = 200
N_BOOT = 2000

FEATURE_SETS = {
    'F0_var': ('f0', 'avg'), 'F0_ent': ('f0', 'avg'), 'F0_if': ('f0', 'avg'),
    'F1_avg': ('power', 'avg'), 'F1_ch': ('power', 'ch'),
    'F2_avg': ('mixed', 'avg'), 'F2_ch': ('mixed', 'ch'),
    'F3_avg': ('power', 'avg'), 'F3_ch': ('power', 'ch'),
}
CLASSIFIERS = {
    'f0': ['knn_pearson', 'knn_euclidean', 'knn_is'],
    'power': ['knn_euclidean', 'knn_pearson', 'knn_is', 'lda'],
    'mixed': ['knn_euclidean', 'knn_pearson', 'lda'],
}
NULL_CONFIGS = [('F1_ch', 'lda'), ('F1_ch', 'knn_euclidean'), ('F0_var', 'knn_euclidean')]


def configs():
    """All (feature set, classifier) pairs, in a fixed order."""
    return [(fs, clf) for fs, (kind, _) in FEATURE_SETS.items() for clf in CLASSIFIERS[kind]]


# =============================================================================
# FEATURE PREPARATION
# =============================================================================

def _floor_zeros(X):
    """Replace non-positive values by the column's smallest positive value (1.0 if
    the column has none). Callers pass every valid window of a data set, so the
    floor can come from a test window; it uses no labels."""
    X = X.astype(float).copy()
    for j in range(X.shape[1]):
        pos = X[:, j] > 0
        X[~pos, j] = X[pos, j].min() if pos.any() else 1.0
    return X


def _zscore_within(X, groups):
    out = np.empty_like(X, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd[sd == 0] = 1.0
        out[m] = (X[m] - mu) / sd
    return out


def _center_log_within(P, groups):
    """Divide positive values by their geometric mean within each group."""
    L = np.log(P)
    out = np.empty_like(L)
    for g in np.unique(groups):
        m = groups == g
        out[m] = L[m] - L[m].mean(axis=0)
    return np.exp(out)


def prepare(fs, X, subjects, norm='none'):
    """Matrices a feature set is classified from.

    Returns dict: 'vec' (for Euclidean, Pearson, LDA), 'lin' (positive, for
    Itakura-Saito), and 'standardize' (whether 'vec' is standardized with
    training statistics before use).
    """
    kind, _ = FEATURE_SETS[fs]
    if kind == 'f0':
        vec = X.astype(float)
        lin = _floor_zeros(X)
        if norm == 'subject':
            vec = _zscore_within(vec, subjects)
            lin = _center_log_within(lin, subjects)
        return {'vec': vec, 'lin': lin, 'standardize': False}
    if kind == 'power':
        P = _floor_zeros(X)               # zero power -> the column's smallest positive value
        vec = np.log10(P)
        lin = P
        if norm == 'subject':
            vec = _zscore_within(vec, subjects)
            lin = _center_log_within(P, subjects)
        return {'vec': vec, 'lin': lin, 'standardize': True}
    vec = X.astype(float)
    if norm == 'subject':
        vec = _zscore_within(vec, subjects)
    return {'vec': vec, 'lin': None, 'standardize': True}


def _standardize(A_tr, A_te):
    mu = A_tr.mean(axis=0)
    sd = A_tr.std(axis=0)
    sd[sd == 0] = 1.0
    return (A_tr - mu) / sd, (A_te - mu) / sd


# =============================================================================
# CLASSIFIERS
# =============================================================================

def _knn_vote(D, y_tr, n_classes, k=K_NEIGHBORS):
    k = min(k, D.shape[1])
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    votes = np.zeros((D.shape[0], n_classes))
    rows = np.repeat(np.arange(D.shape[0]), k)
    np.add.at(votes, (rows, y_tr[idx].ravel()), 1)
    return votes.argmax(axis=1)


def predict(clf, prep, tr, te, y, n_classes):
    """Train on windows `tr`, predict windows `te`."""
    vec = prep['vec']
    if clf == 'knn_is':
        D = itakura_saito_cross(prep['lin'][te], prep['lin'][tr])
        return _knn_vote(D, y[tr], n_classes)
    A_tr, A_te = vec[tr], vec[te]
    if prep['standardize']:
        A_tr, A_te = _standardize(A_tr, A_te)
    if clf == 'lda':
        model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        return model.fit(A_tr, y[tr]).predict(A_te)
    if clf == 'knn_euclidean':
        D = cdist(A_te, A_tr, metric='euclidean')
    elif clf == 'knn_pearson':
        with np.errstate(invalid='ignore', divide='ignore'):
            D = cdist(A_te, A_tr, metric='correlation')
        D = np.nan_to_num(D, nan=1.0)
    else:
        raise ValueError(clf)
    return _knn_vote(D, y[tr], n_classes)


def scores(y_true, y_pred):
    return {
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
    }


# =============================================================================
# STATISTICS HELPERS
# =============================================================================

def bootstrap_mean_ci(values, rng, n_boot=N_BOOT):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def perm_pvalue_bacc(y, pred, classes, rng, n_perm=N_PERM_SUBJECT):
    """Permutation p-value of one subject's balanced accuracy (labels shuffled)."""
    def bacc(Y):
        rec = []
        for c in classes:
            m = Y == c
            n = m.sum(axis=-1)
            rec.append(np.where(n > 0, ((pred == c) & m).sum(axis=-1) / np.maximum(n, 1), np.nan))
        return np.nanmean(np.stack(rec), axis=0)
    obs = bacc(y[None, :])[0]
    Y = np.stack([rng.permutation(y) for _ in range(n_perm)])
    null = bacc(Y)
    return float(obs), float((1 + np.sum(null >= obs - 1e-12)) / (1 + n_perm))


def bh_reject(p, q=0.05):
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, dtype=bool)
    rej[order[:k]] = True
    return rej


def holm(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (n - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


# =============================================================================
# EVALUATIONS
# =============================================================================

def _valid(fs, data):
    """Windows usable by a feature set (F2 needs at least 4 IMFs)."""
    return ~np.any(np.isnan(data[fs]), axis=1)


def evaluate_split(variant, data, rng):
    """Original protocol: stratified 70/30 split of windows, several seeds, and a
    label-permutation null at seed 42."""
    y_all = data['labels']
    n_classes = int(y_all.max()) + 1
    rows, null_rows = [], []
    for fs, clf in configs():
        ok = _valid(fs, data)
        X, y, subj = data[fs][ok], y_all[ok], data['subjects'][ok]
        prep = prepare(fs, X, subj, 'none')
        idx = np.arange(len(y))
        for seed in SPLIT_SEEDS:
            tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
            s = scores(y[te], predict(clf, prep, tr, te, y, n_classes))
            rows.append({'variant': variant, 'feature_set': fs, 'classifier': clf,
                         'seed': seed, 'n_windows': int(len(y)), 'n_test': int(len(te)), **s})
        if (fs, clf) in NULL_CONFIGS:
            for p in range(N_PERM_NULL):
                yp = rng.permutation(y)
                tr, te = train_test_split(idx, test_size=0.3, random_state=42, stratify=yp)
                s = scores(yp[te], predict(clf, prep, tr, te, yp, n_classes))
                null_rows.append({'variant': variant, 'feature_set': fs, 'classifier': clf,
                                  'perm': p, **s})
    return rows, null_rows


def _out_of_fold(fs, clf, data, ok, folds, norm):
    y = data['labels'][ok]
    subj = data['subjects'][ok]
    n_classes = int(data['labels'].max()) + 1
    prep = prepare(fs, data[fs][ok], subj, norm)
    pred = np.full(len(y), -1)
    for tr, te in folds:
        pred[te] = predict(clf, prep, tr, te, y, n_classes)
    return y, subj, pred


def _subject_rows(variant, fs, clf, evaluation, norm, y, subj, pred, rng):
    classes = np.unique(y)
    per = []
    for s in np.unique(subj):
        m = (subj == s) & (pred >= 0)
        if m.sum() == 0 or len(np.unique(y[m])) < len(classes):
            continue
        b, p = perm_pvalue_bacc(y[m], pred[m], classes, rng)
        per.append({'variant': variant, 'feature_set': fs, 'classifier': clf,
                    'evaluation': evaluation, 'norm': norm, 'subject': s,
                    'n_windows': int(m.sum()), 'balanced_accuracy': b, 'p_perm': p})
    rej = bh_reject([r['p_perm'] for r in per])
    for r, flag in zip(per, rej):
        r['bh_significant'] = bool(flag)
    return per


def _summary_row(variant, fs, clf, evaluation, norm, y, pred, per, rng):
    m = pred >= 0
    mean, lo, hi = bootstrap_mean_ci([r['balanced_accuracy'] for r in per], rng)
    return {'variant': variant, 'feature_set': fs, 'classifier': clf,
            'evaluation': evaluation, 'norm': norm, 'n_subjects': len(per),
            'n_windows': int(m.sum()),
            'subject_bacc_mean': mean, 'subject_bacc_ci_low': lo, 'subject_bacc_ci_high': hi,
            'subject_bacc_median': float(np.median([r['balanced_accuracy'] for r in per])) if per else np.nan,
            'n_subjects_above_chance_bh': int(sum(r['bh_significant'] for r in per)),
            **{f'pooled_{k}': v for k, v in scores(y[m], pred[m]).items()}}


def evaluate_within(variant, data, rng):
    """Stratified 5-fold cross-validation inside each subject."""
    summary, per_subject = [], []
    for fs, clf in configs():
        ok = _valid(fs, data)
        y = data['labels'][ok]
        subj = data['subjects'][ok]
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
        y, subj, pred = _out_of_fold(fs, clf, data, ok, folds, 'none')
        per = _subject_rows(variant, fs, clf, 'within', 'none', y, subj, pred, rng)
        per_subject += per
        summary.append(_summary_row(variant, fs, clf, 'within', 'none', y, pred, per, rng))
    return summary, per_subject


def evaluate_heldout(variant, data, rng):
    """Subjects held out: leave one subject out, or 10 subject folds for EEG MMI."""
    summary, per_subject = [], []
    for fs, clf in configs():
        ok = _valid(fs, data)
        subj = data['subjects'][ok]
        idx = np.arange(ok.sum())
        if len(np.unique(subj)) > 40:
            folds = list(GroupKFold(n_splits=10).split(idx, groups=subj))
        else:
            folds = list(LeaveOneGroupOut().split(idx, groups=subj))
        for norm in ('none', 'subject'):
            y, s_, pred = _out_of_fold(fs, clf, data, ok, folds, norm)
            per = _subject_rows(variant, fs, clf, 'heldout', norm, y, s_, pred, rng)
            per_subject += per
            summary.append(_summary_row(variant, fs, clf, 'heldout', norm, y, pred, per, rng))
    return summary, per_subject


def paired_tests(per_subject_rows, rng):
    """Wilcoxon signed-rank tests across subjects, Holm-corrected within each
    (variant, evaluation, norm) family. The test ranks the paired differences
    rounded to 10 decimal places, so exact zeros drop and exact ties share a
    rank; p = 1 when every rounded difference is zero. The bootstrap interval,
    mean, and median use the unrounded differences."""
    import pandas as pd
    if len(per_subject_rows) == 0:
        return []
    df = pd.DataFrame(per_subject_rows)
    out = []
    comparisons = []
    for prep in ('avg', 'ch'):
        for a, b in (('F2', 'F1'), ('F3', 'F1'), ('F3', 'F2')):
            comparisons.append((f'{a}_{prep}', f'{b}_{prep}'))
    for fam in ('F1', 'F2', 'F3'):
        comparisons.append((f'{fam}_ch', f'{fam}_avg'))
    for (variant, evaluation, norm), g in df.groupby(['variant', 'evaluation', 'norm']):
        fam_rows = []
        for clf in ('knn_euclidean', 'knn_pearson', 'lda'):
            for a, b in comparisons:
                A = g[(g.feature_set == a) & (g.classifier == clf)].set_index('subject')['balanced_accuracy']
                B = g[(g.feature_set == b) & (g.classifier == clf)].set_index('subject')['balanced_accuracy']
                common = A.index.intersection(B.index)
                if len(common) < 5:
                    continue
                d = (A[common] - B[common]).values
                # rounded so that exact zeros drop and exact ties share a rank
                d_rank = np.round(d, 10)
                if not np.any(d_rank):
                    p = 1.0
                else:
                    p = float(wilcoxon(d_rank, zero_method='wilcox').pvalue)
                mean, lo, hi = bootstrap_mean_ci(d, rng)
                fam_rows.append({'variant': variant, 'evaluation': evaluation, 'norm': norm,
                                 'classifier': clf, 'a': a, 'b': b, 'n_subjects': len(common),
                                 'mean_diff': mean, 'diff_ci_low': lo, 'diff_ci_high': hi,
                                 'median_diff': float(np.median(d)), 'p_wilcoxon': p})
        if fam_rows:
            adj = holm([r['p_wilcoxon'] for r in fam_rows])
            for r, pa in zip(fam_rows, adj):
                r['p_holm'] = float(pa)
            out += fam_rows
    return out


# =============================================================================
# CLUSTERING
# =============================================================================

def clustering(variant, data):
    """Average-linkage clusters on each distance and K-Means on the features,
    scored against labels and subjects."""
    rows = []
    y_all = data['labels']
    n_classes = len(np.unique(y_all))
    for fs, (kind, _) in FEATURE_SETS.items():
        ok = _valid(fs, data)
        y = y_all[ok]
        subj = data['subjects'][ok]
        prep = prepare(fs, data[fs][ok], subj, 'none')
        vec = prep['vec']
        if prep['standardize']:
            vec, _ = _standardize(vec, vec)
        dists = [d for d in ('knn_pearson', 'knn_euclidean', 'knn_is') if d in CLASSIFIERS[kind]]
        for d in dists:
            if d == 'knn_is':
                D = itakura_saito_cross(prep['lin'], prep['lin'])
            elif d == 'knn_euclidean':
                D = squareform(pdist(vec, 'euclidean'))
            else:
                with np.errstate(invalid='ignore', divide='ignore'):
                    D = squareform(np.nan_to_num(pdist(vec, 'correlation'), nan=1.0))
            np.fill_diagonal(D, 0.0)
            lab = AgglomerativeClustering(n_clusters=n_classes, metric='precomputed',
                                          linkage='average').fit_predict(D)
            rows.append(_cluster_row(variant, fs, 'average_linkage', d.replace('knn_', ''),
                                     lab, y, subj, D, data, ok))
            del D
        lab = KMeans(n_clusters=n_classes, n_init=10, random_state=42).fit_predict(vec)
        D = squareform(pdist(vec, 'euclidean'))
        rows.append(_cluster_row(variant, fs, 'kmeans', 'euclidean', lab, y, subj, D, data, ok))
        del D
    return rows


def _cluster_row(variant, fs, method, distance, lab, y, subj, D, data, ok):
    sizes = np.bincount(lab)
    try:
        sil = float(silhouette_score(D, lab, metric='precomputed'))
    except ValueError:
        sil = np.nan
    row = {'variant': variant, 'feature_set': fs, 'method': method, 'distance': distance,
           'n_windows': int(len(y)),
           'largest_cluster_share': float(sizes.max() / sizes.sum()),
           'ari_labels': float(adjusted_rand_score(y, lab)),
           'nmi_labels': float(normalized_mutual_info_score(y, lab)),
           'ari_subjects': float(adjusted_rand_score(subj, lab)),
           'nmi_subjects': float(normalized_mutual_info_score(subj, lab)),
           'silhouette': sil}
    if fs.startswith('F0'):
        rows_ = data['f0_rows'][ok]
        row['ari_f0_row_count'] = float(adjusted_rand_score(rows_, lab))
        row['nmi_f0_row_count'] = float(normalized_mutual_info_score(rows_, lab))
    return row
