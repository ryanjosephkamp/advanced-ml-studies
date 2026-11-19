"""
features.py - Windows and features for the EMD feature study.

For one data set variant this module:
1. cuts the raw recordings into labeled windows with scripts/data_loader.py
   (one row per channel, quality checks applied);
2. builds two versions of each window:
     P-avg  the original preprocessing: channels averaged, then z-scored;
     P-ch   each channel kept separate, linear trend removed, in microvolts;
3. decomposes every P-avg signal and every P-ch channel with EMD (PyEMD), in
   worker processes, and computes the feature sets:
     F0  the original features: the rows PyEMD returns (IMFs plus the residue),
         zero-padded or truncated to 8, then variance, histogram entropy, or mean
         instantaneous frequency per row (P-avg only; kept to measure the bug);
     F1  band power: Welch power in five bands (no EMD);
     F2  fixed IMF features: the first 4 IMFs (residue excluded), with log10
         variance, mean instantaneous frequency, and histogram entropy each;
     F3  Hilbert band power: the instantaneous power of every IMF (residue
         excluded), assigned to the band that holds its instantaneous frequency
         at each sample, averaged over the window;
4. stores class-mean spectra per subject for the spectra figure.

Results are cached as one npz file per variant.
"""

import os
import sys
import time
import numpy as np
from scipy.signal import welch, detrend, hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))

import data_loader as dl                     # noqa: E402
import feature_extraction as fx              # noqa: E402

BANDS = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
BAND_NAMES = ['delta', 'theta', 'alpha_mu', 'beta', 'gamma']
N_IMFS_F2 = 4
F0_SLOTS = 8
SPECTRUM_RANGE = (1.0, 45.0)

# Data set variants: the underlying data set and the window start after the cue
VARIANTS = {
    'eeg_mmi': {'dataset': 'eeg_mmi', 'offset': 0.0},
    'bci_civ_2a': {'dataset': 'bci_civ_2a', 'offset': 0.0},
    'bci_civ_2a_late': {'dataset': 'bci_civ_2a', 'offset': 0.5},
    'eeg_ma': {'dataset': 'eeg_ma', 'offset': 0.0},
}


# =============================================================================
# PER-SIGNAL EMD FEATURES (run in worker processes)
# =============================================================================

def _hilbert_band_power(imfs, sfreq):
    """Mean instantaneous power of the IMFs in each band (signal units squared).

    Each sample of each IMF contributes |analytic|^2 / 2 (the variance of a
    sinusoid of that amplitude) to the band that holds its instantaneous frequency
    at that sample. Samples outside 1 to 45 Hz contribute nothing.
    """
    out = np.zeros(len(BANDS))
    if imfs.shape[0] == 0:
        return out
    analytic = hilbert(imfs, axis=1)
    power = 0.5 * np.abs(analytic) ** 2                       # (k, T)
    phase = np.unwrap(np.angle(analytic), axis=1)
    freq = np.diff(phase, axis=1) * sfreq / (2 * np.pi)       # (k, T-1)
    p = 0.5 * (power[:, 1:] + power[:, :-1])                  # power between samples
    n = freq.shape[1]
    for b, (lo, hi) in enumerate(BANDS):
        out[b] = p[(freq >= lo) & (freq < hi)].sum() / n
    return out


def _signal_features(task):
    """EMD features for one signal. task = (signal, sfreq, want_f0, rescale)."""
    from PyEMD import EMD
    x, sfreq, want_f0, rescale = task
    emd = EMD()
    scale = 1.0
    if rescale:
        # P-ch: decompose at unit variance (PyEMD's stopping thresholds are absolute),
        # then scale the IMFs back to microvolts
        scale = float(np.std(x)) or 1.0
    emd.emd(x / scale)
    imfs, residue = emd.get_imfs_and_residue()
    imfs = np.asarray(imfs) * scale
    residue = np.asarray(residue) * scale
    n_imfs = imfs.shape[0]

    res = {'n_imfs': n_imfs}
    if n_imfs >= N_IMFS_F2:
        first = imfs[:N_IMFS_F2]
        var = np.var(first, axis=1)
        res['F2'] = np.concatenate([np.log10(np.maximum(var, 1e-300)),
                                    fx._compute_inst_freq(first, sfreq),
                                    fx._compute_entropy(first)])
    else:
        res['F2'] = np.full(3 * N_IMFS_F2, np.nan)
    res['F3'] = _hilbert_band_power(imfs, sfreq)

    if want_f0:
        # The original pipeline's rows: PyEMD's emd() output, which appends the
        # residue when it is not all zeros, then zero-padded or truncated to 8 rows
        rows = imfs if np.allclose(residue, 0) else np.vstack([imfs, residue[None, :]])
        res['f0_rows'] = rows.shape[0]
        if rows.shape[0] >= F0_SLOTS:
            rows = rows[:F0_SLOTS]
        else:
            rows = np.vstack([rows, np.zeros((F0_SLOTS - rows.shape[0], rows.shape[1]))])
        res['F0_var'] = fx._compute_variance(rows)
        res['F0_ent'] = fx._compute_entropy(rows)
        res['F0_if'] = fx._compute_inst_freq(rows, sfreq)
    return res


def _worker_init():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'


def _run_emd(tasks, workers, verbose, label):
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, initializer=_worker_init) as pool:
            out = pool.map(_signal_features, tasks, chunksize=64)
    else:
        out = [_signal_features(t) for t in tasks]
    if verbose:
        dt = time.time() - t0
        print(f"    EMD {label}: {len(tasks)} signals in {dt:.1f} s ({len(tasks) / max(dt, 1e-9):.0f}/s)")
    return out, time.time() - t0


# =============================================================================
# WELCH BAND POWER AND SPECTRA
# =============================================================================

def welch_psd(x, sfreq):
    """Welch PSD along the last axis with 1-s segments and 50% overlap."""
    nper = int(round(sfreq))
    return welch(x, fs=sfreq, nperseg=nper, noverlap=nper // 2, axis=-1)


def band_power(f, psd):
    """Mean PSD in each band; psd (..., n_freqs) -> (..., n_bands)."""
    return np.stack([psd[..., (f >= lo) & (f < hi)].mean(axis=-1) for lo, hi in BANDS], axis=-1)


# =============================================================================
# BUILD ONE VARIANT
# =============================================================================

def build_variant(variant, raw_dir, cache_dir, workers=2, max_subjects=None, verbose=True):
    """Compute (or load) every feature set for one data set variant.

    Args:
        variant: key of VARIANTS
        raw_dir: folder with one subfolder of raw files per data set
        cache_dir: folder for the cached feature file
        workers: worker processes for EMD
        max_subjects: use only the first N subjects (for timing slices)
    Returns:
        dict of arrays (see the keys saved below)
    """
    spec = VARIANTS[variant]
    tag = '' if max_subjects is None else f'_n{max_subjects}'
    path = os.path.join(cache_dir, f'{variant}{tag}_features.npz')
    if os.path.exists(path):
        if verbose:
            print(f"  {variant}: loading cached features")
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}

    t_start = time.time()
    key = spec['dataset']
    source = dl.find_raw_dir(key, None, raw_dir)
    win = dl.extract_windows(key, source, n_subjects=max_subjects, window_length=2.0,
                             offset=spec['offset'], verbose=verbose)
    sfreq = float(win['sfreq'])
    windows_uv = win['windows'] * 1e6                       # (n, C, T)
    n, C, T = windows_uv.shape

    p_avg = dl.average_and_zscore(win['windows'])           # (n, T), unitless
    p_ch = detrend(windows_uv, axis=2, type='linear')       # (n, C, T), microvolts

    # F1: Welch band power
    f, psd_avg = welch_psd(p_avg, sfreq)
    _, psd_ch = welch_psd(p_ch, sfreq)
    F1_avg = band_power(f, psd_avg)                         # (n, 5)
    F1_ch = band_power(f, psd_ch).reshape(n, C * len(BANDS))

    # Spectra for the figure: per subject and class, log10 PSD of each channel
    # and of the plain channel average (microvolts, not z-scored)
    _, psd_mean = welch_psd(detrend(windows_uv.mean(axis=1), axis=1, type='linear'), sfreq)
    keep_f = (f >= SPECTRUM_RANGE[0]) & (f <= SPECTRUM_RANGE[1])
    subjects = np.unique(win['subjects'])
    classes = np.unique(win['labels'])
    spec_arr = np.full((len(subjects), len(classes), C + 1, keep_f.sum()), np.nan)
    logp = np.log10(np.concatenate([psd_ch, psd_mean[:, None, :]], axis=1)[..., keep_f])
    for i, s in enumerate(subjects):
        for j, c in enumerate(classes):
            m = (win['subjects'] == s) & (win['labels'] == c)
            if m.any():
                spec_arr[i, j] = logp[m].mean(axis=0)

    # EMD features
    tasks = [(p_avg[i], sfreq, True, False) for i in range(n)]
    out_avg, t_avg = _run_emd(tasks, workers, verbose, f'{variant} P-avg')
    tasks = [(p_ch[i, c], sfreq, False, True) for i in range(n) for c in range(C)]
    out_ch, t_ch = _run_emd(tasks, workers, verbose, f'{variant} P-ch')

    F0 = {k: np.array([o[k] for o in out_avg]) for k in ('F0_var', 'F0_ent', 'F0_if')}
    result = {
        'labels': win['labels'],
        'subjects': win['subjects'].astype(str),
        'artifact': win['artifact'],
        'channel_names': np.array(win['channel_names'], dtype=str),
        'sfreq': np.array(sfreq),
        'F0_var': F0['F0_var'], 'F0_ent': F0['F0_ent'], 'F0_if': F0['F0_if'],
        'f0_rows': np.array([o['f0_rows'] for o in out_avg]),
        'F1_avg': F1_avg, 'F1_ch': F1_ch,
        'F2_avg': np.array([o['F2'] for o in out_avg]),
        'F2_ch': np.array([o['F2'] for o in out_ch]).reshape(n, C * 3 * N_IMFS_F2),
        'F3_avg': np.array([o['F3'] for o in out_avg]),
        'F3_ch': np.array([o['F3'] for o in out_ch]).reshape(n, C * len(BANDS)),
        'n_imfs_avg': np.array([o['n_imfs'] for o in out_avg]),
        'n_imfs_ch': np.array([o['n_imfs'] for o in out_ch]).reshape(n, C),
        'spec_freqs': f[keep_f],
        'spec_subjects': subjects.astype(str),
        'spec_classes': classes,
        'spectra': spec_arr,
        'example_signal': p_avg[0],
        'dropped_flat': np.array(win['dropped'].get('flat_windows', 0)),
        'dropped_files_sfreq': np.array([w['file'] for w in win['dropped'].get('files_wrong_sfreq', [])], dtype=str),
        'emd_seconds': np.array([t_avg, t_ch]),
        'build_seconds': np.array(time.time() - t_start),
    }
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, **result)
    if verbose:
        print(f"  {variant}: {n} windows, {C} channels, built in {time.time() - t_start:.1f} s")
    return result
