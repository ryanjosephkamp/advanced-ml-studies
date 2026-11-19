#!/usr/bin/env python3
"""
data_loader.py - EEG Data Loading and Preprocessing

Loads the three EEG datasets (EEG MMI, BCI CIV 2a, EEG MA), cuts them into
fixed-length windows, and caches the result.

How this works:
1. `extract_windows()` reads the raw EDF/GDF files and returns one window per
   trial (EEG MMI, BCI CIV 2a) or per 2-second step (EEG MA), with every selected
   channel kept separate, in volts, plus the label, the subject, and quality flags.
   The study code in `study/` uses this directly.
2. `load_dataset()` builds the pipeline's input from those windows: the channels
   of each window are averaged into one signal, which is z-scored. The result is
   cached as a compressed npz file, keyed by dataset, window length, subject
   count, and channel set.

Quality checks applied while windowing:
- EEG MMI: the sampling rate is read from each file. Files that are not 160 Hz
  (S088, S092, and S100 in the published data set) are skipped and listed in the
  metadata, so every window has the same length in seconds and in samples.
- Every dataset: a window in which any selected channel holds the same value for
  0.1 s or longer is dropped and counted. This removes the padded end of each EEG
  MA recording (about 1 s of held and zero values), which is not EEG.
- BCI CIV 2a: trials that the data set marks as artifacts (event 1023) are kept
  and flagged in `artifact`.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 4 – ML for Signal Analysis using EMD
Date: March 8, 2026
"""

import os
import json
import glob
import hashlib
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# DEFAULT CHANNEL SELECTIONS AND CONSTANTS
# =============================================================================

# EEG MMI: channel names have dot padding (e.g., 'C3..' for C3)
EEG_MMI_DEFAULT_CHANNELS = ['C3..', 'Cz..', 'C4..']
# Motor imagery runs (Task 2: imagine opening and closing the left or right fist)
EEG_MMI_IMAGERY_RUNS = [4, 8, 12]
# Expected sampling rate; files at other rates are skipped
EEG_MMI_SFREQ = 160.0
# Annotation labels: T1 = left fist, T2 = right fist (T0 = rest, not used)
EEG_MMI_EVENT_LABELS = {'T1': 0, 'T2': 1}

# BCI CIV 2a: channels prefixed with 'EEG-'
BCI_DEFAULT_CHANNELS = ['EEG-C3', 'EEG-Cz', 'EEG-C4']
# Cue events for the 4 classes
BCI_EVENT_CODES = {'769': 'left_hand', '770': 'right_hand', '771': 'both_feet', '772': 'tongue'}
BCI_EVENT_LABELS = {'769': 0, '770': 1, '771': 2, '772': 3}
# Event marking a trial rejected by the data set's artifact screening
BCI_ARTIFACT_CODE = '1023'

# EEG MA: channels prefixed with 'EEG '
EEG_MA_DEFAULT_CHANNELS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG Fz']

# A window is dropped when any channel holds one value for this long (seconds)
FLAT_RUN_SECONDS = 0.1

# Bumped whenever the windowing changes, so older cache files are never reused
CACHE_VERSION = 2


# =============================================================================
# DATASET REGISTRY
# =============================================================================

DATASET_REGISTRY = {
    'eeg_mmi': {
        'name': 'EEG Motor Movement/Imagery Dataset',
        'short_name': 'EEG MMI',
        'source_url': 'https://physionet.org/content/eegmmidb/1.0.0/',
        'raw_subdir': 'eeg_mmi/raw_download/raw_data',
        'raw_aliases': ['eeg_mmi'],
        'preprocessed_subdir': 'eeg_mmi/preprocessed_data',
    },
    'bci_civ_2a': {
        'name': 'BCI Competition IV Dataset 2a',
        'short_name': 'BCI CIV 2a',
        'source_url': 'http://www.bbci.de/competition/iv/',
        'raw_subdir': 'bci_civ_2a/raw_download/raw_data',
        'raw_aliases': ['bci_civ_2a', 'bci_iv_2a'],
        'preprocessed_subdir': 'bci_civ_2a/preprocessed_data',
    },
    'eeg_ma': {
        'name': 'EEG Mental Arithmetic Dataset',
        'short_name': 'EEG MA',
        'source_url': 'https://physionet.org/content/eegmat/1.0.0/',
        'raw_subdir': 'eeg_ma/raw_download/raw_data',
        'raw_aliases': ['eeg_ma', 'eegmat'],
        'preprocessed_subdir': 'eeg_ma/preprocessed_data',
    },
}

DEFAULT_DATASETS = ['eeg_mmi', 'bci_civ_2a', 'eeg_ma']

DEFAULT_CHANNELS = {
    'eeg_mmi': EEG_MMI_DEFAULT_CHANNELS,
    'bci_civ_2a': BCI_DEFAULT_CHANNELS,
    'eeg_ma': EEG_MA_DEFAULT_CHANNELS,
}

LABEL_NAMES = {
    'eeg_mmi': {0: 'left_hand', 1: 'right_hand'},
    'bci_civ_2a': {0: 'left_hand', 1: 'right_hand', 2: 'both_feet', 3: 'tongue'},
    'eeg_ma': {0: 'resting', 1: 'mental_arithmetic'},
}


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def find_raw_dir(dataset_key, data_dir, raw_dir=None):
    """
    Locate a dataset's raw files.

    With `raw_dir`, looks for a subfolder named after the dataset
    (e.g. raw_dir/eeg_mmi, raw_dir/bci_iv_2a, raw_dir/eegmat). Without it, uses
    the in-repository location data/<dataset>/raw_download/raw_data.
    """
    reg = DATASET_REGISTRY[dataset_key]
    if raw_dir is None:
        return os.path.join(data_dir, reg['raw_subdir'])
    for alias in reg['raw_aliases']:
        candidate = os.path.join(raw_dir, alias)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"No raw folder for {dataset_key} under the raw directory "
        f"(looked for: {', '.join(reg['raw_aliases'])})")


def extract_windows(dataset_key, raw_dir, channels=None, n_subjects=None,
                    window_length=2.0, offset=0.0, verbose=True):
    """
    Cut a dataset into labeled windows, one channel per row.

    Args:
        dataset_key: One of 'eeg_mmi', 'bci_civ_2a', 'eeg_ma'
        raw_dir: Folder holding the dataset's raw files (see find_raw_dir)
        channels: Channel names (None = the dataset's defaults)
        n_subjects: Max subjects to load, in sorted order (None = all)
        window_length: Window length in seconds
        offset: Start of each window relative to the cue, in seconds
                (EEG MMI and BCI CIV 2a; ignored for EEG MA)
        verbose: Print progress

    Returns:
        dict with keys:
            windows: float array (n_windows, n_channels, n_samples), volts
            labels: int array (n_windows,)
            subjects: str array (n_windows,), e.g. 'S001', 'A01', 'Subject00'
            artifact: bool array (n_windows,), True for BCI CIV 2a trials marked 1023
            sfreq, channel_names, label_names, n_subjects, dropped (dict)
    """
    if channels is None:
        channels = DEFAULT_CHANNELS[dataset_key]
    if dataset_key == 'eeg_mmi':
        out = _windows_eeg_mmi(raw_dir, channels, n_subjects, window_length, offset, verbose)
    elif dataset_key == 'bci_civ_2a':
        out = _windows_bci_civ_2a(raw_dir, channels, n_subjects, window_length, offset, verbose)
    elif dataset_key == 'eeg_ma':
        out = _windows_eeg_ma(raw_dir, channels, n_subjects, window_length, verbose)
    else:
        raise ValueError(f"No loader for dataset: {dataset_key}")

    # Drop windows in which any channel is flat for FLAT_RUN_SECONDS or longer
    max_run = int(round(FLAT_RUN_SECONDS * out['sfreq']))
    runs = longest_flat_run(out['windows'])            # (n_windows,)
    keep = runs < max_run
    out['dropped']['flat_windows'] = int(np.sum(~keep))
    out['dropped']['flat_windows_by_label'] = {
        int(l): int(np.sum((~keep) & (out['labels'] == l))) for l in np.unique(out['labels'])}
    for key in ('windows', 'labels', 'subjects', 'artifact'):
        out[key] = out[key][keep]

    out['channel_names'] = list(channels)
    out['label_names'] = LABEL_NAMES[dataset_key]
    out['n_subjects'] = int(len(np.unique(out['subjects'])))
    if verbose:
        d = out['dropped']
        print(f"  Windows kept: {len(out['labels'])}; flat windows dropped: {d['flat_windows']}"
              + (f"; files skipped for sampling rate: {len(d['files_wrong_sfreq'])}"
                 if d.get('files_wrong_sfreq') else ""))
    return out


def longest_flat_run(windows):
    """Longest run of identical consecutive samples in any channel, per window.

    Args:
        windows: array (n_windows, n_channels, n_samples)
    Returns:
        int array (n_windows,)
    """
    same = np.diff(windows, axis=2) == 0               # (n, C, T-1)
    n, c, t = same.shape
    flat = same.reshape(n * c, t)
    best = np.zeros(n * c, dtype=int)
    cur = np.zeros(n * c, dtype=int)
    for j in range(t):
        cur = np.where(flat[:, j], cur + 1, 0)
        best = np.maximum(best, cur)
    return best.reshape(n, c).max(axis=1) + 1


def average_and_zscore(windows):
    """Average the channels of each window and z-score the result.

    Args:
        windows: array (n_windows, n_channels, n_samples)
    Returns:
        array (n_windows, n_samples)
    """
    signals = windows.mean(axis=1)
    mean = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (signals - mean) / std


def load_dataset(dataset_key, data_dir, n_subjects=None, channels=None,
                 window_length=2.0, verbose=True, raw_dir=None, cache_dir=None):
    """
    Load a dataset as channel-averaged, z-scored windows (the pipeline's input).

    Tries the cache first (an exact match on dataset, cache version, window
    length, subject count, and channel set), then builds from the raw files.

    Args:
        dataset_key: One of 'eeg_mmi', 'bci_civ_2a', 'eeg_ma'
        data_dir: Repository data directory (used for default raw and cache paths)
        n_subjects: Max subjects to load (None = all)
        channels: List of channel names (None = defaults)
        window_length: Window length in seconds
        verbose: Print progress info
        raw_dir: Folder holding one subfolder per dataset (None = in-repo default)
        cache_dir: Where cache files are read and written (None = in-repo default)

    Returns:
        dict with keys: signals, labels, subjects, artifact, label_names, sfreq,
                        channel_names, n_subjects, dataset_name, dataset_key,
                        source_url, metadata
    """
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_key}. Choose from {list(DATASET_REGISTRY.keys())}")

    reg = DATASET_REGISTRY[dataset_key]
    if channels is None:
        channels = DEFAULT_CHANNELS[dataset_key]
    if cache_dir is None:
        cache_dir = os.path.join(data_dir, reg['preprocessed_subdir'])
    else:
        cache_dir = os.path.join(cache_dir, dataset_key)

    ch_str = '_'.join(sorted(channels))
    ch_hash = int(hashlib.md5(ch_str.encode()).hexdigest(), 16) % 100000
    n_str = 'all' if n_subjects is None else str(n_subjects)
    cache_name = f"{dataset_key}_v{CACHE_VERSION}_w{window_length}_n{n_str}_ch{ch_hash}.npz"
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        if verbose:
            print(f"  Loading preprocessed data from cache: {cache_name}")
        return _load_from_cache(cache_path, reg)

    source = find_raw_dir(dataset_key, data_dir, raw_dir)
    if verbose:
        print(f"  Loading raw data ({reg['short_name']})")
    win = extract_windows(dataset_key, source, channels=channels, n_subjects=n_subjects,
                          window_length=window_length, verbose=verbose)

    result = {
        'signals': average_and_zscore(win['windows']),
        'labels': win['labels'],
        'subjects': win['subjects'],
        'artifact': win['artifact'],
        'label_names': win['label_names'],
        'sfreq': win['sfreq'],
        'channel_names': win['channel_names'],
        'n_subjects': win['n_subjects'],
        'dataset_name': reg['name'],
        'dataset_key': dataset_key,
        'source_url': reg['source_url'],
        'metadata': {
            'n_segments': int(len(win['labels'])),
            'dropped': win['dropped'],
            'cache_version': CACHE_VERSION,
        },
    }
    _save_to_cache(result, cache_path, verbose)
    return result


# =============================================================================
# CACHE I/O
# =============================================================================

def _load_from_cache(cache_path, reg):
    """Load preprocessed dataset from npz cache."""
    data = np.load(cache_path, allow_pickle=False)
    label_names = {int(k): v for k, v in json.loads(str(data['label_names'])).items()}
    return {
        'signals': data['signals'],
        'labels': data['labels'],
        'subjects': data['subjects'],
        'artifact': data['artifact'],
        'label_names': label_names,
        'sfreq': float(data['sfreq']),
        'channel_names': [str(c) for c in data['channel_names']],
        'n_subjects': int(data['n_subjects']),
        'dataset_name': reg['name'],
        'dataset_key': str(data['dataset_key']),
        'source_url': reg['source_url'],
        'metadata': json.loads(str(data['metadata'])),
    }


def _save_to_cache(result, cache_path, verbose):
    """Save preprocessed dataset to npz cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        signals=result['signals'],
        labels=result['labels'],
        subjects=np.asarray(result['subjects'], dtype=str),
        artifact=np.asarray(result['artifact'], dtype=bool),
        label_names=json.dumps(result['label_names']),
        sfreq=result['sfreq'],
        channel_names=np.asarray(result['channel_names'], dtype=str),
        n_subjects=result['n_subjects'],
        dataset_key=result['dataset_key'],
        metadata=json.dumps(result.get('metadata', {})),
    )
    if verbose:
        print(f"  Saved preprocessed data to cache: {os.path.basename(cache_path)}")


# =============================================================================
# EEG MMI
# =============================================================================

def _windows_eeg_mmi(raw_dir, channels, n_subjects, window_length, offset, verbose):
    """
    EEG Motor Movement/Imagery Dataset: runs 4, 8, and 12 (imagined left or right
    fist), one window per T1 (left) or T2 (right) cue, starting `offset` seconds
    after the cue. Labels come from the annotation names, not from event codes.
    Files whose sampling rate is not 160 Hz are skipped and listed.
    """
    import mne

    subject_dirs = sorted(glob.glob(os.path.join(raw_dir, 'S[0-9][0-9][0-9]')))
    if n_subjects is not None:
        subject_dirs = subject_dirs[:n_subjects]

    n_samples = int(round(window_length * EEG_MMI_SFREQ))
    shift = int(round(offset * EEG_MMI_SFREQ))
    windows, labels, subjects = [], [], []
    wrong_sfreq, out_of_range = [], 0

    for subj_dir in tqdm(subject_dirs, desc="Loading EEG MMI subjects", disable=not verbose):
        subj_id = os.path.basename(subj_dir)
        for run_num in EEG_MMI_IMAGERY_RUNS:
            edf_file = os.path.join(subj_dir, f"{subj_id}R{run_num:02d}.edf")
            if not os.path.exists(edf_file):
                continue
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            if raw.info['sfreq'] != EEG_MMI_SFREQ:
                wrong_sfreq.append({'file': os.path.basename(edf_file),
                                    'sfreq': float(raw.info['sfreq'])})
                continue
            data = raw.get_data(picks=channels)          # (n_channels, n_times), volts
            for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
                if desc not in EEG_MMI_EVENT_LABELS:
                    continue
                start = int(round(onset * EEG_MMI_SFREQ)) + shift
                end = start + n_samples
                if start < 0 or end > data.shape[1]:
                    out_of_range += 1
                    continue
                windows.append(data[:, start:end])
                labels.append(EEG_MMI_EVENT_LABELS[desc])
                subjects.append(subj_id)

    return {
        'windows': np.array(windows),
        'labels': np.array(labels, dtype=int),
        'subjects': np.array(subjects, dtype=str),
        'artifact': np.zeros(len(labels), dtype=bool),
        'sfreq': EEG_MMI_SFREQ,
        'dropped': {
            'files_wrong_sfreq': wrong_sfreq,
            'subjects_wrong_sfreq': sorted({w['file'][:4] for w in wrong_sfreq}),
            'out_of_range': out_of_range,
        },
    }


# =============================================================================
# BCI CIV 2a
# =============================================================================

def _windows_bci_civ_2a(raw_dir, channels, n_subjects, window_length, offset, verbose):
    """
    BCI Competition IV Dataset 2a, training sessions (A01T to A09T): one window per
    cue (events 769 to 772), starting `offset` seconds after the cue. Trials the
    data set marks as artifacts (event 1023 at the trial start, 2 s before the cue)
    are kept and flagged.
    """
    import mne

    gdf_files = sorted(glob.glob(os.path.join(raw_dir, 'A0[1-9]T.gdf')))
    if n_subjects is not None:
        gdf_files = gdf_files[:n_subjects]

    windows, labels, subjects, artifact = [], [], [], []
    out_of_range = 0
    sfreq = None

    for gdf_file in tqdm(gdf_files, desc="Loading BCI CIV 2a subjects", disable=not verbose):
        raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
        sfreq = float(raw.info['sfreq'])
        n_samples = int(round(window_length * sfreq))
        shift = int(round(offset * sfreq))
        data = raw.get_data(picks=channels)
        onsets = np.asarray(raw.annotations.onset)
        descs = np.asarray(raw.annotations.description)
        rejected = onsets[descs == BCI_ARTIFACT_CODE]
        subj_id = os.path.basename(gdf_file)[:3]
        for onset, desc in zip(onsets, descs):
            if desc not in BCI_EVENT_LABELS:
                continue
            start = int(round(onset * sfreq)) + shift
            end = start + n_samples
            if start < 0 or end > data.shape[1]:
                out_of_range += 1
                continue
            windows.append(data[:, start:end])
            labels.append(BCI_EVENT_LABELS[desc])
            subjects.append(subj_id)
            # A 1023 marker sits at the trial start, 2 s before the cue
            artifact.append(bool(np.any((rejected >= onset - 2.5) & (rejected <= onset))))

    return {
        'windows': np.array(windows),
        'labels': np.array(labels, dtype=int),
        'subjects': np.array(subjects, dtype=str),
        'artifact': np.array(artifact, dtype=bool),
        'sfreq': sfreq if sfreq is not None else 250.0,
        'dropped': {'out_of_range': out_of_range},
    }


# =============================================================================
# EEG MA
# =============================================================================

def _windows_eeg_ma(raw_dir, channels, n_subjects, window_length, verbose):
    """
    EEG Mental Arithmetic Dataset (PhysioNet EEGMAT): each subject has a rest
    recording (`_1`, before the task) and an arithmetic recording (`_2`). Each is
    cut into consecutive, non-overlapping windows. The flat-window check in
    extract_windows() removes the padded end of each recording.
    """
    import mne

    edf_files = sorted(glob.glob(os.path.join(raw_dir, 'Subject*_*.edf')))
    recordings = {}
    for f in edf_files:
        subj_id, condition = os.path.basename(f).replace('.edf', '').split('_')
        recordings.setdefault(subj_id, {})[condition] = f
    subject_ids = sorted(recordings)
    if n_subjects is not None:
        subject_ids = subject_ids[:n_subjects]

    windows, labels, subjects = [], [], []
    sfreq = None
    for subj_id in tqdm(subject_ids, desc="Loading EEG MA subjects", disable=not verbose):
        for condition, label in [('1', 0), ('2', 1)]:   # 0 = resting, 1 = arithmetic
            if condition not in recordings[subj_id]:
                continue
            raw = mne.io.read_raw_edf(recordings[subj_id][condition], preload=True, verbose=False)
            sfreq = float(raw.info['sfreq'])
            n_samples = int(round(window_length * sfreq))
            data = raw.get_data(picks=channels)
            for w in range(data.shape[1] // n_samples):
                windows.append(data[:, w * n_samples:(w + 1) * n_samples])
                labels.append(label)
                subjects.append(subj_id)

    return {
        'windows': np.array(windows),
        'labels': np.array(labels, dtype=int),
        'subjects': np.array(subjects, dtype=str),
        'artifact': np.zeros(len(labels), dtype=bool),
        'sfreq': sfreq if sfreq is not None else 500.0,
        'dropped': {},
    }


# =============================================================================
# DEMO / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    """Quick self-test: load three EEG MA subjects and print basic info."""
    import sys
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from author_info import print_console_header

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default=None,
                        help='Folder with one subfolder per dataset (default: data/<dataset>/raw_download/raw_data)')
    parser.add_argument('--cache-dir', default=None,
                        help='Folder for cache files (default: data/<dataset>/preprocessed_data)')
    args = parser.parse_args()

    print_console_header("data_loader.py")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    result = load_dataset('eeg_ma', data_dir, n_subjects=3, verbose=True,
                          raw_dir=args.raw_dir, cache_dir=args.cache_dir)
    print(f"  Signals shape: {result['signals'].shape}")
    print(f"  Labels shape: {result['labels'].shape}")
    print(f"  Label names: {result['label_names']}")
    print(f"  Sampling freq: {result['sfreq']}")
    print(f"  Channels: {result['channel_names']}")
    print(f"  Subjects loaded: {result['n_subjects']}")
