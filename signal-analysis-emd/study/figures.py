"""
figures.py - Figures for the EMD feature study, drawn from the result files.

Every figure is a PNG at 200 dpi on a white background, one column wide so it
stays readable on a phone, with 95% intervals where the data has them. Colors
(blue, orange, aqua) follow one fixed order and are paired with marker shapes,
so no comparison depends on color alone.

Figures:
    fig1_heldout.png       balanced accuracy on held-out subjects (LDA)
    fig2_within.png        balanced accuracy within subjects (LDA)
    fig3_emd_vs_bandpower.png   per-subject differences, EMD features minus band power
    fig4_spectra.png       right-minus-left spectra at C3, C4, and their average
    fig5_leak.png          the original window split against held-out subjects (KNN)
    fig6_null.png          permutation null for the original split, with the
                           March 2026 and corrected-baseline results marked
    fig7_imf_example.png   one window's IMFs and residue
    fig8_clustering.png    cluster agreement with classes and with subjects
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, GRAY = '#0b0b0b', '#52514e', '#e3e2de', '#8a8984'

FAMILY = {  # feature family -> (label, color, marker)
    'F1': ('Band power', BLUE, 'o'),
    'F2': ('IMF features', ORANGE, 's'),
    'F3': ('Hilbert band power', AQUA, 'D'),
}
PREP_LABEL = {'ch': 'per channel', 'avg': 'channels averaged'}
DATASET_LABEL = {
    'eeg_mmi': 'EEG MMI: left vs. right hand',
    'bci_civ_2a': 'BCI IV 2a: four imagery classes',
    'bci_civ_2a_late': 'BCI IV 2a, window 0.5-2.5 s',
    'eeg_ma': 'EEGMAT: rest vs. arithmetic',
}
CHANCE = {'eeg_mmi': 0.5, 'bci_civ_2a': 0.25, 'bci_civ_2a_late': 0.25, 'eeg_ma': 0.5}
MAIN = ['eeg_mmi', 'bci_civ_2a', 'eeg_ma']
ROWS = [('F1', 'ch'), ('F1', 'avg'), ('F2', 'ch'), ('F2', 'avg'), ('F3', 'ch'), ('F3', 'avg')]


def _style():
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'lines.linewidth': 2, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    })


def _save(fig, out):
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)


def _footnote(fig, text):
    fig.text(0.01, 0.0, text, ha='left', va='top', fontsize=11, color=INK2)


def _chance(ax, x, label=True, vertical=True):
    if vertical:
        ax.axvline(x, color=INK2, ls='--', lw=1.2, zorder=1)
        if label:
            ax.text(x, 1.0, ' chance', transform=ax.get_xaxis_transform(), va='bottom',
                    ha='center', color=INK2, fontsize=11)
    else:
        ax.axhline(x, color=INK2, ls='--', lw=1.2, zorder=1)


def _row_label(fam, prep):
    return f"{FAMILY[fam][0]}, {PREP_LABEL[prep]}"


def _dot_panel(ax, sub, variant, title):
    ys = np.arange(len(ROWS))[::-1]
    for y, (fam, prep) in zip(ys, ROWS):
        r = sub[sub.feature_set == f'{fam}_{prep}']
        if r.empty:
            continue
        r = r.iloc[0]
        _, color, marker = FAMILY[fam]
        m, lo, hi = r.subject_bacc_mean, r.subject_bacc_ci_low, r.subject_bacc_ci_high
        ax.plot([lo, hi], [y, y], color=color, lw=2.5, solid_capstyle='round', zorder=2)
        ax.plot(m, y, marker=marker, ms=9, color=color, zorder=3,
                mfc=color if prep == 'ch' else 'white', mew=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([_row_label(f, p) for f, p in ROWS])
    ax.grid(axis='y', visible=False)
    _chance(ax, CHANCE[variant])
    ax.set_title(title, loc='left', pad=18)


def fig_dots(summary, evaluation, norm, out, variants, title_suffix, note=None):
    sub_all = summary[(summary.evaluation == evaluation) & (summary.norm == norm)
                      & (summary.classifier == 'lda')]
    fig, axes = plt.subplots(len(variants), 1, figsize=(7.2, 3.1 * len(variants)), squeeze=False)
    for ax, v in zip(axes[:, 0], variants):
        sub = sub_all[sub_all.variant == v]
        n = int(sub.n_subjects.max()) if not sub.empty else 0
        _dot_panel(ax, sub, v, f"{DATASET_LABEL[v]} ({n} subjects)")
        lo = min(CHANCE[v] - 0.05, sub.subject_bacc_ci_low.min() - 0.02)
        hi = max(CHANCE[v] + 0.1, sub.subject_bacc_ci_high.max() + 0.02)
        ax.set_xlim(lo, hi)
    axes[-1, 0].set_xlabel('Balanced accuracy\n(mean over subjects, 95% interval)')
    fig.suptitle(title_suffix, x=0.01, ha='left', fontsize=14, y=1.0)
    fig.tight_layout()
    if note:
        _footnote(fig, note)
    _save(fig, out)


def fig_emd_vs_bandpower(paired, out):
    panels = [('heldout', 'subject', MAIN, 'Held-out subjects, standardized'),
              ('within', 'none', ['eeg_mmi', 'bci_civ_2a'], 'Within subjects')]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.6), gridspec_kw={'height_ratios': [3, 2]})
    lows, highs = [], []
    for ax, (evaluation, norm, variants, title) in zip(axes, panels):
        rows = []
        for v in variants:
            for a, fam in (('F2_ch', 'F2'), ('F3_ch', 'F3')):
                r = paired[(paired.variant == v) & (paired.evaluation == evaluation)
                           & (paired.norm == norm) & (paired.classifier == 'lda')
                           & (paired.a == a) & (paired.b == 'F1_ch')]
                if not r.empty:
                    rows.append((v, fam, r.iloc[0]))
        ys = np.arange(len(rows))[::-1]
        for y, (v, fam, r) in zip(ys, rows):
            _, color, marker = FAMILY[fam]
            ax.plot([r.diff_ci_low, r.diff_ci_high], [y, y], color=color, lw=2.5, zorder=2)
            ax.plot(r.mean_diff, y, marker=marker, ms=9, color=color, zorder=3)
            ax.text(max(r.diff_ci_high, 0) + 0.004, y, f"p = {r.p_holm:.2g}", va='center',
                    fontsize=11, color=INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{DATASET_LABEL[v].split(':')[0]}: {FAMILY[f][0]}" for v, f, _ in rows])
        ax.grid(axis='y', visible=False)
        ax.axvline(0, color=INK2, lw=1.2, ls='--')
        ax.set_title(title, loc='left')
        lows.append(min(r.diff_ci_low for _, _, r in rows))
        highs.append(max(r.diff_ci_high for _, _, r in rows))
    from matplotlib.ticker import MultipleLocator
    for ax in axes:
        ax.set_xlim(min(lows) - 0.015, max(max(highs), 0) + 0.075)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
    axes[-1].set_xlabel('Difference in balanced accuracy vs. band power')
    fig.tight_layout()
    _footnote(fig, 'Mean over subjects with 95% interval; LDA, channels kept separate.\n'
                   'p: Wilcoxon signed-rank across subjects, Holm-corrected.')
    _save(fig, out)


def fig_spectra(spectra, out):
    variants = ['eeg_mmi', 'bci_civ_2a']
    fig, axes = plt.subplots(len(variants), 1, figsize=(7.2, 7.4), sharex=True)
    for ax, v in zip(axes, variants):
        sub = spectra[spectra.variant == v]
        chans = [c for c in sub.channel.unique() if c != 'average']
        series = [(chans[0], 'C3', BLUE, '-'), (chans[-1], 'C4', ORANGE, '-'),
                  ('average', 'C3, Cz, C4 averaged', AQUA, '-')]
        ax.axvspan(8, 30, color='#f2f1ee', zorder=0)
        ax.axhline(0, color=INK2, lw=1.2, ls='--')
        for ch, label, color, ls in series:
            s = sub[sub.channel == ch].sort_values('freq_hz')
            ax.fill_between(s.freq_hz, s.ci_low_db, s.ci_high_db, color=color, alpha=0.18, lw=0)
            ax.plot(s.freq_hz, s.mean_db, color=color, ls=ls, label=label)
        n = int(sub.n_subjects.iloc[0])
        ax.set_title(f"{DATASET_LABEL[v].split(':')[0]}: right minus left hand ({n} subjects)", loc='left')
        ax.set_ylabel('Power difference (dB)')
        ax.text(19, 0.98, 'mu and beta, 8-30 Hz', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=11, color=INK2)
    axes[0].legend(loc='lower right', frameon=False)
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.tight_layout()
    _save(fig, out)


def fig_leak(split_summary, summary, out):
    rows = [('F0_var', 'Original IMF variance'), ('F0_ent', 'Original IMF entropy'),
            ('F0_if', 'Original IMF frequency'), ('F1_ch', 'Band power, per channel'),
            ('F2_ch', 'IMF features, per channel'), ('F3_ch', 'Hilbert band power, per channel')]
    fig, axes = plt.subplots(len(MAIN), 1, figsize=(7.2, 3.0 * len(MAIN)))
    for ax, v in zip(axes, MAIN):
        ys = np.arange(len(rows))[::-1]
        for y, (fs, _) in zip(ys, rows):
            a = split_summary[(split_summary.variant == v) & (split_summary.feature_set == fs)
                              & (split_summary.classifier == 'knn_euclidean')]
            b = summary[(summary.variant == v) & (summary.feature_set == fs)
                        & (summary.classifier == 'knn_euclidean') & (summary.evaluation == 'heldout')
                        & (summary.norm == 'none')]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            ax.plot([a.balanced_accuracy_mean, b.subject_bacc_mean], [y, y], color=GRID, lw=3, zorder=1)
            ax.plot([a.balanced_accuracy_min, a.balanced_accuracy_max], [y + 0.12] * 2, color=BLUE, lw=2.5)
            ax.plot(a.balanced_accuracy_mean, y + 0.12, 'o', color=BLUE, ms=8, zorder=3,
                    label='70/30 split of windows' if y == ys[0] else None)
            ax.plot([b.subject_bacc_ci_low, b.subject_bacc_ci_high], [y - 0.12] * 2, color=ORANGE, lw=2.5)
            ax.plot(b.subject_bacc_mean, y - 0.12, 's', color=ORANGE, ms=8, zorder=3,
                    label='Held-out subjects' if y == ys[0] else None)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[1] for r in rows])
        ax.grid(axis='y', visible=False)
        _chance(ax, CHANCE[v])
        ax.set_title(DATASET_LABEL[v], loc='left', pad=18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 1.04), ncol=2,
               frameon=False, fontsize=12)
    axes[-1].set_xlabel('Balanced accuracy, KNN (k = 5, Euclidean)')
    fig.tight_layout()
    _footnote(fig, 'Split of windows: mean and range over 10 seeds.\n'
                   'Held-out subjects (unstandardized): mean over subjects, 95% interval.')
    _save(fig, out)


def _baseline_values(project_dir, baseline_run):
    # The March 9, 2026 run's comparison table, copied unchanged from
    # results/run_2026-03-09_15-21-15/comparative_analysis/comparison_table.csv
    march = os.path.join(project_dir, 'results', 'study', 'march_2026_comparison_table.csv')
    if baseline_run is None:
        runs = sorted(r for r in glob.glob(os.path.join(project_dir, 'results', 'run_*'))
                      if not r.endswith('run_2026-03-09_15-21-15'))
        baseline_run = runs[-1] if runs else None
    out = {}
    for name, path in (('March 2026 run', march),
                       ('Corrected baseline', os.path.join(baseline_run, 'comparative_analysis',
                                                           'comparison_table.csv') if baseline_run else '')):
        if path and os.path.exists(path):
            d = pd.read_csv(path)
            out[name] = d[d.task == 'classification']
    return out


def fig_null(null, project_dir, baseline_run, out):
    base = _baseline_values(project_dir, baseline_run)
    fig, axes = plt.subplots(len(MAIN), 1, figsize=(7.2, 2.9 * len(MAIN)))
    for ax, v in zip(axes, MAIN):
        vals = null[null.variant == v].f1_weighted.values
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ax.hist(vals, bins=30, color='#d7d6d1', edgecolor='white', lw=0.5)
        ax.axvspan(lo, hi, color='#f2f1ee', zorder=0)
        top = ax.get_ylim()[1]
        for k, (name, d) in enumerate(base.items()):
            x = d[d.dataset_key == v].f1_score.values
            color, marker = (GRAY, '|') if k == 0 else (BLUE, '|')
            ax.plot(x, np.full(len(x), top * (1.08 + 0.12 * k)), marker, color=color, ms=16, mew=2.5,
                    label=name, clip_on=False)
        ax.set_ylim(0, top * 1.3)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(DATASET_LABEL[v], loc='left', pad=8)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=11,
                   handlelength=0.8)
    axes[-1].set_xlabel('Weighted F1, 70/30 split of windows (seed 42)')
    fig.tight_layout()
    _footnote(fig, 'Gray bars: 600 label shuffles of the corrected windows, 200 for each of\n'
                   'three configurations, pooled; shaded: their middle 95%.')
    _save(fig, out)


def fig_imf_example(example, out):
    cols = [c for c in example.columns if c.startswith('imf_')]
    n = len(cols) + 2
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.25 * n + 0.6), sharex=True)
    t = example.time_s.values
    axes[0].plot(t, example.signal, color=INK, lw=1.2)
    axes[0].set_ylabel('Signal', rotation=0, ha='right', va='center')
    for ax, c in zip(axes[1:-1], cols):
        ax.plot(t, example[c], color=BLUE, lw=1.2)
        ax.set_ylabel(f"IMF {c.split('_')[1]}", rotation=0, ha='right', va='center')
    axes[-1].plot(t, example.residue, color=ORANGE, lw=1.6)
    axes[-1].set_ylabel('Residue', rotation=0, ha='right', va='center')
    for ax in axes:
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', labelsize=10)
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('One EEG MMI window (C3, Cz, C4 averaged, z-scored)', loc='left')
    fig.tight_layout(h_pad=0.3)
    _save(fig, out)


def fig_clustering(clusters, out):
    rows = [('F0_var', 'Original IMF variance'), ('F0_ent', 'Original IMF entropy'),
            ('F0_if', 'Original IMF frequency'), ('F1_avg', 'Band power, averaged'),
            ('F1_ch', 'Band power, per channel'), ('F2_avg', 'IMF features, averaged'),
            ('F2_ch', 'IMF features, per channel'), ('F3_avg', 'Hilbert power, averaged'),
            ('F3_ch', 'Hilbert power, per channel')]
    fig, axes = plt.subplots(len(MAIN), 1, figsize=(7.2, 3.7 * len(MAIN)))
    for ax, v in zip(axes, MAIN):
        sub = clusters[(clusters.variant == v) & (clusters.method == 'kmeans')]
        ys = np.arange(len(rows))[::-1]
        for y, (fs, _) in zip(ys, rows):
            r = sub[sub.feature_set == fs]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot(r.ari_labels, y + 0.15, 'o', color=BLUE, ms=8,
                    label='Agreement with classes' if y == ys[0] else None)
            ax.plot(r.ari_subjects, y - 0.15, 's', color=ORANGE, ms=8,
                    label='Agreement with subjects' if y == ys[0] else None)
            if fs.startswith('F0') and not np.isnan(r.get('ari_f0_row_count', np.nan)):
                ax.plot(r.ari_f0_row_count, y, 'D', color=AQUA, ms=8,
                        label='Agreement with EMD row count' if fs == 'F0_var' else None)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[1] for r in rows])
        ax.grid(axis='y', visible=False)
        ax.axvline(0, color=INK2, lw=1.2, ls='--')
        ax.set_title(DATASET_LABEL[v], loc='left')
    axes[0].legend(loc='lower right', frameon=False, fontsize=11)
    axes[-1].set_xlabel('Adjusted Rand index (0 = chance, 1 = identical)')
    fig.tight_layout()
    _footnote(fig, 'k-means clusters, one per class, compared with the classes, the subjects,\n'
                   'and (original features) the number of rows EMD returned per window.')
    _save(fig, out)


def make_all(out_dir, project_dir, baseline_run=None):
    _style()
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    summary = pd.read_csv(os.path.join(out_dir, 'summary.csv'))
    present = [v for v in MAIN if v in set(summary.variant)]
    fig_dots(summary, 'heldout', 'subject', os.path.join(fig_dir, 'fig1_heldout.png'), present,
             'Held-out subjects, LDA, features standardized within each subject',
             note='Filled markers: channels kept separate.\n'
                  'Open markers: channels averaged and z-scored (the original preprocessing).')
    fig_dots(summary, 'within', 'none', os.path.join(fig_dir, 'fig2_within.png'), present,
             'Within subjects, LDA, 5-fold cross-validation',
             note='Filled markers: channels kept separate. Open: channels averaged.\n'
                  'EEGMAT: rest and arithmetic are separate recordings, so within-subject\n'
                  'scores there can reflect the recording as well as the task.')
    paired = pd.read_csv(os.path.join(out_dir, 'paired_tests.csv'))
    fig_emd_vs_bandpower(paired, os.path.join(fig_dir, 'fig3_emd_vs_bandpower.png'))
    spectra = pd.read_csv(os.path.join(out_dir, 'spectra.csv'))
    if {'eeg_mmi', 'bci_civ_2a'} <= set(spectra.variant):
        fig_spectra(spectra, os.path.join(fig_dir, 'fig4_spectra.png'))
    split_summary = pd.read_csv(os.path.join(out_dir, 'split_summary.csv'))
    fig_leak(split_summary, summary, os.path.join(fig_dir, 'fig5_leak.png'))
    null = pd.read_csv(os.path.join(out_dir, 'split_null.csv'))
    fig_null(null, project_dir, baseline_run, os.path.join(fig_dir, 'fig6_null.png'))
    ex = os.path.join(out_dir, 'imf_example.csv')
    if os.path.exists(ex):
        fig_imf_example(pd.read_csv(ex), os.path.join(fig_dir, 'fig7_imf_example.png'))
    clusters = pd.read_csv(os.path.join(out_dir, 'clustering.csv'))
    fig_clustering(clusters, os.path.join(fig_dir, 'fig8_clustering.png'))
