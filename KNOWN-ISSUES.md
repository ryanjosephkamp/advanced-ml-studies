# Known issues

This file lists known and potential problems in these four studies: a claim that may be stronger than its evidence, a
result that depends on a choice the text doesn't state, or something that can't yet be reproduced from this repository.
They were found in reviews after release. Each entry says what was observed, where, and its status. An entry marked
"potential" has not been rerun.

I'm working through them as time allows. A fixed entry stays here with the commit that fixed it.

If you find something else, please [open an issue](https://github.com/ryanjosephkamp/advanced-ml-studies/issues).

Line numbers refer to the files as of November 25, 2026.

| Study | Entries |
|---|---|
| [Parzen-window classifiers on Iris](#iris-parzen-classifier) | KI-01 to KI-05 |
| [Hopfield networks](#classical-hopfield-networks) | KI-06 to KI-09 |
| [Random projections before multiclass SVMs](#random-projections-multiclass-svm) | KI-10 to KI-16 |
| [Empirical mode decomposition features for EEG](#signal-analysis-emd) | KI-17 to KI-20 |

## iris-parzen-classifier

### Claims and wording

#### KI-01. The polynomial kernels' Problem 3 result is credited to the units; centering explains most of it

- **Kind:** claim. **Status:** open, November 25, 2026.
- **Where:** `paper/sections/results.tex`, line 55 ("depend partly on the units") and line 27 ("reflect what the rule
  can express"); `paper/sections/limitations.tex`, line 3 ("Raw features"); `README.md`, line 26; `chapter/chapter.md`,
  line 99.
- **Observed:** standardizing both centers and rescales each feature, and the study runs only the two together. On
  Problem 3, over the same splits as Table II, centering each feature without rescaling raises the degree-2 kernel's
  mean test accuracy from 0.574 to 0.851 (standardizing gives 0.853), while dividing by the standard deviation without
  centering, which is the change of units alone, gives 0.677. For the degree-3 kernel both parts matter: 0.546 raw,
  0.698 centered, 0.668 rescaled, 0.771 standardized. With centering alone, the Gaussian's lead over the degree-2 kernel
  on Problem 3 falls from 0.35 to 0.07. The limitations' suggested reason (the rule mostly compares vector lengths) was
  not tested.
- **Affects:** "depend partly on the units" should say that for the degree-2 kernel the uncentered origin, more than the
  units, is what breaks the rule; "reflect what the rule can express" holds for the rule on uncentered measurements,
  since the same rule reaches 0.85 once the features are centered. Table II has no centering-only or rescaling-only arm.
- **To check:** add centering-only and rescaling-only arms to the standardized comparison in `repeated_splits.py` (each
  training split's means and standard deviations, applied to its test rows).

#### KI-02. The standardized comparison gives no interval for the change it discusses

- **Kind:** reporting. **Status:** open, November 25, 2026.
- **Where:** `repeated_splits.py`, the standardized comparison (lines 277 to 291);
  `repeated_splits/standardized_feature_results.csv`; Table II.
- **Observed:** the comparison reports a 95% interval for standardized accuracy and the mean paired change, with the
  shares of splits on which accuracy rose or fell, but no interval for the mean paired change. For the degree-2 kernel
  on Problem 3, an interval from the per-seed means is +0.278 ± 0.005. No reading in the paper changes.
- **Affects:** Table II and the CSV.

### Methods and results

#### KI-03. The Gaussian's lead is for one untuned width; a narrower width scores higher

- **Kind:** method. **Status:** open, November 25, 2026.
- **Where:** `kamp_hw1.py` and `repeated_splits.py` (σ = 1/√2, the assignment's width); the results' kernel comparison
  and Table I; `paper/sections/limitations.tex`, line 5 ("One width").
- **Observed:** on the study's own splits and raw features, σ = 0.3 gives mean test accuracy 0.951, 0.952, and 0.945 on
  Problems 2 to 4, against 0.938, 0.926, and 0.929 at σ = 1/√2 (σ = 0.5: 0.954, 0.943, 0.942).
- **Affects:** "the Gaussian kernel wins" holds for the assignment's width against untuned polynomial kernels, not for
  tuned kernels. The limitations say the width was not shown to be good, not that a narrower width is better.
- **To check:** change the Gaussian width in the repeated-split run.

### Reproducibility

#### KI-04. Only summary tables are committed

- **Kind:** reproducibility. **Status:** open, November 25, 2026.
- **Where:** `repeated_splits.py`, lines 246 and 294; `repeated_splits/`.
- **Observed:** the script writes two summary CSVs and two figures, and no per-split accuracies, so the intervals,
  percentiles, and shares can be checked only by rerunning it. The run is deterministic and takes a few minutes on a
  laptop.
- **To check:** `python repeated_splits.py`.

#### KI-05. The revised scripts keep the original assignment's dates

- **Kind:** provenance. **Status:** fixed, November 25, 2026, in the commit that added this file.
- **Where:** `kamp_hw1.py` (header lines 5 to 7); also `../classical-hopfield-networks/scripts/kamp_hw2.py`,
  `../random-projections-multiclass-svm/scripts/kamp_hw3.py`, and `../signal-analysis-emd/scripts/kamp_hw4.py`.
- **Observed:** all four scripts were revised in November 2026 for this repository, but each opened with its original
  assignment date (January to March 2026), so a reader could take it for the submitted code. `kamp_hw1.py`'s docstring
  points to `repeated_splits.py`, and `kamp_hw3.py` documents a `--legacy-seeding` option, both added in November.
- **Fix:** each header now has the line `Revised: November 2026, for this repository` under its date.
- **Affects:** the script headers only.

## classical-hopfield-networks

### Claims and wording

#### KI-06. The file and fresh-noise columns differ in how the noise is made, not only in its draws

- **Kind:** claim. **Status:** open, November 25, 2026.
- **Where:** `paper/sections/characters.tex`, line 4; Table III (`paper/table_characters.tex`).
- **Observed:** the test file flips exactly 13 bits of each character; the 20% file flips 518 bits chosen over the whole
  18-character array, 21 to 39 per character; the study's fresh queries flip exactly 14 or 29 bits per character
  (`study/run_study.py`, lines 156 to 159). The text says only that the fresh columns draw new noise.
- **Affects:** differences between the file and fresh columns of Table III mix the noise construction with repetition.

#### KI-07. "Nothing more" than nearest-pattern lookup is slightly stronger than the data

- **Kind:** claim. **Status:** open, November 25, 2026.
- **Where:** `paper/sections/characters.tex`, line 26 ("the network retrieves the closest stored pattern and nothing
  more").
- **Observed:** at β = 64 with fresh noise, the modern network's recall count differs from nearest-pattern lookup's in 1
  of 200 draws at 10% noise and 4 of 200 at 20%, never by more than one pattern. The sentence before it gives the
  agreement on average.
- **Affects:** that sentence ("behaves almost exactly as nearest-pattern lookup").

### Methods and results

#### KI-08. The modern network's results stop at 10 updates, and its records say every query converged

- **Kind:** result. **Status:** open, November 25, 2026.
- **Where:** `study/hopfield_rules.py`, lines 158 to 169 (`modern_update`, `max_iter=10`); `study/run_study.py`, lines
  181 to 182 (records one sweep and "converged" for every modern query); `paper/sections/design.tex`, line 12 ("up to 10
  times or until x stops changing"); `paper/sections/limitations.tex`, line 3 (the cap statement covers the classical
  rules only); Table II (`paper/table_capacity.tex`, modern rows); Table III (β = 16 row);
  `study/results/synthetic_recall.csv.gz` (sweeps and converged columns).
- **Observed:** with the study's own seeds, 47.3%, 15.8%, and 4.8% of the 219,600 synthetic queries at β = 4, 16, and 64
  are still changing at the 10th update. The published numbers sit between two other answers. At 10% noise, P50 for β =
  16 at ρ = 0.7 is above 40 after one update, 14.3 as published, and 13.1 at convergence; at ρ = 0.8 it is 12.2, 4.5,
  and 4.1; for β = 64 at ρ = 0.9 it is above 40, 16.6, and 15.4. On the characters at β = 16, the test file gives 18 of
  18 recalled after one update, 14 as published, and 12 at convergence, and the "Fixed" count is 14 as published against
  12 at convergence. At β = 64 every character query has converged within the cap. Run to convergence, a few synthetic
  queries take thousands of updates (up to 4,224).
- **Affects:** the modern network's capacities in Table II and the text, the β = 16 row of Table III, and the
  description of the method: the numbers are for a renormalized, sign-output variant of the update of Ramsauer et al.,
  stopped at 10 updates, and they depend on that number. The records' sweeps and converged values for the modern network
  and nearest-pattern lookup are placeholders, not measurements.
- **To check:** count the iterations inside `modern_update`, and rerun the modern rows with one update and to
  convergence.

#### KI-09. The P50 intervals leave out interpolation error

- **Kind:** method. **Status:** open, November 25, 2026.
- **Where:** `study/analyze.py` (P50 by log-linear interpolation); Table II; the capacities in the abstract, the text,
  and `README.md`.
- **Observed:** P50 is interpolated between tested loads that step by 3 to 10 above 9 patterns. For example, the Hebbian
  rule's 21.4 at ρ = 0 lies between P = 18 (exact recall 0.753) and P = 24 (0.335). Its bootstrap interval, 21.0 to
  21.9, resamples the 50 pattern sets only. The limitations say values above 9 are interpolated across wide gaps, but
  not that the intervals leave this out.
- **Affects:** the intervals on interpolated capacities read as tighter than the measurement supports.

## random-projections-multiclass-svm

### Claims and wording

#### KI-10. "The variance order does not explain the accuracy order" is broader than the data

- **Kind:** claim. **Status:** open, November 25, 2026.
- **Where:** `paper/sections/discussion.tex`, line 7; `paper/sections/distortion.tex`, line 17; `README.md`, line 29.
- **Observed:** the family with the highest variance, one nonzero per column, has the lowest mean accuracy in 30 of 30
  (data set, k) cells. Across the families run in each cell (five; four on Glass), the rank correlation of variance with
  accuracy is negative in 30 of 30 cells (mean −0.54); without one nonzero per column, it is negative in 14, positive in
  14, and zero in 2 (mean −0.01). Neighbor agreement after projection, which the text mentions as "not shown",
  correlates positively with accuracy in 30 of 30 cells (mean 0.85).
- **Affects:** the claim holds among the Gaussian, Eq. 3, Eq. 4, and very sparse matrices; one nonzero per column has
  both the highest variance and the lowest accuracy. `chapter/chapter.md` (line 93) already words it narrowly.

#### KI-11. One comparison is counted both as equivalent and as a significant loss

- **Kind:** claim. **Status:** open, November 25, 2026.
- **Where:** `paper/sections/classification.tex`, line 26; the abstract's equivalence count; Table IV
  (`paper/table_comparisons.tex`); `README.md`, lines 25 to 26.
- **Observed:** one nonzero per column against the Gaussian on Breast Cancer at k = 16 has a mean difference of −1.25
  points with a 95% interval of [−1.58, −0.91], inside the ±2-point margin, and a Holm-adjusted p below 0.001. So the
  equivalent comparisons and those that differ after Holm's correction overlap by one.
- **Affects:** the counts read as separate groups; the two calls are compatible, but the text should say so or count
  that cell once.

#### KI-12. The 100-draw repeat and the tuned study differ in more than repetition

- **Kind:** claim. **Status:** open, November 25, 2026.
- **Where:** `paper/sections/assignment.tex`, line 23; `study/replicate.py`; `study/run_svm.py`.
- **Observed:** the repeat keeps the assignment's pipeline: the older Iris file, min-max scaling fit on all rows,
  default SVM settings, k = 2, one split, the original seeding, and the best kernel chosen on the test set. The tuned
  study scales on training rows only, sets `coef0 = 1` for the polynomial kernels, standardizes after projecting,
  selects by inner cross-validation, and draws a new split per draw. The text compares the repeat's Seeds difference
  with the tuned study's without listing these differences.
- **Affects:** that comparison shows how much the two pipelines differ, not the effect of repetition alone.

#### KI-13. Related work on random projections for SVMs is not cited

- **Kind:** citation. **Status:** open, November 25, 2026.
- **Where:** `paper/sections/introduction.tex`; `paper/sections/references.tex`.
- **Observed:** none of the paper's 18 references is on classification after random projection. Paul, Boutsidis,
  Magdon-Ismail, and Drineas, "Random Projections for Support Vector Machines," AISTATS 2013 (PMLR 31:498–506), study
  margin preservation for linear SVMs under projection. The paper makes no novelty claim that the missing work would
  undercut.
- **Affects:** the introduction should cite it and say this study is empirical, multiclass, and with nonlinear kernels.

### Methods and results

#### KI-14. The pooled test and leaving out Iris were chosen after the per-cell results

- **Kind:** method. **Status:** open, November 25, 2026.
- **Where:** `paper/sections/design.tex`, line 10 ("A pooled test … is used for the overall deficit"); the abstract and
  conclusion ("the data sets other than Iris"); `README.md`.
- **Observed:** the planned analysis was the 144 primary comparisons with Holm's correction and the ±2-point margin. The
  pooled per-draw test and the pool without Iris were added after the per-cell results were seen, and the design section
  lists the pooled test with the planned statistics without saying so. With Iris, the pooled deficits against the
  Gaussian are 0.73 points for Eq. 3 (p = 0.005), 1.47 for Eq. 4 (p = 0.0006; 95% interval reaching 2.30), and 0.85 for
  very sparse (p = 0.040); without Iris they are 0.39 (p = 0.10), 0.16 (p = 0.61), and 0.61 (p = 0.08). The zero-column
  reason for leaving Iris out applies to Eq. 4 only.
- **Affects:** the abstract's "within about a point … other than Iris": with Iris, Eq. 3 and very sparse matrices are
  still within a point but significantly behind, and only Eq. 4 needs the exclusion. The body reports the with-Iris
  numbers (`paper/sections/classification.tex`, line 44; `paper/sections/discussion.tex`, line 3).

#### KI-15. PCA is fit before the inner cross-validation folds, like the scalers

- **Kind:** method. **Status:** open, potential, November 25, 2026.
- **Where:** `study/run_svm.py`, lines 67 to 91; `paper/sections/limitations.tex`, line 3.
- **Observed:** min-max scaling, PCA (for the PCA baseline), and the standardization after projection are all fit on the
  whole outer training set before the 3-fold inner search. The limitations name the scalers, not PCA. The outer test
  rows are never used in a fit.
- **Affects:** the PCA baseline has the same small leak within the training set as the scalers; its size was not
  measured.

### Reproducibility

#### KI-16. A code comment says min-max scaling helps the Johnson–Lindenstrauss lemma hold

- **Kind:** code comment. **Status:** fixed, November 25, 2026, in the commit that added this file.
- **Where:** `scripts/random_projection.py`, lines 48 to 50.
- **Observed:** the docstring said scaling "allows the Johnson–Lindenstrauss lemma to hold more effectively". The lemma
  holds for any fixed set of points; scaling only changes which distances are the ones preserved.
- **Fix:** the docstring now says scaling keeps any one feature's units from dominating the distances the projection
  approximately preserves, and that the lemma holds for any fixed set of points.
- **Affects:** that docstring.

## signal-analysis-emd

### Claims and wording

#### KI-17. "Within about a point of band power" is a mean, not an equivalence result

- **Kind:** claim. **Status:** open, November 25, 2026.
- **Where:** `paper/sections/abstract.tex`, finding (1).
- **Observed:** on held-out subjects with LDA, the per-channel Hilbert band power minus Welch band power is −0.61 points
  on EEG MMI standardized (95% interval −2.17 to 0.91) and −1.05 unstandardized (−2.62 to 0.62); every Holm-adjusted p
  is 1.0. No equivalence test was run. `paper/sections/emd.tex` and `README.md` say so; the abstract does not.
- **Affects:** finding (1) in the abstract.

### Methods and results

#### KI-18. Hilbert band power leaves out samples with negative instantaneous frequency

- **Kind:** method. **Status:** open, potential, November 25, 2026.
- **Where:** `study/features.py`, lines 58 to 75; `scripts/feature_extraction.py`, line 141.
- **Observed:** each sample's power goes to the band that holds its signed instantaneous frequency, so samples where the
  unwrapped phase runs backward count toward no band. The mean-frequency feature uses the absolute frequency. The paper
  describes neither choice. The share of power left out can't be computed for the study from the committed files, which
  hold features, not IMFs. In the one committed example window (`results/study/imf_example.csv`, the first
  channel-averaged EEG MMI window), 0.22% of the IMFs' power is at negative frequencies, and 0.14% would fall in a study
  band if the absolute frequency were used.
- **Affects:** Hilbert band power may be slightly low against Welch band power. On held-out subjects it is already
  within about a point of band power, so the room to move is small.

#### KI-19. BCI IV 2a keeps its artifact-flagged windows, with no check of their effect

- **Kind:** method. **Status:** open, potential, November 25, 2026.
- **Where:** `paper/sections/data.tex`, line 15; `results/study/diagnostics.json`.
- **Observed:** the 262 windows that carry the data set's artifact flag (263 in the late-window variant) are kept, as
  the data section says. No analysis compares the results with them removed, and the committed files hold no per-window
  predictions, so it can't be checked from this repository.
- **Affects:** the BCI IV 2a scores are not artifact-free estimates.

#### KI-20. The data section doesn't say how EEGMAT was filtered before release

- **Kind:** data. **Status:** open, potential, November 25, 2026.
- **Where:** `paper/sections/data.tex`, line 4.
- **Observed:** the paper gives no high-pass, low-pass, or notch settings for the released EEGMAT recordings. The data
  set's own description mentions "a high-pass filter with a 30 Hz cut-off frequency" and a 50 Hz notch. Read literally,
  that would leave little signal in the bands below 30 Hz; if it means a low-pass, little would remain in the 30 to 45
  Hz band. This repository doesn't show which was applied.
- **Affects:** the EEGMAT results for the bands on the wrong side of that filter. Each EDF file's prefiltering field
  would settle what was applied.
