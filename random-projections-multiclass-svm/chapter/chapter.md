# *Shape-shifting*: geometry preservation with sparse random projections

*Coursework for CS 7063 Advanced Machine Learning, Spring 2026 (Submitted February 14, 2026); Extended November 2026.*

You can squeeze data into fewer dimensions with a matrix of random numbers, without looking at the data at all. It sounds like it shouldn't work, but there's a theorem that says it often does. The matrix can also be mostly zeros, which makes it cheap. This chapter uses five small data sets and support vector machines (SVMs) to test whether that cheap version loses anything important.

## The idea

Take a table with d columns, multiply it by a d by k matrix of random numbers, and divide by √k. Each row becomes k numbers. The Johnson–Lindenstrauss lemma says that for enough columns k (for these data sets, enough would be more than 100 columns, more than any of them has features), distances between rows stay roughly the same. The matrices in the assignment come from Dimitris Achlioptas: one is filled with +1 and −1 (Eq. 3), and the other has zeros two thirds of the time and plus or minus a constant otherwise (Eq. 4).

<!-- [[demo (3D): Iris as 150 points colored by species, on three of its four measurements. Choose a random matrix (Gaussian, Eq. 3, Eq. 4, very sparse, or one per column) and a target size of 2 or 3, press "Draw a matrix", and watch the cloud move into its projected shape. The matrix is drawn as a grid, with the dropped measurements (all-zero rows) and any all-zero column marked, and a tally of your draws counts how often a whole column comes out zero and how often both petal rows do, against their theory (35.6% and 19.8% for Eq. 4 at k = 2). It runs live in the page on the 150 flowers, so it is an exploration, and it shows no accuracy.]] -->
<!-- github -->

*Plate 1, live:* [Draw a matrix](https://ryanjosephkamp.github.io/advanced-ml-studies/random-projections-multiclass-svm/#plate-1), a live plate on the book's site.

<!-- /github -->

## What the assignment found

The assignment projected each data set to two dimensions, trained four kinds of SVM, and compared the two matrices. In the submitted table, Eq. 4 came out ahead on three of five data sets, and the report explained why. Seeds, for example, looked 7.9 points better with Eq. 4.

On Iris, Eq. 4 scored 68.9% under every kernel. That was not a collapse. The seed-42 matrix happened to have zeros in both petal rows, so the projection used only sepal length and sepal width, one in each column. A random Eq. 4 matrix has zeros in both petal rows with probability 19.8%.

## One draw is not an answer

The table came from one random matrix per method and one split of the data. The extension repeats the same pipeline for 100 draws of the matrices. The report's Seeds result for Eq. 4, 92.1%, is higher than 92 of the other 99 draws (it ties 3 and is below 4); their median is 81.0%. Across the 100 draws, Eq. 4 minus Eq. 3 is −6.2 points on Iris (95% interval −10.1 to −2.4) and 3.1 on Seeds (0.5 to 5.6), and the other three data sets show no clear difference. Those intervals are not adjusted across the five data sets, they all use the report's one split, and the report picked each kernel on the test set.

The headline itself was not rare: 52 of the 100 draws would also have put Eq. 4 ahead on three or more data sets. What one draw cannot show is which differences hold on average: only Seeds (for Eq. 4) and Iris (for Eq. 3) have intervals that exclude zero. In the tuned study below, which also changes the pipeline (scaling, tuning, and a new split for each draw), the Seeds difference at k = 2 is only 1.3 points, with an interval from −1.1 to 3.8.

<!-- [[figure (pop-out): study/figures/fig7_original_pipeline_100_draws.png. Best-kernel test accuracy of the original pipeline at k = 2 for Eq. 3 and Eq. 4, one point per draw over 100 draws, with a bar at each median and a diamond at the report's draw.]] -->
<!-- github -->

![One row per data set (Iris, Breast Cancer, Wine, Seeds, and Glass) of best-kernel test accuracy at k = 2 over 100 draws of the original pipeline: a square per draw for Eq. 3 and a triangle per draw for Eq. 4, a bar at each median, and a white diamond at the report's draw (seed 42).](../study/figures/fig7_original_pipeline_100_draws.png)

*Plate 2.* Best-kernel test accuracy at *k* = 2 for Eq. 3 (squares) and Eq. 4 (triangles), one point per draw over 100 draws, with each median and the report's draw (seed 42). The best kernel is chosen on the test set, as in the report.

<!-- /github -->

<!-- [[demo: the original pipeline's 100 committed draws as tickets in a drum. Draw once to pick one at random and read its table as the report read its own, with whether it would have given the report's conclusion (Eq. 4 ahead on three or more data sets, as in 52 of the 100 draws). A board shows every draw's best-kernel accuracy, with the report's draw marked.]] -->
<!-- github -->

*Plate 3, live:* [One hundred draws](https://ryanjosephkamp.github.io/advanced-ml-studies/random-projections-multiclass-svm/#plate-3), a live plate on the book's site.

<!-- /github -->

## The study

The extension compares five kinds of matrix: a Gaussian one (the dense reference), Eq. 3, Eq. 4, a "very sparse" one that keeps about one entry in √d, and one with a single nonzero entry per column on average. It uses target sizes from 1 to 24 columns, 100 draws per setting with a fresh train/test split each time, and SVMs whose settings are tuned by cross-validation on the training rows. Two references give points of comparison: all the original features, and principal component analysis (PCA), which looks at the data to choose its directions. In each draw, every matrix sees the same split, so the comparisons are paired. Glass, whose six classes are unbalanced, is scored by balanced accuracy.

<!-- [[figure (pop-out): study/figures/fig1_accuracy_vs_k.png. Accuracy against target size in five stacked panels, one per data set (balanced accuracy for Glass), for all families (very sparse is not run on Glass), PCA, all features, and the majority-class line (chance balanced accuracy for Glass).]] -->
<!-- github -->

![Five stacked panels, Iris, Breast Cancer, Wine, Seeds, and Glass, of test accuracy (balanced accuracy for Glass) against target size k on a log scale: a line with a 95% band for each matrix family (four on Glass, where very sparse is not run), a dashed line for PCA, a gray line and band for all features, and a dotted line for the majority class (chance for Glass).](../study/figures/fig1_accuracy_vs_k.png)

*Plate 4.* Test accuracy of the cross-validated model against target size k for each of the five data sets (balanced accuracy for Glass), 100 draws each; bands are 95% intervals. The gray line is all features, with its 95% interval as a band; the dashed line is PCA to k; and the dotted line is the majority class (chance for Glass). The horizontal axis is k on a log scale. Very sparse is not run on Glass.

<!-- /github -->

## Sparse against dense

Across the 144 primary comparisons, 28 differ after Holm's correction for multiple comparisons, and every one is a loss: 27 are the single-nonzero-per-column matrices, and one is Eq. 4 on Iris at k = 2 (−9.0 points, by a paired t-test; a Wilcoxon test with the same correction does not confirm it). The 144 comparisons are the 114 of Eq. 3, Eq. 4, very sparse, and one per column with the Gaussian, plus 30 of Eq. 4 with Eq. 3, none of which survives. Intervals inside ±2 points, which show equivalence, occur in 52 of them, mostly at larger k. One comparison is in both groups: one per column on Breast Cancer at k = 16 is equivalent and also a significant loss. Pooled over target sizes and then equally over the data sets each family was run on, the paired differences of Eq. 3, Eq. 4, and the very sparse matrix from the Gaussian matrix stay within about a point outside Iris, though single cells can be further off. This pooled summary, and leaving out Iris, came after the per-cell results. That is not the same as equal. With Iris, pooled the same way, Eq. 3 trails the Gaussian by 0.73 points, Eq. 4 by 1.47, and the very sparse matrix by 0.85. Without Iris, Eq. 4 trails by only 0.16.

<!-- [[figure (pop-out): study/figures/fig2_paired_difference_vs_gaussian.png. The paired difference from the Gaussian matrix for every (data set, k), with 95% intervals and a shaded band for plus or minus two points.]] -->
<!-- github -->

![Five stacked panels, one per data set, of Eq. 3, Eq. 4, very sparse (not run on Glass), and one per column minus the Gaussian matrix, in points of accuracy (balanced accuracy for Glass), against target size k on a log scale, each with its 95% interval bar, around a line at zero and a shaded band from −2 to +2 points.](../study/figures/fig2_paired_difference_vs_gaussian.png)

*Plate 5.* The paired difference from the Gaussian matrix for every data set and target size k, in points of accuracy (balanced accuracy for Glass), with 95% intervals and a shaded band for plus or minus two points. Very sparse is not run on Glass.

<!-- /github -->

## Why Iris is the exception

Iris has four features. With Eq. 4 at k = 2, an average of 1.84 of them are dropped, because their rows in the matrix are all zeros. A column of the matrix is all zero in 37 of the 100 draws, and then the projection has fewer than two dimensions. In those draws, Eq. 4 trails the Gaussian by 25.9 points. In the other 63 draws, Eq. 4 is 0.9 points ahead, no detectable difference (p = 0.59).

## One nonzero per column

Making each entry nonzero with probability 1/d, so that a column has one nonzero entry on average and sometimes none, is the sparsest option, and it loses everywhere. It trails the Gaussian in all 30 cells, and after the correction in 27, by 1.3 to 12.1 points. On Breast Cancer, the gap shrinks as k grows, to 0.5 points at k = 24.

## Against PCA, and against all features

PCA looks at the data, and random projection does not. At two dimensions, the Gaussian matrix trails PCA by 5.3 points on Iris and 23.8 on Wine. A random matrix comes within two points of all features only at larger k: from 12 columns on Breast Cancer, which has 30 features; from 4 to 6 of 7 on Seeds; and at 12 on Wine, which has 13 features and so saves little, and there only in the means. It never does on Iris or Glass.

<!-- [[demo: a heat map of mean accuracy over target size and matrix family, one data set at a time (balanced accuracy for Glass), with PCA and all features as extra rows. Tap a cell to read its value, its interval, and its paired differences where it has them. Only the measured cells are drawn: the families sit at s = 1, 3, √d, and d.]] -->
<!-- github -->

*Plate 6, live:* [Every cell the study measured](https://ryanjosephkamp.github.io/advanced-ml-studies/random-projections-multiclass-svm/#plate-6), a live plate on the book's site.

<!-- /github -->

## Does distortion explain the results?

The squared-distance ratio of a pair of points, after projection, has mean 1 and a variance that depends on the matrix. Over 20,000 draws, the measured variance matches the formula to within about 4.0% in every cell. Eq. 3 has less variance than the Gaussian, Eq. 4 the same, and matrices sparser than Eq. 4 more. (The very sparse matrix is denser than Eq. 4 on Iris and Seeds, where d is small.) Theory therefore gave no reason to expect Eq. 4 to beat the Gaussian. Nor does the variance order match the accuracy ranking: Eq. 3, with the lowest variance, still trails the Gaussian in 23 of 30 cells.

<!-- [[figure (pop-out): study/figures/fig3_distortion_variance_vs_theory.png. Measured variance against the formula for each data set, from the study's 100-draw check, whose ratios run from 0.52 to 1.60; the within-4.0% result in the text comes from the 20,000-draw check.]] -->
<!-- github -->

![Five stacked panels, one per data set, of k times the variance of the squared-distance ratio q, averaged over 2,000 sampled pairs, against target size k: a marker per family for the variance measured over the study's 100 draws, and a dashed line per family for the formula 2 + (s − 3)κ at the panel's mean κ, where the Gaussian's and Eq. 4's lines coincide at 2. Very sparse is not run on Glass.](../study/figures/fig3_distortion_variance_vs_theory.png)

*Plate 7.* Each pair's variance of the squared-distance ratio q over the study's 100 draws, averaged over 2,000 sampled pairs and multiplied by k, against the formula 2 + (s − 3)κ at each data set's mean κ (study/results/variance_check.csv). The measured variance runs from 0.52 to 1.60 times the formula; the within-4.0% result in the text comes from the 20,000-draw check.

<!-- /github -->

<!-- [[figure (pop-out): study/figures/fig4_distortion_tails_vs_bound.png. The share of Breast Cancer pairs that shrink or stretch by a given factor, against the Gaussian's exact law and per-pair bound (a reference for the other families; the bound is not proved for s > 3).]] -->
<!-- github -->

![Two panels for Breast Cancer, against target size k: the share of pairs whose squared distance shrinks to half or less (top) and grows by half or more (bottom), one line per family, with the Gaussian's exact law as a solid line and the per-pair bound as a dashed line.](../study/figures/fig4_distortion_tails_vs_bound.png)

*Plate 8.* Share of Breast Cancer pairs with *q* ≤ 0.5 (top) and *q* ≥ 1.5 (bottom) against target size *k*, for each family, with the Gaussian's exact law and per-pair bound.

<!-- /github -->

## What this does not show

- It does not show that sparse matrices preserve structure better than dense ones. The two positive intervals, Eq. 4 against the Gaussian on Glass at k = 3 and Eq. 4 against Eq. 3 on Breast Cancer at k = 6, do not survive the correction.
- "No detectable difference" at small k is not "the same": for the primary comparisons that do not survive Holm's correction, the intervals are up to 5.3 points wide on each side.
- It uses five small tabular data sets, SVMs, and a 70/30 split. Very sparse projections in high dimensions, where Li, Hastie, and Church worked, are not tested.
- The Johnson–Lindenstrauss guarantee is not met at any target size here. That means no guarantee, not that distortion must be large.
- Distortion is measured and compared with theory, but no experiment here shows that it causes the accuracy differences.

## Run it yourself

`python scripts/kamp_hw3.py --datasets iris breast_cancer wine seeds glass --legacy-seeding` reproduces the assignment's numbers. The study runs with the scripts in `study/` (`run_svm.py` and `run_distortion.py` each stop after nine minutes and resume when rerun, so repeat each until it no longer prints "rerun to resume"), and `study/analyze.py` and `study/figures.py` rebuild the summaries and figures from the committed results. The committed per-draw values are rounded, so rebuilding from them can change ties or decisions exactly at a threshold.
