# *Eyeing the Iris*: four kernels, four problems

*Coursework for CS 7063 Advanced Machine Learning, Spring 2026; Extended November 2026.*

A classifier can be as simple as asking which class average a flower is closer to. This chapter asks how much the answer depends on what "closer" means. The study takes one such classifier, gives it four ways to measure similarity, and tries them on four sorting problems drawn from the Iris flowers: all four on the three harder problems, and the plain dot product on the easy one. It also asks a second question that sounds easy: does it help to train on more of the data?

## The classifier

Each class gets an average, and a new flower goes to the class whose average is nearer. Nothing is tuned. Every training flower of a class counts equally, and the only things taken from the training flowers are the two class averages and a bias term, which turns the comparison into a comparison of distances to the two averages. Given the data and the kernel, the classifier is fixed. It returns a label, not a probability.

The only subtle part is the word "nearer". A kernel is a rule for how similar two flowers are. Measuring the distance to a class average through a kernel means taking the average in the kernel's feature space, so what counts as near depends on the kernel. With a Gaussian kernel, the rule turns out to compare two Parzen density estimates, one per class, against a threshold set by the bias term, which is why the paper calls it a Parzen-style classifier.

<!-- [[demo (3D): Iris on three of its four measurements, the 150 flowers colored by species, with the chosen problem's two class averages drawn as stars and the distance between them. The reader picks Problems 1 to 4 and the three measurements drawn; on Problems 1 to 3, each pooled species' own average appears as a ring. The case turns and zooms, and a flower or an average can be picked to read it. On Problem 3, the "rest" average sits next to versicolor, which shows why the problem is hard for a rule that compares averages of the raw measurements.]] -->
<!-- github -->

*Plate 1, live:* [Iris in three measurements](https://ryanjosephkamp.github.io/advanced-ml-studies/iris-parzen-classifier/#plate-1), a live plate on the book's site.

<!-- /github -->

## The four kernels

- **Dot product.** Similarity is the dot product of the two measurement vectors, which grows with both their alignment and their lengths. The rule becomes the nearest-centroid rule on the raw measurements, with a flat boundary (a hyperplane in the four measurements).
- **Polynomial, degree 2 and degree 3.** The boundary is a fixed surface of that degree in the four measurements, set by moments of the two classes' measurements.
- **Gaussian.** Similarity falls off smoothly with distance. The width is σ = 1/√2 ≈ 0.71.

The width came with the assignment and is the same in every run. It was not tuned, so nothing was chosen on test data.

## The data and the four problems

The data are the 150 Iris flowers that ship with scikit-learn: 50 each of setosa, versicolor, and virginica, each with four measurements in centimeters. The measurements are used as they are, with no scaling. Four binary problems are built from them.

<!-- [[table: the four problems. Columns: problem, one class, the other class, linearly separable (yes only for Problem 1). Problem 1 is setosa against the other two species. Problem 2 is virginica against the other two. Problem 3 is versicolor against the other two. Problem 4 is versicolor against virginica, 100 flowers.]] -->
<!-- github -->

The four problems. Problems 1 to 3 set one species against the other two, so their classes are in a 1:2 ratio.

<table><thead><tr><th>Problem</th><th>One class</th><th>The other class</th><th>Linearly separable</th></tr></thead><tbody><tr><td>1</td><td>Setosa</td><td>Versicolor and virginica</td><td>Yes</td></tr><tr><td>2</td><td>Virginica</td><td>Setosa and versicolor</td><td>No</td></tr><tr><td>3</td><td>Versicolor</td><td>Setosa and virginica</td><td>No</td></tr><tr><td>4</td><td>Versicolor</td><td>Virginica (setosa removed)</td><td>No</td></tr></tbody></table>

<!-- /github -->

Setosa sits apart from the other two species, so Problem 1 can be solved with a flat boundary. Versicolor sits between setosa and virginica, so on Problem 3, the "rest" class has one part on each side of it.

## How the test was run

The original coursework evaluated five training shares, from 90/10 to 50/50, with one random split each. All five splits came from the same shuffle, so each test set sat inside the next larger one, and every share was a single draw. With 15 test flowers at 90/10 on Problems 1 to 3, one mistake moves accuracy by almost seven points.

This version repeats the procedure on 500 random splits per share, using seeds 0 to 499, which include the original seed. The results are the mean test accuracy over the splits, a 95% confidence interval for that mean, and the 2.5th to 97.5th percentiles of single splits. The interval covers the choice of splits only, because every split draws from the same 150 flowers. The baseline is the majority-class rate: the larger class's share of each test set, averaged over splits. It's at least what always answering with one fixed label would score. On Problem 4, whose two classes are the same size, a fixed label scores about 0.5.

## What happened

<!-- [[figure (pop-out): repeated_split_by_kernel.png. Test accuracy at 80/20 for each kernel tested on each problem: mean, 2.5th to 97.5th percentiles of single splits, the original split, and the mean majority-class rate, with the paper's caption.]] -->
<!-- github -->

![Four panels, one per problem, of test accuracy at 80/20 over 500 random splits: for each kernel tested, a filled marker at the mean, a bar spanning the 2.5th to 97.5th percentiles of single splits, and an open marker at the original split, with a dashed line at the mean majority-class rate. Problem 1's panel shows the dot product only; on Problems 2 to 4, the Gaussian's mean is the highest, and on Problem 3, the other three kernels' means fall below the dashed line.](../repeated_splits/repeated_split_by_kernel.png)

*Plate 2.* Test accuracy at 80/20 over 500 random splits, with 30 test flowers on Problems 1 to 3 and fewer on Problem 4, which leaves out setosa. Filled markers are means, bars span the 2.5th to 97.5th percentiles of single splits, open markers are the original split (seed 42), and the dashed line is the mean majority-class rate.

<!-- /github -->

**The Gaussian kernel wins where the problems are not linearly separable.** On Problems 2, 3, and 4, it averages 0.938, 0.926, and 0.929 (0.931 averaged over the three). The dot product, degree 2, and degree 3 average 0.775, 0.757, and 0.747. On the same split, the Gaussian is at least as accurate as the best other kernel in 99.2%, 99.7%, and 95.2% of the 2,500 splits on those three problems.

<!-- [[table: Table I of the paper. Test accuracy by problem and kernel, pooled over the five training shares: mean, 95% interval of the mean (computed from the 500 per-seed means), 2.5th to 97.5th percentiles of single splits, and the mean of the five original single-split runs, with the paper's caption.]] -->
<!-- github -->

**Table I.** Test accuracy by problem and kernel, pooled over the five training shares (2,500 splits per row). The 95% CI of the mean is computed from the 500 per-seed means (each the mean of one seed's five splits) and covers the choice of splits only. “Original” is the mean of the five original single-split runs.

<table><thead><tr><th>Prob.</th><th>Kernel</th><th>Mean</th><th>95% CI of mean</th><th>2.5th to 97.5th percentiles of splits</th><th>Original</th></tr></thead><tbody><tr><td>1</td><td>Dot product</td><td>0.987</td><td>0.986–0.988</td><td>0.933–1.000</td><td>1.000</td></tr><tr><td>2</td><td>Dot product</td><td>0.821</td><td>0.816–0.825</td><td>0.667–0.933</td><td>0.788</td></tr><tr><td><span>2</span></td><td>Polynomial, <i>d</i>=2</td><td>0.865</td><td>0.861–0.869</td><td>0.733–1.000</td><td>0.828</td></tr><tr><td><span>2</span></td><td>Polynomial, <i>d</i>=3</td><td>0.880</td><td>0.876–0.883</td><td>0.767–1.000</td><td>0.909</td></tr><tr><td><span>2</span></td><td>Gaussian</td><td><b>0.938</b></td><td>0.935–0.941</td><td>0.842–1.000</td><td>0.951</td></tr><tr><td>3</td><td>Dot product</td><td>0.622</td><td>0.617–0.627</td><td>0.467–0.800</td><td>0.674</td></tr><tr><td><span>3</span></td><td>Polynomial, <i>d</i>=2</td><td>0.574</td><td>0.569–0.580</td><td>0.400–0.747</td><td>0.628</td></tr><tr><td><span>3</span></td><td>Polynomial, <i>d</i>=3</td><td>0.546</td><td>0.539–0.552</td><td>0.333–0.860</td><td>0.555</td></tr><tr><td><span>3</span></td><td>Gaussian</td><td><b>0.926</b></td><td>0.923–0.929</td><td>0.800–1.000</td><td>0.966</td></tr><tr><td>4</td><td>Dot product</td><td>0.881</td><td>0.877–0.886</td><td>0.738–1.000</td><td>0.924</td></tr><tr><td><span>4</span></td><td>Polynomial, <i>d</i>=2</td><td>0.832</td><td>0.827–0.837</td><td>0.667–1.000</td><td>0.875</td></tr><tr><td><span>4</span></td><td>Polynomial, <i>d</i>=3</td><td>0.817</td><td>0.812–0.821</td><td>0.644–1.000</td><td>0.859</td></tr><tr><td><span>4</span></td><td>Gaussian</td><td><b>0.929</b></td><td>0.925–0.932</td><td>0.800–1.000</td><td>0.987</td></tr></tbody></table>

<!-- /github -->

**On Problem 3, the other kernels fall below the baseline.** The dot product, degree 2, and degree 3 average 0.622, 0.574, and 0.546. The baseline, the larger class's share of each test set, is about 0.668, close to what always answering "rest" scores. A rule that compares class averages of the raw measurements, as the dot product does, has little to work with when one class lies between the two parts of the other. The Gaussian averages 0.926. This fits its window being small next to the distances between the average measurements of neighboring species: 3.2 cm from setosa to versicolor and 1.6 cm from versicolor to virginica, against a width of about 0.71.

<!-- [[demo (2D, an exploration beyond the study): the classifier refit in the page on the two petal measurements of every flower in the chosen problem, with its decision score over the petal plane drawn as stipple, the boundary where the score crosses zero, and the flowers as points. The reader picks the problem and the kernel, moves a probe to read the score, and can change the Gaussian's width, which the study fixed. The study used all four measurements and computed no decision boundaries, so the plate shows no accuracy.]] -->
<!-- github -->

*Plate 3, live:* [Where the rule draws the line](https://ryanjosephkamp.github.io/advanced-ml-studies/iris-parzen-classifier/#plate-3), a live plate on the book's site.

<!-- /github -->

**Problem 1 is easy, but not perfect.** Setosa can be separated with a flat boundary, and the dot-product rule gets 0.987 on the random test sets. On the original splits, it got every test flower right, yet it misclassified one versicolor flower in each training set (two at 90/10), a flower nearest its own species' average. The rule, though, compares the setosa average with the average of the other two species together, which virginica pulls away from that flower, and the flower lies nearer the setosa average.

**The polynomial kernels underfit.** Their training accuracy is as low as their test accuracy. Over random splits, the two averages differ by at most 0.022, except for degree 3 on Problem 3 (0.618 against 0.546). The only things taken from the training flowers are two class averages and a bias, so there is little to memorize with. The degree-3 gap likely comes from each training flower counting toward its own class average.

**More training data barely changed the average.** Averaged over the 13 problem–kernel pairs, mean test accuracy is 0.817, 0.816, 0.817, 0.817, and 0.816 at 90/10 down to 50/50. The largest difference between shares is under 0.002. Single pairs move more, by up to 0.033. A rule built from class averages may already estimate them well from 75 flowers (50/50 on Problems 1 to 3); that explanation fits the result but was not tested. The kernel matters far more: on Problem 3, the averages of the best and worst kernels differ by 0.380.

<!-- [[figure (pop-out): repeated_split_by_ratio.png. Mean test accuracy over Problems 2 to 4 for each kernel by training share, with the 2.5th to 97.5th percentiles of the three-problem mean on single splits and the original split, with the paper's caption.]] -->
<!-- github -->

![Mean test accuracy over Problems 2 to 4 for each of the four kernels at training shares from 50% to 90%, with bars spanning the 2.5th to 97.5th percentiles of the three-problem mean on single splits and open markers at the original split. The Gaussian's mean sits well above the other three at every share, and each kernel's mean is nearly flat across shares.](../repeated_splits/repeated_split_by_ratio.png)

*Plate 4.* Mean test accuracy over Problems 2 to 4 by training share, 500 random splits per share. Bars span the 2.5th to 97.5th percentiles of the three-problem mean on single splits, and open markers are the original split.

<!-- /github -->

**The original split was a favorable draw.** On the original splits, the average over all 13 problem–kernel pairs ranged from 0.824 to 0.862 across shares, with no steady trend. The five test sets are nested, so the five values are not independent, and the spread reflects which flowers each set happened to hold. The original split beat the mean of its 500 random splits in 46 of 65 runs. For the Gaussian on Problems 2 to 4, it scored 0.968, against 0.931 over random splits.

## What standardizing changes

The dot-product and polynomial kernels depend on the measurements' origin and units, and the measurements here are raw centimeters. As an added comparison, repeating the evaluation on standardized features (scaled with each problem's training rows), on the same splits, changes one result a lot, and makes Problem 1 perfect. On Problem 3, the degree-2 kernel rises from 0.574 to 0.853 and degree 3 from 0.546 to 0.771. On Problems 2 and 4, the polynomial and dot-product kernels move by at most 0.019. The Gaussian drops on Problems 2 and 3 (0.938 to 0.904 and 0.926 to 0.908) and still has the highest mean on Problems 2 to 4. On Problem 3, its lead over degree 2 shrinks from 0.35 to 0.06.

<!-- [[table: Table II of the paper, raw against standardized test accuracy for all 13 problem–kernel pairs, with the share of splits on which standardizing helped or hurt.]] -->
<!-- github -->

**Table II.** Test accuracy on raw and standardized features, same 2,500 splits per row. “Higher / lower” is the share of splits on which standardizing raised or lowered accuracy.

<table><thead><tr><th>Prob.</th><th>Kernel</th><th>Raw</th><th>Standardized</th><th>Change</th><th>Higher / lower</th></tr></thead><tbody><tr><td>1</td><td>Dot product</td><td>0.987</td><td>1.000</td><td>+0.013</td><td>45.2% / 0.0%</td></tr><tr><td>2</td><td>Dot product</td><td>0.821</td><td>0.836</td><td>+0.016</td><td>52.1% / 5.3%</td></tr><tr><td><span>2</span></td><td>Polynomial, <i>d</i>=2</td><td>0.865</td><td>0.872</td><td>+0.007</td><td>44.0% / 32.1%</td></tr><tr><td><span>2</span></td><td>Polynomial, <i>d</i>=3</td><td>0.880</td><td>0.881</td><td>+0.001</td><td>44.2% / 32.1%</td></tr><tr><td><span>2</span></td><td>Gaussian</td><td>0.938</td><td>0.904</td><td>−0.034</td><td>1.8% / 71.0%</td></tr><tr><td>3</td><td>Dot product</td><td>0.622</td><td>0.646</td><td>+0.024</td><td>61.0% / 19.8%</td></tr><tr><td><span>3</span></td><td>Polynomial, <i>d</i>=2</td><td>0.574</td><td>0.853</td><td>+0.278</td><td>99.7% / &lt;0.1%</td></tr><tr><td><span>3</span></td><td>Polynomial, <i>d</i>=3</td><td>0.546</td><td>0.771</td><td>+0.225</td><td>92.0% / 5.4%</td></tr><tr><td><span>3</span></td><td>Gaussian</td><td>0.926</td><td>0.908</td><td>−0.018</td><td>12.2% / 50.0%</td></tr><tr><td>4</td><td>Dot product</td><td>0.881</td><td>0.881</td><td>−0.001</td><td>34.3% / 34.9%</td></tr><tr><td><span>4</span></td><td>Polynomial, <i>d</i>=2</td><td>0.832</td><td>0.850</td><td>+0.018</td><td>48.0% / 22.0%</td></tr><tr><td><span>4</span></td><td>Polynomial, <i>d</i>=3</td><td>0.817</td><td>0.808</td><td>−0.009</td><td>35.8% / 39.0%</td></tr><tr><td><span>4</span></td><td>Gaussian</td><td>0.929</td><td>0.928</td><td>−0.001</td><td>31.3% / 32.1%</td></tr></tbody></table>

<!-- /github -->

## What this does not show

- One Gaussian width, from the assignment. Nothing here shows it's a good width.
- Standardizing centers and scales the measurements at once, and this study doesn't test the two separately.
- The rule weights both classes equally, which ignores the 1:2 class ratio on the one-versus-rest problems.
- The random splits reuse the same 150 flowers, so the intervals describe the choice of splits, not new data.
- No significance tests were run, and the classifier gives labels, not probabilities, so calibration was not studied. Decision boundaries were not studied either.
- It is one small data set.

## Run it yourself

`pip install -r requirements.txt` installs the package versions used. `python kamp_hw1.py` runs the original single-split study. `python repeated_splits.py` runs the random-split evaluation and the standardized-feature comparison, and takes seconds on a laptop. Both load Iris through scikit-learn.
