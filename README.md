# Advanced ML Studies

Four studies from CS 7063, spring 2026. Each folder holds one project's code, data notes, results,
and figures.

| Folder | Study |
|---|---|
| [`iris-parzen-classifier`](iris-parzen-classifier/) | Parzen-window classifiers on Iris |
| [`classical-hopfield-networks`](classical-hopfield-networks/) | Hopfield networks and the capacity of their recall rules |
| [`random-projections-multiclass-svm`](random-projections-multiclass-svm/) | Random projections before multiclass SVMs |
| [`signal-analysis-emd`](signal-analysis-emd/) | Empirical mode decomposition features for EEG |

Known and potential problems in these studies, and the status of each, are listed in
[`KNOWN-ISSUES.md`](KNOWN-ISSUES.md).

The course book's copies of each study's figures, in its day and night inks, are in each project's `figures/site/`.
Each project's `figures/site/render.py` draws them with the study's own plotting code (using
[`tools/site_figures.py`](tools/site_figures.py)) and checks that they plot exactly what the study's figures plot.

Code is under the MIT License. Each project's `data/README.md` or README credits its data.
