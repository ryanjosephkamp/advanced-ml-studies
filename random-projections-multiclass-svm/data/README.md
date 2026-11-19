# Data

The five data sets used by `scripts/` and `study/`, stored here so both run without a download.
All come from the UCI Machine Learning Repository (https://archive.ics.uci.edu). Each data set's
UCI page lists its license (Creative Commons Attribution 4.0 International, CC BY 4.0); cite the
sources below when you use them.

| Folder | Data set (UCI ID) | Samples | Features | Classes | Files |
|---|---|---|---|---|---|
| `iris/` | Iris (53) | 150 | 4 | 3 | `iris.data`, `bezdekIris.data`, `iris.names`, `Index` (UCI's files) |
| `breast_cancer/` | Breast Cancer Wisconsin (Diagnostic) (17) | 569 | 30 | 2 | `wdbc.data`, `wdbc.names` (UCI's files) |
| `wine/` | Wine (109) | 178 | 13 | 3 | `wine_cached.csv`, `wine_cached_meta.csv` |
| `seeds/` | Seeds (236) | 210 | 7 | 3 | `seeds_cached.csv`, `seeds_cached_meta.csv` |
| `glass/` | Glass Identification (42) | 214 | 9 | 6 | `glass_cached.csv`, `glass_cached_meta.csv` |

Notes:

- **Iris:** `iris.data` is UCI's original file, which differs from Fisher's published data in
  rows 35 and 38. `bezdekIris.data` has the corrections and is identical to scikit-learn's
  `load_iris()`. The February 13, 2026 run used `iris.data`; the study uses `bezdekIris.data`.
- **Breast Cancer:** identical to scikit-learn's `load_breast_cancer()` features. Labels here are
  malignant = 1, benign = 0 (scikit-learn codes malignant as 0).
- **Wine, Seeds, Glass:** converted from the UCI files by `scripts/data_loader.py` on February 13,
  2026. `<name>_cached.csv` holds the features and a 0-based `target`; `<name>_cached_meta.csv`
  maps each index to the original class label (Glass has no class 4). Wine is identical to
  scikit-learn's `load_wine()`; scikit-learn has no copy of Seeds or Glass.

Sources:

- R. A. Fisher, "The use of multiple measurements in taxonomic problems," *Annals of Eugenics*,
  vol. 7, no. 2, pp. 179–188, 1936.
- W. H. Wolberg, W. N. Street, and O. L. Mangasarian, "Image analysis and machine learning applied
  to breast cancer diagnosis and prognosis," *Analytical and Quantitative Cytology and Histology*,
  vol. 17, no. 2, pp. 77–87, 1995.
- S. Aeberhard, D. Coomans, and O. de Vel, "Comparative analysis of statistical pattern
  recognition methods in high dimensional settings," *Pattern Recognition*, vol. 27, no. 8,
  pp. 1065–1077, 1994.
- M. Charytanowicz, J. Niewczas, P. Kulczycki, P. A. Kowalski, S. Łukasik, and S. Żak, "Complete
  gradient clustering algorithm for features analysis of X-ray images," in *Information
  Technologies in Biomedicine*, Springer, 2010, pp. 15–24.
- B. German, "Glass Identification," UCI Machine Learning Repository, 1987.
