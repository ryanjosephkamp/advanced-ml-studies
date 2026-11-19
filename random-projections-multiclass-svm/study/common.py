"""Shared pieces of the random-projection study: data, projection matrices, seeds, and the grid.

Every random choice is keyed by (draw, dataset), so any single cell can be rerun on its own:
  - draw r uses split seed r (stratified 70/30) and inner cross-validation seed r;
  - its projection matrices come from np.random.default_rng([r, crc32(dataset)]).

Projection families (r_ij scaled to unit variance; E = X R / sqrt(k)):
  gaussian     G, standard normal                                    (dense reference)
  rademacher   sign(G)                                                (the report's Eq. 3, s = 1)
  achlioptas3  sqrt(3) * sign(G) * [U < 1/3]                          (the report's Eq. 4, s = 3)
  verysparse   sqrt(s) * sign(G) * [U < 1/s], s = sqrt(d)             (Li, Hastie, and Church, 2006)
  onepercol    sqrt(s) * sign(G) * [U < 1/s], s = d                   (one nonzero per column on average)
Within a draw, every family but the Gaussian takes G's signs and keeps the entries where U < 1/s, so
the families share their signs: Eq. 3 differs from the Gaussian only in the sizes of its entries, and
the sparse families also in their zeros. G and U are drawn once per (draw, dataset) with KMAX columns;
a target size k uses the first k columns, so the projections are nested across k.
"""
import os
import zlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "study", "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
FIG_DIR = os.path.join(ROOT, "study", "figures")

DATASETS = ["iris", "breast_cancer", "wine", "seeds", "glass"]
LABELS = {"iris": "Iris", "breast_cancer": "Breast Cancer", "wine": "Wine", "seeds": "Seeds",
          "glass": "Glass"}
K_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24]
KMAX = max(K_GRID)
FAMILIES = ["gaussian", "rademacher", "achlioptas3", "verysparse", "onepercol"]
FAMILY_LABELS = {"gaussian": "Gaussian", "rademacher": "Eq. 3 (±1)", "achlioptas3": "Eq. 4 (s = 3)",
                 "verysparse": "Very sparse (s = √d)", "onepercol": "One per column (s = d)",
                 "pca": "PCA", "full": "All features"}
N_DRAWS = 100
TEST_SIZE = 0.3


def load_dataset(key: str, iris_file: str = "bezdekIris.data"):
    """Return (X, y, class_names) from the files in data/. No network access.

    Iris defaults to the corrected UCI file (bezdekIris.data, identical to scikit-learn's copy);
    pass iris_file="iris.data" for the file the February 13 run used.
    """
    if key == "iris":
        df = pd.read_csv(os.path.join(DATA_DIR, "iris", iris_file), header=None).dropna()
        codes, names = pd.factorize(df[4], sort=True)
        return df.iloc[:, :4].to_numpy(float), codes.astype(int), list(names)
    if key == "breast_cancer":
        df = pd.read_csv(os.path.join(DATA_DIR, "breast_cancer", "wdbc.data"), header=None)
        return df.iloc[:, 2:].to_numpy(float), (df[1] == "M").to_numpy(int), ["Benign", "Malignant"]
    df = pd.read_csv(os.path.join(DATA_DIR, key, f"{key}_cached.csv"))
    meta = pd.read_csv(os.path.join(DATA_DIR, key, f"{key}_cached_meta.csv"))
    return (df.drop(columns="target").to_numpy(float), df["target"].to_numpy(int),
            [str(c) for c in meta["class_name"]])


def k_values(d: int) -> list:
    """Target sizes used for a dataset with d features (k < d)."""
    return [k for k in K_GRID if k < d]


def sparsity(family: str, d: int) -> float:
    """s for the sparse families (entries nonzero with probability 1/s); 3 stands in for the Gaussian."""
    return {"gaussian": 3.0, "rademacher": 1.0, "achlioptas3": 3.0,
            "verysparse": float(np.sqrt(d)), "onepercol": float(d)}[family]


def families_for(d: int) -> list:
    """Families run for a dataset. Very sparse equals Eq. 4 when sqrt(d) = 3 (Glass), so it is skipped."""
    return [f for f in FAMILIES if not (f == "verysparse" and np.isclose(np.sqrt(d), 3.0))]


def base_matrices(draw: int, dataset: str, d: int):
    """The draw's Gaussian matrix G and uniform matrix U, both d x KMAX."""
    rng = np.random.default_rng([draw, zlib.crc32(dataset.encode("utf-8"))])
    return rng.standard_normal((d, KMAX)), rng.random((d, KMAX))


def projection_matrix(family: str, G: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    """The d x k matrix R for one family, built from the first k columns of G and U."""
    d = G.shape[0]
    g, u = G[:, :k], U[:, :k]
    if family == "gaussian":
        return g.copy()
    s = sparsity(family, d)
    return np.sqrt(s) * np.sign(g) * (u < 1.0 / s)


def project(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """E = X R / sqrt(k)."""
    return X @ R / np.sqrt(R.shape[1])
