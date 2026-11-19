#!/usr/bin/env python3
"""
data_loader.py - Dataset Loading and Caching

This module handles loading, downloading, and caching datasets for HW3.
It supports the two required datasets (Iris, Breast Cancer) and three
additional datasets (Wine, Seeds, Glass Identification).

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 3 – Random Projections and Multiclass Classification
Date: February 13, 2026
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Project paths
HW3_DIR = SCRIPT_DIR.parent
DATA_DIR = HW3_DIR / "data"


# =============================================================================
# DATASET REGISTRY
# =============================================================================

# The DATASET_REGISTRY dictionary contains entries for each dataset, keyed by a unique identifier (e.g. "iris", "breast_cancer"). 
# Each entry includes the dataset's name, a description, the folder where local files are stored, the name of the cached CSV file, and the UCI ID for downloading.
# This module is responsible for ensuring that all datasets required for the HW3 assignment can be loaded reliably and efficiently, with clear handling of different 
# data sources and formats, and with a consistent interface for accessing the dataset information throughout the rest of the codebase.
DATASET_REGISTRY = {
    "iris": { # Unique key for the Iris dataset, used to identify it in the code and when loading
        "name": "Iris",
        "folder": "iris",
        "cached_csv": "iris_cached.csv",
        "description": "150 samples, 4 features, 3 classes",
        "uci_id": 53,
    },
    "breast_cancer": { # Unique key for the Breast Cancer dataset, used to identify it in the code and when loading
        "name": "Wisconsin Breast Cancer (Diagnostic)",
        "folder": "breast_cancer",
        "cached_csv": "breast_cancer_cached.csv",
        "description": "569 samples, 30 features, 2 classes",
        "uci_id": 17,
    },
    "wine": { # Unique key for the Wine dataset, used to identify it in the code and when loading
        "name": "Wine",
        "folder": "wine",
        "cached_csv": "wine_cached.csv",
        "description": "178 samples, 13 features, 3 classes",
        "uci_id": 109,
    },
    "seeds": { # Unique key for the Seeds dataset, used to identify it in the code and when loading
        "name": "Seeds",
        "folder": "seeds",
        "cached_csv": "seeds_cached.csv",
        "description": "210 samples, 7 features, 3 classes",
        "uci_id": 236,
    },
    "glass": { # Unique key for the Glass Identification dataset, used to identify it in the code and when loading
        "name": "Glass Identification",
        "folder": "glass",
        "cached_csv": "glass_cached.csv",
        "description": "214 samples, 9 features, 6 classes",
        "uci_id": 42,
    },
}

# Default datasets (required by assignment)
DEFAULT_DATASETS = ["iris", "breast_cancer"]


def get_available_datasets() -> list:
    """Return list of all available dataset keys."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_info(dataset_key: str) -> dict:
    """Return registry info for a given dataset key."""
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. "
            f"Available: {get_available_datasets()}"
        )
    return DATASET_REGISTRY[dataset_key]


# =============================================================================
# INDIVIDUAL DATASET LOADERS (from local files)
# =============================================================================

def _load_iris_from_local() -> Optional[dict]:
    """Load Iris dataset from local iris.data file."""
    data_path = DATA_DIR / "iris" / "iris.data"
    if not data_path.exists():
        return None

    # Try to load the Iris dataset from the local iris.data file:
    # If successful, it will read the data into a DataFrame, extract the features and labels, and return a dataset dictionary with all the relevant information. 
    # If any error occurs during this process (e.g., file not found, parsing error), it will catch the exception and print a warning message, then return None to indicate that loading from local file failed.
    try: 
        feature_names = [ # The feature names for the Iris dataset, which correspond to the columns in the iris.data file. These will be used as column names when reading the data and for reference in the dataset dictionary.
            "sepal_length", "sepal_width", "petal_length", "petal_width"
        ]
        col_names = feature_names + ["class"]
        df = pd.read_csv(data_path, header=None, names=col_names)
        # Drop any empty rows (the file may have a trailing blank line)
        df = df.dropna(subset=["class"])
        df = df[df["class"].str.strip() != ""]

        X = df[feature_names].values.astype(np.float64)
        y_raw = df["class"].values
        class_names = sorted(list(set(y_raw)))
        label_map = {name: i for i, name in enumerate(class_names)}
        y = np.array([label_map[c] for c in y_raw])

        # Return a dataset dictionary containing all the relevant information about the Iris dataset, 
        # including the features (X), labels (y), feature names, class names, dataset name, dataset key, 
        # and the number of samples, features, and classes:
        # This dictionary will be used by the rest of the codebase to access the dataset information in a consistent way.
        return { 
            "X": X, # Feature matrix containing the 150 samples and 4 features of the Iris dataset, extracted from the iris.data file
            "y": y, # Label vector containing the class labels for each sample in the Iris dataset, encoded as integers (0, 1, 2) corresponding to the three classes
            "feature_names": feature_names, # List of feature names for the Iris dataset, used as column names when reading the data and for reference in the dataset dictionary
            "class_names": class_names, # List of class names for the Iris dataset, derived from the unique values in the class column of the data, used for reference in the dataset dictionary
            "dataset_name": "Iris", # Human-readable name for the dataset, used in the dataset dictionary and for display purposes in the analysis and report generation
            "dataset_key": "iris", # Unique key for the dataset, used to identify it in the code and when loading, also included in the dataset dictionary
            "n_samples": X.shape[0], # Number of samples in the dataset, computed from the shape of the feature matrix X, included in the dataset dictionary for reference
            "n_features": X.shape[1], # Number of features in the dataset, computed from the shape of the feature matrix X, included in the dataset dictionary for reference
            "n_classes": len(class_names), # Number of classes in the dataset, computed from the length of the class_names list, included in the dataset dictionary for reference
        }
    except Exception as e:
        print(f"  [WARNING] Failed to load Iris from local file: {e}")
        return None


def _load_breast_cancer_from_local() -> Optional[dict]:
    """Load Breast Cancer dataset from local wdbc.data file."""
    data_path = DATA_DIR / "breast_cancer" / "wdbc.data"
    if not data_path.exists():
        return None

    # Try to load the Breast Cancer dataset from the local wdbc.data file:
    # If successful, it will read the data into a DataFrame, extract the features and labels, and return a dataset dictionary with all the relevant information.
    # If any error occurs during this process (e.g., file not found, parsing error), it will catch the exception and print a warning message, then return None to indicate that loading from local file failed.
    try:
        # 32 columns: ID, Diagnosis, 30 features
        # Feature names are not included in the file, so we define them based on the dataset documentation.
        # The first two columns are ID and Diagnosis, followed by 30 feature columns that describe various measurements of the cell nuclei in the breast tissue samples.
        feature_names = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean",
            "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se",
            "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst",
            "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
        ]
        col_names = ["id", "diagnosis"] + feature_names
        df = pd.read_csv(data_path, header=None, names=col_names)

        X = df[feature_names].values.astype(np.float64) # Feature matrix containing the samples and features of the Breast Cancer dataset, extracted from the wdbc.data file
        y_raw = df["diagnosis"].values
        class_names = ["Benign", "Malignant"]  # B=0, M=1
        label_map = {"B": 0, "M": 1}
        y = np.array([label_map[c] for c in y_raw]) # Label vector containing the class labels for each sample in the Breast Cancer dataset, encoded as integers (0, 1) corresponding to the two classes

        # Return a dataset dictionary containing all the relevant information about the Breast Cancer dataset: 
        # This includes the features (X), labels (y), feature names, class names, dataset name, dataset key,
        # and the number of samples, features, and classes. 
        # This dictionary will be used by the rest of the codebase to access the dataset information in a consistent way.
        return {
            "X": X, # Feature matrix containing the samples and features of the Breast Cancer dataset, extracted from the wdbc.data file
            "y": y, # Label vector containing the class labels for each sample in the Breast Cancer dataset, encoded as integers (0, 1) corresponding to the two classes
            "feature_names": feature_names, # List of feature names for the Breast Cancer dataset, defined based on the dataset documentation, used as column names when reading the data and for reference in the dataset dictionary
            "class_names": class_names, # List of class names for the Breast Cancer dataset, defined based on the dataset documentation, used for reference in the dataset dictionary
            "dataset_name": "Wisconsin Breast Cancer (Diagnostic)", # Human-readable name for the dataset, used in the dataset dictionary and for display purposes in the analysis and report generation
            "dataset_key": "breast_cancer", # Unique key for the dataset, used to identify it in the code and when loading, also included in the dataset dictionary
            "n_samples": X.shape[0], # Number of samples in the dataset, computed from the shape of the feature matrix X, included in the dataset dictionary for reference
            "n_features": X.shape[1], # Number of features in the dataset, computed from the shape of the feature matrix X, included in the dataset dictionary for reference
            "n_classes": len(class_names), # Number of classes in the dataset, computed from the length of the class_names list, included in the dataset dictionary for reference
        }
    except Exception as e:
        print(f"  [WARNING] Failed to load Breast Cancer from local file: {e}")
        return None


# =============================================================================
# GENERIC CACHED CSV LOADER / SAVER
# =============================================================================

def _save_to_cache(dataset_dict: dict, cache_path: Path) -> None:
    """Save a dataset dict to a cached CSV file."""
    os.makedirs(cache_path.parent, exist_ok=True)
    X = dataset_dict["X"]
    y = dataset_dict["y"]
    feature_names = dataset_dict["feature_names"]

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    # Also save class name mapping in a separate small file
    df.to_csv(cache_path, index=False)

    # Save metadata
    meta_path = cache_path.parent / (cache_path.stem + "_meta.csv")
    meta_df = pd.DataFrame({
        "class_index": range(dataset_dict["n_classes"]),
        "class_name": dataset_dict["class_names"],
    })
    meta_df.to_csv(meta_path, index=False)


def _load_from_cache(dataset_key: str, cache_path: Path, dataset_name: str) -> Optional[dict]:
    """Load a dataset from its cached CSV file."""
    meta_path = cache_path.parent / (cache_path.stem + "_meta.csv")
    if not cache_path.exists() or not meta_path.exists():
        return None

    # Try to load the dataset from the cached CSV file:
    # If successful, it will read the data into a DataFrame, extract the features and labels, and return a dataset dictionary with all the relevant information.
    # If any error occurs during this process (e.g., file not found, parsing error), it will catch the exception and print a warning message, then return None to indicate that loading from cache failed.
    try:
        df = pd.read_csv(cache_path)
        meta_df = pd.read_csv(meta_path)

        feature_names = [c for c in df.columns if c != "target"]
        X = df[feature_names].values.astype(np.float64)
        y = df["target"].values.astype(int)
        class_names = meta_df["class_name"].tolist()
        # Return a dataset dictionary containing all the relevant information about the dataset loaded from cache:
        return {
            "X": X, # Feature matrix containing the samples and features of the dataset loaded from cache, extracted from the cached CSV file
            "y": y, # Label vector containing the class labels for each sample in the dataset loaded from cache, extracted from the target column of the cached CSV file
            "feature_names": feature_names, # List of feature names for the dataset loaded from cache, derived from the column names of the cached CSV file (excluding the target column), used for reference in the dataset dictionary
            "class_names": class_names, # List of class names for the dataset loaded from cache, derived from the metadata CSV file, used for reference in the dataset dictionary
            "dataset_name": dataset_name, # Name of the dataset loaded from cache, used for reference in the dataset dictionary
            "dataset_key": dataset_key, # Key of the dataset loaded from cache, used for reference in the dataset dictionary
            "n_samples": X.shape[0], # Number of samples in the dataset loaded from cache, derived from the shape of the feature matrix
            "n_features": X.shape[1], # Number of features in the dataset loaded from cache, derived from the shape of the feature matrix
            "n_classes": len(class_names), # Number of classes in the dataset loaded from cache, derived from the length of the class names list
        }
    except Exception as e:
        print(f"  [WARNING] Failed to load {dataset_name} from cache: {e}")
        return None


# =============================================================================
# DOWNLOAD FUNCTIONS (fallback)
# =============================================================================

def _download_dataset_ucimlrepo(dataset_key: str, uci_id: int, dataset_name: str) -> Optional[dict]:
    """Download a dataset from the UCI ML Repository using ucimlrepo."""
    # This function attempts to download the specified dataset from the UCI ML Repository using the ucimlrepo Python API.

    # Try to import the ucimlrepo library and use it to fetch the dataset based on the provided UCI ID: 
    # If successful, it will extract the features, labels, feature names, and class names from the fetched data, 
    # handle any necessary encoding of the target labels, and return a dataset dictionary with all the relevant information. 
    # If any error occurs during this process (e.g., library not installed, download failure, parsing error), it will catch the exception 
    # and print a warning message, then return None to indicate that downloading from UCI ML Repository failed.
    try: 
        from ucimlrepo import fetch_ucirepo
        print(f"  Downloading {dataset_name} from UCI ML Repository (ID={uci_id})...")
        result = fetch_ucirepo(id=uci_id)

        X_df = result.data.features
        y_df = result.data.targets

        feature_names = list(X_df.columns)
        X = X_df.values.astype(np.float64)

        # Handle target column
        y_raw = y_df.iloc[:, 0].values
        # If numeric already, use as-is; else encode
        if np.issubdtype(y_raw.dtype, np.integer) or np.issubdtype(y_raw.dtype, np.floating):
            unique_vals = sorted(set(int(v) for v in y_raw))
            class_names = [str(v) for v in unique_vals]
            label_map = {v: i for i, v in enumerate(unique_vals)}
            y = np.array([label_map[int(v)] for v in y_raw])
        else:
            class_names = sorted(list(set(y_raw)))
            label_map = {name: i for i, name in enumerate(class_names)}
            y = np.array([label_map[c] for c in y_raw])

        # Drop any rows with NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Return a dataset dictionary containing all the relevant information about the downloaded dataset:
        # This includes the features (X), labels (y), feature names, class names, dataset name, dataset key, 
        # and the number of samples, features, and classes. 
        # This dictionary will be used by the rest of the codebase to access the dataset information in a consistent way.
        return {
            "X": X, # Feature matrix containing the samples and features of the downloaded dataset from UCI ML Repository, extracted using the ucimlrepo API
            "y": y, # Label vector containing the class labels for each sample in the downloaded dataset, encoded as integers corresponding to the unique classes in the target column
            "feature_names": feature_names, # List of feature names for the downloaded dataset, extracted from the column names of the features DataFrame obtained from ucimlrepo
            "class_names": class_names, # List of class names for the downloaded dataset, derived from the unique values in the target column of the data, used for reference in the dataset dictionary
            "dataset_name": dataset_name, # Human-readable name for the dataset, used in the dataset dictionary and for display purposes in the analysis and report generation
            "dataset_key": dataset_key, # Unique key for the dataset, used to identify it in the code and when loading, also included in the dataset dictionary
            "n_samples": X.shape[0], # Number of samples in the dataset, computed from the shape of the feature matrix X, included in the dataset dictionary for reference
            "n_features": X.shape[1], # Number of features in the dataset, computed from the shape of the feature matrix X, included in the dataset dictionary for reference
            "n_classes": len(class_names), # Number of classes in the dataset, computed from the length of the class_names list, included in the dataset dictionary for reference
        }
    except Exception as e:
        print(f"  [WARNING] ucimlrepo download failed for {dataset_name}: {e}")
        return None


def _download_dataset_sklearn(dataset_key: str, dataset_name: str) -> Optional[dict]:
    """Fallback: try loading from sklearn.datasets."""
    # This function attempts to load the specified dataset using the sklearn.datasets module as a fallback option.

    # Try to import the sklearn.datasets library and use it to load the dataset based on the provided dataset key:
    # If successful, it will extract the features, labels, feature names, and class names from the loaded data,
    # handle any necessary encoding of the target labels, and return a dataset dictionary with all the relevant information.
    # If any error occurs during this process (e.g., library not installed, dataset not available in sklearn, parsing error), 
    # it will catch the exception and print a warning message, then return None to indicate that loading from sklearn failed.
    try: 
        from sklearn import datasets as sk_datasets

        loader_map = { # Mapping of dataset keys to their corresponding sklearn loader functions
            "iris": sk_datasets.load_iris,
            "breast_cancer": sk_datasets.load_breast_cancer,
            "wine": sk_datasets.load_wine,
        } # Note that seeds and glass are not available in sklearn, so they are not included in this mapping.

        if dataset_key not in loader_map: # e.g., if dataset_key is "seeds" or "glass"
            return None

        print(f"  Loading {dataset_name} from scikit-learn...")
        data = loader_map[dataset_key]()
        X = data.data.astype(np.float64)
        y = data.target.astype(int)
        feature_names = list(data.feature_names)
        class_names = list(data.target_names)

        # Return a dataset dictionary containing all the relevant information about the loaded dataset from sklearn:
        # This includes the features (X), labels (y), feature names, class names, dataset name, dataset key,
        # and the number of samples, features, and classes. This dictionary will be used by the rest of the codebase 
        # to access the dataset information in a consistent way.
        return {
            "X": X, # Feature matrix containing the samples and features of the dataset loaded from sklearn, extracted using the corresponding loader function
            "y": y, # Label vector containing the class labels for each sample in the dataset loaded from sklearn, encoded as integers corresponding to the unique classes in the target variable
            "feature_names": feature_names, # List of feature names for the dataset loaded from sklearn, extracted from the feature_names attribute of the loaded data
            "class_names": class_names, # List of class names for the dataset loaded from sklearn, extracted from the target_names attribute of the loaded data, used for reference in the dataset dictionary
            "dataset_name": dataset_name, # Human-readable name for the dataset, used in the dataset dictionary and for display purposes in the analysis and report generation
            "dataset_key": dataset_key, # Unique key for the dataset, used to identify it in the code and when loading, also included in the dataset dictionary
            "n_samples": X.shape[0], # Number of samples in the dataset, computed from the shape of the feature matrix X, included in the dataset dictionary for reference
            "n_features": X.shape[1], # Number of features in the dataset, computed from the shape of the feature matrix X, included in the dataset dictionary for reference
            "n_classes": len(class_names), # Number of classes in the dataset, computed from the length of the class_names list, included in the dataset dictionary for reference
        }
    except Exception as e:
        print(f"  [WARNING] sklearn fallback failed for {dataset_name}: {e}")
        return None


# Direct-download URLs for datasets not available via ucimlrepo Python API:
# These are used as a last-resort fallback if both local loading and ucimlrepo downloading fail.
_DIRECT_DOWNLOAD_URLS = {
    "seeds": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
        "feature_names": [
            "area", "perimeter", "compactness", "length_of_kernel",
            "width_of_kernel", "asymmetry_coefficient", "length_of_groove",
        ],
        "class_names": ["Kama", "Rosa", "Canadian"],
        "sep": r"\s+",
        "header": None,
        "target_col": -1,   # last column is class (1,2,3)
    },
    "glass": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        "feature_names": [
            "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe",
        ],
        "class_names": ["1", "2", "3", "5", "6", "7"],
        "sep": ",",
        "header": None,
        "id_col": 0,        # first column is ID
        "target_col": -1,   # last column is class
    },
}


def _download_dataset_direct(dataset_key: str, dataset_name: str) -> Optional[dict]:
    """
    Download a dataset directly from its UCI URL.
    
    This function attempts to download the specified dataset directly from its URL on the UCI ML Repository as a last-resort fallback option.
    It checks if the dataset key is in the _DIRECT_DOWNLOAD_URLS dictionary, and if so, it tries to download the dataset from the specified URL, 
    read it into a DataFrame, extract the features and labels, handle any necessary encoding of the target labels, and return a dataset dictionary 
    with all the relevant information. 
    If any error occurs during this process (e.g., download failure, parsing error), it will catch the exception and print a warning message, 
    then return None to indicate that direct downloading failed.
    """
    if dataset_key not in _DIRECT_DOWNLOAD_URLS: 
        return None

    info = _DIRECT_DOWNLOAD_URLS[dataset_key]
    url = info["url"]

    # Try to download the dataset directly from the specified URL, read it into a DataFrame, extract the features and labels, 
    # handle any necessary encoding of the target labels, and return a dataset dictionary with all the relevant information:
    # If any error occurs during this process (e.g., download failure, parsing error), it will catch the exception and print a warning message, 
    # then return None to indicate that direct downloading failed.
    try:
        import urllib.request
        import io

        print(f"  Downloading {dataset_name} from {url} ...")
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")

        df = pd.read_csv(io.StringIO(raw), sep=info["sep"], header=info.get("header"))

        # Drop ID column if present
        if "id_col" in info:
            df = df.drop(df.columns[info["id_col"]], axis=1)

        # Split features and target
        target_col_idx = info["target_col"]
        y_raw = df.iloc[:, target_col_idx].values
        X_cols = [i for i in range(df.shape[1]) if i != (df.shape[1] + target_col_idx if target_col_idx < 0 else target_col_idx)]
        X = df.iloc[:, X_cols].values.astype(np.float64)

        # Encode labels 0..C-1
        unique_vals = sorted(set(int(v) for v in y_raw))
        class_names = info["class_names"]
        label_map = {v: i for i, v in enumerate(unique_vals)}
        y = np.array([label_map[int(v)] for v in y_raw])

        # Drop NaN rows
        valid = ~np.isnan(X).any(axis=1)
        X, y = X[valid], y[valid]

        feature_names = info["feature_names"]

        # Return a dataset dictionary containing all the relevant information about the downloaded dataset:
        # This includes the features (X), labels (y), feature names, class names, dataset name, dataset key, and the number of samples, 
        # features, and classes. This dictionary will be used by the rest of the codebase to access the dataset information in a consistent way.
        result = {
            "X": X, # Feature matrix containing the samples and features of the dataset downloaded directly from the specified URL, extracted by reading the raw data into a DataFrame and processing it according to the specified format
            "y": y, # Label vector containing the class labels for each sample in the dataset downloaded directly from the specified URL, encoded as integers corresponding to the unique classes in the target variable
            "feature_names": feature_names, # List of feature names for the dataset downloaded directly from the specified URL, defined in the _DIRECT_DOWNLOAD_URLS dictionary, used as column names when reading the data and for reference in the dataset dictionary
            "class_names": class_names, # List of class names for the dataset downloaded directly from the specified URL, defined in the _DIRECT_DOWNLOAD_URLS dictionary, used for reference in the dataset dictionary
            "dataset_name": dataset_name, # Human-readable name for the dataset, used in the dataset dictionary and for display purposes in the analysis and report generation
            "dataset_key": dataset_key, # Unique key for the dataset, used to identify it in the code and when loading, also included in the dataset dictionary
            "n_samples": X.shape[0], # Number of samples in the dataset, computed from the shape of the feature matrix X, included in the dataset dictionary for reference
            "n_features": X.shape[1], # Number of features in the dataset, computed from the shape of the feature matrix X, included in the dataset dictionary for reference
            "n_classes": len(class_names), # Number of classes in the dataset, computed from the length of the class_names list, included in the dataset dictionary for reference
        }
        print(f"  ✓ Downloaded {dataset_name} ({X.shape[0]} samples, "
              f"{X.shape[1]} features, {len(class_names)} classes)")
        return result

    except Exception as e:
        print(f"  [WARNING] Direct download failed for {dataset_name}: {e}")
        return None


# =============================================================================
# MAIN LOADING FUNCTION
# =============================================================================

def load_dataset(dataset_key: str) -> dict:
    """
    Load a dataset by key. Checks local files first, then cache, then downloads.

    Args:
        dataset_key: Key from DATASET_REGISTRY (e.g., 'iris', 'breast_cancer')

    Returns:
        Dataset dictionary with keys: X, y, feature_names, class_names,
        dataset_name, dataset_key, n_samples, n_features, n_classes
    """
    # Retrieve the dataset information from the DATASET_REGISTRY using the provided dataset key:
    # This information includes the dataset's name, folder for local files, cached CSV filename, and UCI ID for downloading.
    # This information will be used to determine how to load the dataset, whether from local files, cached CSVs, 
    # or by downloading from the UCI ML Repository or other sources.
    info = get_dataset_info(dataset_key) 
    dataset_name = info["name"] # Human-readable name for the dataset, used for display purposes in the loading process and in the dataset dictionary
    folder = info["folder"] # Folder name for storing local files and cached CSVs, used to construct paths for loading and saving
    cached_csv = info["cached_csv"] # Filename for the cached CSV, used to construct the path for loading and saving
    uci_id = info["uci_id"] # UCI ID for downloading the dataset, used when downloading from the UCI ML Repository

    # Construct the folder path and cache path for the dataset based on the DATA_DIR and the information from the registry.
    folder_path = DATA_DIR / folder
    cache_path = folder_path / cached_csv

    print(f"\n  Loading dataset: {dataset_name} ({dataset_key})")

    # --- Strategy 1: Load from original local files (Iris, Breast Cancer) ---
    local_loaders = {
        "iris": _load_iris_from_local,
        "breast_cancer": _load_breast_cancer_from_local,
    }
    if dataset_key in local_loaders:
        result = local_loaders[dataset_key]()
        if result is not None:
            # Cache it for future runs
            _save_to_cache(result, cache_path)
            print(f"  ✓ Loaded {dataset_name} from local files "
                  f"({result['n_samples']} samples, {result['n_features']} features, "
                  f"{result['n_classes']} classes)")
            return result

    # --- Strategy 2: Load from cached CSV ---
    result = _load_from_cache(dataset_key, cache_path, dataset_name)
    if result is not None:
        print(f"  ✓ Loaded {dataset_name} from cache "
              f"({result['n_samples']} samples, {result['n_features']} features, "
              f"{result['n_classes']} classes)")
        return result

    # --- Strategy 3: Download via ucimlrepo ---
    result = _download_dataset_ucimlrepo(dataset_key, uci_id, dataset_name)
    if result is not None:
        _save_to_cache(result, cache_path)
        print(f"  ✓ Downloaded and cached {dataset_name} "
              f"({result['n_samples']} samples, {result['n_features']} features, "
              f"{result['n_classes']} classes)")
        return result

    # --- Strategy 3b: Direct URL download (for seeds, glass, etc.) ---
    result = _download_dataset_direct(dataset_key, dataset_name)
    if result is not None:
        os.makedirs(folder_path, exist_ok=True)
        _save_to_cache(result, cache_path)
        return result

    # --- Strategy 4: Fallback to sklearn ---
    result = _download_dataset_sklearn(dataset_key, dataset_name)
    if result is not None:
        os.makedirs(folder_path, exist_ok=True)
        _save_to_cache(result, cache_path)
        print(f"  ✓ Loaded {dataset_name} from sklearn and cached "
              f"({result['n_samples']} samples, {result['n_features']} features, "
              f"{result['n_classes']} classes)")
        return result

    raise RuntimeError(
        f"Failed to load dataset '{dataset_key}' from any source. "
        f"Please check your data directory or internet connection."
    )


def load_datasets(dataset_keys: list) -> dict:
    """
    Load multiple datasets.

    Args:
        dataset_keys: List of dataset keys to load

    Returns:
        Dictionary mapping dataset_key -> dataset_dict
    """
    datasets = {}
    for key in dataset_keys:
        datasets[key] = load_dataset(key)
    return datasets


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing data_loader.py...")
    print(f"Available datasets: {get_available_datasets()}")
    print(f"Default datasets: {DEFAULT_DATASETS}")
    # Load and print info about default datasets:
    for key in DEFAULT_DATASETS:
        ds = load_dataset(key)
        print(f"\n{ds['dataset_name']}:")
        print(f"  X shape: {ds['X'].shape}")
        print(f"  y shape: {ds['y'].shape}")
        print(f"  Classes: {ds['class_names']}")
        print(f"  Features: {ds['feature_names'][:5]}{'...' if len(ds['feature_names']) > 5 else ''}")
