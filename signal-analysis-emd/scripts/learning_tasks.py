#!/usr/bin/env python3
"""
learning_tasks.py - Classification and Clustering Learning Tasks

Performs KNN classification and K-Means clustering using precomputed
distance matrices and IMF features, with standard evaluation metrics.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 4 – ML for Signal Analysis using EMD
Date: March 8, 2026
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    accuracy_score, silhouette_score, balanced_accuracy_score,
    adjusted_rand_score, normalized_mutual_info_score
)


# =============================================================================
# CLASSIFICATION
# =============================================================================

def run_classification(features, labels, distance_matrix, measure,
                       label_names=None, n_neighbors=5, test_size=0.3,
                       random_state=42):
    """
    Run KNN *classification* (Option 1 from Task 5 in assignment) using precomputed distance matrix:
        - Performs a train/test split with stratification to maintain class balance.
        - Uses KNN with the precomputed distance matrix to predict labels on the test set.
        - Evaluates performance using F1 score, accuracy, confusion matrix, and classification report.

    Note: If the dataset is too small for meaningful classification, returns an empty result.

    Args:
        features: np.ndarray of shape (n_segments, n_features)
        labels: np.ndarray of shape (n_segments,)
        distance_matrix: np.ndarray of shape (n, n) — precomputed distances
        measure: Name of similarity measure used (for reporting)
        label_names: dict mapping label int → name string
        n_neighbors: Number of neighbors for KNN
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        dict with classification results
    """
    n_samples = len(labels)
    if n_samples < 10:
        return _empty_classification_result(measure, label_names)

    # Adjust n_neighbors if too large
    effective_k = min(n_neighbors, n_samples - int(n_samples * test_size) - 1)
    if effective_k < 1:
        effective_k = 1

    # Train/test split with stratification
    indices = np.arange(n_samples)
    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state,
            stratify=labels
        )
    except ValueError:
        # If stratification fails (e.g., too few samples per class)
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

    # KNN with precomputed distance matrix
    knn = KNeighborsClassifier(n_neighbors=effective_k, metric='precomputed')
    knn.fit(distance_matrix[np.ix_(train_idx, train_idx)], labels[train_idx])
    predictions = knn.predict(distance_matrix[np.ix_(test_idx, train_idx)])

    y_true = labels[test_idx]
    y_pred = predictions

    # Compute metrics
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Per-class F1
    unique_labels = sorted(np.unique(labels))
    target_names = [label_names.get(int(l), f'class_{l}') for l in unique_labels] if label_names else None
    report = classification_report(y_true, y_pred, target_names=target_names,
                                    zero_division=0)

    f1_per_class = {}
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0,
                          labels=unique_labels)
    for lbl, score in zip(unique_labels, f1_scores):
        name = label_names.get(int(lbl), f'class_{lbl}') if label_names else f'class_{lbl}'
        f1_per_class[name] = float(score)

    return {
        'f1_score': float(f1), # Overall weighted F1 score; convert to float for JSON serialization
        'f1_per_class': f1_per_class, # Per-class F1 scores as a dict mapping label names to scores
        'confusion_matrix': cm, # Confusion matrix as a numpy array
        'classification_report': report, # Text report from sklearn's classification_report function
        'accuracy': float(accuracy), # Accuracy score; convert to float for JSON serialization
        'balanced_accuracy': float(balanced_accuracy), # Mean per-class recall (chance = 1 / number of classes)
        'f1_macro': float(f1_macro), # Unweighted mean of per-class F1
        'label_names': label_names, # Original label names dict for reference
        'n_train': len(train_idx), # Number of training samples
        'n_test': len(test_idx), # Number of testing samples
        'n_neighbors': effective_k, # Number of neighbors used in KNN (adjusted if necessary)
        'measure': measure, # Name of similarity measure used (for reporting)
    }


def _empty_classification_result(measure, label_names):
    """Return empty result when classification cannot be performed."""
    return {
        'f1_score': 0.0, # Overall weighted F1 score; convert to float for JSON serialization; set to 0.0 when classification cannot be performed
        'f1_per_class': {}, # Per-class F1 scores as an empty dict when classification cannot be performed
        'confusion_matrix': np.array([]), # Empty confusion matrix when classification cannot be performed
        'classification_report': 'Insufficient data for classification.', # Text report indicating insufficient data for classification
        'accuracy': 0.0, # Accuracy score; convert to float for JSON serialization; set to 0.0 when classification cannot be performed
        'balanced_accuracy': 0.0, # Balanced accuracy; 0.0 when classification cannot be performed
        'f1_macro': 0.0, # Macro F1; 0.0 when classification cannot be performed
        'label_names': label_names, # Original label names dict for reference (can be None)
        'n_train': 0, # Number of training samples; set to 0 when classification cannot be performed
        'n_test': 0, # Number of testing samples; set to 0 when classification cannot be performed
        'n_neighbors': 0, # Number of neighbors used in KNN (adjusted if necessary)
        'measure': measure, # Name of similarity measure used (for reporting)
    }


# =============================================================================
# CLUSTERING
# =============================================================================

def run_clustering(features, labels, distance_matrix, measure,
                   n_clusters=None, random_state=42):
    """
    Run K-Means *clustering* (Option 2 from Task 5 in assignment) and evaluate with silhouette score:
        - Determines the number of clusters (either from unique labels or specified).
        - Performs K-Means clustering on the feature vectors (Euclidean; the distance measure does not
          change the clusters).
        - Scores the clusters with the silhouette under the precomputed distance matrix, and against the
          true labels with the adjusted Rand index (ARI) and normalized mutual information (NMI). The
          silhouette measures cluster separation only; ARI and NMI measure agreement with the classes.

    Args:
        features: np.ndarray of shape (n_segments, n_features)
        labels: np.ndarray of shape (n_segments,) — true labels for reference
        distance_matrix: np.ndarray of shape (n, n) — precomputed distances
        measure: Name of similarity measure used (for reporting)
        n_clusters: Number of clusters (None = infer from unique labels)
        random_state: Random seed

    Returns:
        dict with clustering results
    """
    # Determine number of clusters: either from unique labels or specified. 
    # If n_clusters is None, we infer it from the number of unique labels in the dataset. 
    # This allows us to evaluate how well the clustering algorithm can recover the true class structure when the number of clusters is not predefined.
    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    n_samples = len(labels) # Number of samples in the dataset
    # If the number of samples is less than the number of clusters plus one, we cannot perform meaningful clustering, so we return an empty result.
    if n_samples < n_clusters + 1:
        return _empty_clustering_result(measure, labels, n_clusters)

    # K-Means on feature vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features)

    # Silhouette score using precomputed distance matrix
    try:
        sil = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
    except ValueError:
        sil = -1.0

    return {
        'silhouette_score': float(sil), # Silhouette score; convert to float for JSON serialization
        'ari': float(adjusted_rand_score(labels, cluster_labels)), # Agreement with the true labels (0 = chance)
        'nmi': float(normalized_mutual_info_score(labels, cluster_labels)), # Shared information with the true labels
        'cluster_labels': cluster_labels, # Cluster labels assigned by K-Means
        'true_labels': labels, # Original true labels for reference
        'n_clusters': n_clusters, # Number of clusters used in K-Means
        'measure': measure, # Name of similarity measure used (for reporting)
        'inertia': float(kmeans.inertia_), # Inertia of the K-Means clustering; convert to float for JSON serialization
    }


def _empty_clustering_result(measure, labels, n_clusters):
    """Return empty result when clustering cannot be performed."""
    return {
        'silhouette_score': -1.0, # Silhouette score set to -1.0 to indicate invalid clustering when it cannot be performed
        'ari': 0.0, # Agreement with the true labels; 0.0 when clustering cannot be performed
        'nmi': 0.0, # Shared information with the true labels; 0.0 when clustering cannot be performed
        'cluster_labels': np.zeros(len(labels), dtype=int), # Cluster labels set to zeros when clustering cannot be performed
        'true_labels': labels, # Original true labels for reference (can be returned even when clustering cannot be performed)
        'n_clusters': n_clusters, # Number of clusters that was attempted (can be returned even when clustering cannot be performed)
        'measure': measure, # Name of similarity measure used (for reporting)
        'inertia': 0.0, # Inertia set to 0.0 when clustering cannot be performed
    }
