#!/usr/bin/env python3
"""
get_data.py - Dataset Loading and Preparation Script

This script handles loading and processing of .mat dataset files for the HW2 assignment.
It extracts binary matrices from .mat files, flattens them to row vectors, 
and saves them as CSV files. It also generates validation datasets by applying noise.

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 2 – Hopfield Networks
Date: February 2, 2026
"""

import os
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Optional


def get_project_paths() -> dict:
    """Get the standard project paths."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    valid_datasets_dir = data_dir / "valid_datasets"
    
    return {
        "project": project_dir,
        "data": data_dir,
        "valid_datasets": valid_datasets_dir,
        "scripts": script_dir
    }


def load_mat_file(mat_path: Path, key: str) -> Optional[np.ndarray]:
    """
    Load a .mat file and extract the cell array data.
    
    Args:
        mat_path: Path to the .mat file
        key: The key name for the data in the .mat file
        
    Returns:
        NumPy array of shape (1, n_cells) containing the cell data, or None if failed
    """
    try:
        mat_data = sio.loadmat(str(mat_path))
        if key in mat_data:
            return mat_data[key]
        else:
            # Try to find the data key (not metadata keys)
            for k, v in mat_data.items():
                if not k.startswith('__') and hasattr(v, 'dtype') and v.dtype == object:
                    print(f"  Using key '{k}' from {mat_path.name}")
                    return v
            print(f"  Error: Could not find data key in {mat_path.name}")
            return None
    except Exception as e:
        print(f"  Error loading {mat_path}: {e}")
        return None


def extract_matrices_from_cells(cell_array: np.ndarray) -> list:
    """
    Extract individual matrices from a cell array.
    
    Args:
        cell_array: NumPy array of shape (1, n_cells) with object dtype
        
    Returns:
        List of 2D numpy arrays (the matrices)
    """
    matrices = []
    for i in range(cell_array.shape[1]):
        matrix = cell_array[0, i]
        matrices.append(matrix)
    return matrices


def matrices_to_3d_array(matrices: list) -> np.ndarray:
    """
    Convert a list of 2D matrices to a 3D numpy array for easy indexing.
    
    Args:
        matrices: List of 2D numpy arrays (each 12x12)
        
    Returns:
        NumPy array of shape (n_matrices, 12, 12) allowing access via data[i]
    """
    return np.stack(matrices, axis=0).astype(np.uint8)


def flatten_matrices_to_dataframe(matrices: list) -> pd.DataFrame:
    """
    Flatten a list of 2D matrices to a DataFrame where each row is a flattened matrix.
    
    Args:
        matrices: List of 2D numpy arrays
        
    Returns:
        DataFrame with shape (n_matrices, n_elements)
    """
    flattened = [matrix.flatten() for matrix in matrices]
    df = pd.DataFrame(flattened)
    return df


def dataframe_to_matrices(df: pd.DataFrame, matrix_shape: tuple = (12, 12)) -> np.ndarray:
    """
    Convert a flattened DataFrame back to a 3D array of matrices.
    
    Args:
        df: DataFrame where each row is a flattened matrix
        matrix_shape: Shape of each individual matrix (default 12x12)
        
    Returns:
        NumPy array of shape (n_samples, *matrix_shape) for easy indexing
    """
    n_samples = df.shape[0]
    matrices = np.zeros((n_samples, *matrix_shape), dtype=np.uint8)
    for i in range(n_samples):
        matrices[i] = df.iloc[i].values.reshape(matrix_shape)
    return matrices


def validate_csv(csv_path: Path, expected_cols: int = 144) -> bool:
    """
    Validate that a CSV file contains valid binary data.
    
    Args:
        csv_path: Path to the CSV file
        expected_cols: Expected number of columns (default 144 for 12x12 matrices)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check column count
        if df.shape[1] != expected_cols:
            print(f"  CSV validation failed: expected {expected_cols} columns, got {df.shape[1]}")
            return False
        
        # Check for binary values (0 or 1)
        unique_values = set(df.values.flatten())
        if not unique_values.issubset({0, 1, 0.0, 1.0}):
            print(f"  CSV validation failed: non-binary values found")
            return False
        
        return True
    except Exception as e:
        print(f"  CSV validation failed: {e}")
        return False


def load_or_create_dataset(mat_path: Path, csv_path: Path, mat_key: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load a dataset from CSV if valid, otherwise extract from .mat file.
    
    Args:
        mat_path: Path to the source .mat file
        csv_path: Path to the CSV file to load/save
        mat_key: Key name for data in the .mat file
        
    Returns:
        Tuple of (DataFrame with flattened data, 3D NumPy array of shape (n, 12, 12))
    """
    # Check if CSV already exists and is valid
    if csv_path.exists():
        print(f"  Found existing CSV: {csv_path.name}")
        if validate_csv(csv_path):
            print(f"  Loading from existing CSV...")
            df = pd.read_csv(csv_path)
            # Reconstruct matrices from DataFrame
            matrices = dataframe_to_matrices(df)
            return df, matrices
        else:
            print(f"  Existing CSV is invalid, regenerating from .mat file...")
    
    # Extract from .mat file
    print(f"  Extracting data from {mat_path.name}...")
    cell_array = load_mat_file(mat_path, mat_key)
    
    if cell_array is None:
        raise ValueError(f"Failed to load data from {mat_path}")
    
    matrices_list = extract_matrices_from_cells(cell_array)
    matrices = matrices_to_3d_array(matrices_list)
    df = flatten_matrices_to_dataframe(matrices_list)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path.name}")
    
    return df, matrices


def apply_noise_to_data(data: np.ndarray, noise_percent: float, random_seed: int) -> np.ndarray:
    """
    Apply noise to binary data by randomly flipping a percentage of values.
    
    Args:
        data: Binary numpy array
        noise_percent: Percentage of values to flip (0-100)
        random_seed: Random seed for reproducibility
        
    Returns:
        Noisy binary array with same shape as input
    """
    np.random.seed(random_seed)
    
    # Make a copy to avoid modifying original
    noisy_data = data.copy().astype(np.uint8)
    
    # Calculate number of values to flip
    total_values = noisy_data.size
    n_flip = int(total_values * (noise_percent / 100.0))
    
    # Get random indices to flip
    flat_data = noisy_data.flatten()
    flip_indices = np.random.choice(total_values, size=n_flip, replace=False)
    
    # Flip the values (0 -> 1, 1 -> 0)
    flat_data[flip_indices] = 1 - flat_data[flip_indices]
    
    return flat_data.reshape(noisy_data.shape)


def create_validation_dataset(train_df: pd.DataFrame, noise_percent: float, 
                               random_seed: int, paths: dict) -> Tuple[pd.DataFrame, np.ndarray, Path]:
    """
    Create a validation dataset by applying noise to training data.
    
    Args:
        train_df: Training DataFrame
        noise_percent: Percentage of noise to apply
        random_seed: Random seed for reproducibility
        paths: Dictionary of project paths
        
    Returns:
        Tuple of (validation DataFrame, 3D matrices array, path to validation folder)
    """
    # Create folder name
    folder_name = f"valid_dataset_{random_seed}_{noise_percent}"
    valid_folder = paths["valid_datasets"] / folder_name
    
    mat_filename = f"valid_{random_seed}_{noise_percent}.mat"
    csv_filename = f"valid_{random_seed}_{noise_percent}.csv"
    
    mat_path = valid_folder / mat_filename
    csv_path = valid_folder / csv_filename
    
    # Check if validation dataset already exists
    if valid_folder.exists():
        print(f"  Found existing validation folder: {folder_name}")
        
        if mat_path.exists() and csv_path.exists():
            print(f"  Checking existing validation files...")
            if validate_csv(csv_path):
                print(f"  Loading validation data from existing CSV...")
                valid_df = pd.read_csv(csv_path)
                # Reconstruct matrices from DataFrame
                valid_matrices = dataframe_to_matrices(valid_df)
                return valid_df, valid_matrices, valid_folder
            else:
                print(f"  Existing validation files are invalid, regenerating...")
        else:
            print(f"  Missing validation files, regenerating...")
    
    # Create the validation folder
    valid_folder.mkdir(parents=True, exist_ok=True)
    
    # Convert training data to matrices
    print(f"  Generating validation dataset with {noise_percent}% noise (seed={random_seed})...")
    train_data = train_df.values
    
    # Apply noise to all training data at once
    noisy_data = apply_noise_to_data(train_data, noise_percent, random_seed)
    
    # Create validation DataFrame
    valid_df = pd.DataFrame(noisy_data)
    
    # Save to CSV
    valid_df.to_csv(csv_path, index=False)
    print(f"  Saved validation CSV: {csv_filename}")
    
    # Convert to 12x12 matrices (3D array) and save as .mat file
    n_samples = valid_df.shape[0]
    valid_matrices = np.zeros((n_samples, 12, 12), dtype=np.uint8)
    for i in range(n_samples):
        valid_matrices[i] = noisy_data[i].reshape(12, 12)
    
    # Create cell array structure for .mat file
    cell_array = np.empty((1, n_samples), dtype=object)
    for i in range(n_samples):
        cell_array[0, i] = valid_matrices[i]
    
    # Save as .mat file
    sio.savemat(str(mat_path), {'valid': cell_array})
    print(f"  Saved validation MAT: {mat_filename}")
    
    return valid_df, valid_matrices, valid_folder


def load_all_datasets(noise_percent: float = 20.0, random_seed: int = 42,
                      verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                      np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load or create all three datasets (training, testing, validation).
    
    Args:
        noise_percent: Percentage of noise for validation dataset
        random_seed: Random seed for validation dataset generation
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (train_df, test_df, valid_df, train_data, test_data, valid_data, info_dict)
        - train_df, test_df, valid_df: Flattened DataFrames with shape (18, 144)
        - train_data, test_data, valid_data: 3D NumPy arrays with shape (18, 12, 12)
          Access individual matrices via indexing: train_data[i] for i in 0-17
        - info_dict: Dictionary with metadata about the datasets
    """
    paths = get_project_paths()
    
    # Ensure directories exist
    paths["valid_datasets"].mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "="*60)
        print("DATASET LOADING AND PREPARATION")
        print("="*60)
    
    # Load training dataset
    if verbose:
        print("\n[1/3] Loading training dataset (characters.mat)...")
    train_mat = paths["data"] / "characters.mat"
    train_csv = paths["data"] / "characters.csv"
    train_df, train_data = load_or_create_dataset(train_mat, train_csv, "char3")
    if verbose:
        print(f"  Training data shape: {train_df.shape} (matrices: {train_data.shape})")
    
    # Load testing dataset
    if verbose:
        print("\n[2/3] Loading testing dataset (test.mat)...")
    test_mat = paths["data"] / "test.mat"
    test_csv = paths["data"] / "test.csv"
    test_df, test_data = load_or_create_dataset(test_mat, test_csv, "test")
    if verbose:
        print(f"  Testing data shape: {test_df.shape} (matrices: {test_data.shape})")
    
    # Create/load validation dataset
    if verbose:
        print(f"\n[3/3] Loading validation dataset (noise={noise_percent}%, seed={random_seed})...")
    valid_df, valid_data, valid_folder = create_validation_dataset(
        train_df, noise_percent, random_seed, paths
    )
    if verbose:
        print(f"  Validation data shape: {valid_df.shape} (matrices: {valid_data.shape})")
    
    # Summary info
    info = {
        "noise_percent": noise_percent,
        "random_seed": random_seed,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "valid_shape": valid_df.shape,
        "train_matrices_shape": train_data.shape,
        "test_matrices_shape": test_data.shape,
        "valid_matrices_shape": valid_data.shape,
        "valid_folder": valid_folder,
        "paths": paths
    }
    
    if verbose:
        print("\n" + "="*60)
        print("DATASET LOADING COMPLETE")
        print("="*60)
        print(f"  Training samples:   {train_df.shape[0]} (features: {train_df.shape[1]})")
        print(f"  Testing samples:    {test_df.shape[0]} (features: {test_df.shape[1]})")
        print(f"  Validation samples: {valid_df.shape[0]} (features: {valid_df.shape[1]})")
        print(f"  Matrix shape:       {train_data.shape[1]}x{train_data.shape[2]} (access via data[i])")
        print(f"  Noise applied:      {noise_percent}%")
        print(f"  Random seed:        {random_seed}")
        print("="*60 + "\n")
    
    return train_df, test_df, valid_df, train_data, test_data, valid_data, info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and prepare datasets for HW2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument( # Noise percentage for validation dataset
        "--noise-percent", "-n",
        type=float,
        default=20.0,
        help="Noise percentage for validation dataset (0-100)"
    )
    
    parser.add_argument( # Random seed for validation dataset generation
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for validation dataset generation"
    )
    
    parser.add_argument( # Suppress verbose output
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point when script is run directly."""
    args = parse_args()
    
    # Print console header
    if not args.quiet:
        try:
            from author_info import print_console_header
            print_console_header("get_data.py")
        except ImportError:
            pass
    
    # Load all datasets
    train_df, test_df, valid_df, train_data, test_data, valid_data, info = load_all_datasets(
        noise_percent=args.noise_percent,
        random_seed=args.seed,
        verbose=not args.quiet
    )
    
    return train_df, test_df, valid_df, train_data, test_data, valid_data, info


if __name__ == "__main__":
    main()
