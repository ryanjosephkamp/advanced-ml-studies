#!/usr/bin/env python3
"""
kamp_hw3.py - Main Orchestrator for HW3

Random Projections (Part I) and Multiclass SVM Classification (Part II)

This script coordinates the entire pipeline:
  1. Load datasets (with caching)
  2. Compute dataset statistics (optional)
  3. Perform random projection (Eq.3 and Eq.4)
  4. Train/evaluate SVM classifiers on projected data
  5. Within-dataset analysis per dataset
  6. Between-datasets analysis (if ≥2 datasets)
  7. Generate comprehensive reports

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 3 – Random Projections and Multiclass Classification
Date: February 13, 2026
Revised: November 2026, for this repository

References:
    [1] Johnson, W. B. & Lindenstrauss, J. (1984). Extensions of Lipschitz
        mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
    [2] Achlioptas, D. (2003). Database-friendly random projections:
        Johnson-Lindenstrauss with binary coins. Journal of Computer and System
        Sciences, 66(4), 671-687.
    [3] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
        JMLR, 12, 2825-2830.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path


# ===========================================================================
# PATHS
# ===========================================================================
SCRIPT_DIR = Path(__file__).parent
HW3_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
SCRIPTS_DIR = HW3_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# ===========================================================================
# IMPORTS FROM HELPER MODULES
# ===========================================================================

# Import author info and utilities
from author_info import ( 
    print_console_header,
    TeeLogger,
    COURSE_ASSIGNMENT,
)
# Import pipeline components
from data_loader import (
    load_dataset,
    DATASET_REGISTRY,
    DEFAULT_DATASETS,
)
# Import dataset statistics function (also generates markdown report sections)
from dataset_stats import compute_and_save_statistics
# Import random projection, SVM classification, and analysis functions
from random_projection import run_random_projection
from svm_classifier import run_svm_classification
from within_dataset_analysis import run_within_dataset_analysis
from between_datasets_analysis import run_between_datasets_analysis
# Import report generation functions
from report_generator import (
    generate_comprehensive_markdown_report,
    generate_ieee_latex_report,
)


# ===========================================================================
# CLI ARGUMENT PARSING
# ===========================================================================

def parse_args():
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="HW3 — Random Projections + Multiclass SVM Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python kamp_hw3.py --datasets iris breast_cancer wine --seed 42",
    )
    # Add dataset selection argument
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
        default=DEFAULT_DATASETS,
        help=f"Datasets to process (default: {' '.join(DEFAULT_DATASETS)})",
    )
    # Add "no-stats" flag to skip dataset statistics generation
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Skip dataset statistics generation",
    )
    # Add random seed
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    # Add test size for train/test split
    parser.add_argument(
        "--test-size", type=float, default=0.3,
        help="Test split fraction (default: 0.3)",
    )
    # Add target dimensionality for random projection
    parser.add_argument(
        "--k", type=int, default=2,
        help="Target dimensionality for projection (default: 2)",
    )
    # Add "quiet" flag: the console shows only the run header, one line per dataset, and the final summary;
    # the full printout still goes to the log file in printouts/
    parser.add_argument(
        "--quiet", action="store_true",
        help="Print only a short summary to the console (the log file keeps the full printout)",
    )
    # Add "legacy-seeding" flag to reproduce the February 13, 2026 run, where every dataset's
    # projection matrices came from the same random stream
    parser.add_argument(
        "--legacy-seeding", action="store_true",
        help="Seed every dataset's projection with the same stream, as in the February 13, 2026 run "
             "(reproduces the original numbers); by default each dataset gets its own stream",
    )
    return parser.parse_args()


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    # Parse CLI arguments
    args = parse_args()
    # Start wall clock timer for the entire pipeline
    wall_start = time.time()
    # Generate timestamp for this run (used in output directories and log files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Naming components -------------------------------------------------
    script_name = Path(__file__).stem                    # e.g. "kamp_hw3"
    datasets_tag = "-".join(args.datasets)               # e.g. "iris-breast_cancer-wine-seeds-glass"
    run_tag = f"hw3_run__{script_name}_{datasets_tag}_{timestamp}"
    printout_tag = f"hw3_printout__{script_name}_{datasets_tag}_{timestamp}"

    # --- Directories -------------------------------------------------------
    results_base = HW3_DIR / "results"
    run_dir = str(results_base / run_tag)
    os.makedirs(run_dir, exist_ok=True)

    printouts_dir = HW3_DIR / "printouts" / printout_tag
    os.makedirs(str(printouts_dir), exist_ok=True)
    log_path = str(printouts_dir / f"{printout_tag}.txt")

    reports_dir = HW3_DIR / "reports"
    os.makedirs(str(reports_dir), exist_ok=True)

    # --- TeeLogger ---------------------------------------------------------
    # Initialize TeeLogger to capture console output to a file while still printing to console
    # (This allows us to have a complete log of the run in a text file, while also seeing progress in real-time in the console.)
    tee = TeeLogger(log_path, echo=not args.quiet) 
    # Start logging to file (this will also print to console)
    tee.start()

    try: # Use try/finally to ensure we stop the TeeLogger even if an error occurs
        # --- Header --------------------------------------------------------
        print_console_header(f"kamp_hw3.py – {COURSE_ASSIGNMENT}")
        print(f"\n{'='*72}")
        print(f"  Run timestamp : {timestamp}")
        print(f"  Output dir    : {run_dir}")
        print(f"  Log file      : {log_path}")
        print(f"  Datasets      : {', '.join(args.datasets)}")
        print(f"  Seed          : {args.seed}")
        print(f"  Test size     : {args.test_size}")
        print(f"  k (projection): {args.k}")
        print(f"  Skip stats    : {args.no_stats}")
        print(f"  Seeding       : {'legacy (same stream for every dataset)' if args.legacy_seeding else 'per dataset'}")
        print(f"{'='*72}\n")
        if args.quiet:
            tee.announce(f"Run {run_tag}: datasets {', '.join(args.datasets)}; full log in {log_path}")

        # ===================================================================
        # PER-DATASET PIPELINE
        # ===================================================================
        all_results = {} # Dictionary to store results for all datasets, which will be used for between-datasets analysis and report generation

        # Loop over each selected dataset and run the full pipeline for that dataset
        for ds_idx, ds_key in enumerate(args.datasets, 1):
            ds_start = time.time()
            print(f"\n{'#'*72}")
            print(f"# DATASET {ds_idx}/{len(args.datasets)}: {ds_key}")
            print(f"{'#'*72}\n")

            ds_out_dir = os.path.join(run_dir, ds_key)
            os.makedirs(ds_out_dir, exist_ok=True)

            result_entry = {}

            # 1. LOAD DATASET -----------------------------------------------
            print(f"[{ds_key}] Step 1: Loading dataset...")
            ds = load_dataset(ds_key)
            result_entry["dataset"] = ds
            print(f"  ✓ Loaded {ds['dataset_name']}: "
                  f"{ds['n_samples']} samples, {ds['n_features']} features, "
                  f"{ds['n_classes']} classes\n")

            # 2. DATASET STATISTICS -----------------------------------------
            if not args.no_stats:
                print(f"[{ds_key}] Step 2: Computing dataset statistics...")
                stats_out = compute_and_save_statistics(ds)
                print(f"  ✓ Statistics saved to {stats_out}\n")
            else:
                print(f"[{ds_key}] Step 2: Skipping statistics (--no-stats)\n")

            # 3. RANDOM PROJECTION ------------------------------------------
            print(f"[{ds_key}] Step 3: Running random projection...")
            # Run both Eq.3 and Eq.4 projections in the same function for efficiency, and it returns a dictionary containing all projection results and distortion metrics.
            proj = run_random_projection(
                ds, output_dir=ds_out_dir,
                k=args.k, seed=args.seed,
                legacy_seeding=args.legacy_seeding,
            )
            result_entry["projection"] = proj
            print(f"  ✓ Projection complete. "
                  f"Eq3 ε={proj['distortion_eq3']['epsilon']:.4f}, "
                  f"Eq4 ε={proj['distortion_eq4']['epsilon']:.4f}\n")

            # 4. SVM CLASSIFICATION -----------------------------------------
            # Run SVM classification separately on the Eq.3 and Eq.4 projections, and store the results in the result_entry dictionary under "svm_eq3" and "svm_eq4" keys.
            # This allows us to easily compare the classification performance on the two different projections in the within-dataset analysis step later on.
            # A quick summary of the SVM results is also printed for each projection immediately after running the classification, showing the accuracy for each kernel.
            # The detailed results (including confusion matrices, classification reports, and plots) are saved in the output directory for this dataset and will be included in the final report.
            
            # Loop over the two projection types (Eq.3 and Eq.4) and run SVM classification on each one
            for eq in ["eq3", "eq4"]: 
                print(f"[{ds_key}] Step 4{chr(ord('a') + ['eq3','eq4'].index(eq))}: "
                      f"SVM classification on {eq.upper()} projection...")
                
                svm_res = run_svm_classification(
                    E=proj[f"E_{eq}"], # Projected data for the current projection type (Eq.3 or Eq.4)
                    y=ds["y"], # Original class labels (not projected)
                    class_names=ds["class_names"], # Original class names (not projected)
                    dataset_key=ds_key, # Dataset key (e.g. "iris", "breast_cancer", etc.)
                    dataset_name=ds["dataset_name"], # Dataset name (e.g. "Iris", "Breast Cancer Wisconsin", etc.)
                    projection_label=eq, # Projection label ("eq3" or "eq4") to distinguish the results for the two different projections
                    output_dir=ds_out_dir, # Output directory for this dataset where the results will be saved
                    test_size=args.test_size, # Test split fraction for train/test split (default: 0.3)
                    seed=args.seed, # Random seed for reproducibility
                )
                # Store the SVM results for this projection type in the result_entry dictionary under a key that indicates which projection it corresponds to (e.g. "svm_eq3" or "svm_eq4")
                result_entry[f"svm_{eq}"] = svm_res 
                # Print quick summary
                for ktag, kr in svm_res["kernel_results"].items():
                    print(f"    {kr['kernel_name']:20s} "
                          f"Acc={kr['test']['accuracy']:.4f}")
                print()

            # 5. WITHIN-DATASET ANALYSIS ------------------------------------
            print(f"[{ds_key}] Step 5: Within-dataset analysis...")
            # Run the within-dataset analysis function, which takes the SVM results for both projections, the original dataset dictionary, and the output directory for this dataset. 
            # It returns a dictionary containing all the analysis results (e.g. performance comparisons, visualizations, etc.) which is stored in the result_entry under 
            # "within_eq3" and "within_eq4" keys for later use in the report generation step.
            within_eq3 = run_within_dataset_analysis(
                svm_results=result_entry["svm_eq3"], # The SVM results for the Eq.3 projection, which includes all the performance metrics and plots generated from running SVM classification on the Eq.3 projected data
                projection_results=proj, # Results from the random projection step, which includes the projected data for both Eq.3 and Eq.4, as well as the distortion metrics
                dataset_dict=ds, # Original dataset dictionary containing all the information about the dataset
                output_dir=ds_out_dir, # Output directory for this dataset where the analysis results will be saved
            )
            within_eq4 = run_within_dataset_analysis(
                svm_results=result_entry["svm_eq4"], # The SVM results for the Eq.4 projection, which includes all the performance metrics and plots generated from running SVM classification on the Eq.4 projected data
                projection_results=proj, # Results from the random projection step, which includes the projected data for both Eq.3 and Eq.4, as well as the distortion metrics
                dataset_dict=ds, # Original dataset dictionary containing all the information about the dataset
                output_dir=ds_out_dir, # Output directory for this dataset where the analysis results will be saved
            )
            # Store the within-dataset analysis results for both projections in the result_entry dictionary under keys for their corresponding projection (e.g., "within_eq3" and "within_eq4")
            result_entry["within_eq3"] = within_eq3
            result_entry["within_eq4"] = within_eq4
            print(f"  ✓ Within-dataset analysis complete\n")

            all_results[ds_key] = result_entry

            elapsed = time.time() - ds_start # Compute elapsed time for processing this dataset
            print(f"[{ds_key}] ✓ Dataset pipeline finished in {elapsed:.1f}s\n")
            if args.quiet:
                accs = "; ".join(
                    f"{eq.upper()} " + ", ".join(f"{kr['kernel_name']} {kr['test']['accuracy']:.4f}"
                                                for kr in result_entry[f"svm_{eq}"]["kernel_results"].values())
                    for eq in ["eq3", "eq4"])
                tee.announce(f"[{ds_key}] finished in {elapsed:.1f}s. Test accuracy: {accs}")

        # ===================================================================
        # BETWEEN-DATASETS ANALYSIS
        # ===================================================================
        between_analysis_eq3 = None
        between_analysis_eq4 = None
        if len(args.datasets) >= 2:
            print(f"\n{'#'*72}")
            print(f"# BETWEEN-DATASETS ANALYSIS")
            print(f"{'#'*72}\n")
            # Run the between-datasets analysis function for the Eq.3 projection results across all datasets. 
            # This function will compare the performance of the SVM classifiers on the Eq.3 projected data across the different datasets 
            # and generate visualizations and summary statistics to highlight any patterns or differences in performance between the datasets for this projection type.
            between_analysis_eq3 = run_between_datasets_analysis( 
                all_results=all_results, # Dictionary containing all the results for each dataset, including SVM results and within-dataset analysis results for both projections
                output_dir=run_dir, # Output directory where the between-datasets analysis results will be saved
                proj_label="eq3", # Label for the projection type (Eq.3) to be used in the analysis
            )
            print(f"  ✓ Between-datasets analysis (EQ3) complete\n")
            # Run the between-datasets analysis function for the Eq.4 projection results across all datasets.
            # This function will perform a similar comparison of SVM performance across datasets but for the Eq.4 projection results.
            between_analysis_eq4 = run_between_datasets_analysis(
                all_results=all_results, # Dictionary containing all the results for each dataset, including SVM results and within-dataset analysis results for both projections
                output_dir=run_dir, # Output directory where the between-datasets analysis results will be saved
                proj_label="eq4", # Label for the projection type (Eq.4) to be used in the analysis
            )
            print(f"  ✓ Between-datasets analysis (EQ4) complete\n")

        # ===================================================================
        # REPORT GENERATION
        # ===================================================================
        print(f"\n{'#'*72}")
        print(f"# REPORT GENERATION")
        print(f"{'#'*72}\n")

        # Generate a comprehensive markdown report that includes all the results and analyses from the pipeline.
        # This function takes the complete results dictionary for all datasets, the output directory for this run, 
        # the between-datasets analysis results for both projections, and a project label.
        md_report = generate_comprehensive_markdown_report(
            all_results=all_results, # Dictionary containing all the results for each dataset, including SVM results and within-dataset analysis results for both projections
            run_dir=run_dir, # Output directory where the report will be saved
            between_analysis=between_analysis_eq3, # Between-datasets analysis results for the Eq.3 projection
            proj_label="eq3", # Label for the projection type (Eq.3) to be used in the report
        )

        # Generate an IEEE-style LaTeX report that includes all the results and analyses from the pipeline, formatted according to IEEE guidelines.
        tex_report = generate_ieee_latex_report(
            all_results=all_results, # Dictionary containing all the results for each dataset, including SVM results and within-dataset analysis results for both projections
            run_dir=run_dir, # Output directory where the report will be saved
            between_analysis=between_analysis_eq3, # Between-datasets analysis results for the Eq.3 projection
            proj_label="eq3", # Label for the projection type (Eq.3) to be used in the report
        )

        # ===================================================================
        # FINAL SUMMARY
        # ===================================================================
        total_elapsed = time.time() - wall_start # Compute total elapsed time for the entire pipeline
        # The summary reaches the console in quiet mode too
        say = tee.announce if args.quiet else print
        say(f"\n{'='*72}")
        say(f"  HW3 PIPELINE COMPLETE")
        say(f"{'='*72}")
        say(f"  Total wall time   : {total_elapsed:.1f}s")
        say(f"  Datasets processed: {len(args.datasets)}")
        say(f"  Run directory     : {run_dir}")
        say(f"  Log file          : {log_path}")
        say(f"  Markdown report   : {md_report}")
        say(f"  LaTeX report      : {tex_report}")
        say(f"{'='*72}\n")

    finally: # Ensure we stop the TeeLogger to flush and close the log file properly, even if an error occurs during the pipeline execution
        tee.stop() # Stop logging to file and restore original stdout/stderr


if __name__ == "__main__":
    main()
