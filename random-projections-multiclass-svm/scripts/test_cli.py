#!/usr/bin/env python3
"""
test_cli.py - CLI Argument Test Suite for kamp_hw3.py

This module defines a suite of automated tests to verify that the command-line interface (CLI) 
of the main orchestrator script (kamp_hw3.py) functions correctly under various argument combinations. 

Each test runs kamp_hw3.py as a subprocess with specific CLI arguments and checks for expected outcomes 
such as exit codes, output content, and generated files.

Specifically, each test invokes kamp_hw3.py as a subprocess and checks:
  - Exit code == 0
  - Expected output files/directories exist
  - Key output strings appear in stdout

Author: Ryan Kamp
Course: CS7063 Advanced Machine Learning
Assignment: Assignment 3 – Random Projections and Multiclass Classification
Date: February 13, 2026
"""

import os
import glob
import subprocess
import sys
import time
from pathlib import Path

# ===========================================================================
# Paths
# ===========================================================================
SCRIPT_DIR = Path(__file__).parent
HW3_DIR = SCRIPT_DIR.parent
PYTHON = sys.executable  # Use the same interpreter running this script

RESULTS_DIR = HW3_DIR / "results"
REPORTS_DIR = HW3_DIR / "reports"
PRINTOUTS_DIR = HW3_DIR / "printouts"
WITHIN_REPORTS_DIR = REPORTS_DIR / "within_dataset_reports"
BETWEEN_REPORTS_DIR = REPORTS_DIR / "between_datasets_reports"


# ===========================================================================
# Helpers
# ===========================================================================

def run_pipeline(args_list: list[str], label: str, timeout: int = 300) -> tuple[int, str]:
    """Run kamp_hw3.py with given args. Returns (exit_code, stdout+stderr)."""
    cmd = [PYTHON, str(SCRIPT_DIR / "kamp_hw3.py")] + args_list
    print(f"\n{'─' * 70}")
    print(f"  TEST: {label}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'─' * 70}")
    start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(HW3_DIR), timeout=timeout,
    )
    elapsed = time.time() - start
    output = result.stdout + result.stderr
    return result.returncode, output, elapsed


def find_latest_run_dir() -> Path | None:
    """Return the most recently created hw3_run__* directory."""
    pattern = str(RESULTS_DIR / "hw3_run__*")
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return Path(dirs[-1]) if dirs else None


def find_latest_printout_dir() -> Path | None:
    """Return the most recently created hw3_printout__* directory."""
    pattern = str(PRINTOUTS_DIR / "hw3_printout__*")
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return Path(dirs[-1]) if dirs else None


def check(condition: bool, msg: str):
    """Assert with a descriptive message."""
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {msg}")
    return condition


# ===========================================================================
# Test cases
# ===========================================================================

def test_default_args():
    """Test 1: Default arguments (iris + breast_cancer)."""
    rc, output, elapsed = run_pipeline([], "Default args (iris + breast_cancer)")
    passed = True
    passed &= check(rc == 0, f"Exit code 0 (got {rc})")
    passed &= check("PIPELINE COMPLETE" in output, "Pipeline completion message")
    passed &= check(elapsed < 120, f"Finished in {elapsed:.1f}s (< 120s)")

    run_dir = find_latest_run_dir()
    passed &= check(run_dir is not None, "Run directory created")
    if run_dir:
        passed &= check("iris-breast_cancer" in run_dir.name, "Run dir has correct dataset tag")
        # Within-dataset outputs
        passed &= check((run_dir / "iris").is_dir(), "iris subfolder in run dir")
        passed &= check((run_dir / "breast_cancer").is_dir(), "breast_cancer subfolder in run dir")
        # Between-datasets outputs (eq3 + eq4)
        passed &= check(
            (run_dir / "between_datasets_eq3_analysis_report.md").exists(),
            "between_datasets_eq3 report in run dir",
        )
        passed &= check(
            (run_dir / "between_datasets_eq4_analysis_report.md").exists(),
            "between_datasets_eq4 report in run dir",
        )

    # Reports in organized subfolders
    passed &= check(
        (WITHIN_REPORTS_DIR / "iris_reports" / "iris_eq3_within_dataset_report.md").exists(),
        "iris eq3 within-dataset report in reports/",
    )
    passed &= check(
        (WITHIN_REPORTS_DIR / "breast_cancer_reports" / "breast_cancer_eq4_within_dataset_report.md").exists(),
        "breast_cancer eq4 within-dataset report in reports/",
    )
    passed &= check(
        (BETWEEN_REPORTS_DIR / "between_datasets_eq3_analysis_report.md").exists(),
        "between_datasets eq3 report in reports/",
    )
    passed &= check(
        (BETWEEN_REPORTS_DIR / "between_datasets_eq4_analysis_report.md").exists(),
        "between_datasets eq4 report in reports/",
    )

    # HW3_REPORT.md
    passed &= check(
        (REPORTS_DIR / "HW3_REPORT.md").exists(),
        "HW3_REPORT.md generated",
    )

    # Printout log
    printout_dir = find_latest_printout_dir()
    passed &= check(printout_dir is not None, "Printout directory created")
    if printout_dir:
        txt_files = list(printout_dir.glob("*.txt"))
        passed &= check(len(txt_files) >= 1, "Printout .txt file exists")

    return passed # Return overall pass/fail for this test case


def test_single_dataset():
    """Test 2: Single dataset (no between-datasets analysis expected)."""
    rc, output, elapsed = run_pipeline(
        ["--datasets", "iris"], "Single dataset (iris only)"
    )
    passed = True
    passed &= check(rc == 0, f"Exit code 0 (got {rc})")
    passed &= check("PIPELINE COMPLETE" in output, "Pipeline completion message")
    passed &= check(elapsed < 60, f"Finished in {elapsed:.1f}s (< 60s)")

    # Between-datasets should be skipped with only 1 dataset
    passed &= check(
        "BETWEEN-DATASETS ANALYSIS" not in output,
        "Between-datasets analysis skipped for single dataset",
    )

    run_dir = find_latest_run_dir()
    passed &= check(run_dir is not None, "Run directory created")
    if run_dir:
        passed &= check("iris" in run_dir.name, "Run dir has 'iris' in name")
        passed &= check((run_dir / "iris").is_dir(), "iris subfolder in run dir")

    return passed # Return overall pass/fail for this test case


def test_three_datasets():
    """Test 3: Three datasets (iris + wine + seeds)."""
    rc, output, elapsed = run_pipeline(
        ["--datasets", "iris", "wine", "seeds"],
        "Three datasets (iris, wine, seeds)",
    )
    passed = True
    passed &= check(rc == 0, f"Exit code 0 (got {rc})")
    passed &= check("PIPELINE COMPLETE" in output, "Pipeline completion message")

    run_dir = find_latest_run_dir()
    if run_dir:
        passed &= check("iris-wine-seeds" in run_dir.name, "Run dir has correct dataset tag")
        passed &= check((run_dir / "iris").is_dir(), "iris subfolder")
        passed &= check((run_dir / "wine").is_dir(), "wine subfolder")
        passed &= check((run_dir / "seeds").is_dir(), "seeds subfolder")
        # Between-datasets should run for 3 datasets
        passed &= check(
            (run_dir / "between_datasets_eq3_analysis_report.md").exists(),
            "between_datasets eq3 report generated",
        )
        passed &= check(
            (run_dir / "between_datasets_eq4_analysis_report.md").exists(),
            "between_datasets eq4 report generated",
        )

    # Within-dataset reports for all 3
    for ds in ["iris", "wine", "seeds"]:
        for eq in ["eq3", "eq4"]:
            report = WITHIN_REPORTS_DIR / f"{ds}_reports" / f"{ds}_{eq}_within_dataset_report.md"
            passed &= check(report.exists(), f"{ds} {eq} within-dataset report")

    return passed # Return overall pass/fail for this test case


def test_no_stats_flag():
    """Test 4: --no-stats flag skips dataset statistics."""
    rc, output, elapsed = run_pipeline(
        ["--datasets", "iris", "--no-stats"],
        "--no-stats flag (skip statistics)",
    )
    passed = True
    passed &= check(rc == 0, f"Exit code 0 (got {rc})")
    passed &= check("PIPELINE COMPLETE" in output, "Pipeline completion message")
    # The dataset stats step should be skipped
    passed &= check(
        "Skipping statistics" in output or "DATASET STATISTICS" not in output,
        "Dataset statistics skipped",
    )
    return passed # Return overall pass/fail for this test case


def test_custom_seed():
    """Test 5: Custom --seed value."""
    rc, output, elapsed = run_pipeline(
        ["--datasets", "iris", "--seed", "123", "--no-stats"],
        "Custom seed (--seed 123)",
    )
    passed = True
    passed &= check(rc == 0, f"Exit code 0 (got {rc})")
    passed &= check("PIPELINE COMPLETE" in output, "Pipeline completion message")
    passed &= check("Seed" in output or "seed" in output or "123" in output,
                     "Seed value referenced in output")
    return passed # Return overall pass/fail for this test case


def test_custom_test_size():
    """Test 6: Custom --test-size value."""
    rc, output, elapsed = run_pipeline(
        ["--datasets", "iris", "--test-size", "0.2", "--no-stats"],
        "Custom test size (--test-size 0.2)",
    )
    passed = True
    passed &= check(rc == 0, f"Exit code 0 (got {rc})")
    passed &= check("PIPELINE COMPLETE" in output, "Pipeline completion message")
    return passed # Return overall pass/fail for this test case


def test_custom_k():
    """Test 7: Custom --k value (target dimensionality)."""
    rc, output, elapsed = run_pipeline(
        ["--datasets", "iris", "--k", "3", "--no-stats"],
        "Custom k (--k 3)",
    )
    passed = True
    passed &= check(rc == 0, f"Exit code 0 (got {rc})")
    passed &= check("PIPELINE COMPLETE" in output, "Pipeline completion message")
    return passed # Return overall pass/fail for this test case


def test_quiet_flag():
    """Test 8: --quiet flag reduces output."""
    rc, output, elapsed = run_pipeline(
        ["--datasets", "iris", "--no-stats", "--quiet"],
        "--quiet flag",
    )
    passed = True
    passed &= check(rc == 0, f"Exit code 0 (got {rc})")
    passed &= check("PIPELINE COMPLETE" in output, "Pipeline completion message")
    passed &= check("RANDOM PROJECTION" not in output, "Step-by-step progress is not printed")
    log_dir = find_latest_printout_dir()
    log_text = ""
    if log_dir is not None:
        logs = list(log_dir.glob("*.txt"))
        log_text = logs[0].read_text(encoding="utf-8") if logs else ""
    passed &= check("RANDOM PROJECTION" in log_text, "Log file still holds the full printout")
    return passed # Return overall pass/fail for this test case


def test_all_five_datasets():
    """Test 9: All 5 datasets together."""
    rc, output, elapsed = run_pipeline(
        ["--datasets", "iris", "breast_cancer", "wine", "seeds", "glass"],
        "All 5 datasets",
    )
    passed = True
    passed &= check(rc == 0, f"Exit code 0 (got {rc})")
    passed &= check("PIPELINE COMPLETE" in output, "Pipeline completion message")
    passed &= check(elapsed < 300, f"Finished in {elapsed:.1f}s (< 300s)")

    run_dir = find_latest_run_dir()
    if run_dir:
        for ds in ["iris", "breast_cancer", "wine", "seeds", "glass"]:
            passed &= check((run_dir / ds).is_dir(), f"{ds} subfolder in run dir")

    return passed # Return overall pass/fail for this test case


def test_combined_flags():
    """Test 10: Multiple flags combined."""
    rc, output, elapsed = run_pipeline(
        ["--datasets", "iris", "breast_cancer", "--no-stats",
         "--seed", "99", "--test-size", "0.25", "--k", "2", "--quiet"],
        "Combined flags (--no-stats --seed 99 --test-size 0.25 --k 2 --quiet)",
    )
    passed = True
    passed &= check(rc == 0, f"Exit code 0 (got {rc})")
    passed &= check("PIPELINE COMPLETE" in output, "Pipeline completion message")
    return passed # Return overall pass/fail for this test case


def test_invalid_dataset():
    """Test 11: Invalid dataset name should fail gracefully."""
    rc, output, elapsed = run_pipeline(
        ["--datasets", "nonexistent_dataset"],
        "Invalid dataset name (should fail)",
    )
    passed = True
    passed &= check(rc != 0, f"Non-zero exit code (got {rc})")
    passed &= check(
        "invalid choice" in output.lower() or "error" in output.lower(),
        "Error message about invalid dataset",
    )
    return passed # Return overall pass/fail for this test case



# ===========================================================================
# Runner
# ===========================================================================

# List of all test functions to run in the suite:
ALL_TESTS = [
    test_default_args, # Most comprehensive test and should be run first to verify basic functionality before more specific tests
    test_single_dataset, # Tests handling of single dataset and skipping between-datasets analysis
    test_three_datasets, # Tests handling of multiple datasets and generation of all expected outputs
    test_no_stats_flag, # Tests that --no-stats flag correctly skips dataset statistics step
    test_custom_seed, # Tests that custom seed value is accepted and referenced in output
    test_custom_test_size, # Tests that custom test size value is accepted and does not cause errors
    test_custom_k, # Tests that custom k value is accepted and does not cause errors
    test_quiet_flag, # Tests that --quiet flag does not cause errors and allows pipeline to complete
    test_all_five_datasets, # Tests handling of all 5 datasets together and generation of all expected outputs
    test_combined_flags, # Tests that multiple flags can be combined without causing errors
    test_invalid_dataset, # Tests that invalid dataset name causes graceful failure with appropriate error message
]


def main():
    print("=" * 70)
    print("  HW3 CLI TEST SUITE")
    print(f"  Python: {PYTHON}")
    print(f"  HW3 dir: {HW3_DIR}")
    print("=" * 70)

    overall_start = time.time()
    results = {}

    # Run each test and collect results:
    for test_fn in ALL_TESTS:
        test_name = test_fn.__doc__.split(":")[0].strip() if test_fn.__doc__ else test_fn.__name__
        try:
            passed = test_fn()
        except subprocess.TimeoutExpired:
            print(f"    [FAIL] TIMEOUT — test exceeded time limit")
            passed = False
        except Exception as e:
            print(f"    [FAIL] EXCEPTION: {e}")
            passed = False
        results[test_name] = passed

    # Summary
    total_time = time.time() - overall_start
    n_pass = sum(1 for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)
    n_total = len(results)

    print(f"\n{'=' * 70}")
    print(f"  TEST SUITE SUMMARY")
    print(f"{'=' * 70}")
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{status}]  {name}")
    print(f"{'─' * 70}")
    print(f"  {n_pass}/{n_total} passed, {n_fail} failed")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'=' * 70}")

    sys.exit(0 if n_fail == 0 else 1) # Exit with code 0 if all tests passed, else 1


if __name__ == "__main__":
    main()
