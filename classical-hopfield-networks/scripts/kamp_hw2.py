#!/usr/bin/env python3
"""
kamp_hw2.py - Main Orchestrator Script for HW2

This script orchestrates the complete HW2 workflow:
1. Dataset loading and preparation
2. Descriptive statistics computation (NOT REQUIRED BY ASSIGNMENT)
3. Hopfield network training and testing

Author: Ryan Kamp
Affiliation: Department of Computer Science, University of Cincinnati
Location: Cincinnati, OH, USA
GitHub: https://github.com/ryanjosephkamp

Course: CS7063 Advanced Machine Learning
Assignment: Assignment 2 – Hopfield Networks
Date: February 2, 2026

Usage:
    python kamp_hw2.py [OPTIONS]
    
Examples:
    python kamp_hw2.py                           # Run complete experiment
    python kamp_hw2.py --seed 123 --noise 30     # Custom seed and noise
    python kamp_hw2.py --no-stats                # Skip statistics generation
    python kamp_hw2.py --no-hopfield             # Skip Hopfield experiment
    python kamp_hw2.py --quiet                   # Suppress verbose output
"""

import sys
import argparse
import io
from pathlib import Path
from datetime import datetime

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from get_data import load_all_datasets
from dataset_stats import compute_and_save_statistics
from hopfield_experiment import run_experiment as run_hopfield_experiment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HW2 Main Orchestrator - Complete Hopfield Network Assignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Run complete experiment (stats + Hopfield)
  %(prog)s --seed 123 --noise 30        Custom seed and noise percentage
  %(prog)s --no-stats                   Skip statistics, run Hopfield only
  %(prog)s --no-hopfield                Skip Hopfield, run stats only
  %(prog)s --stats-only                 Alias for --no-hopfield
  %(prog)s --hopfield-only              Alias for --no-stats
  %(prog)s -s 42 -n 15 -q               Custom params, quiet mode
        """
    )
    
    # Data preparation arguments
    data_group = parser.add_argument_group('Data Preparation Options')
    
    data_group.add_argument(
        "--noise-percent", "-n",
        type=float,
        default=20.0,
        metavar="PERCENT",
        help="Noise percentage for validation dataset generation (0-100). Default: 20.0"
    )
    
    data_group.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for reproducible operations. Default: 42"
    )
    
    # Statistics arguments
    stats_group = parser.add_argument_group('Statistics Options')
    
    stats_group.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable descriptive statistics computation and report generation"
    )
    
    stats_group.add_argument(
        "--stats-only",
        action="store_true",
        help="Run statistics only, skip Hopfield experiment"
    )
    
    # Hopfield arguments
    hopfield_group = parser.add_argument_group('Hopfield Network Options')
    
    hopfield_group.add_argument(
        "--no-hopfield",
        action="store_true",
        help="Disable Hopfield network experiment"
    )
    
    hopfield_group.add_argument(
        "--hopfield-only",
        action="store_true",
        help="Run Hopfield experiment only, skip statistics"
    )
    
    hopfield_group.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=100,
        metavar="ITERS",
        help="Maximum iterations for Hopfield pattern recall. Default: 100"
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Options')
    
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose console output"
    )
    
    return parser.parse_args()


def print_header(verbose: bool = True):
    """Print script header."""
    if not verbose:
        return
    
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "ADV_ML HW2 - Hopfield Network Assignment" + " "*12 + "║")
    print("║" + " "*10 + "Data Processing • Statistics • Network Training" + " "*10 + "║")
    print("╚" + "═"*68 + "╝")
    print()


def print_config(args, verbose: bool = True):
    """Print configuration summary."""
    if not verbose:
        return
    
    print("Configuration:")
    print(f"  • Random Seed:        {args.seed}")
    print(f"  • Noise Percentage:   {args.noise_percent}%")
    print(f"  • Statistics:         {'Disabled' if args.no_stats else 'Enabled'}")
    print(f"  • Hopfield Experiment: {'Disabled' if args.no_hopfield else 'Enabled'}")
    if not args.no_hopfield:
        print(f"  • Max Iterations:     {args.max_iterations}")
    print(f"  • Verbose Output:     {'Off' if args.quiet else 'On'}")
    
    print()


def print_summary(data_info: dict, stats_info: dict = None, hopfield_info: dict = None, verbose: bool = True):
    """Print final summary."""
    if not verbose:
        return
    
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " "*25 + "EXECUTION SUMMARY" + " "*26 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    print("📂 Data Files:")
    paths = data_info.get("paths", {})
    if paths:
        print(f"   • Training CSV:    data/characters.csv")
        print(f"   • Testing CSV:     data/test.csv")
        valid_folder = data_info.get("valid_folder")
        if valid_folder:
            print(f"   • Validation:      {valid_folder.relative_to(paths['project'])}/")
    
    print()
    print("📊 Dataset Shapes (DataFrames):")
    print(f"   • Training:        {data_info.get('train_shape', 'N/A')}")
    print(f"   • Testing:         {data_info.get('test_shape', 'N/A')}")
    print(f"   • Validation:      {data_info.get('valid_shape', 'N/A')}")
    
    print()
    print("🔢 Matrix Arrays (indexed access via data[i]):")
    print(f"   • train_data:      {data_info.get('train_matrices_shape', 'N/A')}")
    print(f"   • test_data:       {data_info.get('test_matrices_shape', 'N/A')}")
    print(f"   • valid_data:      {data_info.get('valid_matrices_shape', 'N/A')}")
    
    if stats_info:
        print()
        print("📈 Statistics Output:")
        stats_folder = stats_info.get("stats_folder")
        if stats_folder:
            paths = data_info.get("paths", {})
            if paths.get("project"):
                rel_path = stats_folder.relative_to(paths["project"])
                print(f"   • Folder:          {rel_path}/")
            print(f"   • CSV Report:      {stats_info['csv_file'].name}")
            print(f"   • MD Report:       {stats_info['report_file'].name}")
            print(f"   • Plots:           {len(stats_info.get('plot_files', []))} visualizations")
    
    if hopfield_info:
        print()
        print("🧠 Hopfield Network Results:")
        output_folder = hopfield_info.get("output_folder")
        if output_folder:
            paths = data_info.get("paths", {})
            if paths.get("project"):
                try:
                    rel_path = output_folder.relative_to(paths["project"])
                    print(f"   • Output Folder:   {rel_path}/")
                except ValueError:
                    print(f"   • Output Folder:   {output_folder}/")
        
        # Training patterns recall
        training_recall = hopfield_info.get("training_recall", {})
        if training_recall:
            print(f"   • Training Recall: {training_recall.get('accuracy', 0):.1%} accuracy")
            print(f"                      (Hamming: {training_recall.get('mean_hamming', 0):.2f} avg)")
        
        # Test patterns recall
        test_recall = hopfield_info.get("test_recall", {})
        if test_recall:
            print(f"   • Test Recall:     {test_recall.get('accuracy', 0):.1%} accuracy")
            print(f"                      (Hamming: {test_recall.get('mean_hamming', 0):.2f} avg)")
        
        # Validation patterns recall
        valid_recall = hopfield_info.get("valid_recall", {})
        if valid_recall:
            print(f"   • Valid Recall:    {valid_recall.get('accuracy', 0):.1%} accuracy")
            print(f"                      (Hamming: {valid_recall.get('mean_hamming', 0):.2f} avg)")
        
        # Report file
        report_file = hopfield_info.get("report_file")
        if report_file:
            print(f"   • Report:          {report_file.name}")
        
    print()
    print("✅ Execution completed successfully!")
    print()


class TeeOutput:
    """
    A class that duplicates output to both console and a file.
    This allows capturing all print statements while still displaying them.
    """
    
    def __init__(self, file_path: Path):
        """Initialize TeeOutput with file path."""
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.buffer = io.StringIO()
    
    def write(self, message: str):
        """Write to both stdout and file."""
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write
    
    def flush(self):
        """Flush both streams."""
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        """Close the file and restore stdout."""
        self.file.close()
        sys.stdout = self.stdout


def setup_printout_capture(seed: int, noise_percent: float) -> tuple:
    """
    Set up printout capture to file.
    
    Args:
        seed: Random seed used
        noise_percent: Noise percentage used
        
    Returns:
        Tuple of (TeeOutput object, printout folder path, printout file path)
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create printouts directory structure
    project_dir = Path(__file__).parent.parent
    printouts_dir = project_dir / 'printouts'
    printouts_dir.mkdir(exist_ok=True)
    
    # Create subfolder with naming convention
    subfolder_name = f"main_printout_{seed}_{noise_percent}_{timestamp}"
    printout_folder = printouts_dir / subfolder_name
    printout_folder.mkdir(exist_ok=True)
    
    # Create printout file
    printout_file = printout_folder / "main_printout.txt"
    
    # Set up tee output
    tee = TeeOutput(printout_file)
    sys.stdout = tee
    
    return tee, printout_folder, printout_file


def main():
    """Main entry point."""
    args = parse_args()
    verbose = not args.quiet
    
    # Set up printout capture (captures all console output to file)
    tee, printout_folder, printout_file = setup_printout_capture(
        seed=args.seed,
        noise_percent=args.noise_percent
    )
    
    try:
        # Print console header
        if verbose:
            try:
                from author_info import print_console_header
                print_console_header("kamp_hw2.py")
            except ImportError:
                pass
        
        # Handle exclusive flags
        if args.stats_only:
            args.no_hopfield = True
        if args.hopfield_only:
            args.no_stats = True
        
        # Print header
        print_header(verbose)
        
        # Print configuration
        print_config(args, verbose)
        
        # Validate arguments
        if args.noise_percent < 0 or args.noise_percent > 100:
            print("Error: Noise percentage must be between 0 and 100")
            sys.exit(1)
        
        # Step 1: Load and prepare datasets
        # Returns both DataFrames (flattened) and 3D numpy arrays (matrices)
        train_df, test_df, valid_df, train_data, test_data, valid_data, data_info = load_all_datasets(
            noise_percent=args.noise_percent,
            random_seed=args.seed,
            verbose=verbose
        )
        
        # Step 2: Compute statistics (unless disabled)
        stats_info = None
        if not args.no_stats:
            stats_info = compute_and_save_statistics(
                train_df=train_df,
                test_df=test_df,
                valid_df=valid_df,
                seed=args.seed,
                noise_percent=args.noise_percent,
                verbose=verbose
            )
        
        # Step 3: Run Hopfield network experiment (unless disabled)
        hopfield_info = None
        if not args.no_hopfield:
            if verbose:
                print()
                print("="*70)
                print(" HOPFIELD NETWORK EXPERIMENT")
                print("="*70)
            
            hopfield_info = run_hopfield_experiment(
                train_data=train_data,
                test_data=test_data,
                valid_data=valid_data,
                random_seed=args.seed,
                noise_percent=args.noise_percent,
                max_iterations=args.max_iterations,
                verbose=verbose
            )
        
        # Print summary
        print_summary(data_info, stats_info, hopfield_info, verbose)
        
        # Print printout file location
        if verbose:
            print()
            print("📝 Console Printout Saved:")
            print(f"   • Folder:   {printout_folder.relative_to(printout_folder.parent.parent)}/")
            print(f"   • File:     {printout_file.name}")
            print()
        
        return {
            # Flattened DataFrames (18, 144)
            "train_df": train_df,
            "test_df": test_df,
            "valid_df": valid_df,
            # 3D NumPy arrays (18, 12, 12) - access via data[i] for matrix i
            "train_data": train_data,
            "test_data": test_data,
            "valid_data": valid_data,
            # Metadata
            "data_info": data_info,
            "stats_info": stats_info,
            "hopfield_info": hopfield_info,
            "printout_file": printout_file
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Close tee output and restore stdout
        tee.close()


if __name__ == "__main__":
    main()
