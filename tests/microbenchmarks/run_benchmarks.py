#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convenience script for running microbenchmarks with common configurations."""

# Standard
from pathlib import Path
import argparse
import subprocess
import sys


def run_command(cmd: list[str], verbose: bool = True) -> int:
    """Run a command and return exit code."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
        print("-" * 60)

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run LMCache microbenchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                    # Fast development benchmark
  %(prog)s --comprehensive           # All benchmarks with default settings
  %(prog)s --scaling                 # Just scaling analysis
  %(prog)s --custom --runs 20        # Custom configuration
  %(prog)s --save-results            # Save results to JSON files
        """,
    )

    # Preset configurations
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark for development (fastest)",
    )
    preset_group.add_argument(
        "--scaling", action="store_true", help="Run scaling analysis benchmark"
    )
    preset_group.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run all comprehensive benchmarks (slowest)",
    )
    preset_group.add_argument(
        "--comparison", action="store_true", help="Run single vs batched comparison"
    )
    preset_group.add_argument(
        "--custom", action="store_true", help="Run custom benchmark configuration"
    )

    # Custom options
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of benchmark runs (default: 100)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup runs (default: 3)"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save benchmark results to JSON files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest",
    )

    args = parser.parse_args()

    # Find benchmark directory
    script_dir = Path(__file__).parent
    benchmark_dir = script_dir

    # Build pytest command
    cmd = ["python", "-m", "pytest", str(benchmark_dir)]

    # Add benchmark-specific options
    cmd.extend([f"--benchmark-runs={args.runs}"])
    cmd.extend([f"--benchmark-warmup={args.warmup}"])

    if args.save_results:
        cmd.append("--save-results")

    if args.verbose:
        cmd.append("-v")

    # Always add -s to disable output capturing so we can see benchmark results
    cmd.append("-s")

    # Add test selection based on preset
    if args.quick:
        cmd.append(str(benchmark_dir / "test_weka_gds_bench.py::test_quick_benchmark"))
        print("Running QUICK benchmark (development mode)")

    elif args.scaling:
        cmd.append(
            str(
                benchmark_dir
                / (
                    "test_weka_gds_bench.py::TestWekaGdsBenchmarks::"
                    "test_batched_get_blocking_scaling"
                )
            )
        )
        cmd.extend(["-m", "benchmark"])
        print("Running SCALING analysis benchmark")

    elif args.comprehensive:
        cmd.append(
            str(
                benchmark_dir
                / (
                    "test_weka_gds_bench.py::TestWekaGdsBenchmarks::"
                    "test_batched_get_blocking_comprehensive"
                )
            )
        )
        cmd.extend(["-m", "benchmark"])
        print("Running COMPREHENSIVE benchmark (this may take a while)")

    elif args.comparison:
        cmd.append(
            str(
                benchmark_dir
                / (
                    "test_weka_gds_bench.py::TestWekaGdsBenchmarks::"
                    "test_single_vs_batched_comparison"
                )
            )
        )
        cmd.extend(["-m", "benchmark"])
        print("Running SINGLE vs BATCHED comparison")

    elif args.custom:
        cmd.extend(["-m", "benchmark and not slow"])
        print("Running CUSTOM benchmark configuration")

    else:
        # Default: run all benchmarks except slow ones
        cmd.extend(["-m", "benchmark and not slow"])
        print("Running ALL benchmarks (excluding slow ones)")

    # Add any additional pytest arguments
    if args.pytest_args:
        cmd.extend(args.pytest_args)

    # Run the benchmark
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Runs: {args.runs}, Warmup: {args.warmup}")
    if args.save_results:
        print("Results will be saved to JSON files")
    print()

    exit_code = run_command(cmd, args.verbose)

    if exit_code == 0:
        print("\n✅ Benchmarks completed successfully!")
    else:
        print("\n❌ Benchmarks failed!")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
