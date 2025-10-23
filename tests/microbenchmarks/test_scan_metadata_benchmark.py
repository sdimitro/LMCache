#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
LMCache Metadata Scanning Microbenchmark

This script benchmarks the _scan_metadata() method from WekaGdsBackend by creating
a minimal backend instance and calling the method directly.

Usage:
    # Run as pytest benchmark
    pytest tests/microbenchmarks/test_scan_metadata_benchmark.py -v

    # Run as standalone script
    python tests/microbenchmarks/test_scan_metadata_benchmark.py \
        --weka-path /mnt/weka/bench-cache

    # Run with custom parameters
    python tests/microbenchmarks/test_scan_metadata_benchmark.py \
        --weka-path /mnt/weka/bench-cache \
        --num-iterations 5 \
        --output results.json
"""

#
# Copyright 2025 Serapheim Dimitropoulos <serapheim.dimitropoulos@weka.io>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Standard
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, Mock
import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.config import LMCacheEngineConfig

# Configure logging to suppress INFO messages but keep WARNING and ERROR
# This must be done BEFORE importing lmcache modules
logging.basicConfig(level=logging.WARNING)
for logger_name in [
    "lmcache",
    "lmcache.v1.config",
    "lmcache.v1.storage_backend.weka_gds_backend",
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


@dataclass
class ScanResults:
    """Container for metadata scanning results"""

    scan_time_seconds: float
    num_entries: int
    weka_path: str
    entries_per_second: float


def _print_histogram(values: List[float], num_bins: int = 10, width: int = 50):
    """
    Print a simple ASCII histogram of values.

    Args:
        values: List of values to plot
        num_bins: Number of bins for the histogram
        width: Width of the histogram bars in characters
    """
    if not values:
        print("  No data to plot")
        return

    min_val = min(values)
    max_val = max(values)

    # If all values are the same, just show one bar
    if min_val == max_val:
        print(f"  {min_val:.4f}s | {'█' * width} ({len(values)})")
        return

    # Create bins
    bin_size = (max_val - min_val) / num_bins
    bins = [0] * num_bins

    # Count values in each bin
    for val in values:
        bin_idx = min(int((val - min_val) / bin_size), num_bins - 1)
        bins[bin_idx] += 1

    # Find max count for scaling
    max_count = max(bins)

    # Print histogram
    for i, count in enumerate(bins):
        if count == 0:
            continue
        bin_start = min_val + i * bin_size
        bin_end = min_val + (i + 1) * bin_size
        bar_width = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_width
        print(f"  {bin_start:.4f}-{bin_end:.4f}s | {bar} ({count})")


def create_minimal_backend(weka_path: str, layerwise: bool = False):
    """
    Create a minimal WekaGdsBackend instance with just enough mocking
    to run _scan_metadata().

    Args:
        weka_path: Path to override self.weka_path with
        layerwise: Whether to use layerwise mode

    Returns:
        WekaGdsBackend instance ready for metadata scanning
    """
    # Import here to avoid issues if cufile is not available
    # First Party
    from lmcache.v1.storage_backend.weka_gds_backend import WekaGdsBackend

    # Temporarily disable INFO logging during config creation
    # Use logging.disable() to suppress all logs below WARNING
    logging.disable(logging.INFO)

    try:
        # Create minimal config - we'll override the weka_path anyway
        _ = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=False,
            weka_path="/tmp/dummy",  # Dummy path, will be overridden
            cufile_buffer_size=1024,
        )
    finally:
        logging.disable(logging.NOTSET)  # Re-enable all logging

    # Create minimal metadata
    _ = LMCacheEngineMetadata(
        model_name="benchmark-model",
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 2, 256, 32, 128),
        use_mla=False,
    )

    # Create a mock event loop
    loop = asyncio.new_event_loop()

    # Create a mock memory allocator
    mock_allocator = Mock()
    mock_allocator.base_pointer = 0

    # Mock cufile to avoid needing GPUDirect
    # Standard
    import sys

    mock_cufile = MagicMock()
    mock_cufile.CuFileDriver = MagicMock
    sys.modules["cufile"] = mock_cufile

    try:
        # Create the backend - this will call __init__
        # We need to prevent _scan_metadata from running in __init__
        backend = object.__new__(WekaGdsBackend)

        # Manually set the fields we need
        backend.layerwise = layerwise
        backend.loop = loop
        backend.memory_allocator = mock_allocator
        backend.dst_device = "cuda"
        backend.weka_path = weka_path  # Override with the actual path
        backend.hot_lock = threading.Lock()
        backend.hot_cache = OrderedDict()
        backend.metadata_dirs = set()

        return backend
    finally:
        # Clean up the mock
        if "cufile" in sys.modules:
            del sys.modules["cufile"]


async def benchmark_scan_metadata(
    weka_path: str, layerwise: bool = False
) -> ScanResults:
    """
    Benchmark the _scan_metadata() method by creating a minimal backend
    and timing the scan.

    Args:
        weka_path: Path to the Weka cache directory
        layerwise: Whether to use layerwise cache keys

    Returns:
        ScanResults object with timing and count information
    """
    if not os.path.exists(weka_path):
        print(f"ERROR: Directory not found: {weka_path}")
        return ScanResults(
            scan_time_seconds=0.0,
            num_entries=0,
            weka_path=weka_path,
            entries_per_second=0.0,
        )

    # Create minimal backend instance
    backend = create_minimal_backend(weka_path, layerwise)

    # Time the scan
    start = time.perf_counter()
    await backend._scan_metadata()
    end = time.perf_counter()

    scan_time = end - start
    num_entries = len(backend.hot_cache)
    entries_per_second = num_entries / scan_time if scan_time > 0 else 0

    return ScanResults(
        scan_time_seconds=scan_time,
        num_entries=num_entries,
        weka_path=weka_path,
        entries_per_second=entries_per_second,
    )


def run_scan_metadata_benchmark(
    weka_path: str,
    num_iterations: int = 3,
    layerwise: bool = False,
    output_file: Optional[str] = None,
) -> List[ScanResults]:
    """
    Run the metadata scanning benchmark.

    Args:
        weka_path: Path to the Weka cache directory
        num_iterations: Number of times to run the scan
        layerwise: Whether to use layerwise cache keys
        output_file: Optional path to save results as JSON

    Returns:
        List of ScanResults, one per iteration
    """
    print(f"\n{'=' * 80}")
    print("METADATA SCANNING MICROBENCHMARK")
    print(f"{'=' * 80}")
    print(f"Weka Path: {weka_path}")
    print(f"Iterations: {num_iterations}")
    print(f"Layerwise: {layerwise}")
    print(f"{'=' * 80}\n")

    if not os.path.exists(weka_path):
        print(f"ERROR: Weka path does not exist: {weka_path}")
        sys.exit(1)

    if not os.path.isdir(weka_path):
        print(f"ERROR: Weka path is not a directory: {weka_path}")
        sys.exit(1)

    all_results = []

    print("\nRunning iterations:")
    for iteration in range(num_iterations):
        # Run the scan
        results = asyncio.run(benchmark_scan_metadata(weka_path, layerwise))
        all_results.append(results)

        # Print compact iteration result
        print(
            f"  [{iteration + 1:3d}/{num_iterations}] "
            f"scan time: {results.scan_time_seconds:6.4f}s  "
            f"throughput: {results.entries_per_second:8.2f} entries/s  "
            f"entries: {results.num_entries}"
        )

    # Calculate statistics
    scan_times = [r.scan_time_seconds for r in all_results]
    entry_counts = [r.num_entries for r in all_results]
    throughputs = [r.entries_per_second for r in all_results]

    # Check if entry counts vary (warning condition)
    if len(set(entry_counts)) > 1:
        print("\n⚠️  WARNING: Entry count varied between runs!")
        print(f"    Counts: {set(entry_counts)}")
        print("    This should only happen in exceptional scenarios.\n")

    # Print summary statistics as a table
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    print(
        f"{'Metric':<25} {'Mean':>10} {'Median':>10} "
        f"{'StdDev':>10} {'Min':>10} {'Max':>10}"
    )
    print(f"{'-' * 80}")
    print(
        f"{'Scan Time (s)':<25} {np.mean(scan_times):>10.4f} "
        f"{np.median(scan_times):>10.4f} "
        f"{np.std(scan_times):>10.4f} "
        f"{np.min(scan_times):>10.4f} "
        f"{np.max(scan_times):>10.4f}"
    )
    print(
        f"{'Throughput (entries/s)':<25} {np.mean(throughputs):>10.2f} "
        f"{np.median(throughputs):>10.2f} "
        f"{np.std(throughputs):>10.2f} "
        f"{np.min(throughputs):>10.2f} "
        f"{np.max(throughputs):>10.2f}"
    )
    print(f"{'-' * 80}")
    print(f"Total entries: {entry_counts[0]}")

    # Create a simple histogram of scan times
    print("\nScan Time Distribution (seconds):")
    _print_histogram(scan_times)

    # Save results to file if requested
    if output_file:
        json_results = []
        for i, result in enumerate(all_results):
            json_result = {
                "iteration": i + 1,
                "scan_time_seconds": result.scan_time_seconds,
                "num_entries": result.num_entries,
                "entries_per_second": result.entries_per_second,
                "weka_path": result.weka_path,
            }
            json_results.append(json_result)

        summary = {
            "weka_path": weka_path,
            "num_iterations": num_iterations,
            "layerwise": layerwise,
            "summary_statistics": {
                "scan_time_mean": float(np.mean(scan_times)),
                "scan_time_median": float(np.median(scan_times)),
                "scan_time_std": float(np.std(scan_times)),
                "scan_time_min": float(np.min(scan_times)),
                "scan_time_max": float(np.max(scan_times)),
                "entries_mean": float(np.mean(entry_counts)),
                "entries_median": float(np.median(entry_counts)),
                "throughput_mean": float(np.mean(throughputs)),
                "throughput_median": float(np.median(throughputs)),
            },
            "iterations": json_results,
        }

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return all_results


# Pytest fixtures and test functions


@pytest.mark.benchmark
def test_scan_metadata_default_path():
    """Test metadata scanning with default Weka path"""
    default_path = "/mnt/weka/bench-cache"

    if not os.path.exists(default_path):
        pytest.skip(f"Default Weka path does not exist: {default_path}")

    results = run_scan_metadata_benchmark(
        weka_path=default_path,
        num_iterations=3,
        layerwise=False,
    )

    assert len(results) > 0, "No benchmark results generated"
    assert all(r.num_entries >= 0 for r in results), "Invalid entry counts"
    print(f"\n✅ Metadata scan benchmark completed with {len(results)} iterations")


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "weka_path",
    [
        pytest.param("/mnt/weka/bench-cache", id="bench-cache"),
        pytest.param("/mnt/weka/lmcache", id="lmcache"),
    ],
)
def test_scan_metadata_parametrized(weka_path):
    """Test metadata scanning with parametrized Weka paths"""
    if not os.path.exists(weka_path):
        pytest.skip(f"Weka path does not exist: {weka_path}")

    results = run_scan_metadata_benchmark(
        weka_path=weka_path,
        num_iterations=2,
        layerwise=False,
    )

    assert len(results) > 0, "No benchmark results generated"


if __name__ == "__main__":
    # Allow running as standalone script with command line arguments
    parser = argparse.ArgumentParser(
        description="LMCache Metadata Scanning Microbenchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with default path
  python test_scan_metadata_benchmark.py --weka-path /mnt/weka/bench-cache

  # Run multiple iterations and save results
  python test_scan_metadata_benchmark.py \\
      --weka-path /mnt/weka/bench-cache \\
      --iterations 5 \\
      --output scan_results.json

  # Benchmark with layerwise mode
  python test_scan_metadata_benchmark.py \\
      --weka-path /mnt/weka/bench-cache \\
      --layerwise
        """,
    )

    parser.add_argument(
        "--weka-path",
        type=str,
        required=True,
        help="Path to the Weka cache directory to scan",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run (default: 3)",
    )

    parser.add_argument(
        "--layerwise",
        action="store_true",
        help="Use layerwise cache keys (default: False)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON results (default: None, no file saved)",
    )

    args = parser.parse_args()

    # Run benchmark
    run_scan_metadata_benchmark(
        weka_path=args.weka_path,
        num_iterations=args.iterations,
        layerwise=args.layerwise,
        output_file=args.output,
    )
