#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Microbenchmark to test:
1. os.makedirs() performance when directory already exists
2. File operations in single directory vs 2-level hierarchy
3. Performance with varying numbers of files (1K, 10K, 100K, 1M, 10M)

Usage:
    python benchmarks/weka_directory_benchmark.py /mnt/weka/benchmark_test
"""

# Standard
import argparse
import hashlib
import os
import shutil
import sys
import time


def benchmark_makedirs(base_path: str, num_calls: int = 100000):
    """Benchmark os.makedirs performance when directory already exists."""
    print(f"\n{'=' * 80}")
    print("BENCHMARK 1: os.makedirs() with existing directory")
    print(f"{'=' * 80}")

    test_dir = os.path.join(base_path, "makedirs_test")
    os.makedirs(test_dir, exist_ok=True)

    # Warm up
    for _ in range(100):
        os.makedirs(test_dir, exist_ok=True)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_calls):
        os.makedirs(test_dir, exist_ok=True)
    end = time.perf_counter()

    duration = end - start
    ops_per_sec = num_calls / duration
    ns_per_op = (duration / num_calls) * 1_000_000_000

    print(f"Calls: {num_calls:,}")
    print(f"Total time: {duration:.3f}s")
    print(f"Operations/sec: {ops_per_sec:,.0f}")
    print(f"Nanoseconds/op: {ns_per_op:,.0f} ns")

    shutil.rmtree(test_dir)
    return ns_per_op


def benchmark_single_directory(base_path: str, num_files: int):
    """Benchmark file operations in a single directory."""
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK 2a: Single directory with {num_files:,} files")
    print(f"{'=' * 80}")

    test_dir = os.path.join(base_path, "single_dir_test")
    os.makedirs(test_dir, exist_ok=True)

    # Create files
    print(f"Creating {num_files:,} files...")
    create_start = time.perf_counter()
    for i in range(num_files):
        filepath = os.path.join(test_dir, f"file_{i:08x}.dat")
        with open(filepath, "wb") as f:
            f.write(b"x" * 4096)  # 4KB like weka backend
    create_end = time.perf_counter()
    create_time = create_end - create_start

    files_per_sec = num_files / create_time
    print(f"  Creation time: {create_time:.3f}s ({files_per_sec:,.0f} files/sec)")

    # Benchmark: os.path.exists() on random files
    # Standard
    import random

    sample_files = [
        os.path.join(test_dir, f"file_{i:08x}.dat")
        for i in random.sample(range(num_files), min(1000, num_files))
    ]

    exists_start = time.perf_counter()
    for filepath in sample_files:
        os.path.exists(filepath)
    exists_end = time.perf_counter()
    exists_time = (exists_end - exists_start) / len(sample_files)

    print(f"  os.path.exists(): {exists_time * 1_000_000:.2f} µs/op")

    # Benchmark: open() and read existing files
    read_start = time.perf_counter()
    for filepath in sample_files:
        with open(filepath, "rb") as f:
            f.read(4096)
    read_end = time.perf_counter()
    read_time = (read_end - read_start) / len(sample_files)

    print(f"  open()+read(): {read_time * 1_000_000:.2f} µs/op")

    # Cleanup
    cleanup_start = time.perf_counter()
    shutil.rmtree(test_dir)
    cleanup_end = time.perf_counter()
    cleanup_time = cleanup_end - cleanup_start

    print(f"  Cleanup time: {cleanup_time:.3f}s")

    return {
        "create_time": create_time,
        "exists_time_us": exists_time * 1_000_000,
        "read_time_us": read_time * 1_000_000,
        "cleanup_time": cleanup_time,
    }


def benchmark_two_level_hierarchy(base_path: str, num_files: int):
    """Benchmark file operations in a 2-level directory hierarchy."""
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK 2b: 2-level hierarchy with {num_files:,} files")
    print(f"{'=' * 80}")

    test_dir = os.path.join(base_path, "two_level_test")
    os.makedirs(test_dir, exist_ok=True)

    # Create files distributed across 2-level hierarchy
    print(f"Creating {num_files:,} files in 2-level hierarchy...")
    create_start = time.perf_counter()
    for i in range(num_files):
        # Hash-based distribution (like weka backend)
        hash_val = hashlib.md5(str(i).encode()).hexdigest()
        l1_dir = hash_val[:2]
        l2_dir = hash_val[2:4]
        dir_path = os.path.join(test_dir, l1_dir, l2_dir)
        os.makedirs(dir_path, exist_ok=True)

        filepath = os.path.join(dir_path, f"file_{i:08x}.dat")
        with open(filepath, "wb") as f:
            f.write(b"x" * 4096)  # 4KB
    create_end = time.perf_counter()
    create_time = create_end - create_start

    files_per_sec = num_files / create_time
    print(f"  Creation time: {create_time:.3f}s ({files_per_sec:,.0f} files/sec)")

    # Benchmark: os.path.exists() on random files
    # Standard
    import random

    sample_indices = random.sample(range(num_files), min(1000, num_files))
    sample_files = []
    for i in sample_indices:
        hash_val = hashlib.md5(str(i).encode()).hexdigest()
        l1_dir = hash_val[:2]
        l2_dir = hash_val[2:4]
        filepath = os.path.join(test_dir, l1_dir, l2_dir, f"file_{i:08x}.dat")
        sample_files.append(filepath)

    exists_start = time.perf_counter()
    for filepath in sample_files:
        os.path.exists(filepath)
    exists_end = time.perf_counter()
    exists_time = (exists_end - exists_start) / len(sample_files)

    print(f"  os.path.exists(): {exists_time * 1_000_000:.2f} µs/op")

    # Benchmark: open() and read existing files
    read_start = time.perf_counter()
    for filepath in sample_files:
        with open(filepath, "rb") as f:
            f.read(4096)
    read_end = time.perf_counter()
    read_time = (read_end - read_start) / len(sample_files)

    print(f"  open()+read(): {read_time * 1_000_000:.2f} µs/op")

    # Cleanup
    cleanup_start = time.perf_counter()
    shutil.rmtree(test_dir)
    cleanup_end = time.perf_counter()
    cleanup_time = cleanup_end - cleanup_start

    print(f"  Cleanup time: {cleanup_time:.3f}s")

    return {
        "create_time": create_time,
        "exists_time_us": exists_time * 1_000_000,
        "read_time_us": read_time * 1_000_000,
        "cleanup_time": cleanup_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Weka directory and file operations"
    )
    parser.add_argument(
        "base_path",
        help="Base path on Weka filesystem (e.g., /mnt/weka/benchmark_test)",
    )
    parser.add_argument(
        "--makedirs-calls",
        type=int,
        default=100000,
        help="Number of os.makedirs() calls to benchmark (default: 100000)",
    )
    parser.add_argument(
        "--file-counts",
        type=str,
        default="1000,10000,100000",
        help="Comma-separated list of file counts to test (default: 1000,10000,100000)",
    )
    parser.add_argument(
        "--skip-makedirs",
        action="store_true",
        help="Skip the makedirs benchmark",
    )
    parser.add_argument(
        "--skip-file-ops",
        action="store_true",
        help="Skip the file operations benchmark",
    )

    args = parser.parse_args()

    # Validate base path
    if not os.path.exists(args.base_path):
        print(f"Error: Path '{args.base_path}' does not exist")
        sys.exit(1)

    # Create test directory
    test_base = os.path.join(args.base_path, "weka_benchmark")
    os.makedirs(test_base, exist_ok=True)

    print(f"{'=' * 80}")
    print("Weka Directory Structure Benchmark")
    print(f"Base path: {test_base}")
    print(f"{'=' * 80}")

    try:
        # Benchmark 1: os.makedirs performance
        if not args.skip_makedirs:
            makedirs_ns = benchmark_makedirs(test_base, args.makedirs_calls)

        # Benchmark 2: File operations at various scales
        if not args.skip_file_ops:
            file_counts = [int(x.strip()) for x in args.file_counts.split(",")]

            results = {}
            for num_files in file_counts:
                print(f"\n{'*' * 80}")
                print(f"Testing with {num_files:,} files")
                print(f"{'*' * 80}")

                single_result = benchmark_single_directory(test_base, num_files)
                two_level_result = benchmark_two_level_hierarchy(test_base, num_files)

                results[num_files] = {
                    "single": single_result,
                    "two_level": two_level_result,
                }

            # Summary
            print(f"\n{'=' * 80}")
            print("SUMMARY")
            print(f"{'=' * 80}")

            if not args.skip_makedirs:
                print(
                    f"\nos.makedirs(exist_ok=True) overhead: {makedirs_ns:,.0f} ns/call"
                )

            print("\nFile Operations Comparison:")
            header = (
                f"{'Files':>12} | {'Structure':^15} | {'Create (s)':>12} | "
                f"{'exists() (µs)':>15} | {'read() (µs)':>13} | {'Cleanup (s)':>12}"
            )
            print(header)
            sep = (
                f"{'-' * 12}-+-{'-' * 15}-+-{'-' * 12}-+-"
                f"{'-' * 15}-+-{'-' * 13}-+-{'-' * 12}"
            )
            print(sep)

            for num_files in file_counts:
                single = results[num_files]["single"]
                two_level = results[num_files]["two_level"]

                single_line = (
                    f"{num_files:12,} | {'Single dir':^15} | "
                    f"{single['create_time']:12.3f} | "
                    f"{single['exists_time_us']:15.2f} | "
                    f"{single['read_time_us']:13.2f} | "
                    f"{single['cleanup_time']:12.3f}"
                )
                print(single_line)
                two_level_line = (
                    f"{num_files:12,} | {'2-level':^15} | "
                    f"{two_level['create_time']:12.3f} | "
                    f"{two_level['exists_time_us']:15.2f} | "
                    f"{two_level['read_time_us']:13.2f} | "
                    f"{two_level['cleanup_time']:12.3f}"
                )
                print(two_level_line)

                # Calculate speedup/slowdown
                exists_ratio = single["exists_time_us"] / two_level["exists_time_us"]
                read_ratio = single["read_time_us"] / two_level["read_time_us"]

                if exists_ratio > 1.1:
                    msg = (
                        f"{'':12} | 2-level is {exists_ratio:.1f}x FASTER for exists()"
                    )
                    print(msg)
                elif exists_ratio < 0.9:
                    speedup = 1 / exists_ratio
                    msg = f"{'':12} | Single dir is {speedup:.1f}x FASTER for exists()"
                    print(msg)

                if read_ratio > 1.1:
                    msg = f"{'':12} | 2-level is {read_ratio:.1f}x FASTER for read()"
                    print(msg)
                elif read_ratio < 0.9:
                    speedup = 1 / read_ratio
                    msg = f"{'':12} | Single dir is {speedup:.1f}x FASTER for read()"
                    print(msg)

                print(sep)

    finally:
        # Cleanup
        if os.path.exists(test_base):
            shutil.rmtree(test_base)
        print(f"\nCleaned up benchmark directory: {test_base}")


if __name__ == "__main__":
    main()
