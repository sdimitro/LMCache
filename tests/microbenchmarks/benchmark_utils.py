# SPDX-License-Identifier: Apache-2.0
"""Utilities for microbenchmarking."""

# Standard
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Generator, List, Optional
import json
import statistics
import time

# Third Party
import torch


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    function_name: str
    config_name: str
    batch_size: int
    tensor_shape: tuple
    backend_config: dict

    # Timing results (in seconds)
    mean_time: float
    median_time: float
    std_time: float
    min_time: float
    max_time: float

    # Raw measurements
    raw_times: List[float]

    # Additional metrics
    throughput_mbps: Optional[float] = None  # MB/s
    ops_per_second: Optional[float] = None

    # Metadata
    timestamp: str = ""
    device: str = ""
    cuda_memory_used: Optional[int] = None


class BenchmarkTimer:
    """High-precision timer for benchmarking."""

    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.times: List[float] = []

    @contextmanager
    def time_context(self) -> Generator[None, None, None]:
        """Context manager for timing operations."""
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        start_time = time.perf_counter()

        try:
            yield
        finally:
            torch.cuda.synchronize()  # Ensure completion
            end_time = time.perf_counter()
            self.times.append(end_time - start_time)

    def benchmark_function(self, func: Callable, *args, **kwargs) -> List[float]:
        """Benchmark a function with warmup and multiple runs."""
        self.times.clear()

        # Warmup runs
        for _ in range(self.warmup_runs):
            with self.time_context():
                func(*args, **kwargs)

        # Clear warmup times
        self.times.clear()

        # Actual benchmark runs
        for _ in range(self.benchmark_runs):
            with self.time_context():
                func(*args, **kwargs)

        return self.times.copy()


class BenchmarkRunner:
    """Main benchmark runner that coordinates timing and result collection."""

    def __init__(self, config_name: str = "default"):
        self.config_name = config_name
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        function_name: str,
        func: Callable,
        func_args: tuple,
        func_kwargs: dict,
        batch_size: int,
        tensor_shape: tuple,
        backend_config: dict,
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
    ) -> BenchmarkResult:
        """Run a single benchmark and return results."""

        timer = BenchmarkTimer(warmup_runs, benchmark_runs)
        raw_times = timer.benchmark_function(func, *func_args, **func_kwargs)

        # Calculate statistics
        mean_time = statistics.mean(raw_times)
        median_time = statistics.median(raw_times)
        std_time = statistics.stdev(raw_times) if len(raw_times) > 1 else 0.0
        min_time = min(raw_times)
        max_time = max(raw_times)

        # Calculate throughput if we know the data size
        throughput_mbps = None
        ops_per_second = None

        if batch_size > 0:
            ops_per_second = batch_size / mean_time

            # Estimate data size (rough calculation)
            if tensor_shape:
                elements_per_tensor = 1
                for dim in tensor_shape:
                    elements_per_tensor *= dim
                # Assuming bfloat16 (2 bytes per element)
                bytes_per_tensor = elements_per_tensor * 2
                total_bytes = batch_size * bytes_per_tensor
                throughput_mbps = (total_bytes / (1024 * 1024)) / mean_time

        # Get device info
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        cuda_memory_used = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else None
        )

        result = BenchmarkResult(
            function_name=function_name,
            config_name=self.config_name,
            batch_size=batch_size,
            tensor_shape=tensor_shape,
            backend_config=backend_config,
            mean_time=mean_time,
            median_time=median_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            raw_times=raw_times,
            throughput_mbps=throughput_mbps,
            ops_per_second=ops_per_second,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            device=device,
            cuda_memory_used=cuda_memory_used,
        )

        self.results.append(result)
        return result

    def save_results(self, filepath: str):
        """Save benchmark results to a JSON file."""
        results_data = [asdict(result) for result in self.results]

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

    def load_results(self, filepath: str) -> List[BenchmarkResult]:
        """Load benchmark results from a JSON file."""
        with open(filepath, "r") as f:
            results_data = json.load(f)

        return [BenchmarkResult(**data) for data in results_data]

    def print_summary(self, result: BenchmarkResult):
        """Print a summary of benchmark results."""
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK: {result.function_name}")
        print(f"Config: {result.config_name}")
        print(f"Batch size: {result.batch_size}, Shape: {result.tensor_shape}")
        print(f"{'=' * 60}")
        print(f"Mean time:    {result.mean_time * 1000:.2f} ms")
        print(f"Median time:  {result.median_time * 1000:.2f} ms")
        print(f"Std dev:      {result.std_time * 1000:.2f} ms")
        print(f"Min time:     {result.min_time * 1000:.2f} ms")
        print(f"Max time:     {result.max_time * 1000:.2f} ms")

        if result.ops_per_second:
            print(f"Ops/sec:      {result.ops_per_second:.2f}")
        if result.throughput_mbps:
            print(f"Throughput:   {result.throughput_mbps:.2f} MB/s")

        print(f"Device:       {result.device}")
        if result.cuda_memory_used:
            print(f"CUDA mem:     {result.cuda_memory_used / (1024**2):.2f} MB")


def compare_results(
    baseline: BenchmarkResult, current: BenchmarkResult, threshold_percent: float = 5.0
) -> Dict[str, Any]:
    """Compare two benchmark results and detect regressions."""

    if baseline.function_name != current.function_name:
        raise ValueError("Cannot compare results from different functions")

    mean_diff_percent = (
        (current.mean_time - baseline.mean_time) / baseline.mean_time
    ) * 100
    median_diff_percent = (
        (current.median_time - baseline.median_time) / baseline.median_time
    ) * 100

    is_regression = mean_diff_percent > threshold_percent
    is_improvement = mean_diff_percent < -threshold_percent

    return {
        "baseline_mean": baseline.mean_time,
        "current_mean": current.mean_time,
        "mean_diff_percent": mean_diff_percent,
        "median_diff_percent": median_diff_percent,
        "is_regression": is_regression,
        "is_improvement": is_improvement,
        "threshold_percent": threshold_percent,
    }
