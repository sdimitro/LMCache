# SPDX-License-Identifier: Apache-2.0
"""Configuration settings for microbenchmarks."""

# Standard
from typing import List, Tuple
import dataclasses


@dataclasses.dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Timing configuration
    warmup_runs: int = 3
    benchmark_runs: int = 10
    timeout_seconds: float = 60.0

    # Test data configurations
    batch_sizes: List[int] = dataclasses.field(
        default_factory=lambda: [1, 5, 10, 20, 50]
    )
    tensor_shapes: List[Tuple[int, ...]] = dataclasses.field(
        default_factory=lambda: [
            (2, 16, 8, 128),  # Small
            (2, 32, 16, 128),  # Medium
            (2, 64, 32, 128),  # Large
        ]
    )

    # Backend configurations
    chunk_sizes: List[int] = dataclasses.field(default_factory=lambda: [256, 512, 1024])
    cufile_buffer_sizes: List[int] = dataclasses.field(
        default_factory=lambda: [128, 256, 512]
    )
    gds_io_threads: List[int] = dataclasses.field(default_factory=lambda: [16, 32, 64])

    # Output configuration
    save_results: bool = True
    results_file: str = "benchmark_results.json"
    verbose: bool = True


# Default configuration
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()
