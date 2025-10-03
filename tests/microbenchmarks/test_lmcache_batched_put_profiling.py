#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
LMCache Batched Put Task Profiling Benchmarks

This script benchmarks LMCache batched_submit_put_task performance to identify
bottlenecks and measure I/O throughput for the Weka GDS backend.

The benchmark measures:
1. Time to submit batched put tasks
2. Time for all futures to complete
3. Overall throughput (GB/s)
4. Individual thread/coroutine profiling

Configurable parameters:
- Number of keys to submit in batch
- Size of each memory object
- Number of GDS I/O threads
- Chunk size for LMCache

Uses pyinstrument for statistical profiling.
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
from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse
import json
import os
import time

# Third Party
import numpy as np
import pytest
import torch

# Profiling imports
try:
    # Third Party
    import pyinstrument

    PYINSTRUMENT_AVAILABLE = True
except ImportError:
    PYINSTRUMENT_AVAILABLE = False
    print("WARNING: pyinstrument not available. Install with: pip install pyinstrument")

# LMCache imports
try:
    # First Party
    from lmcache.v1.cache_engine import CacheEngineKey
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.storage_backend.weka_gds_backend import WekaGdsBackend

    LMCACHE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: lmcache not available: {e}")
    LMCACHE_AVAILABLE = False


@dataclass
class BatchedPutResults:
    """Container for batched put profiling results"""

    num_keys: int
    object_size_mb: float
    gds_io_threads: int
    chunk_size: int
    submit_time: float
    completion_time: float
    total_time: float
    submit_throughput_gbps: float
    completion_throughput_gbps: float
    total_throughput_gbps: float
    total_data_mb: float
    profiling_output: Optional[str] = None
    thread_profiling_enabled: bool = False
    errors: int = 0
    error_details: Optional[List[str]] = None

    def __post_init__(self):
        if self.error_details is None:
            self.error_details = []


class DetailedTimer:
    """Context manager for detailed timing measurements"""

    def __init__(self, name: str, collector: Dict[str, List[float]]):
        self.name = name
        self.collector = collector
        self.start_time = None

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        if self.name not in self.collector:
            self.collector[self.name] = []
        self.collector[self.name].append(elapsed)


class AdvancedProfiler:
    """Advanced profiling wrapper for pyinstrument"""

    def __init__(self):
        self.profiler = None
        self.active = False

    def start(self):
        """Start profiling"""
        if PYINSTRUMENT_AVAILABLE:
            self.profiler = pyinstrument.Profiler()
            self.profiler.start()
            self.active = True
        else:
            print("WARNING: pyinstrument not available, skipping profiling")
            self.active = False

    def stop(self) -> Optional[str]:
        """Stop profiling and return results"""
        if not self.active or self.profiler is None:
            return None

        self.profiler.stop()
        output = self.profiler.output_text(unicode=True, color=False)
        return output

    def save_html_report(self, filename: str):
        """Save HTML report"""
        if self.profiler and self.active:
            try:
                html_output = self.profiler.output_html()
                with open(filename, "w") as f:
                    f.write(html_output)
                return True
            except Exception:
                return False
        return False


def generate_test_memory_objects(
    num_objects: int,
    object_size_mb: float,
    memory_allocator,
    dtype: torch.dtype = torch.bfloat16,
) -> List:
    """Generate test memory objects for batched put operations using the allocator

    Args:
        num_objects: Number of memory objects to generate
        object_size_mb: Size of each object in MB
        memory_allocator: CuFileMemoryAllocator instance to use for allocation
        dtype: Data type for tensors

    Returns:
        List of memory objects allocated from the memory allocator
    """
    # First Party
    from lmcache.v1.memory_management import MemoryFormat

    # Calculate tensor shape to match desired MB
    # bfloat16 = 2 bytes per element
    bytes_per_element = 2 if dtype == torch.bfloat16 else 4
    num_elements = int((object_size_mb * 1024 * 1024) / bytes_per_element)

    memory_objects = []

    for i in range(num_objects):
        # Allocate from the memory allocator to get a proper MemoryObj
        shape = torch.Size([num_elements])
        mem_obj = memory_allocator.allocate(
            shape=shape,
            dtype=dtype,
            fmt=MemoryFormat.UNDEFINED,
        )

        if mem_obj is None:
            raise RuntimeError(
                f"Failed to allocate memory object {i + 1}/{num_objects}"
            )

        # Fill with random data to simulate real KV cache data
        mem_obj.tensor.copy_(
            torch.randn(num_elements, dtype=dtype, device=mem_obj.tensor.device)
        )

        memory_objects.append(mem_obj)

    return memory_objects


def generate_test_keys(
    num_keys: int,
    model_name: str = "batched-put-benchmark",
    chunk_size: int = 256,
    num_layers: int = 2,
    num_heads: int = 32,
    head_size: int = 128,
) -> List[CacheEngineKey]:
    """Generate test cache keys for batched put operations"""
    keys = []

    for i in range(num_keys):
        # Create unique token sequence for each key and hash it
        tokens = torch.arange(i * chunk_size, (i + 1) * chunk_size, dtype=torch.long)

        # Compute chunk hash from tokens (similar to TokenDatabase._hash_tokens)
        tokens_tuple = tuple(tokens.cpu().tolist())
        chunk_hash = hash(
            (None, tokens_tuple, None)
        )  # (prefix_hash, tokens, extra_keys)

        # Create a cache key with proper parameters
        key = CacheEngineKey(
            fmt="vllm",
            model_name=f"{model_name}-layers{num_layers}",
            world_size=1,
            worker_id=0,
            chunk_hash=chunk_hash,
            request_configs=None,
        )
        keys.append(key)

    return keys


def setup_weka_backend(
    chunk_size: int,
    weka_path: str = "/mnt/weka/bench-cache",
    cufile_buffer_size: int = 2048,
    gds_io_threads: int = 4,
    num_layers: int = 2,
    num_heads: int = 32,
    head_size: int = 128,
    device: str = "cuda",
    max_batch_size_mb: float = 2000.0,  # Max expected batch size in MB
):
    """Setup Weka GDS backend for benchmarking

    Args:
        max_batch_size_mb: Maximum expected batch size in MB. The memory allocator
            will be sized to accommodate this (with some overhead). Default 2GB.
    """
    if not LMCACHE_AVAILABLE:
        raise RuntimeError("LMCache is not available")

    # Standard
    import asyncio
    import threading

    # First Party
    from lmcache.v1.memory_management import CuFileMemoryAllocator

    # Create config
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=False,
        weka_path=weka_path,
        cufile_buffer_size=cufile_buffer_size,
        extra_config={"gds_io_threads": gds_io_threads},
        remote_url=None,
        remote_serde=None,
        use_layerwise=False,
        save_decode_cache=False,
        enable_blending=False,
        enable_p2p=False,
        error_handling=False,
    )

    # Create an event loop in a separate thread for the backend
    loop = asyncio.new_event_loop()

    def run_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, args=(loop,), daemon=True)
    loop_thread.start()

    # Create memory allocator with reasonable size based on expected workload
    # Add 20% overhead for safety, and align to 4KB boundaries
    allocator_size_mb = max_batch_size_mb * 1.2
    allocator_size_bytes = int(allocator_size_mb * 1024 * 1024)
    # Align to 4KB (4096 bytes) as required by CuFile
    allocator_size_bytes = ((allocator_size_bytes + 4095) // 4096) * 4096

    print(
        f"  Creating CuFile memory allocator: {allocator_size_bytes / (1024**3):.2f} GB"
    )

    try:
        memory_allocator = CuFileMemoryAllocator(
            size=allocator_size_bytes,
            device=device,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Provide helpful error message
            raise RuntimeError(
                f"CUDA OOM when creating CuFile allocator of size "
                f"{allocator_size_bytes / (1024**3):.2f} GB. "
                f"Try reducing --cufile-buffer-size or the workload size "
                f"(num_keys * object_size)."
            ) from e
        raise

    # Create Weka backend with proper parameters
    backend = WekaGdsBackend(
        config=config,
        loop=loop,
        memory_allocator=memory_allocator,
        dst_device=device,
    )

    # Return backend along with loop for cleanup
    # Store loop and thread as tuple for caller to manage
    return backend, loop, loop_thread


def clear_weka_cache_directory(weka_path: str, force: bool = False):
    """Clear the Weka cache directory"""
    if force or input(f"Clear cache directory {weka_path}? (y/N): ").lower().startswith(
        "y"
    ):
        print(f"Clearing cache directory: {weka_path}")
        try:
            # Standard
            import glob
            import subprocess

            items_to_remove = glob.glob(os.path.join(weka_path, "*"))
            if items_to_remove:
                subprocess.run(
                    ["rm", "-rf"] + items_to_remove,
                    check=False,
                    capture_output=True,
                )
                print(
                    f"Cache directory cleared successfully "
                    f"({len(items_to_remove)} items removed)"
                )
            else:
                print("Cache directory was already empty")
        except Exception as e:
            print(f"Warning: Failed to clear cache directory: {e}")


def benchmark_batched_put_scenario(
    num_keys: int,
    object_size_mb: float,
    chunk_size: int = 256,
    gds_io_threads: int = 4,
    num_iterations: int = 3,
    enable_profiling: bool = False,
    save_html_reports: bool = False,
    clear_cache: bool = False,
    weka_path: str = "/mnt/weka/bench-cache",
    cufile_buffer_size: int = 2048,
    device: str = "cuda",
    num_layers: int = 2,
    profiling_sleep_seconds: int = 0,
) -> BatchedPutResults:
    """Benchmark batched_submit_put_task for a specific configuration"""

    print(
        f"\n=== Benchmarking batched_submit_put_task - "
        f"keys={num_keys}, size={object_size_mb}MB, "
        f"gds_threads={gds_io_threads} ==="
    )

    # Clear cache if requested
    if clear_cache:
        clear_weka_cache_directory(weka_path, force=True)

    # Calculate the total batch size for allocator sizing
    total_batch_size_mb = num_keys * object_size_mb

    # Setup backend with appropriately sized allocator
    backend, loop, loop_thread = setup_weka_backend(
        chunk_size=chunk_size,
        weka_path=weka_path,
        cufile_buffer_size=cufile_buffer_size,
        gds_io_threads=gds_io_threads,
        num_layers=num_layers,
        device=device,
        max_batch_size_mb=total_batch_size_mb,
    )

    profiling_output = None
    errors = 0
    error_details = []

    submit_times = []
    completion_times = []
    total_times = []

    try:
        for iteration in range(num_iterations):
            print(f"  Iteration {iteration + 1}/{num_iterations}")

            # Generate fresh test data for each iteration
            keys = generate_test_keys(
                num_keys, chunk_size=chunk_size, num_layers=num_layers
            )
            memory_objects = generate_test_memory_objects(
                num_keys, object_size_mb, backend._memory_allocator
            )

            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Sleep before profiled iteration
            if (
                enable_profiling
                and iteration == (num_iterations - 1)
                and profiling_sleep_seconds > 0
            ):
                print(
                    f"    Profiling iteration - sleeping {profiling_sleep_seconds}"
                    " seconds..."
                )
                time.sleep(profiling_sleep_seconds)
                print("    Starting profiled iteration...")

            # Start profiling on last iteration
            profiler = None
            if enable_profiling and iteration == (num_iterations - 1):
                profiler = AdvancedProfiler()
                profiler.start()

            # Measure submit time
            torch.cuda.synchronize()
            submit_start = time.perf_counter()

            try:
                # Call batched_submit_put_task (returns None, queues async tasks)
                backend.batched_submit_put_task(keys, memory_objects)

            except Exception as e:
                errors += 1
                error_msg = f"Submit error in iteration {iteration}: {str(e)}"
                error_details.append(error_msg)
                print(f"    ERROR: {error_msg}")
                continue

            torch.cuda.synchronize()
            submit_time = time.perf_counter() - submit_start
            submit_times.append(submit_time)

            print(f"    Submit time: {submit_time:.4f}s")

            # Measure completion time by monitoring put_tasks
            completion_start = time.perf_counter()

            try:
                # Wait for all put tasks to complete
                timeout_seconds = 60
                start_wait = time.perf_counter()

                while time.perf_counter() - start_wait < timeout_seconds:
                    with backend.put_lock:
                        pending_tasks = len(backend.put_tasks)

                    if pending_tasks == 0:
                        break

                    time.sleep(0.05)  # Brief sleep to avoid busy waiting

                if pending_tasks > 0:
                    error_msg = (
                        f"Timeout waiting for tasks (still {pending_tasks} pending)"
                    )
                    error_details.append(error_msg)
                    print(f"    WARNING: {error_msg}")
                    errors += 1

            except Exception as e:
                errors += 1
                error_msg = f"Completion error in iteration {iteration}: {str(e)}"
                error_details.append(error_msg)
                print(f"    ERROR: {error_msg}")
                continue

            torch.cuda.synchronize()
            completion_time = time.perf_counter() - completion_start
            completion_times.append(completion_time)

            total_time = submit_time + completion_time
            total_times.append(total_time)

            print(f"    Completion time: {completion_time:.4f}s")
            print(f"    Total time: {total_time:.4f}s")

            # Stop profiling
            if profiler and profiler.active:
                profiling_output = profiler.stop()
                if profiling_output:
                    profile_filename = (
                        f"profile_batched_put_keys_{num_keys}_size_{object_size_mb}MB_"
                        f"threads_{gds_io_threads}.txt"
                    )
                    with open(profile_filename, "w") as f:
                        f.write(
                            f"Batched Put Operation Profile - Keys: {num_keys}, "
                            f"Size: {object_size_mb}MB, Threads: {gds_io_threads}\n"
                        )
                        f.write("=" * 80 + "\n")
                        f.write(profiling_output)
                    print(f"  Profiling saved to {profile_filename}")

                    if save_html_reports:
                        html_filename = (
                            f"profile_batched_put_keys_{num_keys}_size_{object_size_mb}MB_"
                            f"threads_{gds_io_threads}.html"
                        )
                        if profiler.save_html_report(html_filename):
                            print(f"  HTML report saved to {html_filename}")

            # Cleanup memory objects between iterations
            for mem_obj in memory_objects:
                try:
                    # Decrement ref count to allow deallocation
                    while mem_obj.get_ref_count() > 0:
                        mem_obj.ref_count_down()
                    # Free the memory object
                    backend._memory_allocator.free(mem_obj)
                except Exception:
                    pass  # Ignore cleanup errors

            # Cleanup between iterations
            torch.cuda.empty_cache()
            time.sleep(0.5)

        # Calculate statistics
        if submit_times and completion_times:
            avg_submit = np.mean(submit_times)
            avg_completion = np.mean(completion_times)
            avg_total = np.mean(total_times)
        else:
            avg_submit = float("inf")
            avg_completion = float("inf")
            avg_total = float("inf")

        # Calculate throughput
        total_data_mb = num_keys * object_size_mb

        submit_throughput_gbps = (
            (total_data_mb / avg_submit / 1024)
            if avg_submit > 0 and avg_submit != float("inf")
            else 0
        )
        completion_throughput_gbps = (
            (total_data_mb / avg_completion / 1024)
            if avg_completion > 0 and avg_completion != float("inf")
            else 0
        )
        total_throughput_gbps = (
            (total_data_mb / avg_total / 1024)
            if avg_total > 0 and avg_total != float("inf")
            else 0
        )

        # Print timing summary
        print("\n  === Timing Summary ===")
        if submit_times:
            print(f"  Submit times: {[f'{t:.4f}s' for t in submit_times]}")
            print(f"    avg: {avg_submit:.4f}s, std: {np.std(submit_times):.4f}s")
        if completion_times:
            print(f"  Completion times: {[f'{t:.4f}s' for t in completion_times]}")
            print(
                f"    avg: {avg_completion:.4f}s, std: {np.std(completion_times):.4f}s"
            )
        if total_times:
            print(f"  Total times: {[f'{t:.4f}s' for t in total_times]}")
            print(f"    avg: {avg_total:.4f}s, std: {np.std(total_times):.4f}s")

        print(f"\n  Data size: {total_data_mb:.2f} MB")
        print(f"  Submit throughput: {submit_throughput_gbps:.2f} GB/s")
        print(f"  Completion throughput: {completion_throughput_gbps:.2f} GB/s")
        print(f"  Total throughput: {total_throughput_gbps:.2f} GB/s")

        if errors > 0:
            print(f"  ERRORS: {errors}")
            for error in error_details:
                print(f"    - {error}")

        results = BatchedPutResults(
            num_keys=num_keys,
            object_size_mb=object_size_mb,
            gds_io_threads=gds_io_threads,
            chunk_size=chunk_size,
            submit_time=avg_submit,
            completion_time=avg_completion,
            total_time=avg_total,
            submit_throughput_gbps=submit_throughput_gbps,
            completion_throughput_gbps=completion_throughput_gbps,
            total_throughput_gbps=total_throughput_gbps,
            total_data_mb=total_data_mb,
            profiling_output=profiling_output,
            errors=errors,
            error_details=error_details,
        )

        return results

    finally:
        # Cleanup backend
        try:
            backend.close()
        except Exception:
            pass

        # Stop the event loop
        try:
            loop.call_soon_threadsafe(loop.stop)
            time.sleep(0.5)  # Give loop time to stop
        except Exception:
            pass


def run_batched_put_benchmark(
    num_keys_list: Optional[List[int]] = None,
    object_sizes_mb: Optional[List[float]] = None,
    gds_io_threads_list: Optional[List[int]] = None,
    chunk_size: int = 256,
    num_iterations: int = 3,
    enable_profiling: bool = False,
    save_html_reports: bool = False,
    clear_cache: bool = False,
    output_file: Optional[str] = None,
    weka_path: str = "/mnt/weka/bench-cache",
    cufile_buffer_size: int = 2048,
    device: str = "cuda",
    num_layers: int = 2,
    profiling_sleep_seconds: int = 0,
) -> List[BatchedPutResults]:
    """Run comprehensive batched put benchmarks across multiple configurations"""

    if num_keys_list is None:
        num_keys_list = [10, 50, 100]
    if object_sizes_mb is None:
        object_sizes_mb = [8.0, 16.0, 32.0]
    if gds_io_threads_list is None:
        gds_io_threads_list = [4]

    all_results = []

    print("Running batched_submit_put_task profiling benchmark with:")
    print(f"  Number of keys: {num_keys_list}")
    print(f"  Object sizes (MB): {object_sizes_mb}")
    print(f"  GDS I/O threads: {gds_io_threads_list}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Iterations per config: {num_iterations}")
    print(f"  Profiling enabled: {enable_profiling}")
    print(f"  Clear cache: {clear_cache}")
    print(f"  Weka path: {weka_path}")

    first_run = True
    for gds_threads in gds_io_threads_list:
        for num_keys in num_keys_list:
            for obj_size in object_sizes_mb:
                try:
                    results = benchmark_batched_put_scenario(
                        num_keys=num_keys,
                        object_size_mb=obj_size,
                        chunk_size=chunk_size,
                        gds_io_threads=gds_threads,
                        num_iterations=num_iterations,
                        enable_profiling=enable_profiling,
                        save_html_reports=save_html_reports,
                        clear_cache=clear_cache
                        and first_run,  # Only clear on first run
                        weka_path=weka_path,
                        cufile_buffer_size=cufile_buffer_size,
                        device=device,
                        num_layers=num_layers,
                        profiling_sleep_seconds=profiling_sleep_seconds,
                    )
                    all_results.append(results)
                    first_run = False

                    # Cleanup between configurations
                    torch.cuda.empty_cache()
                    time.sleep(1)

                except Exception as e:
                    print(
                        f"ERROR in num_keys={num_keys}, size={obj_size}MB, "
                        f"threads={gds_threads}: {e}"
                    )
                    continue

    # Save results
    if output_file:
        json_results = []
        for result in all_results:
            json_result = {
                "num_keys": result.num_keys,
                "object_size_mb": result.object_size_mb,
                "gds_io_threads": result.gds_io_threads,
                "chunk_size": result.chunk_size,
                "submit_time": result.submit_time
                if result.submit_time != float("inf")
                else None,
                "completion_time": result.completion_time
                if result.completion_time != float("inf")
                else None,
                "total_time": result.total_time
                if result.total_time != float("inf")
                else None,
                "submit_throughput_gbps": result.submit_throughput_gbps,
                "completion_throughput_gbps": result.completion_throughput_gbps,
                "total_throughput_gbps": result.total_throughput_gbps,
                "total_data_mb": result.total_data_mb,
                "has_profiling_data": result.profiling_output is not None,
                "errors": result.errors,
                "error_details": result.error_details,
            }
            json_results.append(json_result)

        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'=' * 100}")
    print("BATCHED PUT PROFILING BENCHMARK SUMMARY")
    print(f"{'=' * 100}")
    print(
        f"{'Keys':<8} {'Size(MB)':<10} {'Threads':<10} {'Submit(s)':<12} "
        f"{'Complete(s)':<12} {'Total(s)':<12} {'Throughput(GB/s)':<18} {'Errors':<10}"
    )
    print(f"{'-' * 100}")

    for result in all_results:
        submit_str = (
            f"{result.submit_time:.4f}"
            if result.submit_time != float("inf")
            else "FAILED"
        )
        complete_str = (
            f"{result.completion_time:.4f}"
            if result.completion_time != float("inf")
            else "FAILED"
        )
        total_str = (
            f"{result.total_time:.4f}"
            if result.total_time != float("inf")
            else "FAILED"
        )
        error_str = str(result.errors) if result.errors > 0 else "-"

        print(
            f"{result.num_keys:<8} {result.object_size_mb:<10.1f} "
            f"{result.gds_io_threads:<10} {submit_str:<12} {complete_str:<12} "
            f"{total_str:<12} {result.total_throughput_gbps:<18.2f} {error_str:<10}"
        )

    return all_results


# Pytest fixtures and test functions


@pytest.mark.benchmark
def test_quick_batched_put_benchmark():
    """Quick batched put benchmark for run_benchmarks.py integration"""
    if not LMCACHE_AVAILABLE:
        pytest.skip("LMCache not available")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    results = run_batched_put_benchmark(
        num_keys_list=[10, 20],
        object_sizes_mb=[8.0, 16.0],
        gds_io_threads_list=[4],
        num_iterations=2,
        enable_profiling=True,
        output_file="quick_batched_put_results.json",
    )

    assert len(results) > 0, "No benchmark results generated"
    print(f"\n✅ Quick batched put benchmark completed with {len(results)} results")


@pytest.mark.benchmark
def test_comprehensive_batched_put_benchmark():
    """Comprehensive batched put benchmark"""
    if not LMCACHE_AVAILABLE:
        pytest.skip("LMCache not available")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    results = run_batched_put_benchmark(
        num_keys_list=[10, 50, 100],
        object_sizes_mb=[8.0, 16.0, 32.0],
        gds_io_threads_list=[2, 4, 8],
        num_iterations=3,
        enable_profiling=True,
        save_html_reports=True,
        output_file="comprehensive_batched_put_results.json",
    )

    assert len(results) > 0, "No benchmark results generated"

    # Should have results for all combinations
    expected_combinations = 3 * 3 * 3  # keys * sizes * threads
    assert len(results) <= expected_combinations, f"Too many results: {len(results)}"

    # Should have at least one successful result
    successful_results = [r for r in results if r.total_time != float("inf")]
    assert len(successful_results) > 0, "No successful benchmark runs"


@pytest.mark.benchmark
def test_thread_scaling_batched_put_benchmark():
    """Benchmark to measure scaling with different numbers of GDS I/O threads"""
    if not LMCACHE_AVAILABLE:
        pytest.skip("LMCache not available")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Fixed workload, varying thread count
    results = run_batched_put_benchmark(
        num_keys_list=[100],
        object_sizes_mb=[16.0],
        gds_io_threads_list=[1, 2, 4, 8, 16],
        num_iterations=5,
        enable_profiling=False,
        output_file="thread_scaling_batched_put_results.json",
    )

    assert len(results) > 0, "No benchmark results generated"

    # Print scaling analysis
    print("\n=== Thread Scaling Analysis ===")
    for result in results:
        if result.total_time != float("inf"):
            print(
                f"Threads: {result.gds_io_threads}, "
                f"Total time: {result.total_time:.4f}s, "
                f"Throughput: {result.total_throughput_gbps:.2f} GB/s"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LMCache Batched Put Task Profiling Benchmark"
    )

    parser.add_argument(
        "--num-keys",
        type=int,
        nargs="+",
        default=[10, 50, 100],
        help="List of key counts to test",
    )

    parser.add_argument(
        "--object-sizes",
        type=float,
        nargs="+",
        default=[8.0, 16.0, 32.0],
        help="List of object sizes (MB) to test",
    )

    parser.add_argument(
        "--gds-io-threads",
        type=int,
        nargs="+",
        default=[4],
        help="List of GDS I/O thread counts to test",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for LMCache",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per configuration",
    )

    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable detailed profiling",
    )

    parser.add_argument(
        "--save-html-reports",
        action="store_true",
        help="Save HTML profiling reports",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache directory before benchmark",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="batched_put_profiling_results.json",
        help="Output file for results",
    )

    parser.add_argument(
        "--weka-path",
        type=str,
        default="/mnt/weka/bench-cache",
        help="Path to Weka mount point for cache storage",
    )

    parser.add_argument(
        "--cufile-buffer-size",
        type=int,
        default=2048,
        help=(
            "CuFile buffer size in MB (default: 2048 MB = 2 GB, "
            "kept for config but allocator sized by workload)"
        ),
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of transformer layers to simulate",
    )

    parser.add_argument(
        "--profiling-sleep-seconds",
        type=int,
        default=0,
        help="Sleep before profiling iteration for external script coordination",
    )

    args = parser.parse_args()

    # Run benchmark
    run_batched_put_benchmark(
        num_keys_list=args.num_keys,
        object_sizes_mb=args.object_sizes,
        gds_io_threads_list=args.gds_io_threads,
        chunk_size=args.chunk_size,
        num_iterations=args.iterations,
        enable_profiling=args.enable_profiling,
        save_html_reports=args.save_html_reports,
        clear_cache=args.clear_cache,
        output_file=args.output,
        weka_path=args.weka_path,
        cufile_buffer_size=args.cufile_buffer_size,
        num_layers=args.num_layers,
        profiling_sleep_seconds=args.profiling_sleep_seconds,
    )
