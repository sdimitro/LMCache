#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
LMCache Layerwise Retrieve Method Profiling Benchmarks

This script benchmarks LMCache layerwise retrieval performance across different
chunk sizes and token counts to identify bottlenecks and overhead sources using
advanced profiling tools.

Adapted from test_lmcache_retrieve_profiling.py to benchmark the layerwise
retrieve_layer() method instead of the standard retrieve() method.

The layerwise approach uses generators to pipeline layer-by-layer retrieval,
which is critical for memory efficiency in large language models.

Uses pyinstrument for statistical profiling and optional line_profiler for
line-by-line analysis.

Model Configuration:
- Configurable number of layers (default: 2 layers, use --num-layers to adjust)
- With 2 layers: Data size per token = 2 layers * 2 (K,V) * 32 heads * 128 head_size
  * 2 bytes = 32KB/token
- With 32 layers: Data size per token = 32 layers * 2 (K,V) * 32 heads * 128 head_size
  * 2 bytes = 512KB/token
- Default token counts (256, 512, 1024, 2048) with 2 layers generate files from
  8MB to 64MB
- Same token counts with 32 layers generate files from 128MB to 1GB
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
import gc
import json
import os
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import GPUConnectorInterface

# Import LMCache components - use EXACT same imports as the working test
try:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngineBuilder, LMCacheEngineMetadata
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.gpu_connector import (
        VLLMBufferLayerwiseGPUConnector,
        VLLMPagedMemLayerwiseGPUConnector,
    )

    LMCACHE_AVAILABLE = True
except ImportError:
    LMCACHE_AVAILABLE = False


# Define profiling utilities directly (copied from the standard test)
@dataclass
class ProfilingResults:
    """Container for profiling results"""

    chunk_size: int
    num_tokens: int
    num_chunks: int
    retrieve_time: float
    throughput_gbps: float
    data_size_mb: float
    detailed_timings: Optional[Dict[str, Dict[str, float]]] = None
    profiling_output: Optional[str] = None
    memory_errors: int = 0
    error_details: Optional[List[str]] = None

    def __post_init__(self):
        if self.detailed_timings is None:
            self.detailed_timings = {}
        if self.error_details is None:
            self.error_details = []


class DetailedTimer:
    """Context manager for detailed timing measurements"""

    def __init__(self, name: str, collector: Dict[str, List[float]]):
        self.name = name
        self.collector = collector
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            if self.name not in self.collector:
                self.collector[self.name] = []
            self.collector[self.name].append(elapsed)


class ProfilerWrapper:
    """Wrapper for different profiling tools"""

    def __init__(self, profiler_type: str = "pyinstrument"):
        self.profiler_type = profiler_type
        self.profiler = None
        self.output = None

    def start(self):
        if self.profiler_type == "pyinstrument":
            try:
                # Third Party
                import pyinstrument

                self.profiler = pyinstrument.Profiler()
                self.profiler.start()
            except ImportError:
                print("Warning: pyinstrument not available, profiling disabled")
        elif self.profiler_type == "line_profiler":
            try:
                # Third Party
                import line_profiler

                self.profiler = line_profiler.LineProfiler()
                self.profiler.enable()
            except ImportError:
                print("Warning: line_profiler not available, profiling disabled")

    def stop(self) -> Optional[str]:
        if self.profiler is None:
            return None

        if self.profiler_type == "pyinstrument":
            self.profiler.stop()
            self.output = self.profiler.output_text(unicode=True, color=False)
            return self.output
        elif self.profiler_type == "line_profiler":
            self.profiler.disable()
            # line_profiler output handling would go here
            return "Line profiler output (not implemented)"

        return None

    def save_html_report(self, filename: str) -> bool:
        if self.profiler is None or self.profiler_type != "pyinstrument":
            return False
        try:
            with open(filename, "w") as f:
                f.write(self.profiler.output_html())
            return True
        except Exception:
            return False


def generate_test_tokens(num_tokens: int) -> torch.Tensor:
    """Generate deterministic test tokens"""
    torch.manual_seed(42)
    return torch.randint(0, 32000, (num_tokens,), dtype=torch.long)


def generate_kv_cache_paged_tensors(
    num_blocks: int,
    device: str = "cuda",
    block_size: int = 16,
    num_layers: int = 2,
    num_heads: int = 32,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> List[torch.Tensor]:
    """Generate paged KV cache tensors for testing"""
    kv_cache = []

    # Create combined K+V tensor for each layer
    for layer_idx in range(num_layers):
        # Combined K+V tensor: [2, num_blocks, block_size, num_heads, head_size]
        layer_tensor = torch.randn(
            2, num_blocks, block_size, num_heads, head_size, dtype=dtype, device=device
        )
        kv_cache.append(layer_tensor)

    return kv_cache


def clear_weka_cache_directory(weka_path: str, force: bool = False):
    """Clear the Weka cache directory"""
    # Standard
    import shutil

    if os.path.exists(weka_path) and force:
        try:
            shutil.rmtree(weka_path)
            print(f"Cleared cache directory: {weka_path}")
        except Exception as e:
            print(f"Warning: Could not clear cache directory {weka_path}: {e}")


def setup_lmcache_metadata(
    model_name: str = "meta-llama/Llama-3.1-7B-Instruct",
    world_size: int = 1,
    worker_id: int = 0,
    num_layers: int = 2,
    num_heads: int = 32,
    head_size: int = 128,
    chunk_size: int = 256,
):
    """Setup LMCache metadata"""
    if not LMCACHE_AVAILABLE:
        raise RuntimeError("LMCache is not available")

    # Include num_layers in model_name to ensure different layer configurations
    # use different cache keys and don't share cache files
    # Note: Can't use '_' as storage backend has assertion against it
    model_name_with_layers = f"{model_name}-layers{num_layers}"

    return LMCacheEngineMetadata(
        model_name=model_name_with_layers,
        world_size=world_size,
        worker_id=worker_id,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=(num_layers, 2, chunk_size, num_heads, head_size),
        use_mla=False,
    )


def setup_layerwise_lmcache_config(
    chunk_size: int,
    use_weka: bool = False,  # Default to local CPU like original test
    weka_path: str = "/mnt/weka/bench-cache",
    cufile_buffer_size: int = 16384,  # MB
    gds_io_threads: int = 4,  # Number of I/O threads
    local_cpu: bool = True,  # Default to True like original test
    max_local_cpu_size: float = 5.0,  # GB
) -> LMCacheEngineConfig:
    """Setup LMCache configuration for layerwise benchmarking"""
    if use_weka:
        return LMCacheEngineConfig.from_defaults(
            chunk_size=chunk_size,
            local_cpu=False,  # MUST disable CPU backend when using Weka exclusively
            max_local_cpu_size=max_local_cpu_size,
            weka_path=weka_path,
            cufile_buffer_size=cufile_buffer_size,
            extra_config={"gds_io_threads": gds_io_threads},
            remote_url=None,
            remote_serde=None,
            use_layerwise=True,  # KEY DIFFERENCE: Enable layerwise mode
            save_decode_cache=False,
            enable_blending=False,
            enable_p2p=False,
        )
    else:
        return LMCacheEngineConfig.from_defaults(
            chunk_size=chunk_size,
            local_cpu=local_cpu,
            max_local_cpu_size=max_local_cpu_size,
            remote_url=None,
            remote_serde=None,
            use_layerwise=True,  # KEY DIFFERENCE: Enable layerwise mode
            save_decode_cache=False,
            enable_blending=False,
            enable_p2p=False,
        )


def benchmark_layerwise_retrieve_scenario(
    chunk_size: int,
    num_tokens: int,
    num_iterations: int = 3,
    enable_profiling: bool = False,
    profiler_type: str = "pyinstrument",
    detailed_timing: bool = True,
    use_weka: bool = False,  # Default to local CPU like original test
    weka_path: str = "/mnt/weka/bench-cache",
    cufile_buffer_size: int = 16384,
    gds_io_threads: int = 4,
    use_local_cpu: bool = True,  # Default to True like original test
    clear_cache: bool = False,
    device: str = "cuda",
    num_layers: int = 2,
    profiling_sleep_seconds: int = 0,
    use_buffer_connector: bool = False,
    verbose: bool = False,
) -> ProfilingResults:
    """Benchmark retrieve_layer() method for a specific configuration"""

    print(
        f"\n=== Benchmarking retrieve_layer() (LAYERWISE) - chunk_size={chunk_size}, "
        f"tokens={num_tokens}, layers={num_layers} ==="
    )

    # Calculate derived values
    num_chunks = (num_tokens + chunk_size - 1) // chunk_size  # Ceiling division
    num_blocks = (num_tokens + 15) // 16  # 16 tokens per block

    # Clear cache directory if requested
    if clear_cache and use_weka:
        clear_weka_cache_directory(weka_path, force=True)

    # Setup configuration and metadata
    config = setup_layerwise_lmcache_config(
        chunk_size=chunk_size,
        use_weka=use_weka,
        weka_path=weka_path,
        cufile_buffer_size=cufile_buffer_size,
        gds_io_threads=gds_io_threads,
        local_cpu=use_local_cpu,
    )
    metadata = setup_lmcache_metadata(chunk_size=chunk_size, num_layers=num_layers)

    # Create layerwise GPU connector
    gpu_connector: Optional[GPUConnectorInterface] = None
    if use_buffer_connector:
        gpu_connector = VLLMBufferLayerwiseGPUConnector(
            hidden_dim_size=4096,  # Typical for 7B model
            num_layers=num_layers,
            use_gpu=True,
            dtype=torch.bfloat16,
            device=device,
            chunk_size=chunk_size,  # Add missing chunk_size parameter
        )
    else:
        gpu_connector = VLLMPagedMemLayerwiseGPUConnector(
            hidden_dim_size=4096,  # Typical for 7B model
            num_layers=num_layers,
            use_gpu=True,  # Need GPU initialization even for local CPU backend
            chunk_size=chunk_size,  # Add missing chunk_size parameter
            dtype=torch.bfloat16,  # Add missing dtype parameter
            device=device,  # Add missing device parameter
        )

    # Create unique engine instance for this configuration
    engine_id = f"benchmark_layerwise_retrieve_{chunk_size}_{num_tokens}_{num_layers}"

    profiling_output = None
    memory_errors = 0
    error_details: List[str] = []

    try:
        # Create engine
        engine = LMCacheEngineBuilder.get_or_create(
            instance_id=engine_id,
            config=config,
            metadata=metadata,
            gpu_connector=gpu_connector,
            broadcast_fn=lambda x, rank: None,  # Dummy broadcast function
            broadcast_object_fn=lambda x, rank: None,  # Dummy broadcast function
        )

        # Generate test data
        tokens = generate_test_tokens(num_tokens)
        kv_cache = generate_kv_cache_paged_tensors(
            num_blocks=num_blocks, device=device, num_layers=num_layers
        )

        # Initialize the GPU connector with KV caches
        engine.post_init(kvcaches=kv_cache)
        # Use deterministic slot mapping for consistent results
        torch.manual_seed(42)
        slot_mapping = torch.randperm(num_blocks * 16, device=device)[:num_tokens]

        # Store data first using layerwise store method
        print(f"  Storing {num_tokens} tokens using layerwise store...")
        layerwise_storer = engine.store_layer(
            tokens, kvcaches=kv_cache, slot_mapping=slot_mapping, sync=True
        )

        # Execute the layerwise store generator
        try:
            # First call to initialize
            next(layerwise_storer)

            # Process each layer
            for layer_id in range(num_layers):
                next(layerwise_storer)

            # Final call to complete
            next(layerwise_storer)
        except StopIteration:
            pass  # Generator completed normally

        # Wait for async operations to complete (critical for Weka backend)
        if use_weka:
            print("  Waiting for async store operations to complete...")
            _wait_for_weka_async_operations(
                engine.storage_manager, timeout=30.0, verbose=True
            )

            # Verify that keys are actually findable after async completion
            print("  Verifying cache availability after async completion...")
            _verify_layerwise_cache_availability(engine, tokens)

        # Benchmark retrieve_layer operation
        retrieve_times = []
        detailed_timings: Dict[str, List[float]] = {}

        for iteration in range(num_iterations):
            print(f"  Iteration {iteration + 1}/{num_iterations}")

            # Prepare fresh cache tensors for retrieve operation
            retrieved_cache = generate_kv_cache_paged_tensors(
                num_blocks=num_blocks, device=device, num_layers=num_layers
            )
            # Force garbage collection and clear CUDA cache to free memory from
            # previous iterations
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Profile retrieve_layer operation on last iteration (after warmup)
            if enable_profiling and iteration == num_iterations - 1:
                if profiling_sleep_seconds > 0:
                    print(
                        f"  Sleeping for {profiling_sleep_seconds}s before profiling..."
                    )
                    time.sleep(profiling_sleep_seconds)

                retrieve_profiler = ProfilerWrapper(profiler_type)
                retrieve_profiler.start()

            # Time the layerwise retrieve operation
            if detailed_timing and iteration == num_iterations - 1:
                with DetailedTimer("layerwise_retrieve_total", detailed_timings):
                    ret_mask = _execute_layerwise_retrieve(
                        engine, tokens, retrieved_cache, slot_mapping
                    )
                # Get the time for this specific iteration
                if detailed_timings.get("layerwise_retrieve_total"):
                    iteration_retrieve_time = detailed_timings[
                        "layerwise_retrieve_total"
                    ][-1]
            else:
                start_time = time.perf_counter()
                ret_mask = _execute_layerwise_retrieve(
                    engine, tokens, retrieved_cache, slot_mapping
                )
                torch.cuda.synchronize()
                iteration_retrieve_time = time.perf_counter() - start_time
                retrieve_times.append(iteration_retrieve_time)

            # Wait for async operations to complete after each retrieve (critical for
            # memory management)
            if use_weka:
                _wait_for_weka_async_operations(
                    engine.storage_manager, timeout=10.0, verbose=False
                )

            retrieved_tokens = torch.sum(ret_mask).item()
            print(f"    Layerwise retrieve time: {iteration_retrieve_time:.4f}s")
            print(f"    Retrieved: {retrieved_tokens}/{num_tokens} tokens")

            # Comprehensive cleanup between iterations to prevent memory buildup
            _cleanup_memory_and_sync(engine, use_weka=use_weka)

            # Stop profiling and save results
            if enable_profiling and iteration == num_iterations - 1:
                retrieve_output = retrieve_profiler.stop()
                save_html_reports = (
                    os.getenv("SAVE_HTML_REPORTS", "false").lower() == "true"
                )

                if retrieve_output:
                    profile_filename = (
                        f"profile_layerwise_retrieve_chunk_{chunk_size}_tokens_"
                        f"{num_tokens}_layers_{num_layers}.txt"
                    )
                    with open(profile_filename, "w") as f:
                        f.write(
                            f"Layerwise Retrieve Operation Profile - "
                            f"Chunk Size: {chunk_size}, Tokens: {num_tokens}, "
                            f"Layers: {num_layers}\n"
                        )
                        f.write("=" * 80 + "\n")
                        f.write(retrieve_output)
                    print(f"  Layerwise retrieve profiling saved to {profile_filename}")
                    profiling_output = retrieve_output  # Store for results

                    if save_html_reports:
                        html_filename = (
                            f"profile_layerwise_retrieve_chunk_{chunk_size}_tokens_"
                            f"{num_tokens}_layers_{num_layers}.html"
                        )
                        if retrieve_profiler.save_html_report(html_filename):
                            print(
                                f"  Layerwise retrieve HTML report saved to "
                                f"{html_filename}"
                            )

        # Print timing summary for this configuration
        if retrieve_times:
            avg_retrieve_time = sum(retrieve_times) / len(retrieve_times)
            min_retrieve_time = min(retrieve_times)
            max_retrieve_time = max(retrieve_times)
        else:
            # Use detailed timing if available
            if detailed_timings.get("layerwise_retrieve_total"):
                avg_retrieve_time = detailed_timings["layerwise_retrieve_total"][-1]
                min_retrieve_time = avg_retrieve_time
                max_retrieve_time = avg_retrieve_time
            else:
                avg_retrieve_time = float("inf")
                min_retrieve_time = float("inf")
                max_retrieve_time = float("inf")

        print(f"  Average layerwise retrieve time: {avg_retrieve_time:.4f}s")
        print(f"  Min layerwise retrieve time: {min_retrieve_time:.4f}s")
        print(f"  Max layerwise retrieve time: {max_retrieve_time:.4f}s")

        # Calculate data size and throughput
        # Each token:
        # num_layers * 2 (K,V) * 32 heads * 128 head_size * 2 bytes (bfloat16)
        data_size_bytes = num_tokens * num_layers * 2 * 32 * 128 * 2
        data_size_mb = data_size_bytes / (1024 * 1024)
        throughput_gbps = (
            (data_size_mb / avg_retrieve_time / 1024)
            if (avg_retrieve_time > 0 and avg_retrieve_time != float("inf"))
            else 0
        )

        print(f"  Data size: {data_size_mb:.2f} MB")
        print(f"  Throughput: {throughput_gbps:.2f} GB/s")

        if memory_errors > 0:
            print(f"  Memory allocation errors: {memory_errors}")
            for error in error_details:
                print(f"    {error}")

        results = ProfilingResults(
            chunk_size=chunk_size,
            num_tokens=num_tokens,
            num_chunks=num_chunks,
            retrieve_time=avg_retrieve_time,
            throughput_gbps=throughput_gbps,
            data_size_mb=data_size_mb,
            profiling_output=profiling_output,
            memory_errors=memory_errors,
            error_details=error_details,
        )

        return results

    except Exception as e:
        # Standard
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"  ERROR in layerwise retrieve benchmark: {error_msg}")
        error_details.append(error_msg)
        return ProfilingResults(
            chunk_size=chunk_size,
            num_tokens=num_tokens,
            num_chunks=num_chunks,
            retrieve_time=float("inf"),
            throughput_gbps=0,
            data_size_mb=0,
            profiling_output=None,
            memory_errors=1,
            error_details=error_details,
        )

    finally:
        # Comprehensive cleanup before destroying the engine
        try:
            _cleanup_memory_and_sync(engine, use_weka=use_weka)
        except Exception as cleanup_error:
            print(f"  Warning: Error during memory cleanup: {cleanup_error}")

        # Clean up
        try:
            LMCacheEngineBuilder.destroy(engine_id)
        except Exception as cleanup_error:
            print(f"  Warning: Error during engine cleanup: {cleanup_error}")


def _verify_layerwise_cache_availability(engine, tokens):
    """Verify that layerwise keys are actually findable in storage"""
    found_keys = 0
    total_keys = 0

    for start, end, key in engine.token_database.process_tokens(tokens=tokens):
        keys_multi_layer = key.split_layers(engine.num_layers)

        for layer_id, layer_key in enumerate(keys_multi_layer):
            total_keys += 1
            backend_location = engine.storage_manager.contains(layer_key)
            if backend_location:
                found_keys += 1
            else:
                print(f"    ✗ Layer {layer_id} key NOT found: {layer_key.to_string()}")

    print(f"    Cache verification: {found_keys}/{total_keys} keys found")

    if found_keys == 0:
        print(
            "    WARNING: No keys found after async completion - "
            "this will cause retrieve to fail"
        )
    if found_keys != total_keys:
        print(
            f"    WARNING: {found_keys}/{total_keys} keys found - "
            "this will cause retrieve to fail"
        )


def _wait_for_weka_async_operations(
    storage_manager, timeout: float = 30.0, verbose: bool = False
):
    """Wait for Weka backend async operations to complete (both put and get)"""
    # Standard
    import asyncio
    import time

    start_time = time.time()

    # Get the Weka backend
    weka_backend = storage_manager.storage_backends.get("WekaGdsBackend")
    if weka_backend is None:
        return

    if verbose:
        print(f"    Waiting for Weka async operations (timeout: {timeout}s)...")

    while time.time() - start_time < timeout:
        # Check if there are any pending put tasks
        pending_put_tasks = 0
        if hasattr(weka_backend, "put_tasks") and hasattr(weka_backend, "put_lock"):
            with weka_backend.put_lock:  # Correct attribute name
                pending_put_tasks = len(weka_backend.put_tasks)

        # Also wait for any remaining asyncio tasks in the backend's loop
        pending_async_tasks = 0
        try:

            async def count_pending_tasks():
                current_task = asyncio.current_task(weka_backend.loop)
                tasks = [
                    task
                    for task in asyncio.all_tasks(weka_backend.loop)
                    if not task.done() and task is not current_task
                ]
                return len(tasks)

            future = asyncio.run_coroutine_threadsafe(
                count_pending_tasks(), weka_backend.loop
            )
            pending_async_tasks = future.result(
                timeout=1.0
            )  # Short timeout for each check
        except Exception:
            pass  # Ignore timeout/errors in this check

        total_pending = pending_put_tasks + pending_async_tasks
        if total_pending == 0:
            if verbose:
                print(
                    f"    All async operations completed "
                    f"({time.time() - start_time:.2f}s)"
                )
            return

        if verbose:
            print(
                f"    Still waiting... {pending_put_tasks} put tasks, "
                f"{pending_async_tasks} async tasks pending"
            )
        time.sleep(0.1)  # Small delay to avoid busy waiting

    if verbose:
        print(f"    WARNING: Timeout waiting for async operations after {timeout}s")


def _force_allocator_cleanup(allocator):
    """Force cleanup of memory allocator by resetting its internal state"""
    try:
        # Import the required classes
        # First Party
        from lmcache.v1.memory_management import (
            FreeBlock,
            MemoryObjMetadata,
            TensorMemoryObj,
        )

        allocator_type = type(allocator).__name__

        # Check if this is a PagedTensorMemoryAllocator
        if hasattr(allocator, "free_blocks") and hasattr(allocator, "paged_buffers"):
            # Clear existing free blocks
            allocator.free_blocks.clear()

            # Recreate all blocks as free (this is a reset)
            for idx, buf in enumerate(allocator.paged_buffers):
                metadata = MemoryObjMetadata(
                    shape=allocator.shape,
                    dtype=allocator.dtype,
                    address=idx,
                    phy_size=allocator.align_bytes,
                    ref_count=1,
                    pin_count=0,
                    fmt=allocator.fmt,
                )

                mem_obj = TensorMemoryObj(
                    raw_data=buf,
                    metadata=metadata,
                    parent_allocator=allocator,
                )

                allocator.free_blocks.append(mem_obj)

            # Reset stats
            allocator.num_active_allocations = 0
            allocator.total_allocated_size = 0

            return True

        elif hasattr(allocator, "explicit_list"):
            # This is a TensorMemoryAllocator with explicit free list
            allocator.explicit_list.clear()
            total_size = allocator.buffer.numel()

            free_block = FreeBlock(start=0, size=total_size)
            allocator.explicit_list.add(free_block)

            # Reset stats
            allocator.num_active_allocations = 0
            allocator.total_allocated_size = 0

            return True

        elif allocator_type == "CuFileMemoryAllocator" and hasattr(
            allocator, "allocator"
        ):
            # This is a CuFileMemoryAllocator wrapping another allocator
            underlying_allocator = allocator.allocator

            # Recursively clean up the underlying allocator
            return _force_allocator_cleanup(underlying_allocator)

        else:
            # Try to find nested allocators
            if hasattr(allocator, "allocator"):
                nested_allocator = allocator.allocator
                return _force_allocator_cleanup(nested_allocator)
            elif hasattr(allocator, "memory_allocator"):
                nested_allocator = allocator.memory_allocator
                return _force_allocator_cleanup(nested_allocator)
            else:
                return False

    except ImportError:
        return False
    except Exception:
        return False


def _cleanup_memory_and_sync(engine, use_weka: bool = False):
    """Comprehensive memory cleanup and synchronization"""
    # Wait for any pending async operations in Weka backend first
    if use_weka:
        _wait_for_weka_async_operations(
            engine.storage_manager, timeout=5.0, verbose=False
        )

    # Force garbage collection multiple times to ensure cleanup
    for _ in range(3):
        gc.collect()

    # Clear CUDA memory
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Force memory allocator cleanup to prevent memory leaks between iterations
    try:
        # Check the main engine's allocator first
        if hasattr(engine, "memory_allocator"):
            main_allocator = engine.memory_allocator

            # NOTE: This is very slow but good for debugging
            # if hasattr(main_allocator, 'memcheck'):
            #     main_allocator.memcheck()

            success = _force_allocator_cleanup(main_allocator)
            if not success:
                print("    Warning: Memory allocator cleanup may have failed")

    except Exception as cleanup_error:
        print(f"    Warning: Error during allocator cleanup: {cleanup_error}")


def _force_memory_object_cleanup(memory_objects: list):
    """Force cleanup of memory objects by manually decrementing references"""
    if not memory_objects:
        return

    for mem_obj in memory_objects:
        if mem_obj is None:
            continue

        try:
            # Force unpin if pinned
            if hasattr(mem_obj, "is_pinned") and mem_obj.is_pinned:
                while mem_obj.is_pinned:
                    mem_obj.unpin()

            # Force ref count down to zero
            if hasattr(mem_obj, "get_ref_count"):
                ref_count = mem_obj.get_ref_count()
                for _ in range(ref_count):
                    mem_obj.ref_count_down()

        except Exception:
            pass  # Ignore cleanup errors


def _execute_layerwise_retrieve(engine, tokens, kvcaches, slot_mapping) -> torch.Tensor:
    """Execute layerwise retrieve operation using the generator pattern"""
    # Get the layerwise retriever generator
    layerwise_retriever = engine.retrieve_layer(
        tokens, kvcaches=kvcaches, slot_mapping=slot_mapping, sync=True
    )

    # Collect memory objects for manual cleanup
    retrieved_memory_objects = []

    # Execute the layerwise retrieval pattern based on the actual implementation
    try:
        # First call gets the number of retrieved tokens (or None if no cache found)
        retrieved_token_count = next(layerwise_retriever)

        # Process each layer
        num_layers = engine.num_layers
        for layer_id in range(num_layers):
            layer_result = next(layerwise_retriever)
            # layer_result should be None for most layers

            # Try to collect memory objects from the layer result if available
            if layer_result is not None and hasattr(layer_result, "__iter__"):
                try:
                    retrieved_memory_objects.extend(
                        [obj for obj in layer_result if obj is not None]
                    )
                except (TypeError, AttributeError):
                    pass  # layer_result might not be iterable

        # Final call to get the return mask
        ret_mask = next(layerwise_retriever)

        # If we got None for retrieved_token_count, create a zero mask
        if retrieved_token_count is None:
            ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        # Force cleanup of any collected memory objects
        _force_memory_object_cleanup(retrieved_memory_objects)

        return ret_mask

    except StopIteration:
        # If the generator stops early, return zero mask
        # Still try to cleanup any collected memory objects
        _force_memory_object_cleanup(retrieved_memory_objects)
        return torch.zeros(len(tokens), dtype=torch.bool, device="cpu")


def run_layerwise_retrieve_benchmark(
    chunk_sizes: Optional[List[int]] = None,
    token_counts: Optional[List[int]] = None,
    num_iterations: int = 3,
    enable_profiling: bool = False,
    profiler_type: str = "pyinstrument",
    output_file: Optional[str] = None,
    use_weka: bool = False,  # Default to local CPU like original test
    weka_path: str = "/mnt/weka/bench-cache",
    cufile_buffer_size: int = 16384,
    gds_io_threads: int = 4,
    use_local_cpu: bool = True,  # Default to True like original test
    clear_cache: bool = False,
    device: str = "cuda",
    num_layers: int = 2,
    profiling_sleep_seconds: int = 0,
    use_buffer_connector: bool = False,
    verbose: bool = False,
) -> List[ProfilingResults]:
    """Run comprehensive layerwise retrieve benchmarks across multiple configurations"""

    if chunk_sizes is None:
        chunk_sizes = [256, 512, 1024]
    if token_counts is None:
        token_counts = [512, 1024, 2048]

    all_results = []

    print("Running layerwise retrieve_layer() profiling benchmark with:")
    print(f"  Chunk sizes: {chunk_sizes}")
    print(f"  Token counts: {token_counts}")
    print(f"  Iterations per config: {num_iterations}")
    print(f"  Profiling enabled: {enable_profiling}")
    print(f"  Profiler type: {profiler_type}")
    print(f"  Clear cache: {clear_cache}")
    print(f"  Backend: {'Weka' if use_weka else 'Local CPU'}")
    print("  Layerwise mode: ENABLED")
    print(
        f"  GPU Connector: {'Buffer' if use_buffer_connector else 'PagedMem'} Layerwise"
    )

    for chunk_size in chunk_sizes:
        for num_tokens in token_counts:
            try:
                results = benchmark_layerwise_retrieve_scenario(
                    chunk_size=chunk_size,
                    num_tokens=num_tokens,
                    num_iterations=num_iterations,
                    enable_profiling=enable_profiling,
                    profiler_type=profiler_type,
                    use_weka=use_weka,
                    weka_path=weka_path,
                    cufile_buffer_size=cufile_buffer_size,
                    gds_io_threads=gds_io_threads,
                    use_local_cpu=use_local_cpu,
                    clear_cache=clear_cache,
                    device=device,
                    num_layers=num_layers,
                    profiling_sleep_seconds=profiling_sleep_seconds,
                    use_buffer_connector=use_buffer_connector,
                    verbose=verbose,
                )
                all_results.append(results)
            except Exception as e:
                print(
                    f"ERROR benchmarking chunk_size={chunk_size}, "
                    f"tokens={num_tokens}: {e}"
                )
                # Add failed result
                all_results.append(
                    ProfilingResults(
                        chunk_size=chunk_size,
                        num_tokens=num_tokens,
                        num_chunks=(num_tokens + chunk_size - 1) // chunk_size,
                        retrieve_time=float("inf"),
                        throughput_gbps=0,
                        data_size_mb=0,
                        profiling_output=None,
                        memory_errors=1,
                        error_details=[str(e)],
                    )
                )

    # Print summary
    print("\n" + "=" * 80)
    print("LAYERWISE RETRIEVE BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Chunk Size':<12} {'Tokens':<8} {'Chunks':<8} {'Time (s)':<10} "
        f"{'Throughput (GB/s)':<18} {'Data Size (MB)':<15}"
    )
    print("-" * 80)

    for result in all_results:
        print(
            f"{result.chunk_size:<12} {result.num_tokens:<8} {result.num_chunks:<8} "
            f"{result.retrieve_time:<10.4f} {result.throughput_gbps:<18.2f} "
            f"{result.data_size_mb:<15.2f}"
        )

    # Save results to file if requested
    if output_file:
        results_data = {
            "benchmark_type": "layerwise_retrieve",
            "configuration": {
                "chunk_sizes": chunk_sizes,
                "token_counts": token_counts,
                "num_iterations": num_iterations,
                "use_weka": use_weka,
                "cufile_buffer_size": cufile_buffer_size,
                "gds_io_threads": gds_io_threads,
                "num_layers": num_layers,
                "use_buffer_connector": use_buffer_connector,
            },
            "results": [
                {
                    "chunk_size": r.chunk_size,
                    "num_tokens": r.num_tokens,
                    "num_chunks": r.num_chunks,
                    "retrieve_time": r.retrieve_time,
                    "throughput_gbps": r.throughput_gbps,
                    "data_size_mb": r.data_size_mb,
                    "memory_errors": r.memory_errors,
                }
                for r in all_results
            ],
        }

        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nLayerwise benchmark results saved to {output_file}")

    return all_results


@pytest.mark.benchmark
def test_quick_layerwise_retrieve_benchmark():
    """Quick layerwise retrieve benchmark entry point for run_benchmarks.py"""
    if not LMCACHE_AVAILABLE:
        pytest.skip("LMCache not available")

    # Quick test with smaller parameters
    results = run_layerwise_retrieve_benchmark(
        chunk_sizes=[256, 512],
        token_counts=[512, 1024],
        num_iterations=2,
        enable_profiling=False,
        use_weka=True,
        clear_cache=True,
        num_layers=2,
        use_buffer_connector=False,  # Test PagedMem connector
    )

    # Basic validation
    assert len(results) > 0, "Should have benchmark results"
    successful_results = [r for r in results if r.retrieve_time != float("inf")]
    assert len(successful_results) > 0, "Should have at least one successful benchmark"


@pytest.mark.benchmark
def test_comprehensive_layerwise_retrieve_benchmark():
    """Comprehensive layerwise retrieve benchmark"""
    if not LMCACHE_AVAILABLE:
        pytest.skip("LMCache not available")

    # More comprehensive test
    results = run_layerwise_retrieve_benchmark(
        chunk_sizes=[256, 512, 1024],
        token_counts=[512, 1024, 2048],
        num_iterations=3,
        enable_profiling=True,
        use_weka=True,
        clear_cache=True,
        num_layers=4,  # Test with more layers
        use_buffer_connector=True,  # Test Buffer connector
    )

    # Validation
    assert len(results) > 0, "Should have benchmark results"
    successful_results = [r for r in results if r.retrieve_time != float("inf")]
    assert len(successful_results) > 0, "Should have at least one successful benchmark"

    # Performance validation - layerwise should be reasonable
    for result in successful_results:
        assert result.throughput_gbps > 0, "Should have positive throughput"
        assert result.data_size_mb > 0, "Should have positive data size"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LMCache Layerwise Retrieve Profiling Benchmark"
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[256, 512, 1024],
        help="Chunk sizes to test",
    )
    parser.add_argument(
        "--token-counts",
        nargs="+",
        type=int,
        default=[512, 1024, 2048],
        help="Token counts to test",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per configuration",
    )
    parser.add_argument(
        "--enable-profiling", action="store_true", help="Enable detailed profiling"
    )
    parser.add_argument(
        "--profiler-type",
        choices=["pyinstrument", "line_profiler"],
        default="pyinstrument",
        help="Profiler type to use",
    )
    parser.add_argument(
        "--output-file", help="Output file for benchmark results (JSON)"
    )
    parser.add_argument(
        "--use-weka",
        action="store_true",
        default=False,
        help="Use Weka backend (default: False, uses Local CPU)",
    )
    parser.add_argument(
        "--use-local-cpu",
        action="store_true",
        help="Use local CPU backend instead of Weka",
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear cache before benchmark"
    )
    parser.add_argument(
        "--weka-path", default="/mnt/weka/bench-cache", help="Weka cache directory path"
    )
    parser.add_argument(
        "--cufile-buffer-size", type=int, default=16384, help="CuFile buffer size in MB"
    )
    parser.add_argument(
        "--gds-io-threads", type=int, default=4, help="Number of GDS I/O threads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of model layers"
    )
    parser.add_argument(
        "--use-buffer-connector",
        action="store_true",
        help="Use Buffer layerwise connector instead of PagedMem",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output for troubleshooting",
    )

    args = parser.parse_args()

    if not LMCACHE_AVAILABLE:
        print(
            "ERROR: LMCache is not available. Please install LMCache to run benchmarks."
        )
        exit(1)

    # Determine backend: use_weka takes precedence over use_local_cpu
    use_weka_backend = args.use_weka and not args.use_local_cpu

    run_layerwise_retrieve_benchmark(
        chunk_sizes=args.chunk_sizes,
        token_counts=args.token_counts,
        num_iterations=args.iterations,  # Fixed parameter name
        enable_profiling=args.enable_profiling,
        profiler_type=args.profiler_type,
        output_file=args.output_file,
        use_weka=use_weka_backend,
        weka_path=args.weka_path,
        cufile_buffer_size=args.cufile_buffer_size,
        gds_io_threads=args.gds_io_threads,
        clear_cache=args.clear_cache,
        num_layers=args.num_layers,
        use_buffer_connector=args.use_buffer_connector,
        use_local_cpu=not use_weka_backend,
        verbose=args.verbose,
    )
