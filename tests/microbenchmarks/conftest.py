# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and fixtures for microbenchmarks."""

# Standard
from typing import List
import asyncio
import os
import shutil
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import CuFileMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.storage_backend import WekaGdsBackend

# Local
from .benchmark_config import DEFAULT_BENCHMARK_CONFIG, BenchmarkConfig


def pytest_addoption(parser):
    """Add custom command line options for benchmarks."""
    parser.addoption(
        "--benchmark-config",
        action="store",
        default="default",
        help="Benchmark configuration to use",
    )
    parser.addoption(
        "--benchmark-warmup",
        action="store",
        type=int,
        default=3,
        help="Number of warmup runs",
    )
    parser.addoption(
        "--benchmark-runs",
        action="store",
        type=int,
        default=10,
        help="Number of benchmark runs",
    )
    parser.addoption(
        "--save-results",
        action="store_true",
        default=False,
        help="Save benchmark results to file",
    )


@pytest.fixture(scope="session")
def benchmark_config(request) -> BenchmarkConfig:
    """Get benchmark configuration from command line or use default."""
    config = DEFAULT_BENCHMARK_CONFIG

    # Override with command line options
    config.warmup_runs = request.config.getoption("--benchmark-warmup")
    config.benchmark_runs = request.config.getoption("--benchmark-runs")
    config.save_results = request.config.getoption("--save-results")

    return config


@pytest.fixture(scope="session")
def weka_test_dir() -> str:
    """Test directory for Weka filesystem."""
    return "/mnt/weka/microbench-cache"


@pytest.fixture
def event_loop():
    """Create an event loop for async operations."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_thread(event_loop):
    """Create a dedicated thread for async operations."""
    thread = threading.Thread(target=event_loop.run_forever)
    thread.start()

    yield event_loop

    if event_loop.is_running():
        event_loop.call_soon_threadsafe(event_loop.stop)
    if thread.is_alive():
        thread.join()


@pytest.fixture
def test_config_factory():
    """Factory for creating test configurations with different parameters."""

    def _create_config(
        weka_path: str = "/mnt/weka/microbench-cache",
        chunk_size: int = 256,
        cufile_buffer_size: int = 128,
        gds_io_threads: int = 32,
    ) -> LMCacheEngineConfig:
        return LMCacheEngineConfig.from_defaults(
            chunk_size=chunk_size,
            weka_path=weka_path,
            lmcache_instance_id="microbench_instance",
            cufile_buffer_size=cufile_buffer_size,
            extra_config={"gds_io_threads": gds_io_threads},
        )

    return _create_config


@pytest.fixture
def weka_backend_factory(test_config_factory, async_thread, weka_test_dir):
    """Factory for creating WekaGDS backends with different configurations."""
    # Use WeakSet to avoid keeping strong references that prevent GC
    # Standard
    import weakref

    backend_refs = weakref.WeakSet()

    def _create_backend(
        chunk_size: int = 256,
        cufile_buffer_size: int = 128,
        gds_io_threads: int = 32,
    ) -> WekaGdsBackend:
        config = test_config_factory(
            weka_path=weka_test_dir,
            chunk_size=chunk_size,
            cufile_buffer_size=cufile_buffer_size,
            gds_io_threads=gds_io_threads,
        )

        backend = WekaGdsBackend(
            config,
            async_thread,
            CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2),
            dst_device="cuda:0",
        )
        backend_refs.add(backend)  # Weak reference won't prevent GC
        return backend

    # Ensure test directory exists
    os.makedirs(weka_test_dir, exist_ok=True)

    yield _create_backend

    # Cleanup any remaining backends (weak references may have been GC'd)
    for backend in list(backend_refs):  # Copy to avoid iteration issues
        try:
            backend.close()
        except Exception:
            pass

    if os.path.exists(weka_test_dir):
        try:
            shutil.rmtree(weka_test_dir, ignore_errors=True)
        except Exception:
            pass


@pytest.fixture
def test_key_factory():
    """Factory for creating test cache keys."""

    def _create_key(
        fmt: str = "vllm",
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        world_size: int = 8,
        worker_id: int = 0,
        chunk_hash: int = None,
    ) -> CacheEngineKey:
        if chunk_hash is None:
            # Standard
            import random

            chunk_hash = random.randint(0, 2**63 - 1)

        return CacheEngineKey(
            fmt=fmt,
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            chunk_hash=chunk_hash,
        )

    return _create_key


@pytest.fixture
def memory_obj_factory():
    """Factory for creating test memory objects."""

    def _create_memory_obj(
        backend: WekaGdsBackend,
        shape: tuple = (2, 16, 8, 128),
        dtype: torch.dtype = torch.bfloat16,
    ) -> MemoryObj:
        return backend.memory_allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_T2D)

    return _create_memory_obj


@pytest.fixture
def test_data_generator(test_key_factory, memory_obj_factory):
    """Generate test data for benchmarks."""

    def _generate_data(
        backend: WekaGdsBackend,
        batch_size: int,
        tensor_shape: tuple = (2, 16, 8, 128),
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[List[CacheEngineKey], List[MemoryObj]]:
        keys = []
        memory_objs = []

        for i in range(batch_size):
            # Generate unique hash for each key
            key = test_key_factory(
                chunk_hash=hash(f"bench_{i}_{batch_size}_{tensor_shape}") % (2**63)
            )
            memory_obj = memory_obj_factory(backend, tensor_shape, dtype)

            keys.append(key)
            memory_objs.append(memory_obj)

        return keys, memory_objs

    return _generate_data


@pytest.fixture
def populated_backend(weka_backend_factory, test_data_generator):
    """Create a backend populated with test data."""

    def _create_populated_backend(
        batch_size: int,
        tensor_shape: tuple = (2, 16, 8, 128),
        backend_config: dict = None,
    ) -> tuple[WekaGdsBackend, List[CacheEngineKey]]:
        if backend_config is None:
            backend_config = {}

        backend = weka_backend_factory(**backend_config)
        keys, memory_objs = test_data_generator(backend, batch_size, tensor_shape)

        # Store all data in the backend
        futures = backend.batched_submit_put_task(keys, memory_objs)
        for future in futures:
            future.result()  # Wait for completion

        # Verify all data is stored
        for key in keys:
            assert backend.contains(key), f"Key {key} not found in backend"

        # Free original memory objects - we don't need them after storage
        for memory_obj in memory_objs:
            if memory_obj is not None:
                memory_obj.ref_count_down()

        # Clear the list to help GC
        memory_objs.clear()

        return backend, keys

    return _create_populated_backend


def pytest_configure(config):
    """Configure pytest for benchmark runs."""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")
    config.addinivalue_line("markers", "slow: mark test as slow running")
