# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import os
import shutil
import threading

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import CuFileMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.storage_backend import WekaGdsBackend


def create_test_config(
    weka_path: str = "/mnt/weka/test-cache",
    chunk_size: int = 256,
    cufile_buffer_size: int = 128,
    gds_io_threads: int = 32,
):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        weka_path=weka_path,
        lmcache_instance_id="test_instance",
        cufile_buffer_size=cufile_buffer_size,
        extra_config={"gds_io_threads": gds_io_threads},
    )
    return config


def create_test_key(
    fmt: str = "vllm",
    model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
    world_size: int = 8,
    worker_id: int = 0,
    chunk_hash: int = -3811288445880773366,
) -> CacheEngineKey:
    return CacheEngineKey(
        fmt=fmt,
        model_name=model_name,
        world_size=world_size,
        worker_id=worker_id,
        chunk_hash=chunk_hash,
    )


def create_test_backend(
    config: LMCacheEngineConfig, loop: asyncio.AbstractEventLoop
) -> WekaGdsBackend:
    weka_backend = WekaGdsBackend(
        config,
        loop,
        CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2),
        dst_device="cuda:0",
    )
    assert weka_backend is not None
    assert weka_backend._memory_allocator is not None
    assert isinstance(weka_backend._memory_allocator, CuFileMemoryAllocator)
    return weka_backend


def create_test_memory_obj(
    backend: WekaGdsBackend, shape=(2, 16, 8, 128), dtype=torch.bfloat16
) -> MemoryObj:
    memory_obj = backend._memory_allocator.allocate(
        shape, dtype, fmt=MemoryFormat.KV_T2D
    )
    assert memory_obj is not None, "Failed to allocate memory object"
    return memory_obj


def init_and_teardown(test_func):
    WEKA_DIR = "/mnt/weka/test-cache"
    thread_loop = None
    thread = None
    try:
        os.makedirs(WEKA_DIR, exist_ok=True)
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        weka_backend = create_test_backend(create_test_config(), thread_loop)
        test_func(weka_backend)
    finally:
        # Properly shutdown the event loop
        if thread_loop is not None:
            if thread_loop.is_running():
                thread_loop.call_soon_threadsafe(thread_loop.stop)
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)
            # Close the loop to clean up resources
            thread_loop.close()

        # We rmtree AFTER we ensure that the thread loop is done.
        # This way we don't hit any race conditions in rmtree()
        # where temp files are renamed while we try to unlink them.
        # We also take care of any other errors with ignore_errors=True
        # so if we want to run tests in parallel in the future they
        # don't make each other fail.
        if os.path.exists(WEKA_DIR):
            shutil.rmtree(WEKA_DIR, ignore_errors=True)


def submit_put_task_test(backend: WekaGdsBackend):
    """
    Test that submit_put_task properly:
    1. Adds the key to hot_cache after completion
    2. Creates an arena file on disk with correct size (data + metadata)
    """
    # Create a test key and verify it doesn't exist yet
    k = create_test_key()
    assert not backend.contains(k, False), "Key should not exist before put"

    # Create a memory object to store
    memory_obj = create_test_memory_obj(
        backend, shape=(2, 16, 8, 128), dtype=torch.bfloat16
    )
    tensor_size = memory_obj.get_size()

    # Submit the put task
    future = backend.submit_put_task(k, memory_obj)
    assert future is not None, "submit_put_task should return a Future"

    # Wait for the task to complete
    future.result()

    # Verify the key is now in hot_cache
    assert backend.contains(k, False), "Key should be in hot_cache after put completes"

    # Get the metadata from hot_cache to find arena details
    with backend.hot_lock:
        metadata = backend.hot_cache[k]

    # Construct the arena file path
    arena_id = metadata.arena_id
    arena_path = os.path.join(
        backend._arena_manager._working_directory, f"{arena_id}.arena"
    )

    # Verify the arena file was created
    assert os.path.exists(arena_path), f"Arena file should exist at {arena_path}"

    # Verify the arena file size matches expectations
    # Arena should contain: metadata (4KB max) + tensor data
    _METADATA_MAX_SIZE = 4 * 1024  # 4KB
    expected_min_size = tensor_size + _METADATA_MAX_SIZE
    actual_size = os.path.getsize(arena_path)

    # The arena allocates space for metadata + data, so size should be at least that
    assert actual_size >= expected_min_size, (
        f"Arena size should be at least {expected_min_size} bytes "
        f"(tensor: {tensor_size} + metadata: {_METADATA_MAX_SIZE}), "
        f"but got {actual_size} bytes"
    )

    # Verify metadata fields are correct
    assert metadata.size == tensor_size, "Metadata size should match tensor size"
    assert metadata.shape == memory_obj.get_shape(), (
        "Metadata shape should match tensor shape"
    )
    assert metadata.arena_id == arena_id, "Metadata should contain correct arena_id"


def submit_put_task_multiple_keys_test(backend: WekaGdsBackend):
    """
    Test that multiple keys can be stored and all appear in hot_cache.
    """
    keys = [
        create_test_key(chunk_hash=0x1111111111111111),
        create_test_key(chunk_hash=0x2222222222222222),
        create_test_key(chunk_hash=0x3333333333333333),
    ]

    futures = []
    for key in keys:
        memory_obj = create_test_memory_obj(
            backend, shape=(2, 8, 4, 64), dtype=torch.bfloat16
        )
        future = backend.submit_put_task(key, memory_obj)
        futures.append(future)

    # Wait for all tasks to complete
    for future in futures:
        future.result()

    # Verify all keys are in hot_cache
    for key in keys:
        assert backend.contains(key, False), (
            f"Key {key.chunk_hash:x} should be in hot_cache"
        )


def test_weka_backend_submit_put():
    """Test suite for submit_put_task functionality."""
    init_and_teardown(submit_put_task_test)


def test_weka_backend_submit_put_multiple():
    """Test suite for submit_put_task with multiple keys."""
    init_and_teardown(submit_put_task_multiple_keys_test)
