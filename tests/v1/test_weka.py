# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import os
import shutil
import threading
import time
import unittest.mock

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
    assert weka_backend.memory_allocator is not None
    assert isinstance(weka_backend.memory_allocator, CuFileMemoryAllocator)
    return weka_backend


def create_test_memory_obj(
    backend: WekaGdsBackend, shape=(2, 16, 8, 128), dtype=torch.bfloat16
) -> MemoryObj:
    memory_obj = backend.memory_allocator.allocate(
        shape, dtype, fmt=MemoryFormat.KV_T2D
    )
    assert memory_obj is not None, "Failed to allocate memory object"
    return memory_obj


def init_and_teardown(test_func):
    WEKA_DIR = "/mnt/weka/test-cache"
    try:
        os.makedirs(WEKA_DIR, exist_ok=True)
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        weka_backend = create_test_backend(create_test_config(), thread_loop)
        test_func(weka_backend)
    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()
        # We rmtree AFTER we ensure that the thread loop is done.
        # This way we don't hit any race conditions in rmtree()
        # where temp files are renamed while we try to unlink them.
        # We also take care of any other errors with ignore_errors=True
        # so if we want to run tests in parallel in the future they
        # don't make each other fail.
        if os.path.exists(WEKA_DIR):
            shutil.rmtree(WEKA_DIR)


def basic_store_load_test(backend: WekaGdsBackend):
    k = create_test_key()
    assert not backend.contains(k, False)
    assert not backend.exists_in_put_tasks(k)

    memory_obj = create_test_memory_obj(backend)
    future = backend.submit_put_task(k, memory_obj)
    assert future is not None
    assert backend.exists_in_put_tasks(k)
    assert not backend.contains(k, False)
    future.result()
    assert backend.contains(k, False)
    assert not backend.exists_in_put_tasks(k)

    returned_memory_obj = backend.get_blocking(k)
    assert returned_memory_obj is not None
    assert returned_memory_obj.get_size() == memory_obj.get_size()
    assert returned_memory_obj.get_shape() == memory_obj.get_shape()
    assert returned_memory_obj.get_dtype() == memory_obj.get_dtype()

    k_does_not_exist = create_test_key(chunk_hash=0xDEADBEEF)
    assert not backend.contains(k_does_not_exist, False)
    assert not backend.exists_in_put_tasks(k_does_not_exist)
    returned_memory_obj = backend.get_blocking(k_does_not_exist)
    assert returned_memory_obj is None


def test_weka_backend_sanity():
    init_and_teardown(basic_store_load_test)


def basic_batch_store_load_test(backend: WekaGdsBackend):
    keys = []
    for chunk_hash in [0xDEADBEEF, 0xCAFEBABE, 0xBADB0E]:
        keys.append(create_test_key(chunk_hash=chunk_hash))
    memory_objs = [create_test_memory_obj(backend) for _ in range(len(keys))]

    backend.batched_submit_put_task(keys, memory_objs)

    # Wait for all put tasks to complete by monitoring the backend's state
    def wait_for_put_tasks_completion():
        """Wait for all put tasks to complete by checking put_tasks and asyncio loop"""
        # Standard
        import asyncio
        import time

        timeout = 30.0  # 30 second timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if all our specific keys are no longer in put_tasks
            keys_still_pending = [
                key for key in keys if backend.exists_in_put_tasks(key)
            ]

            if not keys_still_pending:
                # Also wait for any remaining asyncio tasks in the backend's loop
                # to complete
                async def wait_for_loop_tasks():
                    current_task = asyncio.current_task(backend.loop)
                    tasks = [
                        task
                        for task in asyncio.all_tasks(backend.loop)
                        if not task.done() and task is not current_task
                    ]
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)

                # Wait for any remaining async tasks to complete
                future = asyncio.run_coroutine_threadsafe(
                    wait_for_loop_tasks(), backend.loop
                )
                try:
                    future.result(timeout=5.0)
                except Exception:
                    pass  # Ignore timeout/errors in cleanup

                break

            time.sleep(0.1)  # Small delay to avoid busy waiting
        else:
            raise TimeoutError(f"Put tasks did not complete within {timeout} seconds")

    wait_for_put_tasks_completion()

    for key in keys:
        assert backend.contains(key)

    returned_memory_objs = backend.batched_get_blocking(keys)
    assert returned_memory_objs is not None
    assert len(returned_memory_objs) == len(keys)
    for returned_memory_obj, memory_obj in zip(
        returned_memory_objs, memory_objs, strict=True
    ):
        assert returned_memory_obj is not None
        assert returned_memory_obj.get_size() == memory_obj.get_size()
        assert returned_memory_obj.get_shape() == memory_obj.get_shape()
        assert returned_memory_obj.get_dtype() == memory_obj.get_dtype()

    k_does_not_exist = create_test_key(chunk_hash=0xDEADB0E)
    assert not backend.contains(k_does_not_exist, False)
    assert not backend.exists_in_put_tasks(k_does_not_exist)
    returned_memory_objs = backend.batched_get_blocking([k_does_not_exist])
    assert returned_memory_objs is not None
    assert len(returned_memory_objs) == 1
    assert returned_memory_objs[0] is None
    assert not backend.exists_in_put_tasks(k_does_not_exist)
    assert not backend.contains(k_does_not_exist, False)


def test_weka_backend_batch_store_load():
    init_and_teardown(basic_batch_store_load_test)


async def basic_batched_get_non_blocking_test(backend: WekaGdsBackend):
    """Test basic functionality of batched_get_non_blocking"""
    # Create test keys and memory objects
    keys = []
    for chunk_hash in [0xDEADBEEF, 0xCAFEBABE, 0xBADB0E]:
        keys.append(create_test_key(chunk_hash=chunk_hash))
    memory_objs = [create_test_memory_obj(backend) for _ in range(len(keys))]

    # Store the objects first
    backend.batched_submit_put_task(keys, memory_objs)

    # Wait for put tasks to complete
    timeout = 30.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        keys_still_pending = [key for key in keys if backend.exists_in_put_tasks(key)]
        if not keys_still_pending:
            break
        await asyncio.sleep(0.1)
    else:
        raise TimeoutError("Put tasks did not complete within timeout")

    # Verify all keys are stored
    for key in keys:
        assert backend.contains(key)

    # Test batched_get_non_blocking
    lookup_id = "test_lookup_001"
    returned_memory_objs = await backend.batched_get_non_blocking(lookup_id, keys)

    # Verify results
    assert returned_memory_objs is not None
    assert len(returned_memory_objs) == len(keys)

    for i, (returned_memory_obj, original_memory_obj) in enumerate(
        zip(returned_memory_objs, memory_objs, strict=True)
    ):
        assert returned_memory_obj is not None, f"Memory object {i} should not be None"
        assert returned_memory_obj.get_size() == original_memory_obj.get_size()
        assert returned_memory_obj.get_shape() == original_memory_obj.get_shape()
        assert returned_memory_obj.get_dtype() == original_memory_obj.get_dtype()
        # Verify reference count was incremented
        assert returned_memory_obj.metadata.ref_count > 0


def test_weka_backend_batched_get_non_blocking_basic():
    """Test basic batched_get_non_blocking functionality"""

    def run_async_test(backend: WekaGdsBackend):
        # Run the async test in the backend's event loop
        future = asyncio.run_coroutine_threadsafe(
            basic_batched_get_non_blocking_test(backend), backend.loop
        )
        future.result(timeout=60.0)

    init_and_teardown(run_async_test)


async def batched_get_non_blocking_missing_keys_test(backend: WekaGdsBackend):
    """Test batched_get_non_blocking with missing keys"""
    # Create some keys that exist and some that don't
    existing_keys = []
    for chunk_hash in [0xDEADBEEF, 0xCAFEBABE]:
        existing_keys.append(create_test_key(chunk_hash=chunk_hash))

    missing_keys = []
    for chunk_hash in [0xBADB0E, 0x12345678]:
        missing_keys.append(create_test_key(chunk_hash=chunk_hash))

    # Store only the existing keys
    memory_objs = [create_test_memory_obj(backend) for _ in range(len(existing_keys))]
    backend.batched_submit_put_task(existing_keys, memory_objs)

    # Wait for put tasks to complete
    timeout = 30.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        keys_still_pending = [
            key for key in existing_keys if backend.exists_in_put_tasks(key)
        ]
        if not keys_still_pending:
            break
        await asyncio.sleep(0.1)
    else:
        raise TimeoutError("Put tasks did not complete within timeout")

    # Verify existing keys are stored and missing keys are not
    for key in existing_keys:
        assert backend.contains(key)
    for key in missing_keys:
        assert not backend.contains(key)

    # Test with missing keys - should return [None] for each missing key
    lookup_id = "test_lookup_missing"
    result = await backend.batched_get_non_blocking(lookup_id, missing_keys)
    assert result is not None
    assert len(result) == len(missing_keys)
    assert all(obj is None for obj in result)


def test_weka_backend_batched_get_non_blocking_missing_keys():
    """Test batched_get_non_blocking with missing keys"""

    def run_async_test(backend: WekaGdsBackend):
        future = asyncio.run_coroutine_threadsafe(
            batched_get_non_blocking_missing_keys_test(backend), backend.loop
        )
        future.result(timeout=60.0)

    init_and_teardown(run_async_test)


async def batched_get_non_blocking_empty_list_test(backend: WekaGdsBackend):
    """Test batched_get_non_blocking with empty key list"""
    lookup_id = "test_lookup_empty"
    returned_memory_objs = await backend.batched_get_non_blocking(lookup_id, [])

    assert returned_memory_objs is not None
    assert len(returned_memory_objs) == 0


def test_weka_backend_batched_get_non_blocking_empty_list():
    """Test batched_get_non_blocking with empty key list"""

    def run_async_test(backend: WekaGdsBackend):
        future = asyncio.run_coroutine_threadsafe(
            batched_get_non_blocking_empty_list_test(backend), backend.loop
        )
        future.result(timeout=60.0)

    init_and_teardown(run_async_test)


def test_weka_backend_batched_get_non_blocking_disk_read_failure():
    """Test batched_get_non_blocking error handling with missing keys"""

    async def error_handling_test(backend: WekaGdsBackend):
        # Test with a key that was never stored - should fail at hot cache lookup
        missing_key = create_test_key(chunk_hash=0x999999)

        lookup_id = "test_error_handling"
        result = await asyncio.wait_for(
            backend.batched_get_non_blocking(lookup_id, [missing_key]), timeout=5.0
        )
        assert result is not None
        assert len(result) == 1
        assert result[0] is None

    def run_async_test(backend: WekaGdsBackend):
        future = asyncio.run_coroutine_threadsafe(
            error_handling_test(backend), backend.loop
        )
        future.result(timeout=10.0)

    init_and_teardown(run_async_test)


async def batched_get_non_blocking_performance_test(backend: WekaGdsBackend):
    """Test batched_get_non_blocking performance with multiple keys"""
    # Create a larger set of test data to verify performance
    num_keys = 5
    keys = []
    memory_objs = []

    for i in range(num_keys):
        key = create_test_key(chunk_hash=0x100000 + i)
        keys.append(key)
        memory_objs.append(create_test_memory_obj(backend))

    # Store all objects
    backend.batched_submit_put_task(keys, memory_objs)

    # Wait for put tasks to complete
    timeout = 30.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        keys_still_pending = [key for key in keys if backend.exists_in_put_tasks(key)]
        if not keys_still_pending:
            break
        await asyncio.sleep(0.1)
    else:
        raise TimeoutError("Put tasks did not complete within timeout")

    # Measure batched_get_non_blocking performance
    perf_start_time = time.perf_counter()

    lookup_id = "performance_test"
    returned_memory_objs = await backend.batched_get_non_blocking(lookup_id, keys)

    end_time = time.perf_counter()
    elapsed_time = end_time - perf_start_time

    # Verify results
    assert returned_memory_objs is not None
    assert len(returned_memory_objs) == num_keys

    for mem_obj in returned_memory_objs:
        assert mem_obj is not None
        assert mem_obj.metadata.ref_count > 0

    # Performance should be reasonable (less than 5 seconds for 5 objects)
    assert elapsed_time < 5.0, (
        f"batched_get_non_blocking took too long: {elapsed_time:.2f}s"
    )

    print(f"batched_get_non_blocking took {elapsed_time:.3f}s for {num_keys} objects")


def test_weka_backend_batched_get_non_blocking_performance():
    """Test batched_get_non_blocking performance"""

    def run_async_test(backend: WekaGdsBackend):
        future = asyncio.run_coroutine_threadsafe(
            batched_get_non_blocking_performance_test(backend), backend.loop
        )
        future.result(timeout=60.0)

    init_and_teardown(run_async_test)


async def batched_get_non_blocking_layerwise_simulation_test(backend: WekaGdsBackend):
    """Test batched_get_non_blocking in a layerwise-like scenario"""
    # Simulate layerwise retrieval pattern where we retrieve chunks layer by layer
    num_layers = 3
    chunks_per_layer = 2

    # Create keys organized by layers (like in layerwise mode)
    all_keys = []
    all_memory_objs = []
    keys_by_layer = []

    for layer_id in range(num_layers):
        layer_keys = []
        for chunk_id in range(chunks_per_layer):
            # Create unique hash for each layer-chunk combination
            chunk_hash = (layer_id << 16) | chunk_id
            key = create_test_key(chunk_hash=chunk_hash)
            layer_keys.append(key)
            all_keys.append(key)
            all_memory_objs.append(create_test_memory_obj(backend))
        keys_by_layer.append(layer_keys)

    # Store all objects
    backend.batched_submit_put_task(all_keys, all_memory_objs)

    # Wait for put tasks to complete
    timeout = 30.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        keys_still_pending = [
            key for key in all_keys if backend.exists_in_put_tasks(key)
        ]
        if not keys_still_pending:
            break
        await asyncio.sleep(0.1)
    else:
        raise TimeoutError("Put tasks did not complete within timeout")

    # Simulate layerwise retrieval - retrieve one layer at a time
    retrieved_by_layer = []
    for layer_id, layer_keys in enumerate(keys_by_layer):
        lookup_id = f"layerwise_lookup_layer_{layer_id}"
        layer_memory_objs = await backend.batched_get_non_blocking(
            lookup_id, layer_keys
        )

        assert layer_memory_objs is not None
        assert len(layer_memory_objs) == len(layer_keys)

        # Verify all objects in this layer are valid
        for mem_obj in layer_memory_objs:
            assert mem_obj is not None
            assert mem_obj.metadata.ref_count > 0

        retrieved_by_layer.append(layer_memory_objs)

    # Verify we retrieved all layers correctly
    assert len(retrieved_by_layer) == num_layers
    total_retrieved = sum(len(layer_objs) for layer_objs in retrieved_by_layer)
    assert total_retrieved == len(all_keys)


def test_weka_backend_batched_get_non_blocking_layerwise_simulation():
    """Test batched_get_non_blocking in a layerwise-like scenario"""

    def run_async_test(backend: WekaGdsBackend):
        future = asyncio.run_coroutine_threadsafe(
            batched_get_non_blocking_layerwise_simulation_test(backend), backend.loop
        )
        future.result(timeout=60.0)

    init_and_teardown(run_async_test)


async def batched_get_non_blocking_concurrent_test(backend: WekaGdsBackend):
    """Test concurrent calls to batched_get_non_blocking"""
    # Create test data
    num_concurrent_calls = 3
    keys_per_call = 2
    all_keys = []
    all_memory_objs = []

    for call_id in range(num_concurrent_calls):
        for key_id in range(keys_per_call):
            chunk_hash = (call_id << 8) | key_id
            key = create_test_key(chunk_hash=chunk_hash)
            all_keys.append(key)
            all_memory_objs.append(create_test_memory_obj(backend))

    # Store all objects
    backend.batched_submit_put_task(all_keys, all_memory_objs)

    # Wait for put tasks to complete
    timeout = 30.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        keys_still_pending = [
            key for key in all_keys if backend.exists_in_put_tasks(key)
        ]
        if not keys_still_pending:
            break
        await asyncio.sleep(0.1)
    else:
        raise TimeoutError("Put tasks did not complete within timeout")

    # Create concurrent batched_get_non_blocking calls
    tasks = []
    for call_id in range(num_concurrent_calls):
        start_idx = call_id * keys_per_call
        end_idx = start_idx + keys_per_call
        call_keys = all_keys[start_idx:end_idx]
        lookup_id = f"concurrent_lookup_{call_id}"

        task = asyncio.create_task(
            backend.batched_get_non_blocking(lookup_id, call_keys)
        )
        tasks.append((task, call_keys))

    # Wait for all tasks to complete
    results = []
    for task, expected_keys in tasks:
        memory_objs = await task
        assert memory_objs is not None
        assert len(memory_objs) == len(expected_keys)
        for mem_obj in memory_objs:
            assert mem_obj is not None
            assert mem_obj.metadata.ref_count > 0
        results.append(memory_objs)

    # Verify all concurrent calls succeeded
    assert len(results) == num_concurrent_calls
    total_retrieved = sum(len(result) for result in results)
    assert total_retrieved == len(all_keys)


def test_weka_backend_batched_get_non_blocking_concurrent():
    """Test concurrent calls to batched_get_non_blocking"""

    def run_async_test(backend: WekaGdsBackend):
        future = asyncio.run_coroutine_threadsafe(
            batched_get_non_blocking_concurrent_test(backend), backend.loop
        )
        future.result(timeout=60.0)

    init_and_teardown(run_async_test)


async def batched_get_non_blocking_reference_count_test(backend: WekaGdsBackend):
    """Test that batched_get_non_blocking properly manages reference counts"""
    # Create test data
    keys = []
    for chunk_hash in [0xEF001, 0xEF002]:
        keys.append(create_test_key(chunk_hash=chunk_hash))
    memory_objs = [create_test_memory_obj(backend) for _ in range(len(keys))]

    # Store the objects
    backend.batched_submit_put_task(keys, memory_objs)

    # Wait for put tasks to complete
    timeout = 30.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        keys_still_pending = [key for key in keys if backend.exists_in_put_tasks(key)]
        if not keys_still_pending:
            break
        await asyncio.sleep(0.1)
    else:
        raise TimeoutError("Put tasks did not complete within timeout")

    # Get the objects via batched_get_non_blocking
    lookup_id = "test_refcount"
    returned_memory_objs = await backend.batched_get_non_blocking(lookup_id, keys)

    # Verify reference counts were incremented
    assert len(returned_memory_objs) == len(keys)
    for mem_obj in returned_memory_objs:
        assert mem_obj is not None
        # Reference count should be > 0 because batched_get_non_blocking calls
        # ref_count_up()
        assert mem_obj.metadata.ref_count > 0

    # Manually decrement reference counts to simulate cleanup
    for mem_obj in returned_memory_objs:
        mem_obj.ref_count_down()


def test_weka_backend_batched_get_non_blocking_reference_count():
    """Test that batched_get_non_blocking properly manages reference counts"""

    def run_async_test(backend: WekaGdsBackend):
        future = asyncio.run_coroutine_threadsafe(
            batched_get_non_blocking_reference_count_test(backend), backend.loop
        )
        future.result(timeout=60.0)

    init_and_teardown(run_async_test)


def submit_put_task_cufile_write_failure_test(backend: WekaGdsBackend):
    """Test that submit_put_task handles cuFile write failures gracefully"""
    # Standard
    import unittest.mock

    k = create_test_key()
    memory_obj = create_test_memory_obj(backend)

    # Mock cuFile.CuFile to raise an exception during write
    with unittest.mock.patch.object(backend.cufile, "CuFile") as mock_cufile:
        mock_file = unittest.mock.MagicMock()
        mock_file.write.side_effect = RuntimeError("CuFile write failed")
        mock_cufile.return_value.__enter__ = unittest.mock.MagicMock(
            return_value=mock_file
        )
        mock_cufile.return_value.__exit__ = unittest.mock.MagicMock(return_value=None)

        # Submit the put task
        future = backend.submit_put_task(k, memory_obj)
        assert future is not None
        assert backend.exists_in_put_tasks(k)

        # The future should complete without exception (silent failure)
        future.result()  # Should not raise an exception

        # Verify the key is removed from put_tasks even after failure
        assert not backend.exists_in_put_tasks(k)
        # Key should not be in cache since the operation failed
        assert not backend.contains(k, False)


def test_weka_backend_submit_put_task_cufile_write_failure():
    init_and_teardown(submit_put_task_cufile_write_failure_test)


def submit_put_task_posix_metadata_write_failure_test(backend: WekaGdsBackend):
    """Test that submit_put_task handles POSIX metadata write failures gracefully"""
    # Standard
    import unittest.mock

    k = create_test_key()
    memory_obj = create_test_memory_obj(backend)

    with unittest.mock.patch("builtins.open") as mock_open:
        mock_open.side_effect = IOError("POSIX metadata write failed")

        future = backend.submit_put_task(k, memory_obj)
        assert future is not None
        assert backend.exists_in_put_tasks(k)

        future.result()  # Should not raise an exception

        # Verify the key is removed from put_tasks even after failure
        assert not backend.exists_in_put_tasks(k)
        # Key should not be in cache since the operation failed
        assert not backend.contains(k, False)


def test_weka_backend_submit_put_task_posix_metadata_write_failure():
    init_and_teardown(submit_put_task_posix_metadata_write_failure_test)


def get_blocking_cufile_read_failure_test(backend: WekaGdsBackend):
    """Test that get_blocking handles cuFile read failures gracefully"""
    # Standard
    import unittest.mock

    k = create_test_key()
    memory_obj = create_test_memory_obj(backend)
    future = backend.submit_put_task(k, memory_obj)
    future.result()
    assert backend.contains(k, False)

    with unittest.mock.patch.object(backend.cufile, "CuFile") as mock_cufile:
        mock_file = unittest.mock.MagicMock()
        mock_file.read.side_effect = RuntimeError("CuFile read failed")
        mock_cufile.return_value.__enter__ = unittest.mock.MagicMock(
            return_value=mock_file
        )
        mock_cufile.return_value.__exit__ = unittest.mock.MagicMock(return_value=None)

        # Try to get the object - should handle the error gracefully
        returned_memory_obj = backend.get_blocking(k)
        # Should return None on read failure
        assert returned_memory_obj is None


def test_weka_backend_get_blocking_cufile_read_failure():
    init_and_teardown(get_blocking_cufile_read_failure_test)


def get_blocking_cufile_partial_read_test(backend: WekaGdsBackend):
    """Test that get_blocking handles partial reads gracefully"""
    # Standard
    import unittest.mock

    k = create_test_key()
    memory_obj = create_test_memory_obj(backend)
    future = backend.submit_put_task(k, memory_obj)
    future.result()
    assert backend.contains(k, False)

    expected_size = memory_obj.get_physical_size()

    with unittest.mock.patch.object(backend.cufile, "CuFile") as mock_cufile:
        mock_file = unittest.mock.MagicMock()
        # Return less bytes than expected
        mock_file.read.return_value = expected_size // 2
        mock_cufile.return_value.__enter__ = unittest.mock.MagicMock(
            return_value=mock_file
        )
        mock_cufile.return_value.__exit__ = unittest.mock.MagicMock(return_value=None)

        # Try to get the object - should handle the partial read gracefully
        returned_memory_obj = backend.get_blocking(k)
        # Should return None on partial read
        assert returned_memory_obj is None


def test_weka_backend_get_blocking_cufile_partial_read():
    init_and_teardown(get_blocking_cufile_partial_read_test)


def get_blocking_cufile_negative_return_test(backend: WekaGdsBackend):
    """Test that get_blocking handles negative cuFile read returns gracefully"""
    # Standard
    import unittest.mock

    k = create_test_key()
    memory_obj = create_test_memory_obj(backend)
    future = backend.submit_put_task(k, memory_obj)
    future.result()
    assert backend.contains(k, False)

    with unittest.mock.patch.object(backend.cufile, "CuFile") as mock_cufile:
        mock_file = unittest.mock.MagicMock()
        # Return negative value indicating error
        mock_file.read.return_value = -1
        mock_cufile.return_value.__enter__ = unittest.mock.MagicMock(
            return_value=mock_file
        )
        mock_cufile.return_value.__exit__ = unittest.mock.MagicMock(return_value=None)

        # Try to get the object - should handle the error gracefully
        returned_memory_obj = backend.get_blocking(k)
        # Should return None on error
        assert returned_memory_obj is None
        # Note: Key can still be found via contains() because metadata exists on disk
        # This allows the system to recover from temporary read errors


def test_weka_backend_get_blocking_cufile_negative_return():
    init_and_teardown(get_blocking_cufile_negative_return_test)


def contains_corrupted_metadata_test(backend: WekaGdsBackend):
    """Test that contains() handles corrupted metadata files gracefully"""
    k = create_test_key()
    memory_obj = create_test_memory_obj(backend)
    future = backend.submit_put_task(k, memory_obj)
    future.result()
    assert backend.contains(k, False)

    # Clear hot cache to force disk read
    with backend.hot_lock:
        backend.hot_cache.clear()

    # Get the metadata file path
    path, subdir_key, _, _ = backend._key_to_path(k)
    metadata_path = path + ".metadata"

    # Verify the metadata file exists (skip if test environment doesn't persist files)
    if not os.path.exists(metadata_path):
        # Create a dummy file for testing purposes
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w") as f:
            f.write("dummy metadata")

    # Mock the file reading to simulate a corrupted metadata file that exists but
    # can't be read
    def mock_open_side_effect(*args, **kwargs):
        if args[0] == metadata_path:
            raise OSError("Simulated file read error")
        # For other files, use the real open
        return unittest.mock.DEFAULT

    with unittest.mock.patch(
        "builtins.open", side_effect=mock_open_side_effect
    ) as mock_open:
        # This should not crash, but should return False and log an error
        result = backend.contains(k, False)
        # Should return False because metadata read failed
        assert result is False
        # Verify that file was attempted to be opened
        mock_open.assert_called()

        # Verify that the open call included our metadata file
        calls = mock_open.call_args_list
        metadata_calls = [call for call in calls if call[0][0] == metadata_path]
        assert len(metadata_calls) > 0, f"Expected call to open {metadata_path}"


def test_weka_backend_contains_corrupted_metadata():
    init_and_teardown(contains_corrupted_metadata_test)


# ============================================================================
# Timeout and Hang Threshold Tests
# ============================================================================


def contains_timeout_test(backend: WekaGdsBackend):
    """Test that contains() handles operation timeout gracefully"""
    k = create_test_key()

    # Clear hot cache to force disk read
    with backend.hot_lock:
        backend.hot_cache.clear()

    def slow_try_to_read_metadata(key):
        """Mock function that takes longer than timeout"""
        time.sleep(
            backend.timeout_contains + 1.0
        )  # Sleep longer than configured timeout
        return None

    with unittest.mock.patch.object(
        backend, "_try_to_read_metadata", side_effect=slow_try_to_read_metadata
    ):
        # This should not crash, but should return False and log timeout error
        result = backend.contains(k, False)
        assert result is False


def test_contains_timeout():
    init_and_teardown(contains_timeout_test)


def get_blocking_timeout_test(backend: WekaGdsBackend):
    """Test that get_blocking() handles operation timeout gracefully"""
    k = create_test_key()
    memory_obj = create_test_memory_obj(backend)

    # First put an item in the cache
    future = backend.submit_put_task(k, memory_obj)
    future.result()
    assert backend.contains(k, False)

    def slow_load_bytes_from_disk(key, path, dtype, shape):
        """Mock function that takes longer than timeout"""
        time.sleep(
            backend.timeout_get_blocking + 1.0
        )  # Sleep longer than configured timeout
        return None

    with unittest.mock.patch.object(
        backend,
        "_load_bytes_from_disk_with_allocation",
        side_effect=slow_load_bytes_from_disk,
    ):
        # This should not crash, but should return None and log timeout error
        result = backend.get_blocking(k)
        assert result is None


def test_get_blocking_timeout():
    init_and_teardown(get_blocking_timeout_test)


def batched_get_blocking_timeout_test(backend: WekaGdsBackend):
    """Test that batched_get_blocking() handles operation timeout gracefully"""
    k1 = create_test_key(chunk_hash=123)
    k2 = create_test_key(chunk_hash=456)
    memory_obj1 = create_test_memory_obj(backend)
    memory_obj2 = create_test_memory_obj(backend)

    # First put items in the cache
    future1 = backend.submit_put_task(k1, memory_obj1)
    future2 = backend.submit_put_task(k2, memory_obj2)
    future1.result()
    future2.result()

    def slow_batched_get_blocking(keys):
        """Mock function that takes longer than timeout"""
        time.sleep(
            backend.timeout_batched_get_blocking + 1.0
        )  # Sleep longer than configured timeout
        return [None] * len(keys)

    with unittest.mock.patch.object(
        backend, "_batched_get_blocking", side_effect=slow_batched_get_blocking
    ):
        # This should not crash, but should return [None, None] and log timeout error
        result = backend.batched_get_blocking([k1, k2])
        assert result == [None, None]


def test_batched_get_blocking_timeout():
    init_and_teardown(batched_get_blocking_timeout_test)


def contains_hang_threshold_test(backend: WekaGdsBackend):
    """Test that contains() handles hang threshold gracefully"""
    k = create_test_key()

    # Clear hot cache to force disk reads
    with backend.hot_lock:
        backend.hot_cache.clear()

    def slow_try_to_read_metadata(key):
        """Mock function that always times out"""
        time.sleep(
            backend.timeout_contains + 1.0
        )  # Sleep longer than configured timeout
        return None

    # Reset failure counter first
    backend.op_manager.reset_failure_count()

    with unittest.mock.patch.object(
        backend, "_try_to_read_metadata", side_effect=slow_try_to_read_metadata
    ):
        # First, trigger timeouts to reach the hang threshold
        for i in range(backend.op_manager._hang_threshold):
            result = backend.contains(create_test_key(chunk_hash=i), False)
            assert result is False

        # Verify we've reached the failure count
        assert (
            backend.op_manager.get_failure_count() >= backend.op_manager._hang_threshold
        )

        # Now the next call should trigger hang threshold
        result = backend.contains(k, False)
        # Should still return False, but due to hang threshold
        assert result is False


def test_contains_hang_threshold():
    init_and_teardown(contains_hang_threshold_test)


def get_blocking_hang_threshold_test(backend: WekaGdsBackend):
    """Test that get_blocking() handles hang threshold gracefully"""
    k = create_test_key()
    memory_obj = create_test_memory_obj(backend)

    # First put an item in the cache
    future = backend.submit_put_task(k, memory_obj)
    future.result()

    def slow_load_bytes_from_disk(key, path, dtype, shape):
        """Mock function that always times out"""
        time.sleep(
            backend.timeout_get_blocking + 1.0
        )  # Sleep longer than configured timeout
        return None

    # Reset failure counter first
    backend.op_manager.reset_failure_count()

    with unittest.mock.patch.object(
        backend,
        "_load_bytes_from_disk_with_allocation",
        side_effect=slow_load_bytes_from_disk,
    ):
        # First, trigger timeouts to reach the hang threshold
        for i in range(backend.op_manager._hang_threshold):
            test_key = create_test_key(chunk_hash=i)
            # Put each test key in cache first
            test_memory_obj = create_test_memory_obj(backend)
            test_future = backend.submit_put_task(test_key, test_memory_obj)
            test_future.result()

            result = backend.get_blocking(test_key)
            assert result is None

        # Verify we've reached the failure count
        assert (
            backend.op_manager.get_failure_count() >= backend.op_manager._hang_threshold
        )

        # Now the next call should trigger hang threshold
        result = backend.get_blocking(k)
        # Should still return None, but due to hang threshold
        assert result is None


def test_get_blocking_hang_threshold():
    init_and_teardown(get_blocking_hang_threshold_test)


def batched_get_blocking_hang_threshold_test(backend: WekaGdsBackend):
    """Test that batched_get_blocking() handles hang threshold gracefully"""
    k1 = create_test_key(chunk_hash=123)
    k2 = create_test_key(chunk_hash=456)
    memory_obj1 = create_test_memory_obj(backend)
    memory_obj2 = create_test_memory_obj(backend)

    # First put items in the cache
    future1 = backend.submit_put_task(k1, memory_obj1)
    future2 = backend.submit_put_task(k2, memory_obj2)
    future1.result()
    future2.result()

    def slow_batched_get_blocking(keys):
        """Mock function that always times out"""
        time.sleep(
            backend.timeout_batched_get_blocking + 1.0
        )  # Sleep longer than configured timeout
        return [None] * len(keys)

    # Reset failure counter first
    backend.op_manager.reset_failure_count()

    with unittest.mock.patch.object(
        backend, "_batched_get_blocking", side_effect=slow_batched_get_blocking
    ):
        # First, trigger timeouts to reach the hang threshold
        # Each batched call counts as one timeout, so we need hang_threshold
        # iterations
        for i in range(backend.op_manager._hang_threshold):
            test_keys = [
                create_test_key(chunk_hash=i * 2),
                create_test_key(chunk_hash=i * 2 + 1),
            ]

            # Put each test key in cache first
            for test_key in test_keys:
                test_memory_obj = create_test_memory_obj(backend)
                test_future = backend.submit_put_task(test_key, test_memory_obj)
                test_future.result()

            result = backend.batched_get_blocking(test_keys)
            assert result == [None, None]

        # Verify we've reached the failure count
        assert (
            backend.op_manager.get_failure_count() >= backend.op_manager._hang_threshold
        )

        # Now the next call should trigger hang threshold
        result = backend.batched_get_blocking([k1, k2])
        # Should still return [None, None], but due to hang threshold
        assert result == [
            None,
            None,
        ]


def test_batched_get_blocking_hang_threshold():
    init_and_teardown(batched_get_blocking_hang_threshold_test)


def cross_operation_hang_threshold_test(backend: WekaGdsBackend):
    """Test that hitting hang threshold for one operation affects all operations"""
    # Set up test keys and data
    k1 = create_test_key(chunk_hash=11111111)
    k2 = create_test_key(chunk_hash=22222222)
    k3 = create_test_key(chunk_hash=33333333)

    memory_obj1 = create_test_memory_obj(backend)
    memory_obj2 = create_test_memory_obj(backend)

    # Put items in cache for get operations
    future1 = backend.submit_put_task(k1, memory_obj1)
    future2 = backend.submit_put_task(k2, memory_obj2)
    future1.result()
    future2.result()

    def slow_load_bytes_from_disk(key, path, dtype, shape):
        """Mock function that always times out"""
        time.sleep(backend.timeout_get_blocking + 1.0)
        return None

    def slow_try_to_read_metadata(key):
        """Mock function that always times out"""
        time.sleep(backend.timeout_contains + 1.0)
        return None

    def slow_batched_get_blocking(keys):
        """Mock function that always times out"""
        time.sleep(backend.timeout_batched_get_blocking + 1.0)
        return [None] * len(keys)

    # Reset failure counter first
    backend.op_manager.reset_failure_count()

    # First, use get_blocking to hit the hang threshold
    with unittest.mock.patch.object(
        backend,
        "_load_bytes_from_disk_with_allocation",
        side_effect=slow_load_bytes_from_disk,
    ):
        # Trigger timeouts with get_blocking to reach the hang threshold
        for i in range(backend.op_manager._hang_threshold):
            test_key = create_test_key(chunk_hash=1000 + i)
            # Put each test key in cache first
            test_memory_obj = create_test_memory_obj(backend)
            test_future = backend.submit_put_task(test_key, test_memory_obj)
            test_future.result()

            result = backend.get_blocking(test_key)
            assert result is None

        # Verify we've reached the failure count
        assert (
            backend.op_manager.get_failure_count() >= backend.op_manager._hang_threshold
        )

    # Now test that ALL operations fail due to shared hang threshold
    # Clear hot cache for contains test
    with backend.hot_lock:
        backend.hot_cache.clear()

    # Test contains() - should fail due to hang threshold reached by get_blocking
    with unittest.mock.patch.object(
        backend, "_try_to_read_metadata", side_effect=slow_try_to_read_metadata
    ):
        contains_result = backend.contains(k3, False)
        # Should return False due to hang threshold, not because of timeout
        assert contains_result is False

    # Test batched_get_blocking() - should fail due to hang threshold
    with unittest.mock.patch.object(
        backend, "_batched_get_blocking", side_effect=slow_batched_get_blocking
    ):
        batched_get_result = backend.batched_get_blocking([k1, k2])
        # Should return [None, None] due to hang threshold
        assert batched_get_result == [None, None]

    # Test get_blocking() again - should still fail due to hang threshold
    with unittest.mock.patch.object(
        backend,
        "_load_bytes_from_disk_with_allocation",
        side_effect=slow_load_bytes_from_disk,
    ):
        get_result = backend.get_blocking(k1)
        # Should return None due to hang threshold
        assert get_result is None


def test_cross_operation_hang_threshold():
    init_and_teardown(cross_operation_hang_threshold_test)


def reset_file_threshold_recovery_test(backend: WekaGdsBackend):
    """Test that the reset file functionality allows recovery from hang threshold"""
    k1 = create_test_key(chunk_hash=111111)
    k2 = create_test_key(chunk_hash=222222)
    k3 = create_test_key(chunk_hash=333333)

    memory_obj1 = create_test_memory_obj(backend)
    memory_obj2 = create_test_memory_obj(backend)
    memory_obj3 = create_test_memory_obj(backend)

    # Put items in cache for get operations
    future1 = backend.submit_put_task(k1, memory_obj1)
    future2 = backend.submit_put_task(k2, memory_obj2)
    future3 = backend.submit_put_task(k3, memory_obj3)
    future1.result()
    future2.result()
    future3.result()

    def slow_load_bytes_from_disk(key, path, dtype, shape):
        """Mock function that always times out"""
        time.sleep(backend.timeout_get_blocking + 1.0)
        return None

    def normal_load_bytes_from_disk(key, path, dtype, shape):
        """Mock function that works normally (doesn't timeout)"""
        return None  # Simulate successful load by returning None quickly

    # Clean up any existing reset file and reset failure counter to ensure clean
    # test state
    reset_file_path = backend.op_manager._reset_file
    if os.path.exists(reset_file_path):
        os.remove(reset_file_path)
        print(f"Cleaned up existing reset file at {reset_file_path}")

    backend.op_manager.reset_failure_count()
    print(
        f"Initial state - Failure count: {backend.op_manager.get_failure_count()}, "
        f"Reset file exists: {os.path.exists(reset_file_path)}"
    )

    # Step 1: Trigger enough failures to reach hang threshold
    with unittest.mock.patch.object(
        backend,
        "_load_bytes_from_disk_with_allocation",
        side_effect=slow_load_bytes_from_disk,
    ):
        # Trigger timeouts with get_blocking to reach the hang threshold
        for i in range(backend.op_manager._hang_threshold):
            test_key = create_test_key(chunk_hash=2000 + i)
            # Put each test key in cache first
            test_memory_obj = create_test_memory_obj(backend)
            test_future = backend.submit_put_task(test_key, test_memory_obj)
            test_future.result()

            result = backend.get_blocking(test_key)
            assert result is None

        # Verify we've reached the failure count threshold
        assert (
            backend.op_manager.get_failure_count() >= backend.op_manager._hang_threshold
        )
        print(
            f"After triggering failures - Failure count: "
            f"{backend.op_manager.get_failure_count()}, "
            f"Reset file exists: {os.path.exists(reset_file_path)}"
        )

    # Step 2: Verify operations fail due to hang threshold (not timeout)
    # We'll use a non-timing-out mock to prove the failure is due to threshold
    with unittest.mock.patch.object(
        backend,
        "_load_bytes_from_disk_with_allocation",
        side_effect=normal_load_bytes_from_disk,
    ):
        # This should fail immediately due to hang threshold, not due to timeout
        result = backend.get_blocking(k1)
        assert result is None  # Should fail due to hang threshold

        # Test contains() - should also fail due to hang threshold
        with backend.hot_lock:
            backend.hot_cache.clear()

        def normal_try_to_read_metadata(key):
            """Mock function that works normally (doesn't timeout)"""
            return None

        with unittest.mock.patch.object(
            backend, "_try_to_read_metadata", side_effect=normal_try_to_read_metadata
        ):
            contains_result = backend.contains(k2, False)
            assert contains_result is False  # Should fail due to hang threshold

    print(
        f"After step 2 - Failure count: {backend.op_manager.get_failure_count()}, "
        f"Reset file exists: {os.path.exists(reset_file_path)}"
    )

    # Step 3: Create reset file and verify operations succeed after reset

    # Ensure reset file doesn't exist initially
    if os.path.exists(reset_file_path):
        os.remove(reset_file_path)

    # Create the reset file
    with open(reset_file_path, "w") as f:
        f.write("reset")

    # Verify the reset file exists
    assert os.path.exists(reset_file_path)

    # Step 4: Now operations should succeed because reset file will be detected
    # IMPORTANT: Don't reset failure count here - let the reset file logic
    # handle it. The failure count should still be >= threshold to trigger the
    # reset file logic
    print(f"Reset file path: {reset_file_path}")
    print(f"Reset file exists before operation: {os.path.exists(reset_file_path)}")
    print(f"Failure count before operation: {backend.op_manager.get_failure_count()}")

    # Check hot_cache state
    with backend.hot_lock:
        hot_cache_keys = list(backend.hot_cache.keys())
        print(f"Keys in hot_cache: {[key.chunk_hash for key in hot_cache_keys]}")
        print(f"k3 in hot_cache: {k3 in backend.hot_cache}")
        if k3 in backend.hot_cache:
            print(f"k3 entry: {backend.hot_cache[k3]}")

    # Add a mock to intercept the OperationManager.run_with_timeout call
    original_run_with_timeout = backend.op_manager.run_with_timeout

    def debug_run_with_timeout(
        func, timeout_seconds, label="default_label", metadata=None
    ):
        print(
            f"DEBUG: run_with_timeout called - "
            f"failure_count={backend.op_manager.get_failure_count()}, "
            f"reset_file_exists={os.path.exists(reset_file_path)}"
        )
        return original_run_with_timeout(func, timeout_seconds, label, metadata)

    with unittest.mock.patch.object(
        backend.op_manager, "run_with_timeout", side_effect=debug_run_with_timeout
    ):

        def normal_try_to_read_metadata_for_reset(key):
            """Mock function that works normally for reset test"""
            return None

        with unittest.mock.patch.object(
            backend,
            "_try_to_read_metadata",
            side_effect=normal_try_to_read_metadata_for_reset,
        ):
            # This should now succeed because the reset file will be detected and
            # failure count reset. Use contains() which always calls
            # run_with_timeout regardless of hot_cache state
            # Use any key
            test_key_for_reset = create_test_key(chunk_hash=9999)
            try:
                contains_result = backend.contains(test_key_for_reset, False)
                print(f"Contains operation succeeded, result: {contains_result}")
                operation_succeeded = True
            except Exception as e:
                print(
                    f"Contains operation failed with exception: {type(e).__name__}: {e}"
                )
                contains_result = False
                operation_succeeded = False

        # Check status after the operation
        print(f"Reset file exists after operation: {os.path.exists(reset_file_path)}")
        print(
            f"Failure count after operation: {backend.op_manager.get_failure_count()}"
        )

        # Verify that the operation succeeded (no hang threshold exception)
        assert operation_succeeded, (
            "Operation should have succeeded after reset file was processed"
        )

        # Verify that the reset file has been removed
        assert not os.path.exists(reset_file_path), (
            f"Reset file should have been removed but still exists at {reset_file_path}"
        )

        # Verify that the failure count has been reset
        assert backend.op_manager.get_failure_count() == 0, (
            f"Failure count should be 0 but is {backend.op_manager.get_failure_count()}"
        )

    # Step 5: Verify continued operations work normally after reset
    with unittest.mock.patch.object(
        backend,
        "_load_bytes_from_disk_with_allocation",
        side_effect=normal_load_bytes_from_disk,
    ):
        # These should all work fine now
        result1 = backend.get_blocking(k1)
        result2 = backend.get_blocking(k2)
        assert result1 is None  # Normal mock response
        assert result2 is None  # Normal mock response

        # Contains should also work
        with backend.hot_lock:
            backend.hot_cache.clear()

        with unittest.mock.patch.object(
            backend, "_try_to_read_metadata", side_effect=normal_try_to_read_metadata
        ):
            contains_result = backend.contains(k3, False)
            assert (
                contains_result is False
            )  # Normal mock response, but no threshold exception


def test_reset_file_threshold_recovery():
    init_and_teardown(reset_file_threshold_recovery_test)


def batched_async_contains_basic_test(backend: WekaGdsBackend):
    """Test batched_async_contains with 5 keys stored, all should be found."""
    # Create 5 test keys
    keys = []
    for i, chunk_hash in enumerate(
        [0xDEADBEEF, 0xCAFEBABE, 0xBADB0E, 0xFEEDFACE, 0xDEADC0DE]
    ):
        keys.append(create_test_key(chunk_hash=chunk_hash))

    # Store all 5 keys
    memory_objs = [create_test_memory_obj(backend) for _ in range(len(keys))]
    backend.batched_submit_put_task(keys, memory_objs)

    # Wait for all put tasks to complete
    def wait_for_put_tasks_completion():
        # Standard
        import time

        timeout = 30.0
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_completed = True
            for key in keys:
                if backend.exists_in_put_tasks(key):
                    all_completed = False
                    break
            if all_completed:
                return
            time.sleep(0.1)
        raise TimeoutError("Put tasks did not complete within timeout")

    wait_for_put_tasks_completion()

    # Verify all keys are stored using regular contains()
    for key in keys:
        assert backend.contains(key, False), f"Key {key} should be in backend"

    # Test batched_async_contains - should find all 5 keys
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            backend.batched_async_contains("test_lookup", keys, pin=False)
        )
        assert result == 5, f"Expected 5 hits, got {result}"
    finally:
        loop.close()


def batched_async_contains_partial_test(backend: WekaGdsBackend):
    """Test batched_async_contains where the 3rd key doesn't exist, should return 2."""
    # Create 5 test keys
    keys = []
    for i, chunk_hash in enumerate(
        [0xDEADBEEF, 0xCAFEBABE, 0xBADB0E, 0xFEEDFACE, 0xDEADC0DE]
    ):
        keys.append(create_test_key(chunk_hash=chunk_hash))

    # Store only keys[0], keys[1], keys[3], keys[4] (skip keys[2])
    keys_to_store = [keys[0], keys[1], keys[3], keys[4]]
    memory_objs = [create_test_memory_obj(backend) for _ in range(len(keys_to_store))]
    backend.batched_submit_put_task(keys_to_store, memory_objs)

    # Wait for put tasks to complete
    def wait_for_put_tasks_completion():
        # Standard
        import time

        timeout = 30.0
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_completed = True
            for key in keys_to_store:
                if backend.exists_in_put_tasks(key):
                    all_completed = False
                    break
            if all_completed:
                return
            time.sleep(0.1)
        raise TimeoutError("Put tasks did not complete within timeout")

    wait_for_put_tasks_completion()

    # Verify stored keys exist and missing key doesn't
    assert backend.contains(keys[0], False), "Key 0 should exist"
    assert backend.contains(keys[1], False), "Key 1 should exist"
    assert not backend.contains(keys[2], False), "Key 2 should not exist"
    assert backend.contains(keys[3], False), "Key 3 should exist"
    assert backend.contains(keys[4], False), "Key 4 should exist"

    # Test batched_async_contains - should return 2 (stops at missing keys[2])
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            backend.batched_async_contains("test_lookup", keys, pin=False)
        )
        assert result == 2, f"Expected 2 hits (stops at missing key[2]), got {result}"
    finally:
        loop.close()


def batched_async_contains_mixed_cache_test(backend: WekaGdsBackend):
    """Test batched_async_contains with mixed hot_cache and disk lookup scenarios."""
    # Create 5 test keys: A, B, C, D, E
    key_a = create_test_key(chunk_hash=0xAAAAAAAA)
    key_b = create_test_key(chunk_hash=0xBBBBBBBB)
    key_c = create_test_key(chunk_hash=0xCCCCCCCC)
    key_d = create_test_key(chunk_hash=0xDDDDDDDD)
    key_e = create_test_key(chunk_hash=0xEEEEEEEE)

    # Step 1: Store keys A and B
    keys_ab = [key_a, key_b]
    memory_objs_ab = [create_test_memory_obj(backend) for _ in range(2)]
    backend.batched_submit_put_task(keys_ab, memory_objs_ab)

    def wait_for_put_tasks(keys_to_wait):
        # Standard
        import time

        timeout = 30.0
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_completed = True
            for key in keys_to_wait:
                if backend.exists_in_put_tasks(key):
                    all_completed = False
                    break
            if all_completed:
                return
            time.sleep(0.1)
        raise TimeoutError("Put tasks did not complete within timeout")

    wait_for_put_tasks(keys_ab)

    # Step 2: Clear hot_cache to force disk lookup for A and B
    with backend.hot_lock:
        backend.hot_cache.clear()
        # Also clear metadata dirs to simulate clean state
        backend.metadata_dirs.clear()

    # Step 3: Store keys C, D, E (these will be in hot_cache)
    keys_cde = [key_c, key_d, key_e]
    memory_objs_cde = [create_test_memory_obj(backend) for _ in range(3)]
    backend.batched_submit_put_task(keys_cde, memory_objs_cde)

    wait_for_put_tasks(keys_cde)

    # Verify state: C, D, E should be in hot_cache; A, B should require disk lookup
    with backend.hot_lock:
        assert key_c in backend.hot_cache, "Key C should be in hot_cache"
        assert key_d in backend.hot_cache, "Key D should be in hot_cache"
        assert key_e in backend.hot_cache, "Key E should be in hot_cache"
        assert key_a not in backend.hot_cache, "Key A should not be in hot_cache"
        assert key_b not in backend.hot_cache, "Key B should not be in hot_cache"

    # Verify all keys exist via regular contains() (this will populate hot_cache)
    assert backend.contains(key_a, False), "Key A should exist on disk"
    assert backend.contains(key_b, False), "Key B should exist on disk"
    assert backend.contains(key_c, False), "Key C should exist"
    assert backend.contains(key_d, False), "Key D should exist"
    assert backend.contains(key_e, False), "Key E should exist"

    # Clear hot_cache again to ensure we test the mixed scenario
    with backend.hot_lock:
        backend.hot_cache.clear()
        backend.metadata_dirs.clear()

    # Re-add C, D, E to hot_cache by storing them again
    backend.batched_submit_put_task(
        keys_cde, [create_test_memory_obj(backend) for _ in range(3)]
    )
    wait_for_put_tasks(keys_cde)

    # Step 4: Test batched_async_contains with [C, A, D, B, E]
    # Expected behavior:
    # - C: found in hot_cache (hit 1)
    # - A: not in hot_cache, found on disk (hit 2)
    # - D: found in hot_cache (hit 3)
    # - B: not in hot_cache, found on disk (hit 4)
    # - E: found in hot_cache (hit 5)
    test_keys = [key_c, key_a, key_d, key_b, key_e]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            backend.batched_async_contains("test_lookup", test_keys, pin=False)
        )
        assert result == 5, (
            f"Expected 5 hits with mixed cache/disk lookup, got {result}"
        )
    finally:
        loop.close()


def test_batched_async_contains_basic():
    """Test basic batched_async_contains functionality with all keys present."""
    init_and_teardown(batched_async_contains_basic_test)


def test_batched_async_contains_partial():
    """Test batched_async_contains stops at first missing key."""
    init_and_teardown(batched_async_contains_partial_test)


def test_batched_async_contains_mixed_cache():
    """Test batched_async_contains with mixed hot_cache and disk scenarios."""
    init_and_teardown(batched_async_contains_mixed_cache_test)


def test_cufile_allocator_with_local_cpu_backend_eviction():
    """
    Test that reproduces the AssertionError when LocalCPUBackend
    tries to evict with a CuFileMemoryAllocator.

    This test simulates the scenario where:
    1. WekaGdsBackend is configured with CuFileMemoryAllocator
    2. LocalCPUBackend is created as always but not used.
    3. StorageManager routes allocations to LocalCPUBackend
       (default when enable_nixl=False)
    4. Memory is full, triggering eviction in LocalCPUBackend
    5. The assertion fails because CuFileMemoryAllocator is not
       MixedMemoryAllocator or NixlCPUMemoryAllocator

    Expected: AssertionError with message about allocator type mismatch
    """
    # First Party
    from lmcache.config import LMCacheEngineMetadata
    from lmcache.v1.event_manager import EventManager
    from lmcache.v1.storage_backend.storage_manager import StorageManager

    WEKA_DIR = "/mnt/weka/test-cache-eviction"
    thread_loop = None
    thread = None
    storage_manager = None

    try:
        os.makedirs(WEKA_DIR, exist_ok=True)

        # Create a small buffer (2 MB) to easily trigger memory pressure
        small_buffer_size = 2  # MB
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            weka_path=WEKA_DIR,
            lmcache_instance_id="test_eviction",
            cufile_buffer_size=small_buffer_size,
            local_cpu=False,  # LocalCPUBackend will be created anyways
            extra_config={"gds_io_threads": 4},
        )

        # Create metadata
        metadata = LMCacheEngineMetadata(
            model_name="test-model",
            world_size=1,
            worker_id=0,
            fmt="vllm",
            kv_dtype=torch.bfloat16,
            kv_shape=(32, 2, 256, 8, 128),  # Large enough to fill memory
        )

        # Create CuFileMemoryAllocator (GPU allocator)
        allocator = CuFileMemoryAllocator(small_buffer_size * 1024**2)

        # Create event loop
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        # Create event manager
        event_manager = EventManager()

        # Create StorageManager - this will create both WekaGdsBackend and
        # LocalCPUBackend
        # Both will receive the same CuFileMemoryAllocator
        # StorageManager will use LocalCPUBackend as allocator_backend (default)
        storage_manager = StorageManager(
            config=config,
            metadata=metadata,
            allocator=allocator,
            event_manager=event_manager,
        )

        # Verify that we have both backends
        assert "LocalCPUBackend" in storage_manager.storage_backends, (
            "LocalCPUBackend should be created"
        )
        assert "WekaGdsBackend" in storage_manager.storage_backends, (
            "WekaGdsBackend should be created"
        )

        # Verify LocalCPUBackend is using the CuFileMemoryAllocator
        local_cpu_backend = storage_manager.storage_backends["LocalCPUBackend"]
        assert isinstance(local_cpu_backend.memory_allocator, CuFileMemoryAllocator), (
            "LocalCPUBackend should have CuFileMemoryAllocator"
        )

        # Fill up memory by allocating until we can't allocate anymore
        # Shape that will take up significant space: (2, 16, 8, 128) bfloat16
        # Size = 2 * 16 * 8 * 128 * 2 bytes = 65,536 bytes = 64 KB per allocation
        shape = (2, 16, 8, 128)
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_T2D

        allocated_objs = []
        # Try to allocate many objects to fill the 2MB buffer
        # 2MB / 64KB = 32 allocations theoretically, but fragmentation may reduce this
        for i in range(40):  # Try more than theoretical max
            memory_obj = storage_manager.allocate(
                shape, dtype, fmt, eviction=False, busy_loop=False
            )
            if memory_obj is None:
                # Memory is full
                break
            allocated_objs.append(memory_obj)

        print(f"Allocated {len(allocated_objs)} objects before running out of memory")

        # Now try to allocate one more with eviction enabled
        # Since LocalCPUBackend.use_hot is True but hot_cache is empty,
        # there are no eviction candidates
        # This will trigger the eviction path in LocalCPUBackend.allocate()
        # which will hit the assertion error
        try:
            memory_obj = storage_manager.allocate(
                shape, dtype, fmt, eviction=True, busy_loop=False
            )
            # If we reach here, the bug has been fixed
            print("SUCCESS: No AssertionError - the bug has been fixed!")
            # In the fixed version, this should return None
            # since there's nothing to evict
            assert memory_obj is None, (
                "Should return None when allocation fails and no eviction candidates"
            )
        except AssertionError as e:
            # This is the expected error in the buggy version
            error_msg = str(e)
            print(f"EXPECTED ERROR (bug reproduced): {error_msg}")

            # Verify this is the specific assertion we're looking for
            # The assertion happens at line 266 in local_cpu_backend.py
            # and has no message, so we check the traceback
            # Standard
            import traceback

            tb = traceback.format_exc()
            assert "local_cpu_backend.py" in tb, (
                f"Expected error from local_cpu_backend.py, got:\n{tb}"
            )
            assert "isinstance(self.memory_allocator, MixedMemoryAllocator)" in tb or (
                "allocate" in tb and "assert" in tb.lower()
            ), f"Expected assertion about allocator type, got:\n{tb}"

            # Re-raise to make the test fail and show that we've reproduced the issue
            raise
        finally:
            # Clean up allocated objects
            for obj in allocated_objs:
                obj.ref_count_down()

    finally:
        # Cleanup
        if storage_manager is not None:
            storage_manager.close()

        if thread_loop is not None:
            if thread_loop.is_running():
                thread_loop.call_soon_threadsafe(thread_loop.stop)
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)
            thread_loop.close()

        if os.path.exists(WEKA_DIR):
            shutil.rmtree(WEKA_DIR, ignore_errors=True)
