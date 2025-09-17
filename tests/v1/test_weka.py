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
    futures = backend.batched_submit_put_task(keys, memory_objs)
    assert futures is not None
    assert len(futures) == 3
    for future in futures:
        assert future is not None
        future.result()
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

    # Reset timeout counter first
    backend.op_manager.reset_timeout_count()

    with unittest.mock.patch.object(
        backend, "_try_to_read_metadata", side_effect=slow_try_to_read_metadata
    ):
        # First, trigger timeouts to reach the hang threshold
        for i in range(backend.op_manager._hang_threshold):
            result = backend.contains(create_test_key(chunk_hash=i), False)
            assert result is False

        # Verify we've reached the timeout count
        assert (
            backend.op_manager.get_timeout_count() >= backend.op_manager._hang_threshold
        )

        # Now the next call should trigger hang threshold
        result = backend.contains(k, False)
        # Should still return False, but due to hang threshold
        assert result is False


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

    # Reset timeout counter first
    backend.op_manager.reset_timeout_count()

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

        # Verify we've reached the timeout count
        assert (
            backend.op_manager.get_timeout_count() >= backend.op_manager._hang_threshold
        )

        # Now the next call should trigger hang threshold
        result = backend.get_blocking(k)
        # Should still return None, but due to hang threshold
        assert result is None


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

    # Reset timeout counter first
    backend.op_manager.reset_timeout_count()

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

        # Verify we've reached the timeout count
        assert (
            backend.op_manager.get_timeout_count() >= backend.op_manager._hang_threshold
        )

        # Now the next call should trigger hang threshold
        result = backend.batched_get_blocking([k1, k2])
        # Should still return [None, None], but due to hang threshold
        assert result == [
            None,
            None,
        ]


def test_contains_timeout():
    init_and_teardown(contains_timeout_test)


def test_get_blocking_timeout():
    init_and_teardown(get_blocking_timeout_test)


def test_batched_get_blocking_timeout():
    init_and_teardown(batched_get_blocking_timeout_test)


def test_contains_hang_threshold():
    init_and_teardown(contains_hang_threshold_test)


def test_get_blocking_hang_threshold():
    init_and_teardown(get_blocking_hang_threshold_test)


def test_batched_get_blocking_hang_threshold():
    init_and_teardown(batched_get_blocking_hang_threshold_test)
