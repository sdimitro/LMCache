# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as ConcurrentTimeoutError
from typing import Any, Callable, List, Optional, Sequence, Tuple
import asyncio
import ctypes
import os
import random
import string
import struct
import threading
import time

# Third Party
import aiofile
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import get_loguru, init_logger
from lmcache.observability import (
    ERROR_ALLOC_FAILURES,
    ERROR_IO_FAILURES,
    ERROR_THRESHOLD,
    ERROR_TIMEOUT,
    LMCStatsMonitor,
)
from lmcache.utils import (
    CacheEngineKey,
    DiskCacheMetadata,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    CuFileMemoryAllocator,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
)

logger = init_logger(__name__)
nu_logger = get_loguru()


class OperationTimeoutError(Exception):
    """Exception raised when operations timeout."""

    pass


class OperationHangThresholdReached(Exception):
    """Exception raised when operations hang threshold is reached."""

    pass


class OperationManager:
    def __init__(
        self,
        num_threads: int = 4,
        hang_threshold: int = 10,
        reset_file: str = "/tmp/lmcache_operation_manager_reset",
    ):
        self.timeout_pool = ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="fs-timeout"
        )
        self._failure_count = 0
        self._failure_lock = threading.Lock()
        self._hang_threshold = hang_threshold
        self._reset_file = reset_file

    def run_with_timeout(
        self,
        func: Callable[[], Any],
        timeout_seconds: float,
        label: str = "default_label",
        metadata: Any = None,
    ) -> Any:
        if self._failure_count >= self._hang_threshold:
            if os.path.exists(self._reset_file):
                os.remove(self._reset_file)
                self.reset_failure_count()
                nu_logger.info(
                    f"Resetting operation manager failure count due to reset file "
                    f"{self._reset_file}"
                )
            else:
                raise OperationHangThresholdReached(
                    f"Operation hang threshold reached. Will not run operation "
                    f"'{label}'",
                    metadata,
                )
        future = self.timeout_pool.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except ConcurrentTimeoutError as err:
            count = self.increment_failure_count()
            raise OperationTimeoutError(
                f"Operation '{label}' timed out after {timeout_seconds} seconds",
                metadata,
                count,
            ) from err

    def shutdown(self):
        self.timeout_pool.shutdown(wait=True)

    def increment_failure_count(self) -> int:
        with self._failure_lock:
            self._failure_count += 1
            return self._failure_count

    def get_failure_count(self) -> int:
        """Get the current count of timed-out operations."""
        with self._failure_lock:
            return self._failure_count

    def reset_failure_count(self) -> int:
        """Reset the timeout counter and return the previous count."""
        with self._failure_lock:
            old_count = self._failure_count
            self._failure_count = 0
            return old_count


_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".weka1"
_METADATA_VERSION = 1
_METADATA_MAX_SIZE = 4096  # reserve 4K for metadata


class UnsupportedMetadataVersion(Exception):
    pass


torch_dtypes = [
    torch.half,
    torch.float16,
    torch.bfloat16,
    torch.float,
    torch.float32,
    torch.float64,
    torch.double,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
dtype_to_idx = {dtype: idx for idx, dtype in enumerate(torch_dtypes)}


def pack_metadata(shape, dtype, size) -> bytes:
    metadata_desc = "<QQQQ" + len(shape) * "Q"
    if struct.calcsize(metadata_desc) > _METADATA_MAX_SIZE:
        # TODO(Serapheim/Ilya): support variable offset for data
        raise ValueError(
            f"Metadata size {struct.calcsize(metadata_desc)} "
            f"exceeds max size {_METADATA_MAX_SIZE}"
        )
    return struct.pack(
        metadata_desc, _METADATA_VERSION, dtype_to_idx[dtype], size, len(shape), *shape
    )


def unpack_metadata(buffer):
    version, dt_idx, size, ndim = struct.unpack_from("<QQQQ", buffer)
    shape_offset = struct.calcsize("<QQQQ")
    if version != _METADATA_VERSION:
        # TODO(Serapheim): When we bump the _METADATA_VERSION for
        # the first time, we need to ensure that we can still
        # read older versions.
        raise UnsupportedMetadataVersion(f"Unsupported metadata version: {version}")
    shape = struct.unpack_from("<" + ndim * "Q", buffer, offset=shape_offset)
    return torch.Size(shape), torch_dtypes[dt_idx], size


def rand_suffix(rand, n: int):
    return "".join(
        rand.choice(string.ascii_uppercase + string.digits) for _ in range(n)
    )


async def save_metadata(path: str, tmp: str, metadata: bytes):
    tmp_path = path + tmp
    async with aiofile.async_open(tmp_path, "wb") as f:
        await f.write(metadata)
    os.rename(tmp_path, path)


class WekaGdsBackend(AllocatorBackendInterface):
    """
    This is a backend that leverages NVIDIA's cuFile API to issue GDS requests
    directly to the Weka Filesystem.  In order to use it, users need to specify
    `weka_path` and `cufile_buffer_size` in their LMCache config.

    NOTE: The `weka_path` does not strictly need to be a WekaFS mount so if you
    want to test the backend without Weka you are free to do so for testing
    purposes. For production though it wouldn't scale as this backend is
    tailored to the performance characteristics of WekaFS. More specifically if
    used with non-Weka filesystems performance will suffer potentially for two
    reasons:
    (1) If GPUDirect is not supported on that other filesystem, then CuFile will
        fall back to POSIX I/O.
    (2) Our cache directory structure creates a lot of small files within a
        single directory and uses 4K block/buffer sizes. These align very well
        with Weka but not other filesystems.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
    ):
        # HACK(Jiayi): cufile import is buggy on some hardware
        # (e.g., without GPUDirect), so it's temporarily put here.
        # Third Party
        import cufile

        self.cufile = cufile

        assert dst_device.startswith("cuda")
        super().__init__(dst_device)

        self.layerwise = config.use_layerwise
        self.loop = loop
        self.memory_allocator = memory_allocator
        assert isinstance(self.memory_allocator, CuFileMemoryAllocator)
        self.dst_device = dst_device

        assert config.weka_path is not None, (
            "Need to specify weka_path for WekaGdsBackend"
        )
        # Construct a descriptive directory name based on metadata
        # Format:
        # {model_name}-{world_size}-{fmt}-{kv_dtype}-{kv_shape}-{worker_id}[-layerwise]
        dtype_str = str(metadata.kv_dtype).replace("torch.", "")
        shape_str = "x".join(map(str, metadata.kv_shape))
        dir_components = [
            # Replace / in model names like "meta/Llama-2-7b"
            metadata.model_name.replace("/", "_"),
            str(metadata.world_size),
            metadata.fmt,
            dtype_str,
            shape_str,
            str(metadata.worker_id),
        ]
        if self.layerwise:
            dir_components.append("layerwise")
        metadata_dir = "-".join(dir_components)
        self.weka_path = os.path.join(config.weka_path, metadata_dir)
        os.makedirs(self.weka_path, exist_ok=True)

        self.hot_lock = threading.Lock()
        self.hot_cache: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()
        self.metadata_dirs: set[str] = set()

        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        self.rand = random.Random(self.dst_device)
        thread_count = config.extra_config.get("gds_io_threads", 4)
        self._thread_pool = ThreadPoolExecutor(
            max_workers=thread_count, thread_name_prefix="weka-gds-io"
        )
        self.op_manager = OperationManager(
            config.extra_config.get("operation_manager_threads", 4),
            config.extra_config.get("operation_hang_threshold", 10),
        )
        self.timeout_contains = config.extra_config.get("timeout_contains", 1.0)
        self.timeout_get_blocking = config.extra_config.get("timeout_get_blocking", 5.0)
        self.timeout_batched_get_blocking = config.extra_config.get(
            "timeout_batched_get_blocking", 5.0
        )

        self.max_alloc_attempts = config.extra_config.get("max_alloc_attempts", 10)
        self.alloc_attempt_delay_secs = config.extra_config.get(
            "allocation_attempt_delay_secs", 0.1
        )
        self.enable_blending = config.extra_config.get("enable_blending", False)

        self._cufile_driver = self.cufile.CuFileDriver()
        assert hasattr(self.memory_allocator, "base_pointer")
        self.cufile_base_pointer = self.memory_allocator.base_pointer
        self.save_metadata_tasks: set[asyncio.Task] = set()
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    def _read_metadata_info(self, filename: str) -> Tuple[torch.Size, torch.dtype, int]:
        # Use O_NOATIME to prevent updating access time and improve performance
        # Instead of using Python's open() and read(), we use the OS's open() and
        # read() because it is faster - the metadata file is small and we don't
        # need any buffering.
        #
        # Additionally, we use O_NOATIME for two reasons:
        # 1. Improve performance
        # 2. To prevent updating the access time and preserve our LRU ordering
        #    when we get rid of the metadata file separation.
        fd = os.open(filename, os.O_RDONLY | os.O_NOATIME)
        try:
            buf = os.read(fd, _METADATA_MAX_SIZE)
        finally:
            os.close(fd)
        return unpack_metadata(buf)

    def _import_key_with_metadata(
        self, key: CacheEngineKey, filename: str, subdir_key: str
    ):
        shape, dtype, size = self._read_metadata_info(filename)
        # Set the appropriate memory format for layerwise operations
        fmt = None
        if self.layerwise:
            fmt = MemoryFormat.KV_T2D
        else:
            fmt = MemoryFormat.KV_2LTD
        metadata = DiskCacheMetadata(
            filename.removesuffix(_METADATA_FILE_SUFFIX), size, shape, dtype, fmt
        )
        with self.hot_lock:
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = metadata
        return metadata

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # TODO(Serapheim): implement pin() semantics
        with self.hot_lock:
            res = key in self.hot_cache
        if res:
            return True
        return self._contains_slow_path(key)

    def _try_to_read_metadata(self, key: CacheEngineKey) -> Optional[DiskCacheMetadata]:
        path, subdir_key, _, _ = self._key_to_path(key)
        path += _METADATA_FILE_SUFFIX
        if os.path.exists(path):
            try:
                return self._import_key_with_metadata(key, path, subdir_key)
            except UnsupportedMetadataVersion:
                logger.error(f"Unsupported metadata version for {path}, ignoring")
            except (OSError, IOError) as e:
                logger.error(
                    f"Failed to read metadata file {path}: {type(e).__name__}: {e}. "
                    f"File may be corrupted or inaccessible. "
                    f"Ignoring cache entry for key {key}."
                )
            except Exception as e:
                nu_logger.error(
                    f"Unexpected error reading metadata file {path}: "
                    f"{type(e).__name__}: {e}. Ignoring cache entry for key {key}."
                )
        return None

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> Tuple[str, str, str, str]:
        hash = str(key.chunk_hash)
        l1_dir = hash[:2]
        l2_dir = hash[2:4]
        key_str = key.to_string()
        assert "_" not in key_str, "key string should not contain `_`"
        return (
            os.path.join(
                self.weka_path,
                l1_dir,
                l2_dir,
                key_str.replace("/", "_") + _DATA_FILE_SUFFIX,
            ),
            l1_dir + l2_dir,
            l1_dir,
            l2_dir,
        )

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Future:
        memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.add(key)

        future = asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(key, memory_obj), self.loop
        )
        return future

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

    async def _async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        path, subdir_key, l1_dir, l2_dir = self._key_to_path(key)
        if subdir_key not in self.metadata_dirs:
            os.makedirs(os.path.join(self.weka_path, l1_dir, l2_dir), exist_ok=True)
            self.metadata_dirs.add(subdir_key)
        tmp = ".tmp" + rand_suffix(self.rand, 8)

        try:
            metadata = await asyncio.to_thread(
                self._save_gds_cufile,
                path,
                tmp,
                kv_chunk,
                self.cufile_base_pointer,
                memory_obj.metadata.address,
            )
        except Exception as e:
            nu_logger.error(
                f"GDS/cuFile write operation failed for key {key} at path {path}: "
                f"tensor_shape={kv_chunk.shape}, tensor_dtype={kv_chunk.dtype}, "
                f"tensor_size_bytes={kv_chunk.nbytes}, error={e}",
                exc_info=True,
            )
            self.stats_monitor.update_weka_gds_error(ERROR_IO_FAILURES)
            with self.put_lock:
                self.put_tasks.discard(key)
            return

        self.insert_key(key, memory_obj)
        memory_obj.ref_count_down()

        try:
            task = asyncio.create_task(
                save_metadata(path + _METADATA_FILE_SUFFIX, tmp, metadata)
            )
            self.save_metadata_tasks.add(task)
            task.add_done_callback(self.save_metadata_tasks.discard)
        except Exception as e:
            nu_logger.error(
                f"POSIX metadata write operation failed for key {key} at path "
                f"{path + _METADATA_FILE_SUFFIX}: metadata_size_bytes={len(metadata)}, "
                f"tmp_suffix={tmp}, error={e}",
                exc_info=True,
            )
            self.stats_monitor.update_weka_gds_error(ERROR_IO_FAILURES)
            with self.hot_lock:
                self.hot_cache.pop(key, None)
        with self.put_lock:
            self.put_tasks.discard(key)

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path, _, _, _ = self._key_to_path(key)
        size = memory_obj.get_size()  # Use logical size to match what's stored in file
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        with self.hot_lock:
            self.hot_cache[key] = DiskCacheMetadata(path, size, shape, dtype)

    async def _async_load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        return self._load_bytes_from_disk_with_allocation(key, path, dtype, shape)

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.hot_lock:
            entry = self.hot_cache.get(key)
        if entry is None:
            return None

        path = entry.path
        dtype = entry.dtype
        shape = entry.shape
        assert dtype is not None
        assert shape is not None
        try:
            return self.op_manager.run_with_timeout(
                lambda: self._load_bytes_from_disk_with_allocation(
                    key, path, dtype, shape
                ),
                self.timeout_get_blocking,
                "get_blocking",
                key,
            )
        except OperationHangThresholdReached:
            nu_logger.error(
                "Get blocking hang threshold reached. Will not run operation",
                exc_info=True,
            )
            self.stats_monitor.update_weka_gds_error(ERROR_THRESHOLD)
            return None
        except OperationTimeoutError:
            nu_logger.error(
                f"Get blocking timed out after {self.timeout_get_blocking} seconds",
                exc_info=True,
            )
            self.stats_monitor.update_weka_gds_error(ERROR_TIMEOUT)
            return None

    def _load_bytes_from_disk_with_memory(
        self,
        key: CacheEngineKey,
        path: str,
        memory_obj: Optional[MemoryObj],
    ) -> Optional[MemoryObj]:
        """
        Load byte array from disk into a pre-allocated memory object.

        Args:
            key: Cache key for error handling
            path: File path to load from
            memory_obj: Pre-allocated memory object to load data into

        Returns:
            The memory object with loaded data, or None if loading failed
        """
        if memory_obj is None:
            return None

        # Read logical size instead of physical size since
        # we only store logical size in file
        logical_size = memory_obj.get_size()
        ret = self._load_gds_cufile(
            path,
            _METADATA_MAX_SIZE,
            ctypes.c_void_p(self.cufile_base_pointer),
            logical_size,
            memory_obj.metadata.address,
        )
        if ret != logical_size:
            if ret < 0:
                nu_logger.error(
                    f"Error loading {path}: ret: {ret} removing entry from cache"
                )
                self.stats_monitor.update_weka_gds_error(ERROR_IO_FAILURES)
                with self.hot_lock:
                    self.hot_cache.pop(key)
            else:
                # TODO(Serapheim): we should probably count errors and
                # remove the entry if it's a persistent problem.
                nu_logger.error(
                    f"Error loading {path}: got only {ret} bytes "
                    f"out of {logical_size}, ignoring"
                )
                self.stats_monitor.update_weka_gds_error(ERROR_IO_FAILURES)
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def _load_bytes_from_disk_with_allocation(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Load byte array from disk by first allocating memory, then loading.

        Args:
            key: Cache key for error handling
            path: File path to load from
            dtype: Data type for memory allocation
            shape: Shape for memory allocation

        Returns:
            A new memory object with loaded data, or None if allocation or
            loading failed
        """
        fmt = None
        if self.layerwise:
            fmt = MemoryFormat.KV_T2D
        else:
            fmt = MemoryFormat.KV_2LTD

        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
        if memory_obj is None:
            nu_logger.error("Memory allocation failed during sync disk load.")
            self.stats_monitor.update_weka_gds_error(ERROR_ALLOC_FAILURES)
            return None

        return self._load_bytes_from_disk_with_memory(key, path, memory_obj)

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        try:
            return self.op_manager.run_with_timeout(
                lambda: self._batched_get_blocking(keys),
                self.timeout_batched_get_blocking,
                "batched_get_blocking",
                len(keys),
            )
        except OperationHangThresholdReached:
            nu_logger.error(
                "Batched get blocking hang threshold reached. Will not run operation",
                exc_info=True,
            )
            self.stats_monitor.update_weka_gds_error(ERROR_THRESHOLD)
            return [None] * len(keys)
        except OperationTimeoutError:
            nu_logger.error(
                f"Batched get blocking timed out after "
                f"{self.timeout_batched_get_blocking} seconds",
                exc_info=True,
            )
            self.stats_monitor.update_weka_gds_error(ERROR_TIMEOUT)
            return [None] * len(keys)

    def _batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        paths: list[str | None] = []
        dtypes: list[torch.dtype | None] = []
        shapes: list[torch.Size | None] = []
        with self.hot_lock:
            for key in keys:
                entry = self.hot_cache.get(key)
                if entry is None:
                    nu_logger.error(f"Lookup failed during get_blocking for {key}")
                    paths.append(None)
                    dtypes.append(None)
                    shapes.append(None)
                    continue
                paths.append(entry.path)
                dtypes.append(entry.dtype)
                shapes.append(entry.shape)

        fmt = None
        if self.layerwise:
            fmt = MemoryFormat.KV_T2D
        else:
            fmt = MemoryFormat.KV_2LTD

        memory_objs: list[MemoryObj | None] = []
        gds_reads, gds_read_bytes = 0, 0
        for dtype, shape, path in zip(dtypes, shapes, paths, strict=True):
            if path is None:
                memory_objs.append(None)
                continue
            memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
            if memory_obj is None:
                nu_logger.error(
                    f"Memory allocation failed during get_blocking for {path}"
                )
                self.stats_monitor.update_weka_gds_error(ERROR_ALLOC_FAILURES)
            else:
                gds_reads += 1
                gds_read_bytes += memory_obj.get_size()
            memory_objs.append(memory_obj)

        start_time = time.perf_counter()
        results = list(
            self._thread_pool.map(
                self._load_bytes_from_disk_with_memory, keys, paths, memory_objs
            )
        )
        total_time = time.perf_counter() - start_time
        logger.info(
            f"Time taken for batched_get: {total_time:.3f}s |"
            f" {gds_read_bytes / 1024 / 1024}MiB | {gds_reads} ops."
        )

        # Report GDS read metrics to stats monitor
        self.stats_monitor.update_weka_gds_read_metrics(gds_reads, gds_read_bytes)

        return results

    async def _async_batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        """
        Asynchronously run the batched get operation in a thread pool.
        This allows the event loop to handle other operations while I/O is happening.
        """
        return await asyncio.to_thread(self._batched_get_blocking, keys)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_gds_cufile(
        self,
        path: str,
        tmp: str,
        kv_chunk: torch.Tensor,
        base_pointer: int,
        device_offset: int,
    ):
        addr = ctypes.c_void_p(base_pointer)
        dev_offset = device_offset
        tmp_path = path + tmp
        offset = _METADATA_MAX_SIZE
        metadata = pack_metadata(kv_chunk.shape, kv_chunk.dtype, kv_chunk.nbytes)
        try:
            with open(tmp_path, "wb") as f:
                f.write(metadata)
            with self.cufile.CuFile(tmp_path, "r+") as f:
                f.write(
                    addr, kv_chunk.nbytes, file_offset=offset, dev_offset=dev_offset
                )
        except Exception as e:
            nu_logger.error(f"Error saving {tmp_path}: {e}", exc_info=True)
            raise e
        os.rename(tmp_path, path)
        return metadata

    def _load_gds_cufile(
        self,
        file_path: str,
        file_offset: int,
        gpu_pointer: ctypes.c_void_p,
        size_in_bytes: int,
        dev_offset: int,
    ) -> int:
        # Read data from disk into a GPU buffer
        try:
            with self.cufile.CuFile(file_path, "r") as f:
                return f.read(
                    gpu_pointer,
                    size_in_bytes,
                    file_offset=file_offset,
                    dev_offset=dev_offset,
                )
        except Exception as e:
            nu_logger.error(f"CuFile read failed for {file_path}: {e}", exc_info=True)
            return -1

    def pin(self, key: CacheEngineKey) -> bool:
        # TODO(Serapheim): Implement this
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        # TODO(Serapheim): Implement this
        return False

    def remove(self, key, force=True):
        raise NotImplementedError("Remote backend does not support remove now.")

    def _contains_slow_path(self, key: CacheEngineKey) -> bool:
        try:
            read_from_disk = self.op_manager.run_with_timeout(
                lambda: self._try_to_read_metadata(key),
                self.timeout_contains,
                "contains",
                key,
            )
            if read_from_disk:
                return True
            return False  # Metadata not found or read failed
        except OperationHangThresholdReached:
            nu_logger.error(
                "Contains hang threshold reached. Will not run operation",
                exc_info=True,
            )
            self.stats_monitor.update_weka_gds_error(ERROR_THRESHOLD)
            return False
        except OperationTimeoutError:
            nu_logger.error(
                f"Contains timed out after {self.timeout_contains} seconds",
                exc_info=True,
            )
            self.stats_monitor.update_weka_gds_error(ERROR_TIMEOUT)
            return False

    async def _async_contains_slow_path(self, key: CacheEngineKey) -> bool:
        """
        Asynchronously check if a key exists using the slow path (FS I/O).
        This runs the blocking operation in a thread pool to avoid blocking
        the event loop.
        """
        return await asyncio.to_thread(self._contains_slow_path, key)

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """
        Check whether keys are in the storage backend.

        :param lookup_id: Identifier for the lookup operation
        :param keys: The keys to check
        :param pin: Whether to pin the keys if they exist
        :return: Number of keys that exist in the storage backend
        """
        num_hit_chunks = 0
        while num_hit_chunks < len(keys):
            # Keep the lock as long as we keep getting hits
            # in the hot cache.
            with self.hot_lock:
                while (
                    num_hit_chunks < len(keys)
                    and keys[num_hit_chunks] in self.hot_cache
                ):
                    if pin:
                        # TODO(Serapheim): implement pin() semantics
                        pass
                    num_hit_chunks += 1

            # If we've processed all keys, return the count
            if num_hit_chunks == len(keys):
                return num_hit_chunks

            # Check the current key that's not in hot cache using async slow path
            current_key = keys[num_hit_chunks]
            if await self._async_contains_slow_path(current_key):
                num_hit_chunks += 1
            else:
                return num_hit_chunks

        return num_hit_chunks

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
    ) -> list[MemoryObj]:
        """
        Non-blocking function to get memory objects from storage.

        :param lookup_id: Identifier for the lookup operation
        :param keys: The keys to retrieve
        :return: List of MemoryObj instances (may contain None for missing keys)
        """
        return await self._async_batched_get_blocking(keys)  # type: ignore[return-value]

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """
        Allocate a memory object of shape and dtype
        evict if necessary.
        """
        logger.debug(
            f"Allocating memory in WekaGDS backend with busy loop: {busy_loop}"
            f" with eviction: {eviction}"
        )
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
        if memory_obj is not None:
            return memory_obj
        if not busy_loop:
            nu_logger.error(
                "WekaGDS allocation failed and busy loop is disabled. Returning None."
            )
            self.stats_monitor.update_weka_gds_error(ERROR_ALLOC_FAILURES)
            return None

        num_attempts = 0
        nu_logger.warning(
            "WekaGDS allocation failed and busy loop is enabled. "
            f"Waiting for {self.alloc_attempt_delay_secs} seconds before retrying."
        )
        while True:
            time.sleep(self.alloc_attempt_delay_secs)

            memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
            if memory_obj is not None:
                break
            num_attempts += 1
            nu_logger.warning(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of WekaGDS backend allocate()"
            )
            if num_attempts >= self.max_alloc_attempts:
                nu_logger.error(
                    "WekaGDS allocation failed after "
                    f"{self.max_alloc_attempts} attempts. Returning None."
                )
                self.stats_monitor.update_weka_gds_error(ERROR_ALLOC_FAILURES)
                if not self.memory_allocator.memcheck():
                    nu_logger.error(
                        "WekaGDS allocation failed and memory allocator "
                        "is inconsistent. This is a bug in the memory allocator."
                    )
                return None
        return memory_obj

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate `batch_size` memory objects of shape and dtype
        evict if necessary.
        """
        logger.debug(
            f"Batched allocating memory in WekaGDS backend"
            f" with busy loop: {busy_loop} with eviction: {eviction}"
        )
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        memory_objs = self.memory_allocator.batched_allocate(
            shape, dtype, batch_size, fmt
        )

        if memory_objs is not None:
            return memory_objs
        if not busy_loop:
            nu_logger.error(
                "WekaGDS batched allocation failed and "
                "busy loop is disabled. Returning None."
            )
            self.stats_monitor.update_weka_gds_error(ERROR_ALLOC_FAILURES)
            return None

        num_attempts = 0
        nu_logger.warning(
            "WekaGDS batched allocation failed and busy loop is enabled. "
            f"Waiting for {self.alloc_attempt_delay_secs} seconds before retrying."
        )
        while True:
            time.sleep(self.alloc_attempt_delay_secs)

            memory_objs = self.memory_allocator.batched_allocate(
                shape, dtype, batch_size, fmt
            )
            if memory_objs:
                break

            num_attempts += 1
            logger.debug(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of WekaGDS backend batched_allocate()"
            )
            if num_attempts >= self.max_alloc_attempts:
                nu_logger.error(
                    "WekaGDS batched allocation failed after "
                    f"{self.max_alloc_attempts} attempts. Returning None."
                )
                self.stats_monitor.update_weka_gds_error(ERROR_ALLOC_FAILURES)
                if not self.memory_allocator.memcheck():
                    nu_logger.error(
                        "WekaGDS batched allocation failed and memory allocator "
                        "is inconsistent. This is a bug in the memory allocator."
                    )
                return None
        return memory_objs

    def close(self) -> None:
        self.op_manager.shutdown()
        self._thread_pool.shutdown(wait=True)
        nu_logger.info("Weka backend closed.")
