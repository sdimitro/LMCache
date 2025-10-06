# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as ConcurrentTimeoutError
from typing import Any, Callable, List, Optional, Sequence, Tuple
import asyncio
import ctypes
import os
import pickle
import struct
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    DiskCacheMetadata,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.weka_specific import wio

logger = init_logger(__name__)


# Mapping of torch dtypes to integer IDs for efficient serialization
TORCH_DTYPE_TO_ID = {
    torch.float16: 0,
    torch.half: 0,  # Alias for float16
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float: 2,  # Alias for float32
    torch.float64: 3,
    torch.double: 3,  # Alias for float64
    torch.uint8: 4,
    torch.float8_e4m3fn: 5,
    torch.float8_e5m2: 6,
}

# Reverse mapping: ID to torch dtype (using canonical types)
ID_TO_TORCH_DTYPE = {
    0: torch.float16,
    1: torch.bfloat16,
    2: torch.float32,
    3: torch.float64,
    4: torch.uint8,
    5: torch.float8_e4m3fn,
    6: torch.float8_e5m2,
}


class TensorMetadata:
    """
    Metadata for a cached tensor, including its location in the arena and
    shape/dtype info.
    """

    def __init__(
        self,
        key: CacheEngineKey,
        arena_id: int,
        arena_offset: int,
        size: int,
        shape: Tuple[int, ...],
        dtype,  # Can be torch.dtype or int
    ):
        self.key = key
        self.arena_id = arena_id
        self.arena_offset = arena_offset
        self.size = size
        self.shape = shape

        # Convert dtype to int ID for efficient storage
        if isinstance(dtype, int):
            self.dtype_id = dtype
        else:
            # Assume it's a torch.dtype
            if dtype not in TORCH_DTYPE_TO_ID:
                raise ValueError(f"Unsupported dtype: {dtype}")
            self.dtype_id = TORCH_DTYPE_TO_ID[dtype]

    def get_torch_dtype(self) -> torch.dtype:
        """Get the torch.dtype for this tensor."""
        return ID_TO_TORCH_DTYPE[self.dtype_id]

    def to_bytes(self) -> bytes:
        """
        Serialize TensorMetadata to bytes using custom struct-based format
        for maximum performance.

        Format:
            - Fixed-size header (57 bytes):
              - 2 unsigned long longs (Q): world_size, worker_id
              - 1 signed long long (q): chunk_hash (can be negative)
              - 3 unsigned long longs (Q): arena_id, arena_offset, size
              - 4 unsigned ints (I): num_dims, fmt_len, model_len, tags_len
              - 1 unsigned byte (B): dtype_id
            - Variable-length data: shape array, fmt string, model_name string, tags

        Returns:
            bytes: Binary representation of this TensorMetadata object.
        """
        # Encode variable-length data
        fmt_bytes = self.key.fmt.encode("utf-8")
        model_bytes = self.key.model_name.encode("utf-8")
        tags_bytes = pickle.dumps(self.key.tags) if self.key.tags else b""

        # Pack fixed-size header (57 bytes)
        # Format: 2Q (unsigned) + 1q (signed) + 3Q (unsigned) + 4I + 1B
        header = struct.pack(
            "!2Qq3Q4IB",
            self.key.world_size,
            self.key.worker_id,
            self.key.chunk_hash,  # signed - can be negative
            self.arena_id,
            self.arena_offset,
            self.size,
            len(self.shape),  # number of dimensions
            len(fmt_bytes),  # length of fmt string
            len(model_bytes),  # length of model_name string
            len(tags_bytes),  # length of tags data (0 if None)
            self.dtype_id,  # dtype as single byte
        )

        # Pack shape tuple as array of signed long longs
        shape_data = struct.pack(f"!{len(self.shape)}q", *self.shape)

        # Concatenate all variable-length data
        return header + shape_data + fmt_bytes + model_bytes + tags_bytes

    @staticmethod
    def from_bytes(data: bytes) -> "TensorMetadata":
        """
        Deserialize TensorMetadata from bytes.

        Args:
            data: Binary data containing a TensorMetadata object.

        Returns:
            TensorMetadata: Deserialized TensorMetadata object.
        """
        # Unpack fixed-size header (57 bytes)
        header_size = struct.calcsize("!2Qq3Q4IB")
        header = struct.unpack("!2Qq3Q4IB", data[:header_size])

        world_size, worker_id, chunk_hash = header[0], header[1], header[2]
        arena_id, arena_offset, size = header[3], header[4], header[5]
        num_dims, fmt_len, model_len, tags_len = (
            header[6],
            header[7],
            header[8],
            header[9],
        )
        dtype_id = header[10]

        # Unpack shape
        offset = header_size
        shape = struct.unpack(f"!{num_dims}q", data[offset : offset + num_dims * 8])
        offset += num_dims * 8

        # Unpack variable-length strings
        fmt_str = data[offset : offset + fmt_len].decode("utf-8")
        offset += fmt_len

        model_name = data[offset : offset + model_len].decode("utf-8")
        offset += model_len

        # Unpack tags (if present)
        tags = None
        if tags_len > 0:
            tags = pickle.loads(data[offset : offset + tags_len])

        # Reconstruct CacheEngineKey
        key = CacheEngineKey(
            fmt=fmt_str,
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            chunk_hash=chunk_hash,
        )
        key.tags = tags

        return TensorMetadata(
            key=key,
            arena_id=arena_id,
            arena_offset=arena_offset,
            size=size,
            shape=shape,
            dtype=dtype_id,  # Pass as int, constructor will handle it
        )

    # # Alternative: Pickle-based implementation (simpler but slightly slower)
    # def to_bytes(self) -> bytes:
    #     """
    #     Serialize TensorMetadata to bytes using pickle for maximum performance.
    #
    #     Returns:
    #         bytes: Pickled representation of this TensorMetadata object.
    #     """
    #     return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # @staticmethod
    # def from_bytes(data: bytes) -> "TensorMetadata":
    #     """
    #     Deserialize TensorMetadata from bytes.
    #
    #     Args:
    #         data: Pickled bytes containing a TensorMetadata object.
    #
    #     Returns:
    #         TensorMetadata: Deserialized TensorMetadata object.
    #     """
    #     return pickle.loads(data)


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
                logger.info(
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


class ArenaMetadata:
    """
    TODO(Serapheim): Document this
    """

    def __init__(self, arena_path: str):
        """
        TODO(Serapheim): Document this
        Args:
            arena_path: Path to the arena to open (creates if doesn't exist)
        """
        self.wio_handle = wio.WioHandle(arena_path, os.O_RDWR | os.O_CREAT)

        self._lock = threading.Lock()
        self._next_offset = ctypes.c_uint64(0)
        self._removed_bytes = ctypes.c_uint64(0)

    def fetch_add_next_offset(self, delta: int) -> int:
        """
        TODO(Serapheim): Document this
        """
        with self._lock:
            old_value = self._next_offset.value
            self._next_offset.value += delta
            return old_value

    def fetch_add_removed_bytes(self, delta: int) -> int:
        """
        Atomically add delta to removed_bytes and return the old value.

        Args:
            delta: Amount to add

        Returns:
            The old value of removed_bytes before incrementing
        """
        with self._lock:
            old_value = self._removed_bytes.value
            self._removed_bytes.value += delta
            return old_value

    def get_size(self) -> int:
        """
        TODO(Serapheim): Document this
        """
        with self._lock:
            return self._next_offset.value

    def close(self) -> None:
        """Close the WioHandle."""
        self.wio_handle.close()


class ArenaManager:
    """
    TODO(Serapheim): Document this
    """

    DEFAULT_ARENA_MAX_SIZE_GB = 32
    DEFAULT_ARENA_MAX_SIZE_BYTES = DEFAULT_ARENA_MAX_SIZE_GB * 1024**3

    def __init__(
        self,
        weka_path: str,
        lmcache_instance_id: str,
        arena_max_size_gb: int = DEFAULT_ARENA_MAX_SIZE_GB,
    ):
        """
        TODO(Serapheim): Document this
        Args:
            arena_path: Path to the arena to open (creates if doesn't exist)
        """
        self._arena_manager_lock = threading.Lock()
        self._arena_map: OrderedDict[int, ArenaMetadata] = OrderedDict()
        self._active_arena_id = -1
        self._arena_max_size_bytes = arena_max_size_gb * 1024**3

        self._working_directory = os.path.join(weka_path, lmcache_instance_id)
        os.makedirs(self._working_directory, exist_ok=True)
        self._scan_arenas()

    def _scan_arenas(self) -> None:
        """
        TODO(Serapheim): Document this
        """
        with self._arena_manager_lock:
            for file in os.listdir(self._working_directory):
                if file.endswith(".arena"):
                    self._arena_map[int(file.split(".")[0])] = ArenaMetadata(
                        os.path.join(self._working_directory, file)
                    )
            self._active_arena_id = (
                max(self._arena_map.keys()) if self._arena_map else -1
            )

    def get_arena_handle(self, arena_id: int) -> wio.WioHandle:
        """
        TODO(Serapheim): Document this
        """
        with self._arena_manager_lock:
            return self._arena_map[arena_id].wio_handle

    def _create_arena_unsafe(self) -> None:
        """
        TODO(Serapheim): Document this

        IMPORTANT: This function must be called with the arena manager lock held.
        """
        self._active_arena_id += 1
        self._arena_map[self._active_arena_id] = ArenaMetadata(
            os.path.join(self._working_directory, f"{self._active_arena_id}.arena")
        )

    def allocate_ondisk_space(self, size: int) -> Tuple[int, int, wio.WioHandle]:
        """
        TODO(Serapheim): Document this
        """
        with self._arena_manager_lock:
            # TODO(Serapheim): May be too many locks?
            if self._active_arena_id == -1:
                logger.debug("Creating arena for the first time")
                self._create_arena_unsafe()
            elif (
                self._arena_map[self._active_arena_id].get_size()
                > self._arena_max_size_bytes
            ):
                logger.debug("Current arena is full, creating a new one")
                self._create_arena_unsafe()

            # TODO(Serapheim): May be too many locks?
            arena_id = self._active_arena_id
            arena_offset = self._arena_map[self._active_arena_id].fetch_add_next_offset(
                size
            )
            arena_handle = self._arena_map[self._active_arena_id].wio_handle
            return (arena_id, arena_offset, arena_handle)

    # TODO(Serapheim): Implement merging arenas
    # TODO(Serapheim): Implement removing old arenas


class CheckpointManager:
    """
    TODO(Serapheim): Document this
    """

    def __init__(self, weka_path: str, lmcache_instance_id: str):
        self._checkpoint_manager_lock = threading.Lock()
        self._checkpoint_current_generation = -1
        self._working_directory = os.path.join(weka_path, lmcache_instance_id)
        os.makedirs(self._working_directory, exist_ok=True)
        # TODO(Serapheim): Thread to occasionally save the checkpoint

    def _valid_checkpoint_file(self, checkpoint_number: int) -> bool:
        """
        TODO(Serapheim): Document this
        """
        return False  # TODO(Serapheim): Implement this

    def import_latest_checkpoint(
        self, hot_cache: OrderedDict[CacheEngineKey, DiskCacheMetadata]
    ) -> None:
        """
        TODO(Serapheim): Document this
        """
        with self._checkpoint_manager_lock:
            # We first scan for all checkpoint files in our working directory
            # and get the one with the latest valid checkpoint generation. If
            # none are found, we start from generation 0.
            checkpoint_numbers = []
            for file in os.listdir(self._working_directory):
                if file.endswith(".checkpoint"):
                    checkpoint_num = int(file.split(".")[0])
                    checkpoint_numbers.append(checkpoint_num)
            if len(checkpoint_numbers) == 0:
                logger.info("No checkpoint metadata found, starting from generation 0")
                self._checkpoint_current_generation = 0
            else:
                checkpoint_numbers.sort(reverse=True)
                max_checkpoint_number = checkpoint_numbers[0]
                if self._valid_checkpoint_file(max_checkpoint_number):
                    logger.info(
                        f"Valid checkpoint generation {max_checkpoint_number} found"
                    )
                    self._checkpoint_current_generation = max_checkpoint_number
                elif len(checkpoint_numbers) > 1:
                    next_checkpoint_number = checkpoint_numbers[1]
                    logger.info(
                        f"Invalid checkpoint generation {max_checkpoint_number}, "
                        f"reverting to generation {next_checkpoint_number}"
                    )
                    self._checkpoint_current_generation = next_checkpoint_number
                    assert self._valid_checkpoint_file(next_checkpoint_number), (
                        "Backup checkpoint generation is also invalid, "
                        "this is a serious error"
                    )
                else:
                    logger.error(
                        "No valid checkpoint generation found, this is a serious error"
                    )
                    raise RuntimeError("No valid checkpoint generation found")

            # We then import the latest checkpoint into our hot cache
            # TODO(Serapheim): Implement this

            # We then look for the journal of our generation and
            # load it into our hot cache
            # TODO(Serapheim): Implement this

    def record_insertions(self, journal_entries: List[TensorMetadata]) -> None:
        """
        TODO(Serapheim): Document this
        """
        pass  # TODO(Serapheim): Implement this

    def record_batch_remove(self, journal_entries: List[str]) -> None:
        """
        TODO(Serapheim): Document this
        """
        pass  # TODO(Serapheim): Implement this


class WekaGdsBackend(StorageBackendInterface):
    """
    TODO(Serapheim): Document this backend
    """

    METADATA_MAX_SIZE = 4 * 1024  # 4KB

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
    ):
        assert dst_device.startswith("cuda")
        super().__init__(dst_device)

        self._loop = loop
        self._dst_device = dst_device
        self._cufile_driver = wio.CuFileDriver()
        self._memory_allocator = memory_allocator
        assert hasattr(self._memory_allocator, "base_pointer")
        self._cufile_base_pointer = self._memory_allocator.base_pointer

        assert config.weka_path is not None, (
            "Need to specify weka_path for WekaGdsBackend"
        )
        self.weka_path = config.weka_path
        os.makedirs(self.weka_path, exist_ok=True)
        self._arena_manager = ArenaManager(self.weka_path, config.lmcache_instance_id)
        self._checkpoint_manager = CheckpointManager(
            self.weka_path, config.lmcache_instance_id
        )

        self.hot_lock = threading.Lock()
        self.hot_cache: OrderedDict[CacheEngineKey, TensorMetadata] = OrderedDict()

        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

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

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # NOTE(Serapheim): We do not support pin() semantics for WekaGdsBackend
        with self.hot_lock:
            return key in self.hot_cache

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        TODO(Serapheim): Document this
        """
        with self.put_lock:
            return key in self.put_tasks

    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Future:
        """
        TODO(Serapheim): Document this
        """
        memory_obj.ref_count_up()
        with self.put_lock:
            self.put_tasks.add(key)
        future = asyncio.run_coroutine_threadsafe(
            self._process_put_task(key, memory_obj), self._loop
        )
        return future

    async def _process_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        TODO(Serapheim): Document this
        """
        # TODO(Serapheim): Describe the order of operations in this function
        tensor = memory_obj.tensor
        assert tensor is not None

        arena_id, arena_offset, arena_handle = (
            self._arena_manager.allocate_ondisk_space(
                tensor.nbytes + self.METADATA_MAX_SIZE
            )
        )
        metadata = TensorMetadata(
            key=key,
            arena_id=arena_id,
            arena_offset=arena_offset,
            size=tensor.nbytes,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )
        metadata_bytes = metadata.to_bytes()
        assert len(metadata_bytes) <= self.METADATA_MAX_SIZE, (
            "Metadata size is too large"
        )

        # First we write the metadata to the arena
        # TODO(Serapheim): Use operation manageger for the write
        # TODO(Serapheim): pwrite in WIO
        ret = os.pwrite(
            arena_handle.get_file_handle(),
            metadata_bytes,
            arena_offset + self.METADATA_MAX_SIZE,
        )
        if ret != len(metadata_bytes):
            logger.error(
                f"Failed to write {key} metadata to arena {arena_id} at "
                f"offset {arena_offset}: {ret} != {len(metadata_bytes)}"
            )
            memory_obj.ref_count_down()
            return

        # Then we write the tensor to the arena
        # TODO(Serapheim): Use operation manageger for the write
        ret = arena_handle.write(
            ctypes.c_void_p(self._cufile_base_pointer),
            tensor.nbytes,
            arena_offset + self.METADATA_MAX_SIZE,
            memory_obj.metadata.address,
        )
        if ret != tensor.nbytes:
            logger.error(
                f"Failed to write {key} tensor to arena {arena_id} at "
                f"offset {arena_offset + self.METADATA_MAX_SIZE}: "
                f"{ret} != {tensor.nbytes}"
            )
            memory_obj.ref_count_down()
            return

        # Then we record the insertion in the journal
        self._checkpoint_manager.record_insertions([metadata])

        # Then we finally add the key to the hot cache and decrement
        # the reference count for the memory object
        with self.hot_lock:
            self.hot_cache[key] = metadata
        memory_obj.ref_count_down()

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        """
        TODO(Serapheim): Document this
        """
        # 1. Ref count up all objects upfront
        for memory_obj in memory_objs:
            memory_obj.ref_count_up()

        # 2. Add all keys to put_tasks in one lock acquisition
        with self.put_lock:
            self.put_tasks.update(keys)

        # 3. Submit a single async task that coordinates parallel saves
        asyncio.run_coroutine_threadsafe(
            self._batch_process_put_task(keys, memory_objs), self._loop
        )

    async def _batch_process_put_task(
        self, keys: Sequence[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        """
        TODO(Serapheim): Document this
        """
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            asyncio.run_coroutine_threadsafe(
                self._process_put_task(key, memory_obj), self._loop
            )

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        # TODO(Serapheim): implement this | do we need this?
        return None

    def _load_bytes_from_arena(
        self,
        key: CacheEngineKey,
        target_arena: int,
        target_offset: int,
        memory_obj: MemoryObj,
    ) -> Optional[MemoryObj]:
        """
        TODO(Serapheim): Document this
        """
        tensor_size = memory_obj.get_size()
        tensor_offset = target_offset + self.METADATA_MAX_SIZE
        arena_handle = self._arena_manager.get_arena_handle(target_arena)
        ret = arena_handle.read(
            ctypes.c_void_p(self._cufile_base_pointer),
            tensor_size,
            tensor_offset,
            memory_obj.metadata.address,
        )
        if ret != tensor_size:
            if ret < 0:
                logger.error(
                    f"Failed to read {key} tensor from arena {target_arena} at "
                    f"offset {tensor_offset}: error code {ret}"
                    "Removing entry from cache"
                )
                with self.hot_lock:
                    self.hot_cache.pop(key)
            else:
                logger.error(
                    f"Part read {key} tensor from arena {target_arena} at "
                    f"offset {tensor_offset}: {ret} != {tensor_size}"
                )
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        TODO(Serapheim): Document this
        """
        # TODO(Serapheim): do we need this?
        with self.hot_lock:
            metadata = self.hot_cache.get(key)
            if metadata is None:
                return None

        memory_obj = self._memory_allocator.allocate(
            metadata.shape, metadata.get_torch_dtype()
        )
        if memory_obj is None:
            logger.error(f"get_blocking: Memory allocation failed for key {key}")
            return None
        return self._load_bytes_from_arena(
            key, metadata.arena_id, metadata.arena_offset, memory_obj
        )

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        """
        TODO(Serapheim): Document this
        """
        arena_ids: list[int | None] = []
        arena_offsets: list[int | None] = []
        dtypes: list[torch.dtype | None] = []
        shapes: list[torch.Size | None] = []
        with self.hot_lock:
            for key in keys:
                entry = self.hot_cache.get(key)
                if entry is None:
                    logger.error(f"batched_get_blocking: Lookup failed for {key}")
                    arena_ids.append(None)
                    arena_offsets.append(None)
                    dtypes.append(None)
                    shapes.append(None)
                    continue
                arena_ids.append(entry.arena_id)
                arena_offsets.append(entry.arena_offset)
                dtypes.append(entry.get_torch_dtype())
                shapes.append(entry.shape)

        memory_objs: list[MemoryObj | None] = []
        gds_reads, gds_read_bytes = 0, 0
        for dtype, shape in zip(dtypes, shapes, strict=True):
            if dtype is None:
                memory_objs.append(None)
                continue
            memory_obj = self._memory_allocator.allocate(shape, dtype)
            if memory_obj is None:
                logger.error(
                    f"batched_get_blocking: Memory allocation failed "
                    f"shape {shape} dtype {dtype}"
                )
            else:
                gds_reads += 1
                gds_read_bytes += memory_obj.get_size()
            memory_objs.append(memory_obj)

        start_time = time.perf_counter()
        results = list(
            self._thread_pool.map(
                self._load_bytes_from_arena, keys, arena_ids, arena_offsets, memory_objs
            )
        )
        total_time = time.perf_counter() - start_time
        logger.info(
            f"Time taken for batched_get_blocking: {total_time:.3f}s |"
            f" {gds_read_bytes / 1024 / 1024}MiB | {gds_reads} ops."
        )
        return results

    def pin(self, key: CacheEngineKey) -> bool:
        # TODO(Serapheim): implement this
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        # TODO(Serapheim): implement this
        return False

    def remove(self, key, force=True):
        # TODO(Serapheim): implement this
        pass

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        # NOTE(Serapheim): We do not support batched_async_contains for WekaGdsBackend
        with self.hot_lock:
            return sum(key in self.hot_cache for key in keys)

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
    ) -> list[MemoryObj]:
        # TODO(Serapheim): implement this
        return []

    def close(self) -> None:
        self.op_manager.shutdown()
        self._thread_pool.shutdown(wait=True)
        logger.info("Weka backend closed.")
