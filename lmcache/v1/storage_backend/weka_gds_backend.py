# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as ConcurrentTimeoutError
from typing import Any, Callable, List, Optional, Sequence
import asyncio
import os
import random
import threading

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

logger = init_logger(__name__)


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


class WekaGdsBackend(StorageBackendInterface):
    """
    TODO(Serapheim): Document this backend
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
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
        self.dst_device = dst_device

        assert config.weka_path is not None, (
            "Need to specify weka_path for WekaGdsBackend"
        )
        self.weka_path = config.weka_path
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

        self._cufile_driver = self.cufile.CuFileDriver()
        assert hasattr(self.memory_allocator, "base_pointer")
        self.cufile_base_pointer = self.memory_allocator.base_pointer
        self.save_metadata_tasks: set[asyncio.Task] = set()

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # TODO(Serapheim): implement this
        return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        # TODO(Serapheim): implement this
        return False

    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Future:
        # TODO(Serapheim): implement this
        return Future()

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        # TODO(Serapheim): implement this
        return None

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        # TODO(Serapheim): implement this
        return None

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        # TODO(Serapheim): implement this
        return [None] * len(keys)

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
        # TODO(Serapheim): implement this
        return 0

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
