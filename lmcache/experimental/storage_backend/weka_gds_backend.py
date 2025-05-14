import asyncio
import aiofile
from concurrent.futures import Future
import datasketches
import os
import random
import string
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import signal
import subprocess

import torch

from gds_client_api import (
    gds_api_init,
    gds_api_cleanup,
    gds_api_write_to_gds,
    gds_api_read_from_gds,
    gds_api_start_profiling,
    gds_api_stop_profiling,
)

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    DiskCacheMetadata,
    _lmcache_nvtx_annotate,
    is_envvar_enabled,
    timing,
)

_DEFAULT_TRACE_BUFFER_SIZE = 1024 * 1024
_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".weka2"
_METADATA_VERSION = 1

logger = init_logger(__name__)


def load_gds(gpu_pci: str, file_path: str, file_offset: int, gpu_pointer,
             size_in_bytes: int) -> int:
    """Load GPU buffer data from disk."""
    # Read data from disk into a CPU buffer
    return gds_api_write_to_gds(gpu_pci, file_path, file_offset, gpu_pointer,
                                size_in_bytes)


# TODO(ilya): move metadata serialization somewhere else
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


def rand_suffix(rand, n: int):
    return ''.join(
        rand.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def metadata_max_size():
    return 4096  # reserve 4KB for metadata


def pack_metadata(shape, dtype, size) -> bytes:
    metadata_desc = "<QQQQ" + len(shape) * "Q"
    if struct.calcsize(metadata_desc) > metadata_max_size():
        # TODO(ilya): support variable offset for data
        raise ValueError(
            f"Metadata size {struct.calcsize(metadata_desc)} exceeds max size {metadata_max_size()}"
        )
    return struct.pack(metadata_desc, _METADATA_VERSION, dtype_to_idx[dtype],
                       size, len(shape), *shape)


class UnsupportedMetadataVersion(Exception):
    pass


def unpack_metadata(buffer):
    version, dt_idx, size, ndim = struct.unpack_from("<QQQQ", buffer)
    shape_offset = struct.calcsize("<QQQQ")
    if version != _METADATA_VERSION:
        raise UnsupportedMetadataVersion(
            f"Unsupported metadata version: {version}"
        )  # TODO(ilya): add support for older versions
    shape = struct.unpack_from("<" + ndim * "Q", buffer, offset=shape_offset)
    return torch.Size(shape), torch_dtypes[dt_idx], size


@_lmcache_nvtx_annotate
@torch.inference_mode()
def save_gds(
    gpu_pci,
    path: str,
    tmp: str,
    kv_chunk: torch.Tensor,
):
    tmp_path = path + tmp
    offset = metadata_max_size()
    metadata = pack_metadata(kv_chunk.shape, kv_chunk.dtype, kv_chunk.nbytes)
    with open(tmp_path, "wb") as f:
        f.write(metadata)
    gds_api_read_from_gds(gpu_pci, tmp_path, offset, kv_chunk.data_ptr(),
                          kv_chunk.nbytes)
    os.rename(tmp_path, path)
    return metadata


async def save_metadata(path: str, tmp: str, metadata: bytes):
    tmp_path = path + tmp
    async with aiofile.async_open(tmp_path, "wb") as f:
        await f.write(metadata)
    os.rename(tmp_path, path)


# TODO: find a better way, running an external command is an overkill,
# the data is already there, just not accessible from Python...
def get_dev_pci(device: str):
    # TODO: this is dumb, but I don't see a way to get an index back from torch.device...
    parts = device.split(":")
    idx = parts[1] if len(parts) > 1 else "0"
    res = subprocess.run(
        f"/usr/bin/nvidia-smi -i {idx} --query-gpu=pci.bus_id --format=csv,noheader",
        capture_output=True,
        text=True,
        check=True,
        shell=True)
    output = res.stdout.strip()
    # TODO: for now drop the first part and convert to lower, to match what GDS implementation expects
    pci_id = output.split(":", 1)[1].lower()
    logger.debug(f"Device PCI ID: {pci_id}")
    return pci_id


# TODO(ilya): this is almost the same as in local_backend, find a way to unify
stats_compression = 100


@dataclass
class GDSStats:
    get_count = 0
    get_time = datasketches.kll_floats_sketch(stats_compression)
    gds_read_time = datasketches.kll_floats_sketch(stats_compression)
    total_read_size = 0
    total_read_time = 0
    total_gds_read_time = 0

    put_b_count = 0
    put_b_time = datasketches.kll_floats_sketch(stats_compression)
    put_b_write_time = datasketches.kll_floats_sketch(stats_compression)
    total_put_b_size = 0
    total_put_b_time = 0
    total_put_b_write_time = 0

    def _print_times(self, name, td):
        logger.info(
            "{name} time (min/median/90pt/max), ms: {mn:.3f}/{med:.3f}/{pt90:.3f}/{mx:.3f}"
            .format(name=name,
                    mn=td.get_min_value() * 1000,
                    med=td.get_quantile(0.5) * 1000,
                    pt90=td.get_quantile(0.9) * 1000,
                    mx=td.get_max_value() * 1000))

    def _print_throughput(self, name, size, acc):
        logger.info(f"{name} throughput: {size/1024/1024/acc} MB/s")

    def display(self):
        logger.info(
            f"Number of gets: {self.get_count}, total read size: {self.total_read_size/1024/1024} MB, total time spent in reads: {self.total_read_time} s"
        )
        if self.get_count:
            self._print_times("get", self.get_time)
            self._print_throughput("get", self.total_read_size,
                                   self.total_read_time)
            self._print_times("gds read", self.gds_read_time)
            self._print_throughput("gds read", self.total_read_size,
                                   self.total_gds_read_time)

        logger.info(
            f"Number of puts: {self.put_b_count}, total write size: {self.total_put_b_size/1024/1024} MB, total time spent in writes: {self.total_put_b_time} s"
        )
        if self.put_b_count:
            self._print_times("put", self.put_b_time)
            self._print_throughput("put", self.total_put_b_size,
                                   self.total_put_b_time)
            self._print_times("gds write", self.put_b_write_time)
            self._print_throughput("gds write", self.total_put_b_size,
                                   self.total_put_b_write_time)


class WekaGdsBackend(StorageBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens on Weka FS using GDS for reads and writes.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
    ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """
        super().__init__(dst_device)

        if not dst_device.startswith("cuda"):
            logger.info(f"Can't use GDS with non-CUDA dst device {dst_device}")
        # TODO: check that local device is on WEKA mount

        self.trace_file = None
        self.trace_nr = 0
        trace_file_base = os.environ.get("WEKA_GDS_TRACE_FILE")
        trace_buffer_size = 0
        if trace_file_base:
            pid = os.getpid()
            self.trace_file = f"{trace_file_base}.{pid}"
            trace_buffer_size = _DEFAULT_TRACE_BUFFER_SIZE
            trace_buffer_size_str = os.environ.get(
                "WEKA_GDS_TRACE_BUFFER_SIZE")
            if trace_buffer_size_str:
                try:
                    trace_buffer_size = int(trace_buffer_size_str)
                except ValueError:
                    logger.error(
                        f"Invalid WEKA_GDS_TRACE_BUFFER_SIZE={trace_buffer_size_str}, using default value {trace_buffer_size}"
                    )
            logger.info(
                f"Will save up to {trace_buffer_size} events to trace files with prefix {self.trace_file}"
            )
            signal.signal(
                signal.SIGUSR1,
                lambda signum, frame: self.signal_handler(signum, frame))
            self.sigterm = signal.signal(
                signal.SIGTERM,
                lambda signum, frame: self.signal_handler(signum, frame))
        gds_api_init(event_buffer_size=trace_buffer_size)
        self.dst_device_pci = get_dev_pci(dst_device)
        self.dst_device = dst_device

        self.closed = False
        self.stats = None
        if is_envvar_enabled("LMCACHE_COLLECT_STATS"):
            logger.info("Collecting stats")
            self.stats = GDSStats()
        self.chunk_size = config.chunk_size
        self.config = config
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = (OrderedDict())
        if config.remote_url is None:
            raise ValueError("Expected config.remote_url to be set (got None)")
        self.path = config.remote_url[len("weka://"):]
        self.subdirs: set[str] = set()
        self.rand = random.Random()
        self.rand.seed(self.dst_device_pci)

        assert self.path is not None, (
            "Need to specify remote url if using WekaGdsBackend")

        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)

        self.update_lock = threading.Lock()

        self.loop = loop
        self.put_tasks: set[CacheEngineKey] = set()
        self.put_lock = threading.Lock()
        self.memory_allocator = memory_allocator
        self.closed = False

        self.use_thread_pool = is_envvar_enabled("WEKA_GDS_USE_THREAD_POOL")
        asyncio.run_coroutine_threadsafe(self.scan_metadata(), self.loop)
        self.save_metadata_tasks: set[asyncio.Task] = set()

    def __str__(self):
        return self.__class__.__name__

    def signal_handler(self, signum, frame):
        if signum == signal.SIGUSR1:
            self.write_trace()
        elif signum == signal.SIGTERM:
            self.write_trace(reenable=False)
            signal.signal(signal.SIGTERM, self.sigterm)
            signal.raise_signal(signal.SIGTERM)
        else:
            logger.error(f"Unknown signal {signum} received")

    def write_trace(self, reenable: bool = True):
        if self.trace_file:
            gds_api_stop_profiling(
                f"{self.trace_file}.{self.trace_nr}.trace.json")
            logger.info(
                f"Trace written to {self.trace_file}.{self.trace_nr}.trace.json"
            )
            self.trace_nr += 1
            if reenable:
                gds_api_start_profiling()
        else:
            logger.info("No trace file specified, not writing trace.")

    def read_metadata(self, key, filename, subdirs):
        with open(filename, 'rb') as f:
            buf = f.read(metadata_max_size())
        shape, dtype, size = unpack_metadata(buf)
        metadata = DiskCacheMetadata(
            filename.removesuffix(_METADATA_FILE_SUFFIX), size, shape, dtype)
        with self.update_lock:
            self.subdirs.add(subdirs)
            self.dict[key] = metadata
        return metadata

    def scan_metadata_subdir(self, path, subdir):
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    dirname = os.path.basename(entry.name)
                    if len(dirname) != 2:
                        continue
                    with os.scandir(os.path.join(path, dirname)) as it2:
                        for fentry in it2:
                            if fentry.is_file() and fentry.name.endswith(
                                    _DATA_FILE_SUFFIX + _METADATA_FILE_SUFFIX):
                                filename = os.path.basename(fentry.name)
                                key_str = filename[:-14].replace('_', '/')
                                key = None
                                try:
                                    key = CacheEngineKey.from_string(key_str)
                                except ValueError as e:
                                    logger.error(
                                        f"Filename {filename} can't be converted back into cache key: {e}"
                                    )
                                    continue
                                # TODO(ilya): think if we want to check the main file is still there.
                                # Normally we only write metadata file _after_ the main file, but
                                # what if it was removed?
                                try:
                                    self.read_metadata(key, fentry.path,
                                                       subdir + dirname)
                                except UnsupportedMetadataVersion:
                                    logger.error(
                                        f"Unsupported metadata version for {fentry.path}, ignoring"
                                    )

    async def scan_metadata(self):
        # TODO(ilya): even though we only run it once on startup, this is still not super scalable,
        # maybe we need to add metadata snapshotting later.
        tasks = []
        start = time.perf_counter()
        with os.scandir(self.path) as it:
            for entry in it:
                if entry.is_dir():
                    dirname = os.path.basename(entry.name)
                    if len(dirname) != 2:
                        continue

                    tasks.append(
                        asyncio.to_thread(self.scan_metadata_subdir,
                                          os.path.join(self.path, dirname),
                                          dirname))
        # TODO(ilya): Can we switch to Python 3.11 and use TaskGroup instead?
        await asyncio.gather(*tasks)
        end = time.perf_counter()
        logger.info(
            f"Read {len(self.dict)} cache entries from persistent storage in {end - start:.2f} seconds"
        )

    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        with self.update_lock:
            res = key in self.dict
        if res:
            return True
        if self._try_to_read_metadata(key):
            return True
        return False

    def _try_to_read_metadata(
            self, key: CacheEngineKey) -> Optional[DiskCacheMetadata]:
        fl = key.chunk_hash[:2]
        sl = key.chunk_hash[2:4]
        path = self._key_to_path(key)
        if os.path.exists(path):
            try:
                return self.read_metadata(key, path, sl + fl)
            except UnsupportedMetadataVersion:
                logger.error(
                    f"Unsupported metadata version for {path}, ignoring")
        return None

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        """
        Convert key to path_name

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            returns the path name
        """
        assert self.path is not None
        hash = key.chunk_hash
        fl = hash[:2]
        sl = hash[2:4]
        key_str = key.to_string()
        assert "_" not in key_str, "key string should not contain `_`"
        return os.path.join(self.path, fl, sl,
                            key_str.replace("/", "_") + _DATA_FILE_SUFFIX)

    def submit_put_task(self, key: CacheEngineKey,
                        memory_obj: MemoryObj) -> Optional[Future]:
        assert memory_obj.tensor is not None

        self.memory_allocator.ref_count_up(memory_obj)

        with self.put_lock:
            self.put_tasks.add(key)

        future = asyncio.run_coroutine_threadsafe(
            self.async_save_bytes_to_disk(key, memory_obj), self.loop)
        return future

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        with self.update_lock:
            entry = self.dict.get(key)
        if entry is None:
            return None

        path = entry.path
        dtype = entry.dtype
        shape = entry.shape
        logger.info(f"Prefetching {key} from disk.")

        assert dtype is not None
        assert shape is not None
        future = asyncio.run_coroutine_threadsafe(
            self.async_load_bytes_from_disk(key, path, dtype, shape),
            self.loop)
        return future

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        with self.update_lock:
            entry = self.dict.get(key)
        if entry is None:
            return None

        with timing(self.stats, 'get_time', 'total_read_time'):
            path = entry.path
            dtype = entry.dtype
            shape = entry.shape
            assert dtype is not None
            assert shape is not None
            memory_obj = self.load_bytes_from_disk(key,
                                                   path,
                                                   dtype=dtype,
                                                   shape=shape)
        if self.stats:
            self.stats.get_count += 1
            self.stats.total_read_size += memory_obj.get_size()
        return memory_obj

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    async def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        path = self._key_to_path(key)

        fl = key.chunk_hash[:2]
        sl = key.chunk_hash[2:4]
        subdir = fl + sl
        if subdir not in self.subdirs:
            os.makedirs(os.path.join(self.path, fl, sl), exist_ok=True)
            self.subdirs.add(subdir)
        tmp = '.tmp' + rand_suffix(self.rand, 8)
        if self.use_thread_pool:
            metadata = await asyncio.to_thread(save_gds, self.dst_device_pci,
                                               path, tmp, kv_chunk)
        else:
            metadata = save_gds(self.dst_device_pci, path, tmp, kv_chunk)

        self.insert_key(key, memory_obj)
        self.memory_allocator.ref_count_down(memory_obj)

        task = asyncio.create_task(
            save_metadata(path + _METADATA_FILE_SUFFIX, tmp, metadata))
        self.save_metadata_tasks.add(task)
        task.add_done_callback(self.save_metadata_tasks.discard)

        with self.put_lock:
            self.put_tasks.remove(key)

    # TODO(Jiayi): use `bytes_read = await f.readinto(buffer)`
    # for better performance (i.e., fewer copy)
    async def async_load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Async load bytearray from disk.
        """
        memory_obj = self.memory_allocator.allocate(shape, dtype)
        if memory_obj is None:
            logger.debug("Memory allocation failed during async disk load.")
            return None
        assert memory_obj.tensor is not None
        assert memory_obj.tensor.is_cuda
        assert torch.device(self.dst_device) == torch.device(
            memory_obj.tensor.device)

        offset = metadata_max_size()
        ret = 0
        if self.use_thread_pool:
            ret = await asyncio.to_thread(load_gds, self.dst_device_pci, path,
                                          offset, memory_obj.tensor.data_ptr(),
                                          memory_obj.get_size())
        else:
            ret = load_gds(self.dst_device_pci, path, offset,
                           memory_obj.tensor.data_ptr(), memory_obj.get_size())
        if ret != memory_obj.get_size():
            if ret < 0:
                logger.error(
                    f"Error loading {path}: {ret}, was the entry GCed? Removing it from cache"
                )
                with self.update_lock:
                    self.dict.pop(key)
            else:
                # TODO(ilya): we should probably count errors and remove the entry
                # if it's a persistent problem
                logger.error(
                    f"Error loading {path}: got only {ret} bytes out of {memory_obj.get_size()}, ignoring"
                )
            self.memory_allocator.ref_count_down(memory_obj)
            return None

        return memory_obj

    # TODO(Jiayi): use memory allocator to redeuce cpu buffer allocation
    # TODO(Jiayi): the pinned cpu memory_obj should directly be passed into
    # gpu connector; this gpu buffer could be avoided
    def load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """
        memory_obj = self.memory_allocator.allocate(shape, dtype)
        if memory_obj is None:
            logger.debug("Memory allocation failed during sync disk load.")
            return None
        assert memory_obj.tensor is not None
        assert memory_obj.tensor.is_cuda
        assert torch.device(self.dst_device) == torch.device(
            memory_obj.tensor.device)

        offset = metadata_max_size()
        with timing(self.stats, 'gds_read_time', 'total_gds_read_time'):
            ret = load_gds(self.dst_device_pci, path, offset,
                           memory_obj.tensor.data_ptr(), memory_obj.get_size())
        if ret != memory_obj.get_size():
            if ret < 0:
                logger.error(
                    f"Error loading {path}: {ret}, was the entry GCed? Removing it from cache"
                )
                with self.update_lock:
                    self.dict.pop(key)
            else:
                # TODO(ilya): we should probably count errors and remove the entry
                # if it's a persistent problem
                logger.error(
                    f"Error loading {path}: got only {ret} bytes out of {memory_obj.get_size()}, ignoring"
                )
            self.memory_allocator.ref_count_down(memory_obj)
            return None
        return memory_obj

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path = self._key_to_path(key)
        size = memory_obj.get_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        with self.update_lock:
            self.dict[key] = DiskCacheMetadata(path, size, shape, dtype)

    def close(self):
        if not self.closed:
            self.closed = True
            if self.trace_file:
                self.write_trace(reenable=False)
            gds_api_cleanup()
            if self.stats:
                logger.info("Stats collected:")
                self.stats.display()

    def __del__(self):
        self.close()
