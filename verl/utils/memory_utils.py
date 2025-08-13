# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import gc
import logging
import inspect
import os
import time
import torch
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime

from verl.utils.device import get_torch_device

logger = logging.getLogger(__name__)


def aggressive_empty_cache(force_sync: bool = True, max_retries: int = 3) -> None:
    """
    More aggressive GPU memory cleanup function, tries to release PyTorch reserved but unallocated memory.

    Args:
        force_sync: Whether to force device synchronization
        max_retries: Maximum number of retries
    """
    device = get_torch_device()
    if not device.is_available():
        return

    for attempt in range(max_retries):
        # Record memory status before cleanup
        before_reserved = device.memory_reserved()
        before_allocated = device.memory_allocated()

        # Run garbage collection
        gc.collect()

        # Clear PyTorch cache
        device.empty_cache()

        # Force synchronization (optional)
        if force_sync:
            device.synchronize()

        # Record memory status after cleanup
        after_reserved = device.memory_reserved()
        after_allocated = device.memory_allocated()

        # Calculate freed memory
        reserved_freed = before_reserved - after_reserved
        allocated_freed = before_allocated - after_allocated

        logger.info(
            f"Memory cleanup attempt {attempt + 1}: Freed {reserved_freed / 1024**3:.2f} GB reserved, "
            f"{allocated_freed / 1024**3:.2f} GB allocated"
        )

        # Stop retrying if little memory was freed
        if reserved_freed < 1024**3:  # less than 1GB
            break


def reset_memory_stats() -> None:
    """Reset GPU memory statistics"""
    if get_torch_device().is_available():
        device = get_torch_device()
        device.reset_peak_memory_stats()
        device.reset_accumulated_memory_stats()


def get_memory_info() -> dict:
    """Get detailed GPU memory information"""
    if not get_torch_device().is_available():
        return {}

    device = get_torch_device()
    device_id = device.current_device()

    return {
        "total_memory_gb": device.get_device_properties(device_id).total_memory / 1024**3,
        "reserved_memory_gb": device.memory_reserved() / 1024**3,
        "allocated_memory_gb": device.memory_allocated() / 1024**3,
        "cached_memory_gb": (device.memory_reserved() - device.memory_allocated()) / 1024**3,
        "max_memory_allocated_gb": device.max_memory_allocated() / 1024**3,
        "max_memory_reserved_gb": device.max_memory_reserved() / 1024**3,
    }


def log_memory_usage(stage: str = "current") -> None:
    """Log GPU memory usage"""
    if not get_torch_device().is_available():
        return

    info = get_memory_info()
    logger.info(
        f"Memory usage [{stage}]: "
        f"Total: {info['total_memory_gb']:.2f} GB, "
        f"Allocated: {info['allocated_memory_gb']:.2f} GB, "
        f"Reserved: {info['reserved_memory_gb']:.2f} GB, "
        f"Cached: {info['cached_memory_gb']:.2f} GB"
    )


def optimize_memory_for_inference() -> None:
    """Optimize GPU memory usage for inference"""
    if not get_torch_device().is_available():
        return

    # Set a more aggressive memory allocation policy
    get_torch_device().set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory

    # Clear cache
    aggressive_empty_cache(force_sync=True)

    logger.info("Optimized GPU memory usage for inference")


def optimize_memory_for_training() -> None:
    """Optimize GPU memory usage for training"""
    if not get_torch_device().is_available():
        return

    # Set a moderate memory allocation policy
    get_torch_device().set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory

    # Clear cache
    aggressive_empty_cache(force_sync=False)

    logger.info("Optimized GPU memory usage for training")

def enable_memory_viz(
    trace_alloc_max_entries: int = 200_000,
    stack_depth: int = 32,
    context: str = "all",      # 'alloc' | 'state' | 'all'
    stacks: str = "all",       # 'python' | 'cpp'(少数版本) | 'all'
    devices=None,              # None=默认；或传 int / [int,...]
    record_context: bool = True,
    verbose: bool = True,
):
    """
    建议在任何大规模 CUDA 分配之前调用；DDP/多进程每个 rank 都要调用。
    该实现会按实际 PyTorch 版本的函数签名，自动选择可用参数。
    """
    f = torch.cuda.memory._record_memory_history
    params = set(inspect.signature(f).parameters.keys())

    def _one_call(dev_kw=None):
        kwargs = {}
        # 通用可选项
        if "context" in params: kwargs["context"] = context
        if "stacks"  in params: kwargs["stacks"]  = stacks
        # 条目数：新老名字兼容
        if "max_entries" in params:
            kwargs["max_entries"] = trace_alloc_max_entries
        elif "trace_alloc_max_entries" in params:
            kwargs["trace_alloc_max_entries"] = trace_alloc_max_entries
        # 栈深（只有部分版本支持）
        if "stack_depth" in params:
            kwargs["stack_depth"] = stack_depth
        # 设备：有的叫 device，有的叫 devices（很少）
        if dev_kw is not None:
            if "device" in params:
                kwargs["device"] = dev_kw
            elif "devices" in params:
                kwargs["devices"] = [dev_kw] if isinstance(dev_kw, int) else dev_kw
        # 旧版本需要 record_context
        if "record_context" in params:
            kwargs["record_context"] = record_context

        try:
            # 不显式传 enabled，以避免各版本类型差异
            f(**kwargs)
            return "native", kwargs
        except TypeError:
            # 极简降级（非常老的 legacy）
            try:
                if "trace_alloc_max_entries" in params and "record_context" in params:
                    f(enabled=True, trace_alloc_max_entries=trace_alloc_max_entries, record_context=True)
                    return "legacy", {"enabled": True, "trace_alloc_max_entries": trace_alloc_max_entries, "record_context": True}
                else:
                    f(enabled=True)
                    return "legacy-min", {"enabled": True}
            except Exception as e:
                raise

    # 调用：无设备参数 → 一次；列表 → 逐个设备调用
    if devices is None or isinstance(devices, (str, int, torch.device)):
        mode, used = _one_call(devices if devices is not None else None)
    else:
        mode, used = "multi-device", {}
        for d in list(devices):
            _mode, _used = _one_call(d)
            used[f"dev{d}"] = _used

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    if verbose:
        rank = int(os.environ.get("RANK", "0") or 0)
        print(f"[memory_viz][rank {rank}] recording enabled ({mode}); args={used}")

def dump_memory_viz(path: str = "mem_snapshot.pickle", verbose: bool = True):
    """在你想抓现场的位置调用（异常前后各一次更好）"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot(path)
    if verbose:
        print(f"[memory_viz] snapshot dumped to {path}")

    
@contextmanager
def cuda_mem_range(name: str):
    """
    让 Memory Viz 里显示一段命名区间，比如 'prefill', 'decode', 'dataloader'
    用法:
        with cuda_mem_range('prefill'):
            run_prefill()
    """
    try:
        torch.cuda.memory._push_range(name)
    except Exception:
        pass
    try:
        yield
    finally:
        try:
            torch.cuda.memory._pop_range()
        except Exception:
            pass
    
def dump_memory_snapshot(out_dir: str = "./mem_snapshots", tag: str = "snapshot") -> str:
    """
    生成一个可被 Memory Viz 加载的 .pickle 文件，返回文件路径
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    rank = os.environ.get("RANK", "0")
    pid = os.getpid()
    fname = f"{tag}_rank{rank}_pid{pid}_{ts}.pickle"
    path = os.path.join(out_dir, fname)
    try:
        if torch.cuda.is_available():
            # 避免还在队列里的 kernel 影响统计
            torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(path)
        print(f"[memory_viz] dumped: {path}")
    except Exception as e:
        print(f"[memory_viz][warn] dump failed: {e}")
    return path

class MemorySnapshotSampler:
    def __init__(self, interval_sec: int = 300, out_dir: str = "./mem_snapshots", tag: str = "periodic"):
        self.interval = interval_sec
        self.out_dir = out_dir
        self.tag = tag
        self._evt = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        print(f"[memory_viz] sampler start interval={self.interval}s")
        self._th.start()

    def stop(self):
        self._evt.set()
        self._th.join(timeout=3)

    def _run(self):
        while not self._evt.is_set():
            try:
                dump_memory_snapshot(self.out_dir, self.tag)
            except Exception as e:
                print(f"[memory_viz][warn] periodic dump failed: {e}")
            self._evt.wait(self.interval)

def dump_gpu_memory(device=None):
    """
    Dumps the GPU memory summary for a given device.

    Args:
        device (int, optional): The device index to dump memory for.
                                If None, dumps for all available devices.
                                Defaults to None.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    if device is not None:
        try:
            print(f"--- Memory Summary for device {device} ---")
            print(torch.cuda.memory_summary(device=device, abbreviated=False))
        except Exception as e:
            print(f"Could not get memory summary for device {device}: {e}")
    else:
        for i in range(torch.cuda.device_count()):
            try:
                print(f"--- Memory Summary for device {i} ---")
                print(torch.cuda.memory_summary(device=i, abbreviated=False))
            except Exception as e:
                print(f"Could not get memory summary for device {i}: {e}")