"""Free-memory guard for long eval runs on a small machine.

Embedding and reranking models are loaded one at a time, and runs stop with a
clear message when free RAM gets low, instead of pushing the machine into swap.
"""
import ctypes
import sys
from collections.abc import Callable

MIN_FREE_GB = 1.5


class LowMemoryError(RuntimeError):
    pass


def free_memory_gb() -> float | None:
    """Free physical memory in GB, or None on platforms where this is not implemented."""
    if sys.platform != "win32":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 1024 ** 3


def require_memory(min_gb: float = MIN_FREE_GB, free: Callable[[], float | None] = free_memory_gb) -> float | None:
    """Raise LowMemoryError if less than `min_gb` is free; return the free amount otherwise."""
    available = free()
    if available is not None and available < min_gb:
        raise LowMemoryError(f"Only {available:.1f} GB free; stopping because the floor is {min_gb:.1f} GB.")
    return available
