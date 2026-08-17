"""Wall-clock, memory and device recording.

Fit and predict are timed separately and never summed: a zero-shot foundation
model has no fit stage and a real predict cost, and that asymmetry is a result,
not an accounting detail.
"""
import os
import platform
import resource
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field


def peak_rss_mb() -> float:
    """Peak resident memory for this process.

    tracemalloc only counts Python allocations, so a model that allocates in C
    would look free. Each model runs in its own process, which makes the
    process-level peak a clean per-model figure.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return peak / divisor


def resolve_device(preference: str = "auto", available: set | None = None) -> str:
    if available is None:
        available = _available_devices()

    if preference == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if candidate in available:
                return candidate
        return "cpu"

    if preference not in available:
        raise RuntimeError(
            f"device {preference!r} was requested but is not available "
            f"(available: {sorted(available)})")
    return preference


def _available_devices() -> set:
    devices = {"cpu"}
    try:
        import torch
    except ImportError:
        return devices
    if torch.cuda.is_available():
        devices.add("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.add("mps")
    return devices


def synchronize(device: str) -> None:
    """Block until queued device work finishes, so timers measure compute
    rather than async launch."""
    if device == "cpu":
        return
    # Never import torch here: a lightgbm worker must not load it.
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def device_info(device: str) -> dict:
    info = {
        "device": device,
        "gpu_model": None,
        "cpu_count": os.cpu_count() or 1,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    torch = sys.modules.get("torch")
    if torch is None:
        return info

    info["torch_version"] = torch.__version__
    if device == "cuda" and torch.cuda.is_available():
        info["gpu_model"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    elif device == "mps":
        info["gpu_model"] = f"Apple {platform.machine()} (MPS)"
    return info


@dataclass
class Measurement:
    seconds: float = 0.0
    peak_memory_mb: float = 0.0
    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    @staticmethod
    def combine(fit: "Measurement", predict: "Measurement", device: str,
                n_series: int, n_params: int | None = None) -> dict:
        return {
            "fit_seconds": fit.seconds,
            "predict_seconds": predict.seconds,
            "peak_memory_mb": max(fit.peak_memory_mb, predict.peak_memory_mb),
            "n_series": n_series,
            "n_params": n_params,
            **device_info(device),
        }


@contextmanager
def measure(device: str = "cpu"):
    """Time a block, tracking peak allocation. Reports even if the block raises."""
    m = Measurement(device=device)
    tracing = not tracemalloc.is_tracing()
    if tracing:
        tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[1]

    start = time.perf_counter()
    try:
        yield m
    finally:
        synchronize(device)
        m.seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1]
        m.peak_memory_mb = max(peak - baseline, 0) / 1024 ** 2
        if tracing:
            tracemalloc.stop()
