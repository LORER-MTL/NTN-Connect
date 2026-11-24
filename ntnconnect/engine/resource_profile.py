# Resource profiling (stdlib only): wall/CPU time + peak memory -> resource_profile.json.
from __future__ import annotations
import json
import sys
import threading
import time
from pathlib import Path


def _win_mem_counters():
    """Return the raw PROCESS_MEMORY_COUNTERS struct on Windows, else None."""
    import ctypes
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    handle = k32.GetCurrentProcess()
    counters = _PMC()
    counters.cb = ctypes.sizeof(_PMC)
    if hasattr(k32, "K32GetProcessMemoryInfo"):
        fn = k32.K32GetProcessMemoryInfo
    else:
        fn = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    if fn(handle, ctypes.byref(counters), counters.cb):
        return counters
    return None


def current_rss_bytes() -> int | None:
    """Current (instantaneous) resident set size in bytes, or None.

    Unlike peak_rss_bytes this tracks the live working set, so a sequence of
    samples traces the memory profile over the run. Stdlib only (no psutil):
    WorkingSetSize on Windows, /proc/self/statm on Linux.
    """
    try:
        if sys.platform.startswith("win"):
            c = _win_mem_counters()
            return int(c.WorkingSetSize) if c is not None else None
        if sys.platform.startswith("linux"):
            import os
            with open("/proc/self/statm") as f:
                resident_pages = int(f.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        # macOS/BSD: no cheap stdlib current-RSS; fall back to the peak.
        return peak_rss_bytes()
    except Exception:
        return None


def peak_rss_bytes() -> int | None:
    """Return the process peak resident set size in bytes, or None if the
    platform does not expose it. Uses only the standard library.

    Note: this is a process-wide, monotonic peak (the high-water mark since
    the process started), so it is exact when one simulation runs per process
    and an upper bound when several run sequentially in the same process.
    """
    try:
        if sys.platform.startswith("win"):
            c = _win_mem_counters()
            return int(c.PeakWorkingSetSize) if c is not None else None
        else:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB, macOS/BSD report bytes.
            return int(ru * 1024) if sys.platform.startswith("linux") else int(ru)
    except Exception:
        return None


class RunProfiler:
    """Measure wall time and CPU time across a run.

    Usage:
        prof = RunProfiler().start()
        ... run ...
        prof.stop().write(sim_dir, extra={...})
    """

    def __init__(self, sample_interval_s: float = 1.0):
        self._t_wall0 = None
        self._t_cpu0 = None
        self.wall_s = None
        self.cpu_s = None
        self.peak_rss_bytes = None
        # Continuous resource trace over the run (elapsed_s, rss_mb, cpu_pct).
        self._sampler = ResourceSampler(sample_interval_s) if sample_interval_s else None

    def start(self) -> "RunProfiler":
        self._t_wall0 = time.perf_counter()
        self._t_cpu0 = time.process_time()
        if self._sampler is not None:
            self._sampler.start()
        return self

    def stop(self) -> "RunProfiler":
        if self._sampler is not None:
            self._sampler.stop()
        if self._t_wall0 is not None:
            self.wall_s = round(time.perf_counter() - self._t_wall0, 3)
            self.cpu_s = round(time.process_time() - self._t_cpu0, 3)
        self.peak_rss_bytes = peak_rss_bytes()
        return self

    def timeseries(self) -> list[dict]:
        return self._sampler.samples() if self._sampler is not None else []

    def as_dict(self, extra: dict | None = None) -> dict:
        d = {
            "wall_s": self.wall_s,
            "cpu_s": self.cpu_s,
            "peak_rss_mb": (round(self.peak_rss_bytes / 1e6, 1)
                            if self.peak_rss_bytes else None),
            "cpu_utilization": (round(self.cpu_s / self.wall_s, 2)
                                if (self.wall_s and self.cpu_s) else None),
            "platform": sys.platform,
        }
        if extra:
            d.update(extra)
        return d

    def write(self, sim_dir, extra: dict | None = None) -> Path | None:
        """Write resource_profile.json into sim_dir. Never raises.

        Also writes resource_timeseries.csv (elapsed_s, rss_mb, cpu_pct) when a
        continuous trace was sampled, so downstream analysis can plot memory and
        CPU as functions of run time.
        """
        try:
            path = Path(sim_dir) / "resource_profile.json"
            path.write_text(json.dumps(self.as_dict(extra), indent=2),
                            encoding="utf-8")
        except Exception:
            path = None
        try:
            ts = self.timeseries()
            if ts:
                import csv
                tp = Path(sim_dir) / "resource_timeseries.csv"
                with tp.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["elapsed_s", "rss_mb", "cpu_pct"])
                    w.writeheader(); w.writerows(ts)
        except Exception:
            pass
        return path


class ResourceSampler:
    """Background thread that samples live RSS + CPU utilisation over a run.

    Sampling runs in a daemon thread so it adds no work to the simulation loop.
    Each sample is (elapsed_s, rss_mb, cpu_pct): rss_mb is the instantaneous
    working set, cpu_pct is the process-CPU fraction consumed since the previous
    sample (0..N_cores*100). Stdlib only - no psutil.
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval = max(0.1, float(interval_s))
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self._last_wall = None
        self._last_cpu = None

    def start(self) -> "ResourceSampler":
        self._t0 = time.perf_counter()
        self._last_wall = self._t0
        self._last_cpu = time.process_time()
        self._thread = threading.Thread(target=self._loop, name="res-sampler",
                                        daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        now = time.perf_counter()
        cpu = time.process_time()
        dw = now - self._last_wall
        cpu_pct = (100.0 * (cpu - self._last_cpu) / dw) if dw > 0 else 0.0
        self._last_wall, self._last_cpu = now, cpu
        rss = current_rss_bytes()
        self._samples.append({
            "elapsed_s": round(now - self._t0, 2),
            "rss_mb": round(rss / 1e6, 1) if rss else None,
            "cpu_pct": round(cpu_pct, 1),
        })

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception:
                pass

    def stop(self) -> "ResourceSampler":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:                       # capture a final point at true end-of-run
            self._sample()
        except Exception:
            pass
        return self

    def samples(self) -> list[dict]:
        return list(self._samples)
