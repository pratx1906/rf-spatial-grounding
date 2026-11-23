"""Log GPU clocks, power, temperature, memory, and host RAM to CSV (SPEC 2.3, 2.5).

    python scripts/gpu_monitor.py --out runs/<run_id> [--interval 5] [--duration 0]

Library use (background thread for the lifetime of a run):
    with GpuMonitor(run_dir):
        train()
    GpuMonitor.summary(run_dir / "gpu_monitor.csv")  # medians reported next to throughput numbers
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil
import pynvml as nv

FIELDS = [
    "t_unix", "sm_clock_mhz", "mem_clock_mhz", "power_w", "temp_c", "util_pct",
    "vram_used_mb", "vram_total_mb", "ram_used_mb", "ram_total_mb", "throttle_reasons",
]


def _safe(fn, *args, default=None):
    try:
        return fn(*args)
    except nv.NVMLError:
        return default  # some queries are unsupported under WSL2; log blanks rather than fail


def sample(handle) -> dict:
    mem = _safe(nv.nvmlDeviceGetMemoryInfo, handle)
    util = _safe(nv.nvmlDeviceGetUtilizationRates, handle)
    power = _safe(nv.nvmlDeviceGetPowerUsage, handle)
    vm = psutil.virtual_memory()
    return {
        "t_unix": round(time.time(), 2),
        "sm_clock_mhz": _safe(nv.nvmlDeviceGetClockInfo, handle, nv.NVML_CLOCK_SM),
        "mem_clock_mhz": _safe(nv.nvmlDeviceGetClockInfo, handle, nv.NVML_CLOCK_MEM),
        "power_w": round(power / 1000, 1) if power is not None else None,
        "temp_c": _safe(nv.nvmlDeviceGetTemperature, handle, nv.NVML_TEMPERATURE_GPU),
        "util_pct": util.gpu if util else None,
        "vram_used_mb": mem.used // 2**20 if mem else None,
        "vram_total_mb": mem.total // 2**20 if mem else None,
        "ram_used_mb": (vm.total - vm.available) // 2**20,
        "ram_total_mb": vm.total // 2**20,
        "throttle_reasons": _safe(nv.nvmlDeviceGetCurrentClocksEventReasons, handle),
    }


class GpuMonitor:
    def __init__(self, run_dir: str | Path, interval_s: float = 5.0, device: int = 0):
        self.path = Path(run_dir) / "gpu_monitor.csv"
        self.interval_s = interval_s
        self.device = device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def _loop(self, duration_s: float = 0.0) -> None:
        nv.nvmlInit()
        try:
            handle = nv.nvmlDeviceGetHandleByIndex(self.device)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists()
            t_end = time.time() + duration_s if duration_s > 0 else float("inf")
            with self.path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                while not self._stop.is_set() and time.time() < t_end:
                    w.writerow(sample(handle))
                    f.flush()
                    self._stop.wait(self.interval_s)
        finally:
            nv.nvmlShutdown()

    def _safe_loop(self) -> None:
        try:
            self._loop()
        except Exception as e:  # noqa: BLE001 - monitoring must never kill or silently corrupt the run
            self.error = f"{type(e).__name__}: {e}"
            print(f"[gpu_monitor] stopped: {self.error}", file=sys.stderr)

    def __enter__(self):
        self._thread = threading.Thread(target=self._safe_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 2)

    @staticmethod
    def summary(csv_path: str | Path, since: float | None = None) -> dict:
        """Medians/peaks over the samples; `since` (unix time) restricts them to the loaded window (SPEC 2.3)."""
        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
        except FileNotFoundError:
            rows = []
        if since is not None:
            rows = [r for r in rows if float(r["t_unix"]) >= since]

        def med(key):
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]
            return round(statistics.median(vals), 1) if vals else None

        def peak(key):
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]
            return max(vals) if vals else None

        return {
            "n_samples": len(rows),
            "median_sm_clock_mhz": med("sm_clock_mhz"),
            "median_temp_c": med("temp_c"),
            "median_power_w": med("power_w"),
            "peak_vram_used_mb": peak("vram_used_mb"),
            "peak_ram_used_mb": peak("ram_used_mb"),
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
    a = p.parse_args()
    mon = GpuMonitor(a.out, a.interval)
    try:
        mon._loop(a.duration)
    except KeyboardInterrupt:
        pass
    print(GpuMonitor.summary(mon.path))


if __name__ == "__main__":
    main()
