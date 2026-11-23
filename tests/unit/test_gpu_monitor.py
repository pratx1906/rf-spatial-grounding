import csv

from scripts.gpu_monitor import FIELDS, GpuMonitor


def test_summary_of_missing_csv_is_all_none(tmp_path):
    s = GpuMonitor.summary(tmp_path / "nope.csv")
    assert s["n_samples"] == 0 and s["median_temp_c"] is None and s["peak_vram_used_mb"] is None


def test_summary_of_header_only_csv(tmp_path):
    p = tmp_path / "gpu_monitor.csv"
    with p.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    assert GpuMonitor.summary(p)["n_samples"] == 0


def test_summary_skips_unsupported_fields(tmp_path):
    p = tmp_path / "gpu_monitor.csv"
    rows = [
        {"sm_clock_mhz": 1700, "temp_c": 60, "power_w": "", "vram_used_mb": 1000, "ram_used_mb": 5},
        {"sm_clock_mhz": 1500, "temp_c": 80, "power_w": "", "vram_used_mb": 3000, "ram_used_mb": 9},
        {"sm_clock_mhz": 1600, "temp_c": 70, "power_w": "", "vram_used_mb": 2000, "ram_used_mb": 7},
    ]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    s = GpuMonitor.summary(p)
    assert s["median_sm_clock_mhz"] == 1600 and s["median_temp_c"] == 70
    assert s["median_power_w"] is None  # unsupported under WSL -> blank, not a crash
    assert s["peak_vram_used_mb"] == 3000


def test_monitor_failure_is_recorded_not_silent(tmp_path, monkeypatch):
    import scripts.gpu_monitor as gm

    def boom():
        raise RuntimeError("NVML unavailable")

    monkeypatch.setattr(gm.nv, "nvmlInit", boom)
    with GpuMonitor(tmp_path, interval_s=0.05) as mon:
        pass
    assert mon.error and "NVML unavailable" in mon.error
