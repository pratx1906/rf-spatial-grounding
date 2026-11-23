"""Estimate disk needed for precompute products, checkpoints, and traces; refuse if too little would remain (SPEC 2.5).

    python scripts/disk_budget.py                       # defaults from configs/env/machine.yaml
    python scripts/disk_budget.py --variants 16 --fine-area-fraction 1.0
    python scripts/disk_budget.py --check               # exit 2 if free space after the job < min_free_disk_fraction

Byte counts per stored element (uncompressed) are listed in BYTES so the estimate is auditable.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "env" / "machine.yaml"

BYTES = {
    # one AoA peak: angle f4, rel_power f4, delay f4, path_type u1, via_obj i4
    "aoa_peak": 17,
    # df_err stats per (sensor, cell, height): mean, std, p90 as f4
    "df_err": 12,
    # radio map: path loss f4 per (sensor, cell, height)
    "radio_map": 4,
    # visibility bitmask byte per (sensor, cell, height)
    "visibility": 1,
    # trust layer per (sensor, cell, band): spread_db, spread_deg, surrogate_unc, trust f4 + flags u1
    "trust": 17,
}


@dataclass
class Budget:
    extent_m: tuple[float, float] = (3000, 3000)
    coarse_grid_m: float = 20
    fine_grid_m: float = 10
    fine_area_fraction: float = 0.15
    heights: int = 3
    sensors: int = 6
    bands: int = 4
    variants: int = 8
    backends_path_loss: int = 5
    backends_aoa: int = 2
    aoa_top_k: int = 8
    compression_ratio: float = 2.0
    lora_ckpt_gb: float = 0.6
    traces_gb: float = 20.0
    keep_checkpoints: int = 4

    def cells(self) -> float:
        area = self.extent_m[0] * self.extent_m[1]
        coarse = area / self.coarse_grid_m**2
        fine = self.fine_area_fraction * area / self.fine_grid_m**2
        return coarse + fine

    def estimate_gb(self) -> dict[str, float]:
        per_sensor_cell = self.cells() * self.heights * self.sensors
        gb = 1e9 * self.compression_ratio
        aoa_one = per_sensor_cell * self.aoa_top_k * BYTES["aoa_peak"]
        nominal_band = (
            self.backends_aoa * aoa_one
            + per_sensor_cell * (BYTES["df_err"] + self.backends_path_loss * BYTES["radio_map"])
        )
        rf_cache = self.bands * nominal_band + per_sensor_cell * BYTES["visibility"]
        rf_cache += self.bands * self.cells() * self.sensors * BYTES["trust"]
        # world_cache: each variant stores the nominal-backend AoA and path loss only
        world_cache = self.variants * self.bands * (aoa_one + per_sensor_cell * BYTES["radio_map"])
        out = {
            "rf_cache": rf_cache / gb,
            "world_cache": world_cache / gb,
            "checkpoints": self.lora_ckpt_gb * self.keep_checkpoints,
            "traces": self.traces_gb,
        }
        out["total"] = sum(out.values())
        return {k: round(v, 2) for k, v in out.items()}


def load_defaults(path: Path = CONFIG) -> tuple[Budget, float, dict]:
    cfg = yaml.safe_load(path.read_text())
    m, pc = cfg["machine"], cfg["precompute"]
    keep = sum(m["keep_checkpoints"].values())
    names = {f.name for f in fields(Budget)}
    b = Budget(**{k: (tuple(v) if k == "extent_m" else v) for k, v in pc.items() if k in names}, keep_checkpoints=keep)
    return b, m["min_free_disk_fraction"], m["paths"]


def physical_targets(target: Path) -> list[Path]:
    """Disks that actually absorb the write.

    Under WSL2, a path on the Linux root (ext4) lives in a virtual disk file on the Windows C: drive, and its
    reported free space is the virtual disk's ceiling, not the real drive's. Check C: as well in that case.
    """
    targets = [target]
    in_wsl = "microsoft" in platform.uname().release.lower()
    if in_wsl and not str(target.resolve()).startswith("/mnt/") and Path("/mnt/c").is_dir():
        targets.append(Path("/mnt/c"))
    return targets


def check(est_total_gb: float, target: Path, min_free_fraction: float) -> tuple[bool, dict]:
    target.mkdir(parents=True, exist_ok=True)
    worst: dict | None = None
    for t in physical_targets(target):
        du = shutil.disk_usage(t)
        free_after = du.free / 1e9 - est_total_gb
        info = {
            "disk": str(t),
            "disk_total_gb": round(du.total / 1e9, 1),
            "disk_free_gb": round(du.free / 1e9, 1),
            "free_after_gb": round(free_after, 1),
            "free_after_fraction": round(free_after / (du.total / 1e9), 3),
            "min_free_fraction": min_free_fraction,
        }
        if worst is None or info["free_after_fraction"] < worst["free_after_fraction"]:
            worst = info
    return worst["free_after_fraction"] >= min_free_fraction, worst


def main(argv: list[str] | None = None) -> int:
    b, min_free, paths = load_defaults()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Budget):
        if f.name == "extent_m":
            p.add_argument("--extent-m", type=float, nargs=2, default=list(b.extent_m))
        else:
            # type from the annotation, not the default value: YAML `20` must not make a float field int-only
            p.add_argument("--" + f.name.replace("_", "-"), type={"int": int, "float": float}[f.type],
                           default=getattr(b, f.name))
    p.add_argument("--target", type=Path, default=REPO / paths["data"], help="directory that will hold the products")
    p.add_argument("--check", action="store_true", help="exit 2 if the job would leave too little free space")
    a = p.parse_args(argv)

    budget = Budget(**{f.name: (tuple(a.extent_m) if f.name == "extent_m" else getattr(a, f.name)) for f in fields(Budget)})
    est = budget.estimate_gb()
    ok, disk = check(est["total"], a.target, min_free)

    print(f"grid cells (coarse+fine): {budget.cells():,.0f}")
    for k, v in est.items():
        print(f"  {k:<12} {v:>9.2f} GB")
    print(f"disk {disk['disk']} (tightest for {a.target}): total {disk['disk_total_gb']} GB, free {disk['disk_free_gb']} GB, "
          f"free after job {disk['free_after_gb']} GB ({disk['free_after_fraction']:.1%}, floor {min_free:.0%})")
    if not ok:
        print("REFUSE: job would drop free disk below the floor. Reduce grid/variants or use the overflow rule (SPEC 2.5).")
        return 2 if a.check else 0
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
