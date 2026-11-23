"""Where living-world variant scenes and caches live (no heavy imports: used from the Windows Sionna venv and inside
Blender too). side: world | belief | nominal (v2 only); stack: v1 (D-031) | v2 (RF phase 2).

Data versions (RF convergence plan 2026-10-23, Task 1): `SCENESCOUT_DATA_VERSION` in {v2 (default), v21, v3} picks
the roots of the v2 stack through ONE helper, `data_roots()`:
  v2   the frozen demo (demo-v2): data/scenarios, rf_cache_v2, world_cache_v2, rf_scene_blender (today's paths)
  v21  the S7 code on the current caches: scenarios in data/scenarios_v21; caches and scenes as v2 (read only)
  v3   converged caches on the simplified RF geometry: data/scenarios_v3, rf_cache_v3, world_cache_v3,
       rf_scene_blender_rf
v3 has two Blender RF scene sources (`SCENESCOUT_RF_GEOM` or data_roots(rf_geom=...)): rf (default) =
rf_scene_blender_rf (Task 3, flat merge), rfr = rf_scene_blender_rfr (Task 3b, plus the RF-relevance pass: thin
sub-wavelength components dropped, roof slabs collapsed). Only rf_scene changes; v2/v21 ignore the env var and refuse
an explicit rf_geom="rfr".
Shared by every version (world-side truth, never re-drawn): world_cache_v2/spectrum_env.yaml.
Not routed: envbuild/blender_objects.py (tool/console object catalogue) always reads the full rf_scene_blender.

Writers call `guard_write(path)`: the demo roots are refused unless --allow-demo-roots or
SCENESCOUT_ALLOW_DEMO_ROOTS=1 is given, and are ALWAYS refused when the data version is not v2.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "data" / "site_kandla"
ENV_VERSION = "SCENESCOUT_DATA_VERSION"
ENV_ALLOW = "SCENESCOUT_ALLOW_DEMO_ROOTS"
VERSIONS = ("v2", "v21", "v3")
ENV_RF_GEOM = "SCENESCOUT_RF_GEOM"
V3_RF_SCENES = {"rf": SITE / "rf_scene_blender_rf", "rfr": SITE / "rf_scene_blender_rfr"}
SENSORS = ("S1", "S2", "S3", "S4", "S5")


class DemoRootError(SystemExit):
    """A writer was about to touch a frozen demo root (exits the CLI with the message)."""


def data_version(version: str | None = None) -> str:
    v = version or os.environ.get(ENV_VERSION) or "v2"
    if v not in VERSIONS:
        raise ValueError(f"unknown data version {v!r} ({ENV_VERSION}); expected one of {VERSIONS}")
    return v


def rf_geom_name(geom: str | None = None) -> str:
    """The v3 Blender RF scene source: rf (Task 3) | rfr (Task 3b); argument, else SCENESCOUT_RF_GEOM, else rf."""
    g = geom or os.environ.get(ENV_RF_GEOM) or "rf"
    if g not in V3_RF_SCENES:
        raise ValueError(f"unknown RF geometry {g!r} ({ENV_RF_GEOM}); expected one of {tuple(V3_RF_SCENES)}")
    return g


def data_roots(version: str | None = None, rf_geom: str | None = None) -> dict[str, Path]:
    """Roots of one data version: scenarios, rf_cache (nominal + belief), world_cache (hidden worlds), rf_scene (the
    Blender RF export the scenes reference), track_model (the v2 track classifier), spectrum_env (shared).
    `rf_geom` (or SCENESCOUT_RF_GEOM) picks the v3 scene source (rf | rfr)."""
    v = data_version(version)
    if v != "v3" and rf_geom not in (None, "rf"):
        raise ValueError(f"rf_geom {rf_geom!r} is a v3 scene source; data version {v} uses rf_scene_blender")
    r = {"scenarios": REPO / "data" / "scenarios", "rf_cache": SITE / "rf_cache_v2",
         "world_cache": SITE / "world_cache_v2", "rf_scene": SITE / "rf_scene_blender",
         "track_model": REPO / "configs" / "track_class_v2.json",
         "spectrum_env": SITE / "world_cache_v2" / "spectrum_env.yaml"}
    if v == "v21":
        r.update(scenarios=REPO / "data" / "scenarios_v21", track_model=REPO / "configs" / "track_class_v21.json")
    elif v == "v3":
        r.update(scenarios=REPO / "data" / "scenarios_v3", rf_cache=SITE / "rf_cache_v3",
                 world_cache=SITE / "world_cache_v3", rf_scene=V3_RF_SCENES[rf_geom_name(rf_geom)],
                 track_model=REPO / "configs" / "track_class_v3.json")
    return r


def df_run_dir(sensor: str, version: str | None = None) -> Path:
    """scripts/df_eval.py --stack v2 output (read by coverage_build): runs/air7311_<S>_v2 (v2, v21) or _v3."""
    v = data_version(version)
    return REPO / "runs" / f"air7311_{sensor}_{'v3' if v == 'v3' else 'v2'}"


def task_tag(version: str | None = None, world_variant: str | None = None) -> str:
    """Seed tag in task ids (k15X-..., k15X21-..., k15X3W4-...): sets of different versions never collide."""
    v = data_version(version)
    return "X" + {"v2": "", "v21": "21", "v3": "3"}[v] + (world_variant or "")


def rel(p: Path) -> str:
    """Repo-relative POSIX string (meta.rf_cache_dir)."""
    return Path(os.path.abspath(p)).relative_to(Path(os.path.abspath(REPO))).as_posix()


def demo_roots() -> tuple[Path, ...]:
    """Everything that backs demo-v2 (frozen)."""
    d = data_roots("v2")
    return (d["scenarios"], d["rf_cache"], d["world_cache"], d["rf_scene"], d["track_model"],
            REPO / "data" / "demo_tasks_kandla.json", REPO / "data" / "demo_tasks_kandla_v2.json",
            *(df_run_dir(s, "v2") for s in SENSORS))


def is_demo_path(path: str | Path) -> Path | None:
    p = Path(os.path.abspath(path))
    for r in demo_roots():
        r = Path(os.path.abspath(r))
        if p == r or p.is_relative_to(r):
            return r
    return None


def allow_demo_env() -> bool:
    return os.environ.get(ENV_ALLOW, "") == "1"


def guard_write(path: str | Path, allow: bool = False, version: str | None = None) -> Path:
    """Return `path` if a writer may write it; raise DemoRootError for a demo root (always under v21/v3; under v2
    unless `allow` (--allow-demo-roots) or SCENESCOUT_ALLOW_DEMO_ROOTS=1)."""
    root = is_demo_path(path)
    if root is not None:
        v = data_version(version)
        if v != "v2":
            raise DemoRootError(f"refusing to write demo root {root} ({path}) under data version {v}: "
                                f"{v} outputs go to their own roots (envbuild.variant_roots.data_roots)")
        if not (allow or allow_demo_env()):
            raise DemoRootError(f"refusing to write demo root {root} ({path}): it backs demo-v2; pass "
                                f"--allow-demo-roots or set {ENV_ALLOW}=1 to rebuild it on purpose")
    return Path(path)


def add_allow_flag(ap) -> None:
    ap.add_argument("--allow-demo-roots", action="store_true",
                    help=f"allow writing the frozen demo-v2 roots (data version v2 only; env {ENV_ALLOW}=1)")


def scene_dir(vid: str, side: str, stack: str = "v1", version: str | None = None) -> Path:
    if stack == "v2":
        r = data_roots(version)
        return {"world": r["world_cache"] / "rf_scene" / vid, "nominal": r["rf_cache"] / "scene",
                "belief": r["rf_cache"] / "ensemble_scene" / vid}[side]
    return (SITE / "world_cache" / "rf_scene" / vid) if side == "world" else (SITE / "rf_cache" / "ensemble_scene" / vid)


def cache_root(vid: str, side: str, stack: str = "v1", version: str | None = None) -> Path:
    if stack == "v2":
        r = data_roots(version)
        return {"world": r["world_cache"] / "rf" / vid, "nominal": r["rf_cache"],
                "belief": r["rf_cache"] / "ensemble" / vid}[side]
    return (SITE / "world_cache" / "rf" / vid) if side == "world" else (SITE / "rf_cache" / "ensemble" / vid)


def lattice_root(root: Path, shift: int | None) -> Path:
    """Cache root of one shifted ray lattice (RF convergence Task 4, ruling P1-2; the Task 5 fallback traces each
    cache on several lattices): <root>/lattice/p<shift>; None = the root itself (the unshifted Sionna lattice).
    A sub dir, so the nominal/ensemble globs (ensemble/*/air7311_*) never pick it up as a variant."""
    return Path(root) if shift is None else Path(root) / "lattice" / f"p{int(shift)}"


def convergence_scene_dir(vid: str, side: str, rf_geom: str | None = None, version: str | None = None) -> Path:
    """v3 scene of one variant on one Blender RF scene source, for the convergence sweep (Task 4): both geometries
    (rf, rfr) side by side under the v3 roots, never the demo roots. World (hidden) variants stay under
    world_cache_v3 (truth guard); nominal and belief scenes under rf_cache_v3."""
    v = data_version(version)
    if v != "v3":
        raise ValueError(f"convergence scenes are v3 only (data version {v})")
    g = rf_geom_name(rf_geom)
    r = data_roots(v, g)
    return (r["world_cache"] if side == "world" else r["rf_cache"]) / "convergence_scene" / g / vid
