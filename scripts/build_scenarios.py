"""Build Kandla scenarios through the AIR7311 observation model (WSL, GPU; takes the GPU lock).

    python scripts/build_scenarios.py --seeds 0-16

Seed 0 -> data/site_kandla (the demo world); seed N -> data/scenarios/kandla_s<N>/ (full SPEC 4 product set; the
RF cache is shared through meta.rf_cache_dir). Splits by seed (SPEC 8.2): 1-12 train, 13-14 val, 15-16 and 51-60 test.

Data versions (SCENESCOUT_DATA_VERSION, envbuild.variant_roots.data_roots): --obs v2 sets go to the version's
scenario root (v2 data/scenarios, v21 data/scenarios_v21, v3 data/scenarios_v3). The demo roots (data/scenarios,
data/site_kandla's v2 caches) are written only with --allow-demo-roots (or SCENESCOUT_ALLOW_DEMO_ROOTS=1), and never
under v21/v3.

Mismatch worlds (plan 2026-10-28 T6, audit B1): --hw-override configs/hardware/mismatch/M2.yaml --tag M2 deep-merges
the file into the world-side hardware config and writes kandla_s<N>_X_M2 (belief side, meta.rf_cache_dir and caches
nominal; tag, override and realised per-node errors in truth/world_variant.yaml only). Every build refuses an
existing output dir (--allow-existing re-enables rebuilding an untagged dir in place).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envbuild.variant_roots import add_allow_flag, data_version, guard_write  # noqa: E402
from scripts.gpu_lock import exit_if_busy, gpu_lock  # noqa: E402


def seed_dir(seed: int) -> Path:
    return REPO / "data" / "site_kandla" if seed == 0 else REPO / "data" / "scenarios" / f"kandla_s{seed}"


def variant_dir(seed: int, variant: str) -> Path:
    return REPO / "data" / "scenarios" / f"kandla_s{seed}_{variant}"


def v2_dir(seed: int, variant: str | None, version: str | None = None) -> Path:
    """RF phase 2 observation sets: kandla_s<N>_X (v2 nominal world) or kandla_s<N>_X_<W> (hidden world), under the
    scenario root of the data version (SCENESCOUT_DATA_VERSION: v2 data/scenarios, v21/v3 data/scenarios_v21|_v3)."""
    from envbuild.variant_roots import data_roots

    return data_roots(version)["scenarios"] / (f"kandla_s{seed}_X" + (f"_{variant}" if variant else ""))


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def load_override(path: str | Path) -> dict:
    """A mismatch-world hardware override (a partial air7311.yaml; configs/hardware/mismatch/M*.yaml)."""
    ov = yaml.safe_load(Path(path).read_text())
    if not isinstance(ov, dict) or not ov:
        raise SystemExit(f"{path}: empty or not a mapping")
    return ov


def refuse_existing(outs: dict) -> None:
    """Never build into an existing scenario dir (data roots are never overwritten; plan 2026-10-28)."""
    have = {s: str(o) for s, o in outs.items() if Path(o).exists()}
    if have:
        raise SystemExit(f"output dirs exist (refusing to overwrite; pick a new --tag or remove them): {have}")


@exit_if_busy
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-16")
    ap.add_argument("--world-variant", default=None, help="RF living-world variant (configs/rf_variants.yaml, D-031)")
    ap.add_argument("--obs", choices=["v1", "v2"], default="v1", help="v2: simulated IQ + detection-driven tracking")
    ap.add_argument("--cast", choices=["default", "varied", "newarea", "na_u", "na_f"], default="default",
                    help="emitter cast (sim/world/kandla_scenario.make_emitters); varied: test seeds only (S7.3)")
    ap.add_argument("--check-roots", action="store_true", help="print the output dirs, check the root guard, exit")
    ap.add_argument("--hw-override", default=None,
                    help="mismatch world: YAML deep-merged into the world-side hardware config (needs --tag, --obs v2)")
    ap.add_argument("--tag", default=None, help="mismatch-world tag (M1..M4): output dir kandla_s<N>_X_<tag>")
    ap.add_argument("--allow-existing", action="store_true",
                    help="rebuild an existing untagged dir in place (never with --tag)")
    add_allow_flag(ap)
    a = ap.parse_args(argv)
    seeds = parse_seeds(a.seeds)
    override = None
    if a.hw_override or a.tag:
        if not (a.hw_override and a.tag):
            raise SystemExit("--hw-override and --tag go together")
        if a.obs != "v2" or a.world_variant:
            raise SystemExit("--hw-override builds are --obs v2 on the nominal world (no --world-variant)")
        if not re.fullmatch(r"M[0-9A-Za-z_]*", a.tag):
            raise SystemExit(f"--tag {a.tag!r}: mismatch tags start with M (M1..M4)")
        if a.allow_existing:
            raise SystemExit("--allow-existing never applies to a mismatch build")
        if 0 in seeds:
            raise SystemExit("mismatch worlds never rebuild seed 0")
        override = load_override(a.hw_override)
    if a.world_variant and 0 in seeds:      # v2 nominal seed 0 writes kandla_s0_X, not the demo world
        raise SystemExit("world variants never rebuild seed 0 (the demo world)")
    version = data_version()
    if version != "v2" and a.obs != "v2":
        raise SystemExit(f"data version {version} holds v2-observation sets only (--obs v2)")
    if a.cast != "default":
        from scripts.make_tasks import split_of

        bad = [s for s in seeds if split_of(s) != "test"]
        if bad:
            raise SystemExit(f"--cast {a.cast} is for test seeds only; refused seeds {bad}")

    def out_of(seed: int) -> Path:
        if a.obs == "v2":
            return v2_dir(seed, a.tag or a.world_variant)
        return variant_dir(seed, a.world_variant) if a.world_variant else seed_dir(seed)

    outs = {seed: guard_write(out_of(seed), a.allow_demo_roots) for seed in seeds}     # refuse before any work
    if a.check_roots:
        print(json.dumps({"version": version, "cast": a.cast, "outs": {s: str(o) for s, o in outs.items()}}))
        return 0
    if not a.allow_existing:
        refuse_existing(outs)
    from envbuild.site_kandla import write_site

    with gpu_lock("df_sim"):
        for seed in seeds:
            t = time.time()
            out = outs[seed]
            out.mkdir(parents=True, exist_ok=a.allow_existing)
            m = write_site(seed, out, rf_backend="air7311", world_variant=a.world_variant, obs_version=a.obs,
                           cast=a.cast, allow_demo_roots=a.allow_demo_roots, hw_override=override,
                           variant_tag=a.tag)
            print(f"seed {seed}: {json.dumps(m['counts'])} {m['version_hash']} ({time.time() - t:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
