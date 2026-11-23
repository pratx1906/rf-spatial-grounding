"""AUDIT_CODE C4 fix, offline (world side, CPU): DF hypothesis labels of a hardware-mismatch world compared the BIASED
peak with UNBIASED path angles (observe_v2 adds each node's heading bias to every peak; observe_air7311._label matches
within 5 deg). This rewrites the labels on `az - heading_bias_deg` (truth/world_variant.yaml) into a copy of the
world dir with suffix `d` (kandla_s15_X_M2 -> kandla_s15_X_M2d); observations are copied byte for byte, the old dir is
kept. Task sets built from the copy carry the tag M<k>d.

    SCENESCOUT_DATA_VERSION=v3 python scripts/relabel_debias.py --dirs kandla_s15_X_M2 kandla_s16_X_M2
    python scripts/relabel_debias.py --check --dirs kandla_s15_X_M0     # bias 0: must reproduce the stored labels

Limits: the label input is the stored bearing (rounded to 0.1 deg; the build used the unrounded peak), so a peak
within 0.05 deg of the 5-deg gate can flip (the --check agreement measures it). M3 (use-time channel phase error)
shifts peaks by an angle-dependent amount, not a constant; its labels are re-derived with the (nominal) heading bias
only, and world_variant.yaml says so.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envbuild.variant_roots import data_roots  # noqa: E402
from sim.geometry import bearing_deg  # noqa: E402
from sim.obs_model.observe_air7311 import _label  # noqa: E402
from sim.obs_model.observe_v2 import _Paths  # noqa: E402


def band_at(bands: list, t: float) -> str:
    cur = sorted(bands, key=lambda x: x[0])[0][1]
    for t0, b in sorted(bands, key=lambda x: x[0]):
        if t0 <= t:
            cur = b
    return cur


def relabel(d: Path, debias: bool = True) -> tuple[list[tuple], dict]:
    meta = yaml.safe_load((d / "meta.yaml").read_text())
    w_root = REPO / meta["rf_cache_dir"]
    sens = {s["id"]: s["pose"] for s in yaml.safe_load((d / "sensors.yaml").read_text())["sensors"]}
    wv = d / "truth" / "world_variant.yaml"
    nodes = (yaml.safe_load(wv.read_text()) or {}).get("nodes", {}) if wv.exists() else {}
    bias = {sid: (float(v.get("heading_bias_deg", 0.0)) if debias else 0.0) for sid, v in nodes.items()}
    tr = duckdb.connect(str(d / "truth" / "truth.duckdb"), read_only=True)
    em = {e: json.loads(b) for e, b in tr.execute("SELECT emitter_id, bands FROM emitters").fetchall()}
    tl = {t: e for t, e, _ in tr.execute("SELECT * FROM track_labels").fetchall() if e in em}
    pts = {}
    for e, t, x, y, z in tr.execute("SELECT emitter_id, t_s, x_m, y_m, z_m FROM emitter_points ORDER BY 1, 2").fetchall():
        pts.setdefault(e, []).append((t, x, y, z))
    old = dict((h, (p, v)) for h, p, v in tr.execute("SELECT hyp_id, path_type, via_obj FROM hyp_labels").fetchall())
    tr.close()
    pts = {e: np.array(v) for e, v in pts.items()}
    ob = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    bear = ob.execute("SELECT hyp_id, track_id, sensor_id, t_s, bearing_deg FROM bearings").fetchall()
    ob.close()
    paths: dict = {}
    out = []
    for hid, tid, sid, t, az in bear:
        lab, via = "spurious", None
        e = tl.get(tid)
        if e is not None:
            p = pts[e]
            i = int(np.argmin(np.abs(p[:, 0] - t)))
            if abs(p[i, 0] - t) <= 2.0:
                x, y, z = p[i, 1:]
                key = (sid, band_at(em[e], t))
                if key not in paths:
                    paths[key] = _Paths(w_root, *key)
                pc = paths[key]
                cell = pc.at(x, y, z) if pc.ok else None
                if cell is not None:
                    zz = pc.z
                    h = zz["h_re"][cell] + 1j * zz["h_im"][cell]
                    paz = zz["az_deg"][cell].copy()
                    s = sens[sid]
                    paz[zz["ptype"][cell] == 0] = bearing_deg(s["x_m"], s["y_m"], x, y)
                    lab, via = _label((az - bias.get(sid, 0.0)) % 360.0, paz, zz["ptype"][cell], zz["via"][cell],
                                      zz["names"], np.abs(h) ** 2)
        out.append((hid, lab, via))
    agree = sum(old.get(h, (None,))[0] == lab for h, lab, _ in out)
    los_old = sum(old.get(h, (None,))[0] == "LOS" for h, _, _ in out)
    los_new = sum(lab == "LOS" for _, lab, _ in out)
    stats = {"dir": d.name, "bearings": len(out), "agree": agree, "agree_frac": round(agree / max(len(out), 1), 4),
             "los_old": los_old, "los_new": los_new, "bias_deg": {k: round(v, 3) for k, v in bias.items()}}
    return out, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="world dir names under the data version's scenario root")
    ap.add_argument("--suffix", default="d")
    ap.add_argument("--check", action="store_true", help="no debias, no write: agreement with the stored labels")
    a = ap.parse_args(argv)
    root = data_roots()["scenarios"]
    for name in a.dirs:
        src = root / name
        labels, st = relabel(src, debias=not a.check)
        if a.check:
            print(json.dumps({"check": True, **st}))
            continue
        dst = root / (name + a.suffix)
        if dst.exists():
            raise SystemExit(f"{dst} exists; refusing to overwrite")
        shutil.copytree(src, dst)
        con = duckdb.connect(str(dst / "truth" / "truth.duckdb"))
        con.execute("DELETE FROM hyp_labels")
        con.executemany("INSERT INTO hyp_labels VALUES (?, ?, ?)", labels)
        con.close()
        wv = dst / "truth" / "world_variant.yaml"
        if wv.exists():
            y = yaml.safe_load(wv.read_text())
            y["labels"] = {"debiased": True, "source_dir": name, "rule": "label on az - heading_bias_deg (AUDIT_CODE C4)",
                           "agree_with_biased_labels": st["agree_frac"], "los_old": st["los_old"],
                           "los_new": st["los_new"]}
            if (y.get("use_phase_sigma_deg") or 0) > 0:
                y["labels"]["note"] = "use-time phase error (M3) not removable by a constant; heading bias only"
            wv.write_text(yaml.safe_dump(y, sort_keys=False))
        print(json.dumps({"written": str(dst.relative_to(REPO)), **st}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
