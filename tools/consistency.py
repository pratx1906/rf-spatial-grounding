"""sensor_consistency (plan 2026-10-28 T3): per-node DF bearing bias from the observations alone. Belief side only.

Method (truth-free self-calibration from opportunistic emitters):
  1. For every (track, time) epoch take each node's strongest-power hypothesis as its bearing (the direct path most
     of the time; weaker multipath ghosts are ignored).
  2. Leave one out: for node s, triangulate the epoch from the OTHER nodes only (>= 2 of them) and take s's signed
     residual = observed bearing - bearing from s to that fix. The track_points estimate is not used because the
     tracker fused node s into it, which would pull the reference towards s's own error.
  3. Fixes outside the site, with a 2-sigma major axis > MAX_FIX_A_M, or closer than MIN_RANGE_M to s are skipped;
     residuals beyond OUTLIER_DEG (a ghost taken as strongest) count as outliers, not as bias evidence.
  4. Per node: bias = median residual, spread = 1.4826 * MAD, n = inliers;
     suspect = n >= MIN_N and |bias| > max(BIAS_FLOOR_DEG, 2.5 * spread / sqrt(n)).
  5. A biased node also tilts the fixes used to test the others (two nodes that view the same coordinate both look
     biased under plain leave-one-out). So the fit is a greedy sparse self-calibration (_fit): declare the node whose
     de-biasing (inside the OTHER nodes' fixes) most lowers the total unexplained bias, plus a 2-degree price per
     declared node; repeat for at most (n_nodes - 1) // 2 nodes; refine (REFINE_PASSES). A node's own residual always
     uses its raw bearings. Synthetic check: 64/64 single-node +-5 deg cases recovered (runs/dbg/t3_consistency_sweep.py).
The window is the whole scenario, or at_time_s +- WINDOW_S; track_ref restricts the epochs to one track.
"""

from __future__ import annotations

import math

import numpy as np

from sim.belief.rf_cache import triangulate
from sim.geometry import angle_diff_deg, bearing_deg
from tools.df import _sigma
from tools.errors import ToolError
from tools.registry import ToolContext, ToolOutput, register
from tools.site import _entity
from tools.types import Strict, r1

WINDOW_S = 90.0
MAX_FIX_A_M = 400.0
MIN_RANGE_M = 50.0
SITE_HALF_M = 3200.0
OUTLIER_DEG = 30.0
MIN_N = 5
BIAS_FLOOR_DEG = 2.0
REFINE_PASSES = 2


class SensorConsistencyArgs(Strict):
    sensor_ref: str | None = None
    track_ref: str | None = None
    at_time_s: float | None = None


def _epochs(ctx: ToolContext, track: str | None, t0: float | None, t1: float | None) -> list[dict]:
    """[{sensor: (bearing_deg, sigma_deg)}] per (track, time) epoch with >= 3 nodes; strongest hyp per node."""
    q = ("SELECT track_id, t_s, sensor_id, arg_max(bearing_deg, power_db * 1000 + snr_db), "
         "arg_max(snr_db, power_db * 1000 + snr_db) FROM bearings WHERE 1 = 1")
    params: list = []
    if track is not None:
        q += " AND track_id = ?"
        params.append(track)
    if t0 is not None:
        q += " AND t_s BETWEEN ? AND ?"
        params += [t0, t1]
    q += " GROUP BY track_id, t_s, sensor_id ORDER BY track_id, t_s, sensor_id"
    ep: dict[tuple, dict] = {}
    for tid, t, s, b, snr in ctx.site.cursor().execute(q, params).fetchall():
        if s in ctx.site.sensors:
            ep.setdefault((tid, t), {})[s] = (float(b), _sigma(float(snr)))
    return [e for e in ep.values() if len(e) >= 3]


def _residuals(ctx: ToolContext, epochs: list[dict], corr: dict[str, float]) -> dict[str, tuple[list[float], int]]:
    sens = ctx.site.sensors
    out: dict[str, tuple[list[float], int]] = {}
    for e in epochs:
        for s, (b, _) in e.items():
            obs = [(sens[o]["pose"]["x_m"], sens[o]["pose"]["y_m"], (bo - corr.get(o, 0.0)) % 360.0, sg)
                   for o, (bo, sg) in e.items() if o != s]
            fix = triangulate(obs)
            if fix is None:
                continue
            xy, cov, _ = fix
            sx, sy = sens[s]["pose"]["x_m"], sens[s]["pose"]["y_m"]
            a_m = 2 * math.sqrt(max(float(np.linalg.eigvalsh(cov)[-1]), 1.0))
            if (not np.all(np.abs(xy) < SITE_HALF_M) or a_m > MAX_FIX_A_M
                    or math.hypot(xy[0] - sx, xy[1] - sy) < MIN_RANGE_M):
                continue
            res, n_out = out.setdefault(s, ([], 0))
            r = angle_diff_deg(b, bearing_deg(sx, sy, float(xy[0]), float(xy[1])))
            if abs(r) > OUTLIER_DEG:
                out[s] = (res, n_out + 1)
            else:
                res.append(r)
    return out


def _stats(res: dict[str, tuple[list[float], int]]) -> dict[str, dict]:
    st = {}
    for s, (r, n_out) in res.items():
        n = len(r)
        if n == 0:
            st[s] = {"bias": None, "spread": None, "n": 0, "n_outliers": n_out, "suspect": False}
            continue
        a = np.array(r)
        bias = float(np.median(a))
        spread = 1.4826 * float(np.median(np.abs(a - bias)))
        suspect = n >= MIN_N and abs(bias) > max(BIAS_FLOOR_DEG, 2.5 * spread / math.sqrt(n))
        st[s] = {"bias": bias, "spread": spread, "n": n, "n_outliers": n_out, "suspect": suspect}
    return st


def _cost(st: dict[str, dict], corr: dict[str, float]) -> float:
    """Total unexplained bias: sum over nodes of |median residual - the correction applied to that node| plus
    BIAS_FLOOR_DEG per node declared biased (a sparsity prior: most nodes are calibrated)."""
    return (sum(abs(v["bias"] - corr.get(s, 0.0)) for s, v in st.items() if v["bias"] is not None)
            + BIAS_FLOOR_DEG * len(corr))


def _fit(ctx: ToolContext, epochs: list[dict]) -> dict[str, dict]:
    """Greedy sparse self-calibration. A biased node tilts the fixes used to test the others, and with redundant
    geometry (two nodes viewing the same coordinate) plain leave-one-out flags both. So: try declaring each node
    biased (de-bias its bearings inside the others' fixes), keep the choice that most lowers _cost, repeat for at most
    (n_nodes - 1) // 2 nodes, then refine the kept corrections. Reported residuals always use the node's raw bearings."""
    corr: dict[str, float] = {}
    st = _stats(_residuals(ctx, epochs, corr))
    cost = _cost(st, corr)
    for _ in range(max(1, (len(st) - 1) // 2)):
        best = None
        for s, v in st.items():
            if s in corr or v["bias"] is None or abs(v["bias"]) <= BIAS_FLOOR_DEG / 2:
                continue
            trial = {**corr, s: v["bias"]}
            st_t = _stats(_residuals(ctx, epochs, trial))
            c = _cost(st_t, trial)
            if c < cost and (best is None or c < best[0]):
                best = (c, trial, st_t)
        if best is None:
            break
        cost, corr, st = best
    for _ in range(REFINE_PASSES):
        new = {s: st[s]["bias"] for s in corr if st[s]["bias"] is not None}
        if all(abs(new[s] - corr[s]) < 0.05 for s in new):
            break
        corr = new
        st = _stats(_residuals(ctx, epochs, corr))
    return st


@register("sensor_consistency", "B", "Per-node DF bearing bias from leave-one-out fixes over all tracks; flags suspect nodes.",
          SensorConsistencyArgs, opt_in=True)
def sensor_consistency(ctx: ToolContext, a: SensorConsistencyArgs) -> ToolOutput:
    want = None
    if a.sensor_ref:
        kind, ent, _ = _entity(ctx, a.sensor_ref)
        if kind != "sensor":
            raise ToolError("INVALID_ARGS", f"{a.sensor_ref} is not a sensor ref", "Pass sensor: refs, e.g. sensor:S1.")
        want = ent["id"]
    track = None
    if a.track_ref:
        kind, track, _ = _entity(ctx, a.track_ref)
        if kind != "trk":
            raise ToolError("INVALID_ARGS", f"{a.track_ref} is not a track ref", "Pass a trk: ref from track_search.")
    t0 = t1 = None
    if a.at_time_s is not None:
        t0, t1 = max(0.0, a.at_time_s - WINDOW_S), a.at_time_s + WINDOW_S
    epochs = _epochs(ctx, track, t0, t1)
    if not epochs:
        raise ToolError("NOT_FOUND", "no epochs with bearings from 3 or more DF nodes in this window",
                        "Widen the window: drop at_time_s or track_ref.")
    st = _fit(ctx, epochs)
    nodes = [{"sensor_ref": f"sensor:{s}", "bias_deg": r1(v["bias"]), "spread_deg": r1(v["spread"]), "n": v["n"],
              "n_outliers": v["n_outliers"], "suspect": v["suspect"]}
             for s, v in sorted(st.items()) if want is None or s == want]
    if want is not None and not nodes:
        nodes = [{"sensor_ref": f"sensor:{want}", "bias_deg": None, "spread_deg": None, "n": 0, "n_outliers": 0,
                  "suspect": False}]
    suspects = [f"sensor:{s}" for s, v in sorted(st.items()) if v["suspect"]]
    data = {"method": "leave_one_out_triangulation", "window_s": [r1(t0), r1(t1)] if t0 is not None else None,
            "n_epochs": len(epochs), "nodes": nodes, "suspect_refs": suspects}
    refs = [n["sensor_ref"] for n in nodes] + [s for s in suspects if want is None and s not in
                                               [n["sensor_ref"] for n in nodes]]
    if track is not None:
        data["track_ref"] = f"trk:{track}"
        refs.append(f"trk:{track}")
    return ToolOutput(data, refs, {
        "residual": "node bearing minus bearing to a fix from the other nodes (deg, + = clockwise)",
        "bias_deg": "median residual; spread_deg = 1.4826 x MAD; n = inlier epochs (|residual| <= 30)",
        "suspect": "|bias| > max(2, 2.5 x spread / sqrt(n)) with n >= 5"})
