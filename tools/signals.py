"""Family S: receiver-level signal analysis on v2 observation sets (RF phase 2, Task 9).

Reads only the belief-side tables written by the v2 observation model: bursts (per detected burst: frequency,
bandwidth, duration, protocol estimate, bearing, SNR, time of arrival) and dwells (receiver-measured noise floor).
On v1 observation sets these tables do not exist and the tools say so.
"""

from __future__ import annotations

import math
from typing import Annotated

import numpy as np
from pydantic import Field

from sim.belief.tdoa import match_toas as _match_toas
from sim.belief.tdoa import tdoa_fix as _tdoa_fix
from tools.errors import ToolError
from tools.registry import ToolContext, ToolOutput, register
from tools.tracks import _track
from tools.types import Band, Strict, pt, r1

UPLINK_OF = {"ocusync_like": {"gfsk"}, "wifi": {"wifi"}, "lora": {"lora"}, "gfsk": {"gfsk", "lora"},
             "fm_video": {"lora"}, "unknown": {"gfsk", "lora", "wifi", "unknown"}}
C0 = 299_792_458.0


def _need_v2(ctx: ToolContext, table: str) -> None:
    ok = ctx.site.cursor().execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                                   [table]).fetchone()[0]
    if not ok:
        raise ToolError("OUT_OF_COVERAGE", f"this observation set has no {table} table (receiver-level data)",
                        "Use track_detail / df_hypotheses; signal tools need a v2 (simulated IQ) observation set.")


class SignalDetailArgs(Strict):
    track_ref: str
    at_time_s: Annotated[float, Field(ge=0)] | None = None


@register("signal_detail", "S", "Bursts of a track: protocol estimate, bandwidth, hop pattern, timing, SNR.",
          SignalDetailArgs)
def signal_detail(ctx: ToolContext, a: SignalDetailArgs) -> ToolOutput:
    _need_v2(ctx, "bursts")
    tid = _track(ctx, a.track_ref)
    q = "SELECT t_s, sensor_id, f_mhz, bw_khz, dur_us, protocol_est, protocol_conf, snr_db FROM bursts WHERE track_id = ?"
    args: list = [tid]
    if a.at_time_s is not None:
        q += " AND abs(t_s - ?) <= 10"
        args.append(a.at_time_s)
    rows = ctx.site.cursor().execute(q, args).fetchall()
    if not rows:
        raise ToolError("NOT_FOUND", f"no bursts associated with trk:{tid}" + (f" near t={a.at_time_s:g}" if a.at_time_s else ""),
                        "Try without at_time_s, or check the track with track_detail.")
    protos = {}
    for r in rows:
        protos[r[5]] = protos.get(r[5], 0) + 1
    best = max(protos, key=protos.get)
    f = np.array([r[2] for r in rows])
    hop_channels = int(len(np.unique(np.round(f, 1))))
    per_epoch = {}
    for r in rows:
        per_epoch.setdefault((r[0], r[1]), []).append(r[2])
    hops = [len(set(np.round(v, 1))) for v in per_epoch.values()]
    return ToolOutput({"ref": f"trk:{tid}", "n_bursts": len(rows), "protocol_estimate": best,
                       "protocol_votes": protos, "protocol_conf": r1(float(np.mean([r[6] for r in rows if r[5] == best]))),
                       "freq_mhz": {"median": round(float(np.median(f)), 3), "min": round(float(f.min()), 3),
                                    "max": round(float(f.max()), 3)},
                       "bandwidth_khz": r1(float(np.median([r[3] for r in rows]))),
                       "burst_us": r1(float(np.median([r[4] for r in rows]))),
                       "hopping": hop_channels > 3, "distinct_channels": hop_channels,
                       "channels_per_dwell": r1(float(np.mean(hops))), "snr_db": r1(float(np.median([r[7] for r in rows]))),
                       "sensors": sorted({f"sensor:{r[1]}" for r in rows})},
                      [f"trk:{tid}"],
                      {"protocol_estimate": "majority of per-burst classifications from signal features (bandwidth, "
                                            "cyclic prefix, preamble, chirp, envelope); not decoded content"})


P_CTRL_MIN = 0.2


class PilotSearchArgs(Strict):
    near_track_ref: str
    max_range_m: Annotated[float, Field(gt=0, le=6000)] = 3000.0


@register("pilot_search", "S", "Ground controllers (pilots) that could be flying a given drone track.",
          PilotSearchArgs)
def pilot_search(ctx: ToolContext, a: PilotSearchArgs) -> ToolOutput:
    _need_v2(ctx, "bursts")
    site = ctx.site
    tid = _track(ctx, a.near_track_ref)
    dr, dp = site.tracks[tid], site.points[tid]
    cands = []
    for cid, tr in site.tracks.items():
        if cid == tid:
            continue
        pc = tr["klass_probs"].get("controller", 0.0)
        if pc < P_CTRL_MIN:          # few controllers are tracked, so they are rarely the argmax class
            continue
        cp = site.points[cid]
        overlap = min(tr["last_seen"], dr["last_seen"]) - max(tr["first_seen"], dr["first_seen"])
        if overlap <= 0:
            continue
        d = float(np.min(np.hypot(cp["x"].mean() - dp["x"], cp["y"].mean() - dp["y"])))
        if d > a.max_range_m:
            continue
        compat = tr["protocol_guess"] in UPLINK_OF.get(dr["protocol_guess"], set())
        score = (0.4 * min(1.0, overlap / max(dr["last_seen"] - dr["first_seen"], 1.0)) + 0.2 * compat
                 + 0.1 * (1 - d / a.max_range_m) + 0.3 * pc)
        cands.append({"ref": f"trk:{cid}", "pos": pt(float(cp["x"].mean()), float(cp["y"].mean()), float(cp["z"].mean())),
                      "uncertainty_m": r1(float(np.median(cp["a"]))) if "a" in cp else None,
                      "protocol": tr["protocol_guess"], "uplink_compatible": compat, "co_active_s": r1(overlap),
                      "distance_to_drone_m": r1(d), "p_controller": round(pc, 2), "score": round(score, 3)})
    cands.sort(key=lambda c: -c["score"])
    return ToolOutput({"drone": f"trk:{tid}", "candidates": cands[:5]}, [f"trk:{tid}"] + [c["ref"] for c in cands[:5]],
                      {"score": "0.4 time overlap + 0.2 uplink protocol compatible + 0.1 proximity + 0.3 p(controller); "
                                f"candidates are tracks with p(controller) >= {P_CTRL_MIN}"})


class TdoaFixArgs(Strict):
    track_ref: str
    at_time_s: Annotated[float, Field(ge=0)]


@register("tdoa_fix", "S", "Hybrid AoA+TDOA position of a track at a time (needs 3+ sensors, wideband bursts).",
          TdoaFixArgs)
def tdoa_fix(ctx: ToolContext, a: TdoaFixArgs) -> ToolOutput:
    _need_v2(ctx, "bursts")
    site = ctx.site
    tid = _track(ctx, a.track_ref)
    rows = site.cursor().execute(
        "SELECT sensor_id, bearing_deg, snr_db, toa_s, bw_khz FROM bursts WHERE track_id = ? AND abs(t_s - ?) <= 5 "
        "ORDER BY snr_db DESC", [tid, a.at_time_s]).fetchall()
    # bearings: the strongest burst with a bearing per node; TOAs: the SAME burst at every node (match_toas), from
    # complete bursts only (a burst cut at the dwell edge has toa_s NULL)
    by, toas, bw_of = {}, {}, {}
    for s, brg, snr, toa, bw in rows:
        if brg is not None:
            by.setdefault(s, {"sensor": s, "brg": brg, "snr": snr, "toa": None, "bw_khz": bw})
        if toa is not None and math.isfinite(toa):
            toas.setdefault(s, []).append(float(toa))
            bw_of.setdefault(s, bw)
    grp = list(by.values())
    if len(grp) < 3:
        raise ToolError("OUT_OF_COVERAGE", f"trk:{tid} has bursts at {len(grp)} sensor(s) near t={a.at_time_s:g}",
                        "TDOA needs 3+ sensors; use track_detail for the AoA estimate.")
    pose = {s: (v["pose"]["x_m"], v["pose"]["y_m"], v["pose"]["z_m"]) for s, v in site.sensors.items()}
    i = site.track_index_at(tid, a.at_time_s)
    p = site.points[tid]
    gnss = 5.0
    aoa = (float(p["x"][i]), float(p["y"][i]))
    # same burst at every node, and consistent with the track's AoA position (Task 2c: rejects mis-associated TOAs)
    matched = _match_toas({s: v for s, v in toas.items() if s in by}, pose, xy=aoa, z_m=float(p["z"][i]))
    for g in grp:
        if g["sensor"] in matched:
            g["toa"], g["bw_khz"] = matched[g["sensor"]], bw_of[g["sensor"]]
    xy, accepted, info = _tdoa_fix(aoa, grp, pose, gnss, z_m=float(p["z"][i]))
    timed = [g for g in grp if g["toa"] is not None]
    bw = min((g["bw_khz"] for g in timed), default=min(g["bw_khz"] for g in grp))
    sig = math.sqrt(2) * gnss * 1e-9 * C0 + C0 / (bw * 1e3)
    return ToolOutput({"ref": f"trk:{tid}", "t_s": r1(a.at_time_s), "pos": pt(float(xy[0]), float(xy[1]), float(p["z"][i])),
                       "aoa_pos": pt(aoa[0], aoa[1], float(p["z"][i])),
                       "shift_m": r1(float(math.hypot(xy[0] - aoa[0], xy[1] - aoa[1]))), "accepted": bool(accepted),
                       "range_diff_sigma_m": r1(sig), "sensors": [f"sensor:{g['sensor']}" for g in grp],
                       "timed_sensors": len(timed), "useful": bw >= 10000},
                      [f"trk:{tid}"] + [f"sensor:{g['sensor']}" for g in grp],
                      {"range_diff_sigma_m": "sqrt(2) x GNSS 5 ns x c + c / bandwidth: about 2 us (600 m) for 500 kHz "
                                             "LoRa, so TDOA only helps wideband (>= 10 MHz) signals",
                       "accepted": "the TDOA refinement is kept only if every range difference is within 3 sigma and "
                                   "every bearing still points at it; otherwise pos = aoa_pos"})


class NoiseFloorArgs(Strict):
    sensor_ref: str
    band: Band
    at_time_s: Annotated[float, Field(ge=0)] | None = None


@register("noise_floor", "S", "Measured noise floor (dBm/MHz) and receiver load at a DF sensor and band.",
          NoiseFloorArgs)
def noise_floor(ctx: ToolContext, a: NoiseFloorArgs) -> ToolOutput:
    _need_v2(ctx, "dwells")
    s = a.sensor_ref.replace("sensor:", "").upper()
    if s not in ctx.site.sensors:
        raise ToolError("UNKNOWN_REF", f"sensor:{s} does not exist", "Use sensor:S1..S5.")
    q = "SELECT t_s, window_mhz, noise_dbm_per_mhz, clipped_frac, n_bursts FROM dwells WHERE sensor_id = ? AND band = ?"
    args: list = [s, a.band]
    if a.at_time_s is not None:
        q += " AND abs(t_s - ?) <= 10"
        args.append(a.at_time_s)
    allrows = ctx.site.cursor().execute(q, args).fetchall()
    rows = [r for r in allrows if r[2] is not None]          # NULL floor = dwell blanked by this node's own jamming
    if not rows:
        if allrows:
            raise ToolError("NOT_FOUND", f"every dwell of sensor:{s} in {a.band} was blanked by its own jamming",
                            "Ask about a time outside the effector_log windows, or another band.")
        raise ToolError("NOT_FOUND", f"no dwells for sensor:{s} in {a.band}", "This sensor may not cover that band.")
    nf = np.array([r[2] for r in rows])
    allr = ctx.site.cursor().execute("SELECT median(noise_dbm_per_mhz) FROM dwells WHERE band = ?", [a.band]).fetchone()[0]
    thermal = -174 + 60 + 3.0
    return ToolOutput({"sensor": f"sensor:{s}", "band": a.band, "noise_dbm_per_mhz": r1(float(np.median(nf))),
                       "noise_max_dbm_per_mhz": r1(float(nf.max())), "thermal_dbm_per_mhz": r1(thermal),
                       "rise_over_thermal_db": r1(float(np.median(nf)) - thermal),
                       "rise_over_network_median_db": r1(float(np.median(nf)) - float(allr)),
                       "clipping_pct": r1(float(np.max([r[3] for r in rows])) * 100), "bursts_per_dwell": r1(float(np.mean([r[4] for r in rows]))),
                       "n_dwells": len(rows), "blanked_dwells": len(allrows) - len(rows)},
                      [f"sensor:{s}"],
                      {"noise_dbm_per_mhz": "receiver-measured floor (20th percentile of sample power) referred to the "
                                            "antenna; rise over thermal = desensitisation by interference"})
