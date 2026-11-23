"""Family B: RF tracks and DF hypotheses from OBSERVED data only (SPEC 6.2 B). No path labels, ever."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
import shapely
from pydantic import Field, model_validator

from sim.geometry import bearing_deg
from tools.errors import ToolError
from tools.registry import ToolContext, ToolOutput, register
from tools.site import _cursor, _entity, region_geom
from tools.types import Band, Klass, Region, Strict, pt, r1

DF_MATCH_WINDOW_S = 5.0


def _track(ctx: ToolContext, ref: str) -> str:
    kind, tid, _ = _entity(ctx, ref)
    if kind != "trk":
        raise ToolError("INVALID_ARGS", f"{ref} is not a track ref", "Pass a trk: ref from track_search.")
    return tid


def _unc(p: dict, i: int) -> dict:
    return {"kind": "ellipse", "a_m": r1(p["a"][i]), "b_m": r1(p["b"][i]), "theta_deg": r1(p["theta"][i])}


def _argmax(probs: dict) -> str:
    return max(probs, key=lambda k: (probs[k], k))


# --- track_search ----------------------------------------------------------------------------------------------
class TrackFilters(Strict):
    band: Band | None = None
    freq_mhz_range: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None
    klass: Klass | None = None
    time_range_s: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None
    region: Region | None = None
    min_confidence: Annotated[float, Field(ge=0, le=1)] | None = None
    group_id: Annotated[str, Field(max_length=40)] | None = None

    @model_validator(mode="after")
    def _ranges(self):
        for name in ("freq_mhz_range", "time_range_s"):
            v = getattr(self, name)
            if v is not None and v[0] > v[1]:
                raise ValueError(f"{name} must be [low, high]")
        return self


class TrackSearchArgs(Strict):
    filters: TrackFilters = TrackFilters()
    sort: Literal["first_seen", "last_seen", "confidence"] = "first_seen"
    limit: Annotated[int, Field(ge=1, le=20)] = 10
    cursor: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _lift_filters(cls, v):
        """Accept filter keys given at the top level (a common slip) by moving them into `filters`."""
        if isinstance(v, dict):
            stray = {k: v[k] for k in TrackFilters.model_fields if k in v}
            if stray:
                v = {k: val for k, val in v.items() if k not in stray}
                v["filters"] = {**(v.get("filters") or {}), **stray}
        return v


@register("track_search", "B", "Find drone/emitter tracks by region, time, band, class; returns total and a page.",
          TrackSearchArgs)
def track_search(ctx: ToolContext, a: TrackSearchArgs) -> ToolOutput:
    site, f = ctx.site, a.filters
    t0, t1 = f.time_range_s or (-np.inf, np.inf)
    rg = region_geom(ctx, f.region) if f.region is not None else None
    if rg is not None:
        shapely.prepare(rg)
    rows = []
    for tid, tr in site.tracks.items():
        p = site.points[tid]
        sel = (p["t"] >= t0) & (p["t"] <= t1)
        if f.band is not None:
            sel &= np.array([b == f.band for b in p["band"]])
        if not sel.any():
            continue
        if f.freq_mhz_range and not (f.freq_mhz_range[0] <= tr["freq_mhz"] <= f.freq_mhz_range[1]):
            continue
        if f.klass and _argmax(tr["klass_probs"]) != f.klass:
            continue
        if f.min_confidence is not None and tr["confidence"] < f.min_confidence:
            continue
        if f.group_id is not None and site.groups.get(tid) != f.group_id:
            continue
        if rg is not None:
            inside = shapely.contains_xy(rg, p["x"], p["y"]) & sel
            if not inside.any():
                continue
            sel = inside
        i = int(np.flatnonzero(sel)[-1])   # latest matching estimate
        rows.append((tid, tr, i))
    key = {"first_seen": lambda r: (r[1]["first_seen"], r[0]), "last_seen": lambda r: (-r[1]["last_seen"], r[0]),
           "confidence": lambda r: (-r[1]["confidence"], r[0])}[a.sort]
    rows.sort(key=key)
    off = _cursor(a.cursor)
    page = rows[off: off + a.limit]
    items = []
    for tid, tr, i in page:
        p = site.points[tid]
        items.append({"ref": f"trk:{tid}", "t_s": r1(p["t"][i]), "pos": pt(p["x"][i], p["y"][i], p["z"][i]),
                      "uncertainty": _unc(p, i), "band": p["band"][i], "klass": _argmax(tr["klass_probs"]),
                      "p_klass": round(max(tr["klass_probs"].values()), 2),
                      "confidence": tr["confidence"], "first_seen": r1(tr["first_seen"]),
                      "last_seen": r1(tr["last_seen"]), "n_sensors": int(p["n"][i]),
                      **({"group_id": site.groups[tid]} if site.groups.get(tid) else {})})
    nxt = f"c{off + a.limit}" if off + a.limit < len(rows) else None
    definition = {"pos": "latest estimate matching the filters"}
    if f.band:
        definition["band_mhz"] = site.bands[f.band]
    if f.region is not None:
        definition["region"] = "any estimate inside the region within time_range_s"
    return ToolOutput({"total": len(rows), "items": items, "next_cursor": nxt}, [i["ref"] for i in items], definition)


# --- track_detail ----------------------------------------------------------------------------------------------
class TrackDetailArgs(Strict):
    track_ref: str
    at_time_s: Annotated[float, Field(ge=0)] | None = None


@register("track_detail", "B", "Position, uncertainty, class, signal features and supporting sensors of a track.",
          TrackDetailArgs)
def track_detail(ctx: ToolContext, a: TrackDetailArgs) -> ToolOutput:
    site = ctx.site
    tid = _track(ctx, a.track_ref)
    tr, p = site.tracks[tid], site.points[tid]
    if a.at_time_s is not None and not (p["t"][0] - 1e-6 <= a.at_time_s <= p["t"][-1] + 1e-6):
        raise ToolError("OUT_OF_COVERAGE", f"trk:{tid} has no estimate at t={a.at_time_s}",
                        f"Valid times: {p['t'][0]:.0f}..{p['t'][-1]:.0f} s.")
    i = site.track_index_at(tid, a.at_time_s)
    t = float(p["t"][i])
    sens = [r[0] for r in site.cursor().execute(
        "SELECT DISTINCT sensor_id FROM bearings WHERE track_id = ? AND abs(t_s - ?) <= ? ORDER BY 1",
        [tid, t, DF_MATCH_WINDOW_S]).fetchall()]
    path = float(np.sum(np.hypot(np.diff(p["x"][: i + 1]), np.diff(p["y"][: i + 1])))) if i else 0.0
    band = p["band"][i]
    data = {
        "ref": f"trk:{tid}", "t_s": r1(t), "pos": pt(p["x"][i], p["y"][i], p["z"][i]), "uncertainty": _unc(p, i),
        "klass_probs": tr["klass_probs"], "confidence": tr["confidence"],
        "signal": {"band": band, "freq_mhz": tr["freq_mhz"] if band == tr["band"] else None,
                   "power_dbm": r1(tr["power_dbm"]), "bandwidth_khz": tr["bandwidth_khz"],
                   "protocol_guess": tr["protocol_guess"]},
        "history": {"first_seen": r1(tr["first_seen"]), "last_seen": r1(tr["last_seen"]), "n_points": int(i + 1),
                    "path_len_m": r1(path), "bands_seen": sorted(set(p["band"][: i + 1]))},
        "supporting_sensors": [f"sensor:{s}" for s in sens],
        "zones": [f"zone:{z}" for z in site.zones_containing(float(p["x"][i]), float(p["y"][i]))],
    }
    if site.groups.get(tid):
        data["group_id"] = site.groups[tid]
    return ToolOutput(data, [f"trk:{tid}"] + data["supporting_sensors"] + data["zones"],
                      {"pos": "latest estimate at or before at_time_s", "supporting_sensors":
                       f"sensors with a bearing on this track within {DF_MATCH_WINDOW_S:.0f} s",
                       "zones": "zones containing the position estimate", "group_id": "swarm group (if any)"})


# --- df_hypotheses ---------------------------------------------------------------------------------------------
class DfHypothesesArgs(Strict):
    track_ref: str
    at_time_s: Annotated[float, Field(ge=0)]
    sensor_refs: Annotated[list[str], Field(max_length=10)] | None = None


@register("df_hypotheses", "B", "Bearing hypotheses per DF sensor for a track at a time. No path labels.",
          DfHypothesesArgs)
def df_hypotheses(ctx: ToolContext, a: DfHypothesesArgs) -> ToolOutput:
    site = ctx.site
    tid = _track(ctx, a.track_ref)
    want = None
    if a.sensor_refs:
        want = []
        for ref in a.sensor_refs:
            kind, ent, _ = _entity(ctx, ref)
            if kind != "sensor":
                raise ToolError("INVALID_ARGS", f"{ref} is not a sensor ref", "Pass sensor: refs, e.g. sensor:S1.")
            want.append(ent["id"])
    con = site.cursor()
    row = con.execute("SELECT t_s FROM bearings WHERE track_id = ? ORDER BY abs(t_s - ?), t_s LIMIT 1",
                      [tid, a.at_time_s]).fetchone()
    if row is None or abs(row[0] - a.at_time_s) > DF_MATCH_WINDOW_S:
        near = con.execute("SELECT min(t_s), max(t_s) FROM bearings WHERE track_id = ?", [tid]).fetchone()
        hint = (f"Bearings for trk:{tid} exist every 10 s between {near[0]:.0f} and {near[1]:.0f} s."
                if near[0] is not None else f"No DF sensor observed trk:{tid}.")
        raise ToolError("OUT_OF_COVERAGE", f"no bearings for trk:{tid} within {DF_MATCH_WINDOW_S:.0f} s of "
                                           f"t={a.at_time_s}", hint)
    t = row[0]
    rows = con.execute("SELECT hyp_id, sensor_id, bearing_deg, snr_db, power_db FROM bearings "
                       "WHERE track_id = ? AND t_s = ? ORDER BY sensor_id, bearing_deg", [tid, t]).fetchall()
    per: dict[str, dict] = {}
    refs = [f"trk:{tid}"]
    p = site.points[tid]
    i = site.track_index_at(tid, t)
    for hid, sid, brg, snr, pwr in rows:
        if want is not None and sid not in want:
            continue
        s = site.sensors[sid]["pose"]
        # Observed-data geometry check: bearing from the sensor to the track's own position estimate at t.
        expected = bearing_deg(s["x_m"], s["y_m"], p["x"][i], p["y"][i])
        entry = per.setdefault(sid, {"sensor": f"sensor:{sid}", "sensor_pos": pt(s["x_m"], s["y_m"]),
                                     "bearing_to_track_estimate_deg": r1(expected), "hyps": []})
        ref = f"hyp:{hid}@{t:g}"
        entry["hyps"].append({"ref": ref, "bearing_deg": r1(brg), "snr_db": r1(snr), "power_db": r1(pwr)})
        refs.append(ref)
    return ToolOutput({"track": f"trk:{tid}", "t_s": r1(t), "sensors": list(per.values())}, refs,
                      {"bearing": "deg clockwise from true north at the sensor", "t_s": "nearest bearing sample",
                       "bearing_to_track_estimate_deg": "sensor to the track's position estimate at t_s; a direct "
                                                        "path should be close to it, a reflection usually is not"})
