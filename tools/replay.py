"""Families F and G (SPEC 6.2): event_search, track_history, sql_readonly. Observed data only."""

from __future__ import annotations

import json
import threading
from typing import Annotated, Literal

import numpy as np
import shapely
from pydantic import Field, model_validator

from tools.errors import ToolError
from tools.registry import ToolContext, ToolOutput, register
from tools.site import _cursor, _entity, region_geom
from tools.tracks import _track
from tools.types import Region, Strict, pt, r1

EventType = Literal["zone_enter", "zone_exit", "loiter", "band_change", "detection_start"]


# --- event_search ----------------------------------------------------------------------------------------------
class EventSearchArgs(Strict):
    event_type: EventType | None = None
    subject_ref: str | None = None
    region: Region | None = None
    time_range_s: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None
    limit: Annotated[int, Field(ge=1, le=20)] = 10
    cursor: str | None = None

    @model_validator(mode="after")
    def _range(self):
        if self.time_range_s and self.time_range_s[0] > self.time_range_s[1]:
            raise ValueError("time_range_s must be [low, high]")
        return self


@register("event_search", "F", "Typed track events (zone_enter/exit, loiter, band_change, detection_start), paged.",
          EventSearchArgs)
def event_search(ctx: ToolContext, a: EventSearchArgs) -> ToolOutput:
    site = ctx.site
    trk = zone = None
    if a.subject_ref:
        kind, ent, _ = _entity(ctx, a.subject_ref)
        if kind == "trk":
            trk = ent
        elif kind == "zone":
            zone = ent.id
        else:
            raise ToolError("INVALID_ARGS", "subject_ref must be a trk: or zone: ref",
                            "Pass a track (trk:) or a zone (zone:); use region for objects.")
    rg = region_geom(ctx, a.region) if a.region is not None else None
    t0, t1 = a.time_range_s or (-np.inf, np.inf)
    rows = []
    for e in site.events:
        if a.event_type and e["type"] != a.event_type:
            continue
        if trk and e["track_id"] != trk:
            continue
        if zone and e["zone_id"] != zone:
            continue
        if not (t0 <= e["t_s"] <= t1):
            continue
        if rg is not None and not rg.contains(shapely.Point(e["x_m"], e["y_m"])):
            continue
        rows.append(e)
    off = _cursor(a.cursor)
    page = rows[off: off + a.limit]
    items = []
    for e in page:
        it = {"ref": f"evt:{e['evt_id']}", "type": e["type"], "track_ref": f"trk:{e['track_id']}", "t_s": r1(e["t_s"]),
              "pos": pt(e["x_m"], e["y_m"])}
        if e["zone_id"]:
            it["zone_ref"] = f"zone:{e['zone_id']}"
        if e["detail"]:
            it["detail"] = e["detail"]
        items.append(it)
    nxt = f"c{off + a.limit}" if off + a.limit < len(rows) else None
    refs = [i["ref"] for i in items] + sorted({i["track_ref"] for i in items}) + sorted(
        {i["zone_ref"] for i in items if "zone_ref" in i})
    return ToolOutput({"total": len(rows), "items": items, "next_cursor": nxt}, refs,
                      {"events": "from track estimates; zone changes need 3 consecutive estimates (6 s) on the new "
                                 "side, the time is the first of them; loiter = 60 s within 40 m"})


# --- track_history ---------------------------------------------------------------------------------------------
class TrackHistoryArgs(Strict):
    track_ref: str
    time_range_s: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None
    step_s: Annotated[float, Field(ge=2, le=120)] = 10.0


@register("track_history", "F", "Downsampled positions with uncertainty of a track over a time range.",
          TrackHistoryArgs)
def track_history(ctx: ToolContext, a: TrackHistoryArgs) -> ToolOutput:
    tid = _track(ctx, a.track_ref)
    p = ctx.site.points[tid]
    t0, t1 = a.time_range_s or (float(p["t"][0]), float(p["t"][-1]))
    sel = np.flatnonzero((p["t"] >= t0) & (p["t"] <= t1))
    if not len(sel):
        raise ToolError("OUT_OF_COVERAGE", f"trk:{tid} has no estimates in [{t0:g}, {t1:g}] s",
                        f"Valid times: {p['t'][0]:.0f}..{p['t'][-1]:.0f} s.")
    step = max(a.step_s, (p["t"][sel[-1]] - p["t"][sel[0]]) / 59.0)
    out, last = [], -np.inf
    for i in sel:
        if p["t"][i] - last >= step - 1e-6:
            out.append({"t_s": r1(p["t"][i]), "pos": pt(p["x"][i], p["y"][i], p["z"][i]),
                        "a_m": r1(p["a"][i]), "b_m": r1(p["b"][i]), "band": p["band"][i]})
            last = p["t"][i]
    dist = float(np.sum(np.hypot(np.diff(p["x"][sel]), np.diff(p["y"][sel]))))
    return ToolOutput({"track": f"trk:{tid}", "points": out, "path_len_m": r1(dist), "step_s": r1(step)},
                      [f"trk:{tid}"], {"step_s": "raised automatically so at most 60 points are returned"})


# --- sql_readonly ----------------------------------------------------------------------------------------------
ALLOWED = {"tracks", "track_points", "bearings", "events"}
DENY_FUNCS = {"getenv", "current_setting", "read_text", "read_blob", "glob"}
ROW_LIMIT = 200
TIMEOUT_S = 2.0


class SqlArgs(Strict):
    query: Annotated[str, Field(min_length=6, max_length=1200)]


def _walk(node, tables: set, funcs: set, kinds: set) -> None:
    if isinstance(node, dict):
        t = node.get("type")
        t = t if isinstance(t, str) else None
        if t == "BASE_TABLE":
            tables.add(str(node.get("table_name", "")).lower())
            if node.get("schema_name") or node.get("catalog_name"):
                tables.add(f"{node.get('catalog_name')}.{node.get('schema_name')}")
        if t:
            kinds.add(t)
        if "function_name" in node:
            funcs.add(str(node["function_name"]).lower())
        for v in node.values():
            _walk(v, tables, funcs, kinds)
    elif isinstance(node, list):
        for v in node:
            _walk(v, tables, funcs, kinds)


def _deny(msg: str) -> ToolError:
    return ToolError("NOT_PERMITTED", msg, f"Only single SELECT statements over {', '.join(sorted(ALLOWED))} "
                                           f"are allowed (no files, no table functions); max {ROW_LIMIT} rows.")


@register("sql_readonly", "G", "Read-only SELECT over observed tables (tracks, track_points, bearings, events).",
          SqlArgs)
def sql_readonly(ctx: ToolContext, a: SqlArgs) -> ToolOutput:
    con = ctx.site.cursor()
    q = a.query.strip().rstrip(";").strip()
    ast = json.loads(con.execute("SELECT json_serialize_sql(?)", [q]).fetchone()[0])
    if ast.get("error"):
        raise _deny(f"not a SELECT statement or does not parse: {str(ast.get('error_message', ''))[:120]}")
    if len(ast.get("statements", [])) != 1:
        raise _deny("exactly one statement is allowed")
    tables, funcs, kinds = set(), set(), set()
    _walk(ast["statements"][0], tables, funcs, kinds)
    if "TABLE_FUNCTION" in kinds:
        raise _deny("table functions (file readers etc.) are not allowed")
    bad = {t for t in tables if t not in ALLOWED}
    if bad:
        raise _deny(f"tables not allowed: {sorted(bad)}")
    if funcs & DENY_FUNCS:
        raise _deny(f"functions not allowed: {sorted(funcs & DENY_FUNCS)}")
    timer = threading.Timer(TIMEOUT_S, con.interrupt)
    timer.start()
    try:
        cur = con.execute(f"SELECT * FROM ({q}) AS _q LIMIT {ROW_LIMIT + 1}")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    except Exception as e:  # noqa: BLE001 - surfaced as an instruction, never a crash
        msg = str(e).splitlines()[0][:160]
        if "interrupt" in msg.lower():
            raise ToolError("TIMEOUT", f"query exceeded {TIMEOUT_S:g} s", "Add WHERE filters or aggregate.") from None
        raise ToolError("INVALID_ARGS", f"query failed: {msg}", "Check column names: tracks(track_id, band, "
                        "freq_mhz, confidence, first_seen, last_seen, group_id, ...), track_points(track_id, t_s, x_m, "
                        "y_m, z_m, a_m, b_m, band), bearings(hyp_id, track_id, sensor_id, t_s, bearing_deg, snr_db, "
                        "power_db), events(evt_id, type, track_id, zone_id, t_s, x_m, y_m, detail).") from None
    finally:
        timer.cancel()
    trunc = len(rows) > ROW_LIMIT
    rows = [[round(v, 2) if isinstance(v, float) else v for v in r] for r in rows[:ROW_LIMIT]]
    return ToolOutput({"columns": cols, "rows": rows, "n_rows": len(rows), "truncated": trunc}, [],
                      {"tables": sorted(ALLOWED), "row_limit": ROW_LIMIT},
                      ["rows are raw data: cite refs from other tools for claims"] if rows else [])
