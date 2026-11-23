"""FastAPI gateway for the operator UI (SPEC 7.7, ENV SPEC 13).

    python -m serve.gateway [--port 8080] [--site data/site_kandla]   # or SCENESCOUT_SITE=...
    (model mode needs the LLM server: python -m serve.llm)

Serves the 3D operator console (ui/kandla/, at /) and the legacy 2D stub UI (/stub), belief-side site/track data
converted to lon/lat for the map, and a Server-Sent Events stream of the agent's trace. Only observed (belief) data
is exposed; truth is never read here (SPEC 12.1).

Guardrails (demo-v2 S5): binds 127.0.0.1 unless --host says otherwise; one running question per client (429
otherwise); /api/ask takes GET or POST; each question has the wall-clock budget of the env config
(SCENESCOUT_WALLCLOCK_S overrides) and writes a `question` record to the append-only audit log runs/audit_log/;
POST /api/gate records the operator's two-step decision on a verified answer that requires human confirmation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness.audit import AuditLog, OperatorGate  # noqa: E402
from harness.env import AkshaEnv, site_view  # noqa: E402
from serve import geo_export  # noqa: E402
from serve.agent import LLM_URL, MODEL, run_expert, run_model, run_task, warm_up  # noqa: E402
from tasks.base import Task  # noqa: E402

DEFAULT_SITE = REPO / "data" / "site_kandla"
# demo sets (demo-v2 S6.1): the presenter questions, their recordings and the scenario they run on. Active set:
# SCENESCOUT_DEMO_SET=v1|v2|v2b, else the newest of v2b / v2 whose recordings exist, else v1. v2b = the 12 v2
# questions + 10 more on the same scenario (runs/s78/demo_v2b.md).
DEMO_SETS = {
    "v1": {"site": DEFAULT_SITE, "tasks": REPO / "data" / "demo_tasks_kandla.json",
           "replays": REPO / "data" / "demo_replays"},
    "v2": {"site": REPO / "data" / "scenarios" / "kandla_s0_X", "tasks": REPO / "data" / "demo_tasks_kandla_v2.json",
           "replays": REPO / "data" / "demo_replays_v2"},
    "v2b": {"site": REPO / "data" / "scenarios" / "kandla_s0_X", "tasks": REPO / "data" / "demo_tasks_kandla_v2b.json",
            "replays": REPO / "data" / "demo_replays_v2b"},
}


def active_demo_set() -> str:
    forced = os.environ.get("SCENESCOUT_DEMO_SET")
    if forced:
        if forced not in DEMO_SETS:
            raise ValueError(f"SCENESCOUT_DEMO_SET={forced!r}: expected one of {sorted(DEMO_SETS)}")
        return forced
    for name in ("v2b", "v2"):
        d = DEMO_SETS.get(name)
        if d is None:
            continue
        has_rec = d["replays"].exists() and any(d["replays"].glob("*__*.json"))
        if has_rec and d["tasks"].exists() and d["site"].exists():
            return name
    return "v1"


def default_site() -> Path:
    """SCENESCOUT_SITE, else the active demo set's scenario."""
    env = os.environ.get("SCENESCOUT_SITE")
    return Path(env) if env else DEMO_SETS[active_demo_set()]["site"]


def _demo_set_of_site() -> dict | None:
    """The demo set running on SITE_DIR; the active set wins when several share a scenario (v2 and v2b)."""
    here = Path(SITE_DIR).resolve()
    sets = [DEMO_SETS[active_demo_set()]] + list(DEMO_SETS.values())
    return next((s for s in sets if Path(s["site"]).resolve() == here), None)


SITE_DIR = default_site()
UI = REPO / "ui"
UI_3D = UI / "kandla"
CESIUM_TOKEN_FILE = REPO / "token.txt"
AUDIT = AuditLog(REPO / "runs" / "audit_log")
GATE = OperatorGate(AUDIT, min_confirm_delay_s=float(os.environ.get("SCENESCOUT_GATE_DELAY_S", "1.0")))

app = FastAPI(title="SceneScout")
if UI_3D.exists():
    app.mount("/ui", StaticFiles(directory=UI_3D), name="ui3d")
WEB_WORLD = DEFAULT_SITE / "web"   # Draco GLB tiles of the Blender world (envbuild/blender/export_web.py); geometry only
if WEB_WORLD.exists():
    app.mount("/world", StaticFiles(directory=WEB_WORLD), name="webworld")


def set_site(path: str | Path) -> None:
    global SITE_DIR
    p = Path(path)
    SITE_DIR = p if p.is_absolute() else (REPO / p)


def _site():
    return site_view(SITE_DIR)


def _demo_file() -> Path | None:
    """Demo questions for the active site: SCENESCOUT_DEMO_TASKS, else data/demo_tasks_<site>.json; the procedural
    stub uses data/demo_tasks.json. Missing file -> no demo chips."""
    env = os.environ.get("SCENESCOUT_DEMO_TASKS")
    if env:
        return Path(env)
    ds = _demo_set_of_site()
    if ds is not None:
        return ds["tasks"]
    name = SITE_DIR.resolve().name
    if name == "stub_site":
        return REPO / "data" / "demo_tasks.json"
    short = name.removeprefix("site_")
    return REPO / "data" / f"demo_tasks_{short}.json"


def _demo_tasks() -> list[Task]:
    f = _demo_file()
    if f is None or not f.exists():
        return []
    try:
        return [Task.from_row(r) for r in json.loads(f.read_text())]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


@app.get("/")
def index():
    page = UI_3D / "index.html"
    return FileResponse(page if page.exists() else UI / "index.html")


@app.get("/stub")
def stub_index():
    return FileResponse(UI / "index.html")


@app.get("/api/cesium_token")
def cesium_token():
    """The ion token is read at request time and only ever handed to the browser; it is never logged."""
    try:
        tok = CESIUM_TOKEN_FILE.read_text().strip()
    except OSError:
        tok = ""
    if not tok:
        raise HTTPException(404, "no Cesium ion token configured (token.txt at the repo root)")
    return JSONResponse({"token": tok}, headers={"Cache-Control": "no-store"})


@app.get("/api/meta")
def meta():
    site = _site()
    llm = _llm_probe()
    t = site.meta.get("time") or {}
    return {"site": site.meta.get("site"), "data_version": site.version, "duration_s": t.get("duration_s"),
            "t0_iso": t.get("t0_iso"), "llm": llm, "model": served_model(), "rf_backend": site.meta.get("rf_backend"),
            "demo_set": next((k for k, v in DEMO_SETS.items() if v is _demo_set_of_site()), None),
            "demo": [{"task_id": t.task_id, "question": t.question, "template": t.template, "area": t.area}
                     for t in _demo_tasks()]}


@app.get("/api/site")
def site_geometry():
    """Objects, zones and sensors in lon/lat degrees (polygon = [[lon, lat], ...]) plus meta and fidelity.
    Each object/zone also keeps its ENU ring as `poly` and the legacy keys `extent`/`shore_y` for the /stub UI."""
    site = _site()
    geo = geo_export.site_geo(str(SITE_DIR.resolve()), site.version, site)

    def coords(g):
        g = max(g.geoms, key=lambda p: p.area) if g.geom_type == "MultiPolygon" else g
        return [[round(x, 1), round(y, 1)] for x, y in g.exterior.coords] if g.geom_type == "Polygon" else []

    enu_obj = {o.ref: coords(o.geom) for o in site.objects.values()}
    enu_zone = {z.ref: coords(z.geom) for z in site.zones.values()}
    return {
        "meta": geo["meta"],
        "objects": [{**o, "poly": enu_obj.get(o["ref"], [])} for o in geo["objects"]],
        "zones": [{**z, "poly": enu_zone.get(z["ref"], [])} for z in geo["zones"]],
        "sensors": [{**s, "x": s["x_m"], "y": s["y_m"]} for s in geo["sensors"]],
        "extent": site.meta.get("extent_m"), "shore_y": site.meta.get("shore_y_m"),
    }


@app.get("/api/tracks")
def tracks(step_s: float = Query(4.0, ge=2.0, le=60.0), frame: str = "geo"):
    """Observed track estimates, downsampled for the time slider.
    frame=geo (default): pts = [[t_s, lon, lat, z_m, a_m, b_m, theta_deg], ...]
    frame=enu (legacy /stub UI): pts = [[t_s, x_m, y_m, a_m, b_m, theta_deg], ...]"""
    site = _site()
    if frame == "enu3":                    # site ENU with altitude: the three.js console works in metres, no globe
        out = []
        for tid, tr in site.tracks.items():
            p = site.points[tid]
            keep = [i for i in range(len(p["t"])) if p["t"][i] % step_s == 0 or i == len(p["t"]) - 1]
            out.append({"ref": f"trk:{tid}", "band": tr["band"], "confidence": tr["confidence"],
                        "klass": max(tr["klass_probs"], key=tr["klass_probs"].get), "group": site.groups.get(tid),
                        "klass_probs": tr["klass_probs"],
                        "pts": [[float(p["t"][i]), round(float(p["x"][i]), 1), round(float(p["y"][i]), 1),
                                 round(float(p["z"][i]), 1), round(float(p["a"][i]), 1), round(float(p["b"][i]), 1),
                                 round(float(p["theta"][i]), 1)] for i in keep]})
        return {"tracks": out, "fields": ["t_s", "x_m", "y_m", "z_m", "a_m", "b_m", "theta_deg"]}
    if frame == "enu":
        out = []
        for tid, tr in site.tracks.items():
            p = site.points[tid]
            keep = [i for i in range(len(p["t"])) if p["t"][i] % step_s == 0 or i == len(p["t"]) - 1]
            out.append({"ref": f"trk:{tid}", "band": tr["band"], "confidence": tr["confidence"],
                        "klass": max(tr["klass_probs"], key=tr["klass_probs"].get),
                        "pts": [[float(p["t"][i]), round(float(p["x"][i]), 1), round(float(p["y"][i]), 1),
                                 round(float(p["a"][i]), 1), round(float(p["b"][i]), 1),
                                 round(float(p["theta"][i]), 1)] for i in keep]})
        return {"tracks": out}
    return {"tracks": geo_export.tracks_geo(str(SITE_DIR.resolve()), site.version, site, float(step_s)),
            "fields": ["t_s", "lon", "lat", "z_m", "a_m", "b_m", "theta_deg"],
            "notes": {"z_m": "metres above ground", "theta_deg": "semi-major axis bearing, clockwise from north"}}


@app.get("/api/bearings")
def bearings(t: float | None = None, track: str | None = Query(None, max_length=40)):
    """Observed DF bearings nearest to t (within 10 s), grouped by sensor: helper for drawing rays."""
    site = _site()
    if track is not None:
        tid = track.split(":", 1)[-1].split("@", 1)[0]
        if tid not in site.tracks:
            raise HTTPException(404, f"unknown track {track}")
    return geo_export.bearings_at(site, t, track)


_ACTIVE: dict[str, float] = {}           # client -> start time of its running question
_ACTIVE_LOCK = threading.Lock()


def _wallclock_s() -> float:
    from harness.config import EnvConfig

    return float(os.environ.get("SCENESCOUT_WALLCLOCK_S") or EnvConfig.load().max_wallclock_s)


def _claim(client: str, stale_s: float) -> bool:
    with _ACTIVE_LOCK:
        t0 = _ACTIVE.get(client)
        if t0 is not None and time.time() - t0 < stale_s:
            return False
        _ACTIVE[client] = time.time()
        return True


def _release(client: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.pop(client, None)


@app.get("/api/ask")
def ask(request: Request, q: str = Query("", max_length=400), mode: str = "model", task_id: str | None = None):
    return _ask(request, q, mode, task_id)


@app.post("/api/ask")
def ask_post(request: Request, body: dict = Body(...)):
    """Same as GET /api/ask with a JSON body {q, mode?, task_id?}."""
    q = body.get("q") or ""
    if not isinstance(q, str) or len(q) > 400:
        raise HTTPException(422, "q must be a string of at most 400 characters")
    return _ask(request, q, str(body.get("mode") or "model"), body.get("task_id"))


def _ask(request: Request, q: str, mode: str, task_id: str | None):
    client = request.client.host if request.client else "unknown"
    wall = _wallclock_s()
    if mode != "scripted" and not q.strip():
        raise HTTPException(400, "empty question")
    if not _claim(client, stale_s=wall + 60.0):
        return JSONResponse({"ok": False, "code": "busy", "message": "a question from this client is still running"},
                            status_code=429)
    try:
        model = served_model() if mode == "model" else "scripted-expert"
        env = AkshaEnv(SITE_DIR, model_id=model, trace_dir=REPO / "runs" / "ui_traces")
        if mode == "scripted":
            task = next((t for t in _demo_tasks() if t.task_id == task_id), None)
            if task is None:
                raise HTTPException(404, "scripted mode needs a demo task_id")
            events = run_expert(env, task)
        else:
            req = {"request_id": uuid.uuid4().hex[:12], "question": q.strip(), "mode": mode, "client": client,
                   "method": request.method}
            kw = {"model": model, "max_wallclock_s": wall, "audit": AUDIT, "request": req}
            demo = next((t for t in _demo_tasks() if t.question == q.strip()), None)
            # a presenter-menu question runs with its full demo task (drone payload, clarification); free text bare
            events = run_task(env, demo, **kw) if demo else run_model(env, q, **kw)
    except BaseException:
        _release(client)
        raise

    def sse():
        try:
            for ev in events:
                if ev.get("type") == "transcript":
                    continue                       # full model context: for RL/debugging, not for the browser
                if ev.get("type") == "answer" and ev.get("answer_hash"):
                    GATE.register(env.episode_id, ev["answer_hash"], bool(ev.get("gate_required")))
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        except Exception as e:  # noqa: BLE001 - surfaced to the operator, never a silent hang
            err = {"type": "error", "code": "internal", "message": f"{type(e).__name__}: {e}"}
            yield f"data: {json.dumps(err)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            _release(client)

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/gate")
def gate(request: Request, body: dict = Body(...)):
    """Operator decision on a verified answer that requires human confirmation (two steps; harness/audit.py).
    Request: {episode_id, answer_hash, operator_id, decision: acknowledge|dispute|defer, step: review|confirm,
    confirm_token (confirm only)}."""
    code, out = GATE.submit(body, client=request.client.host if request.client else "")
    return JSONResponse(out, status_code=code, headers={"Cache-Control": "no-store"})


@app.get("/api/gate/{episode_id}")
def gate_status(episode_id: str):
    st = GATE.status(episode_id)
    if st is None:
        raise HTTPException(404, "no answer was delivered for this episode_id")
    return st


def _replays_dir() -> Path:
    """Recordings of the active site's demo set (v1 data/demo_replays, v2 data/demo_replays_v2)."""
    ds = _demo_set_of_site()
    return Path(ds["replays"]) if ds is not None else DEMO_SETS["v1"]["replays"]


@app.get("/api/replay")
def replay(task_id: str = Query(..., max_length=80), who: str = Query("sft", pattern="^[a-z_]+$")):
    """A recorded run of a demo question (scripts/record_demo.py): labelled 'recorded' in the console."""
    f = _replays_dir() / f"{Path(task_id).name}__{who}.json"
    if not f.exists():
        raise HTTPException(404, f"no recording of {task_id} by {who}")
    return JSONResponse(json.loads(f.read_text()))


@app.get("/api/replays")
def replays():
    out, rep = {}, _replays_dir()
    for f in sorted(rep.glob("*__*.json")) if rep.exists() else []:
        tid, who = f.stem.rsplit("__", 1)
        out.setdefault(tid, {})[who] = json.loads(f.read_text())["summary"]
    return out


RF_METRICS = {
    "ghost_risk": ("site_ghost_risk", 0.0, 1.0, "Ghost risk: share of model-ensemble members whose strongest arrival "
                                                 "is off the direct bearing (best sensor); steps of 0.25"),
    "trust": ("site_trust", 0.0, 1.0, "Model trust: agreement of the RF model ensemble (tide, ships, materials) "
                                       "at the best sensor"),
    "df_err": ("df_err_deg", 0.0, 30.0, "DF accuracy: expected bearing error (deg) at the best sensor"),
    "blind_spot": ("snr_db", 0.0, 5.0, "Blind spots: number of DF nodes that detect a drone here (SNR >= 6 dB); "
                                       "fewer than 2 means no cross-fix"),
}
DETECT_SNR_DB = 6.0


@app.get("/api/rfmap")
def rfmap(band: str = Query(..., max_length=24), height: float = 40.0, metric: str = Query(..., max_length=16)):
    """Belief-side RF map layer on the coverage union grid (D-031) as float16 base64, for the console heat layer."""
    import base64

    import numpy as np

    if metric not in RF_METRICS:
        raise HTTPException(404, f"unknown metric {metric}")
    rf = _site().rf_dir
    key, vmin, vmax, legend = RF_METRICS[metric]
    f = rf / ("coverage" if metric in ("df_err", "blind_spot") else "maps") / f"{Path(band).name}.npz"
    if not f.exists():
        raise HTTPException(404, f"no {metric} map for band {band}")
    z = np.load(f)
    ih = int(np.argmin(np.abs(z["heights"] - height)))
    if metric == "df_err":
        with warnings.catch_warnings():                       # cells outside every sensor's radius stay NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanmin(np.where(np.isfinite(z["df_err_deg"][:, ih]), z["df_err_deg"][:, ih], np.nan), axis=0)
        v = np.clip(v, vmin, vmax)
    elif metric == "blind_spot":
        s_ = z["snr_db"][:, ih]
        v = (np.nan_to_num(s_, nan=-99.0) >= DETECT_SNR_DB).sum(0).astype("f4")
        v[~np.isfinite(s_).any(0)] = np.nan                    # outside every node's cache
    else:
        v = z[key][ih]
    xs, ys = z["xs"], z["ys"]
    return {"band": band, "height_m": float(z["heights"][ih]), "metric": metric, "nx": len(xs), "ny": len(ys),
            "x0": float(xs[0]), "y0": float(ys[0]), "dx": float(xs[1] - xs[0]), "dy": float(ys[1] - ys[0]),
            "vmin": vmin, "vmax": vmax, "legend": legend,
            "values": base64.b64encode(np.asarray(v, "<f2").tobytes()).decode()}


_PROBE: dict = {"t": 0.0, "v": None}


def _llm_probe() -> dict:
    """vLLM model list, cached 5 s so a 10 s console poll stays cheap. No client retries: with vLLM down the default
    two retries made every /api/meta (console boot) wait ~1.4 s (S6)."""
    if _PROBE["v"] is None or time.time() - _PROBE["t"] > 5.0:
        try:
            models = [m.id for m in OpenAI(base_url=LLM_URL, api_key="local", timeout=2, max_retries=0).models.list().data]
            v = {"ok": True, "models": models}
        except Exception as e:  # noqa: BLE001
            v = {"ok": False, "models": [], "error": type(e).__name__}
        _PROBE.update(t=time.time(), v=v)
    return _PROBE["v"]


def served_model() -> str:
    """SCENESCOUT_SERVED_MODEL if set; else the SFT adapter when vLLM serves one named `sft`; else the base name."""
    env = os.environ.get("SCENESCOUT_SERVED_MODEL")
    if env:
        return env
    return "sft" if "sft" in (_llm_probe().get("models") or []) else MODEL


@app.get("/api/health")
def health():
    return JSONResponse({"ok": True, "site": str(SITE_DIR.name), "llm": _llm_probe(), "model": served_model()})


_BLENDER_CACHE: dict = {}
_BLENDER_SKIP = {"KW_power", "KW_pipelines", "KW_rail", "KW_fences", "KW_traffic", "KW_cranes", "KW_yard"}
_BLENDER_SPLIT = {"KW_buildings_ml", "KW_stockpiles"}       # tiled meshes: one entry per connected footprint
_KIND_COLLECTION = {"tank": "KW_tanks", "tank_farm": "KW_tank_farms", "jetty": "KW_jetties", "ship": "KW_ships"}
_BL_KIND = {"KW_ships": "ship", "KW_tank_farms": "tank_farm", "KW_stockpiles": "stockpile", "KW_tanks": "tank",
            "KW_jetties": "jetty"}


@dataclass
class _CatObj:
    """Adapter: a SiteView object in the shape envbuild.blender_objects.catalogue expects (twin Obj fields)."""

    id: str
    name: str
    kind: str
    geom: object
    height_m: float | None = None
    material: str | None = None
    aliases: tuple = ()
    tags: tuple = ()

    @classmethod
    def of(cls, o):
        return cls(o.id, o.name, o.kind, o.geom, o.height_m, o.material, tuple(o.aliases), tuple(o.tags))


def _ring(g) -> list[list[float]]:
    g = max(g.geoms, key=lambda p: p.area) if g.geom_type == "MultiPolygon" else g
    return [[round(x, 1), round(y, 1)] for x, y in g.exterior.coords] if g.geom_type == "Polygon" else []


def _components(xyz, cell_m: float = 4.0) -> list:
    """Split a tiled mesh into footprints: occupied cell_m grid cells of its vertices, grouped when within two cells
    of each other (flood fill), convex hull per group."""
    import numpy as np
    from shapely.geometry import MultiPoint

    ij = np.floor(xyz[:, :2] / cell_m).astype(int)
    cells: dict[tuple[int, int], list[int]] = {}
    for n, (i, j) in enumerate(map(tuple, ij)):
        cells.setdefault((i, j), []).append(n)
    label: dict[tuple[int, int], int] = {}
    groups: list[list[int]] = []
    for c0 in cells:
        if c0 in label:
            continue
        label[c0] = len(groups)
        stack, members = [c0], []
        while stack:
            c = stack.pop()
            members += cells[c]
            for di in (-2, -1, 0, 1, 2):
                for dj in (-2, -1, 0, 1, 2):
                    nb = (c[0] + di, c[1] + dj)
                    if nb in cells and nb not in label:
                        label[nb] = len(groups)
                        stack.append(nb)
        groups.append(members)
    out = []
    for members in groups:
        pts = xyz[members]
        if len(pts) >= 3:
            hull = MultiPoint([tuple(p) for p in pts[:, :2][:: max(1, len(pts) // 500)]]).convex_hull
            if hull.geom_type == "Polygon" and hull.area > 4.0:
                out.append((hull, float(pts[:, 2].min()), float(pts[:, 2].max())))
    return out


@app.get("/api/blender_objects")
def blender_objects():
    """The Blender world's objects for console picking, focus outlines, framing and place names (S4.2; STRICT rule:
    Blender is the only production geometry). Site ENU metres; belief-side data only (scene geometry + the site's
    object register; no truth, no world_cache).
    in_tools=true: the active site's objects passed through envbuild.blender_objects.catalogue (refs exactly as the
    tools emit them; modelled heights; the twin's synthetic ships are dropped there). in_tools=false: Blender-only
    structures from rf_scene_blender/objects.json (ML buildings and stockpiles split into footprints, observed ships
    the site's tools do not know, ...); the `clutter` group and linear / illustrative collections are skipped."""
    import numpy as np
    import pyarrow.parquet as pq
    from shapely.geometry import MultiPoint

    from envbuild import blender_objects as bo

    site = _site()
    key = (str(Path(SITE_DIR).resolve()), site.version)
    if key in _BLENDER_CACHE:
        return _BLENDER_CACHE[key]
    f = bo.BL / "objects.json"
    if not f.exists():
        raise HTTPException(404, "no Blender export (data/site_kandla/rf_scene_blender/objects.json)")
    man = json.loads(f.read_text())
    shapes: dict[str, list[dict]] = {}
    for r in man["shapes"]:
        if r["group"] == "structure":
            shapes.setdefault(r["object"], []).append(r)
    try:
        ns = {r["id"]: r.get("name_source") for r in
              pq.read_table(Path(site.dir) / "objects.parquet", columns=["id", "name_source"]).to_pylist()}
    except Exception:  # noqa: BLE001 - older sites have no name_source column
        ns = {}
    cat, _ = bo.catalogue([_CatObj.of(o) for o in site.objects.values()]) if site.objects else ([], {})
    out, seen = [], set()
    for o in cat:
        if o.kind == "road" or o.id not in site.objects:
            continue
        sh = shapes.get(o.id, [])
        coll = sh[0]["collection"] if sh else _KIND_COLLECTION.get(o.kind, "KW_buildings")
        if coll in _BLENDER_SKIP:
            continue
        c = o.geom.representative_point()
        top = max((r["bbox"][5] for r in sh), default=None)
        out.append({"ref": f"obj:{o.id}", "id": o.id, "name": o.name, "kind": o.kind, "collection": coll,
                    "fidelity": sh[0]["fidelity"] if sh else "estimated",
                    "name_source": ns.get(o.id) or "synthetic", "ring": _ring(o.geom),
                    "center": [round(c.x, 1), round(c.y, 1)],
                    "base_m": round(min(r["bbox"][2] for r in sh), 2) if sh else 0.0,
                    "top_m": round(float(top if top is not None else (o.height_m or 0.0)), 2), "in_tools": True})
        seen.add(o.id)
    for name, sh in sorted(shapes.items()):
        coll = sh[0]["collection"]
        if name in seen or coll in _BLENDER_SKIP:
            continue
        xyz = np.concatenate([bo._ply_xyz(bo.BL / "meshes" / f"{r['shape']}.ply") for r in sh])
        if coll in _BLENDER_SPLIT:
            parts = _components(xyz)
        else:
            hull = MultiPoint([tuple(p) for p in xyz[:, :2][:: max(1, len(xyz) // 2000)]]).convex_hull
            parts = [(hull, float(xyz[:, 2].min()), float(xyz[:, 2].max()))] if hull.geom_type == "Polygon" else []
        for k, (g, z0, z1) in enumerate(parts):
            oid = name if len(parts) == 1 else f"{name}#{k}"
            c = g.representative_point()
            out.append({"ref": f"bl:{oid}", "id": oid, "name": sh[0].get("name"), "kind": _BL_KIND.get(coll, "building"),
                        "collection": coll, "fidelity": sh[0]["fidelity"],
                        "name_source": "blender" if sh[0].get("name") else "synthetic", "ring": _ring(g),
                        "center": [round(c.x, 1), round(c.y, 1)], "base_m": round(z0, 2), "top_m": round(z1, 2),
                        "in_tools": False})
    res = {"source": "blender", "frame": man.get("frame"),
           "geometry": "Blender port model: OSM footprints with modelled roofs, ML-detected buildings, Sentinel-2 "
                       "ships; heights estimated; simulation only",
           "n_in_tools": sum(o["in_tools"] for o in out), "objects": out}
    _BLENDER_CACHE[key] = res
    return res


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only; 0.0.0.0 exposes the "
                                                       "unauthenticated console to the network)")
    p.add_argument("--site", default=None, help="site directory (default $SCENESCOUT_SITE, else the active demo set's "
                                                       "scenario: $SCENESCOUT_DEMO_SET=v1|v2, v2 once v2 recordings "
                                                       "exist)")
    return p.parse_args(argv)


WARMUP: dict = {"state": "off"}


def warm_site() -> dict:
    """Fill the caches the console's first load reads (site geometry, tracks at the console's step, Blender objects;
    the last took ~5 s cold). Best effort, run in a thread at gateway start (S6 boot-time trim)."""
    t0 = time.time()
    try:
        site_geometry()
        tracks(step_s=2.0, frame="enu3")
        blender_objects()
        res = {"ok": True}
    except Exception as e:  # noqa: BLE001 - the endpoint reports the same error when the console asks
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    res["s"] = round(time.time() - t0, 1)
    print(f"[gateway] site caches warm {res}", flush=True)
    return res


def warm_llm(wait_s: float = 900.0, poll_s: float = 5.0) -> dict:
    """Wait (up to wait_s) for vLLM to list the served model, then send one warm-up request with the agent's exact
    first-turn prefix (serve.agent.warm_up): the first operator question then skips the cold ~7k-token prefill and
    adapter load (demo-v2 S5.6). Best effort; result kept in WARMUP and printed."""
    WARMUP.update(state="waiting")
    t_end = time.time() + wait_s
    while time.time() < t_end:
        _PROBE["t"] = 0.0                                   # fresh probe each poll
        probe = _llm_probe()
        model = served_model()
        if probe.get("ok") and model in (probe.get("models") or []):
            res = warm_up(AkshaEnv(SITE_DIR, model_id=model), model=model)
            WARMUP.update(state="done" if res.get("ok") else "failed", model=model, **res)
            print(f"[gateway] LLM warm-up {WARMUP}", flush=True)
            return dict(WARMUP)
        time.sleep(poll_s)
    WARMUP.update(state="timeout")
    print("[gateway] LLM warm-up: vLLM not ready, skipped", flush=True)
    return dict(WARMUP)


def main() -> None:
    a = parse_args()
    if a.site:
        set_site(a.site)
    ds = _demo_set_of_site()
    print(f"[gateway] site {SITE_DIR}; demo set "
          f"{next((k for k, v in DEMO_SETS.items() if v is ds), 'none (site has no demo set)')}", flush=True)
    threading.Thread(target=warm_site, name="site-warmup", daemon=True).start()
    if os.environ.get("SCENESCOUT_WARMUP", "1") != "0":
        threading.Thread(target=warm_llm, name="llm-warmup", daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
