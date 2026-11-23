"""Capture the console (C-route) shots of video/shot_list.md as deterministic 1920x1080 30 fps clips.

    uv run --no-project --with playwright python scripts/capture_console_clips.py [--url http://localhost:8085]
           [--shots S03,S06] [--frames-dir <tmp>] [--keep-frames]

Needs a running gateway (python -m serve.gateway) and ffmpeg on PATH. The console runs in video mode (?video=1: chrome
off, public-safe labels) with the capture hook (?capture=1: window.ssCapture). Time and camera are driven from here:
every frame sets the camera, advances a virtual clock by 1/30 s (recorded agent events fire on that clock), renders one
frame and screenshots it. Output: video/public/clips/<shot>.mp4 (H.264 CRF 12) and video/public/clips/capture_log.json.
Every clip has 2 s handles before and after its move.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from playwright.async_api import async_playwright

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "video" / "public" / "clips"
REPLAYS = REPO / "data" / "demo_replays_v2b"
TASKS = REPO / "data" / "demo_tasks_kandla_v2b.json"
FPS = 30
HANDLE = 2.0
NEVER = 1e9                                      # schedule time for recorded events a shot leaves out


# ------------------------------------------------------------------ camera paths
def ease(k: float) -> float:
    k = min(1.0, max(0.0, k))
    return 4 * k ** 3 if k < 0.5 else 1 - (-2 * k + 2) ** 3 / 2


def lerp_pose(a: dict, b: dict, k: float) -> dict:
    dh = ((b["heading"] - a["heading"] + 540) % 360) - 180
    return {"target": [a["target"][i] + (b["target"][i] - a["target"][i]) * k for i in range(3)],
            "dist": math.exp(math.log(a["dist"]) + (math.log(b["dist"]) - math.log(a["dist"])) * k),
            "heading": (a["heading"] + dh * k) % 360, "pitch": a["pitch"] + (b["pitch"] - a["pitch"]) * k}


def move(a: dict, b: dict, move_s: float, eased: bool = True):
    """hold a for the head handle, move to b over move_s, hold b for the tail handle"""
    def f(t: float) -> dict:
        k = (t - HANDLE) / move_s
        return lerp_pose(a, b, ease(k) if eased else min(1.0, max(0.0, k)))
    return f


def drift(a: dict, b: dict, total_s: float):
    """constant-speed drift over the whole clip, handles included (cross-dissolve friendly)"""
    return lambda t: lerp_pose(a, b, t / total_s)


def pose(target, dist, heading, pitch) -> dict:
    return {"target": [float(target[0]), float(target[1]), float(target[2]) if len(target) > 2 else 0.0],
            "dist": float(dist), "heading": float(heading), "pitch": float(pitch)}


def rec(task_id: str, who: str = "sft") -> dict:
    return json.loads((REPLAYS / f"{task_id}__{who}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the browser side
class Console:
    def __init__(self, browser, url: str):
        self.browser, self.url, self.pg = browser, url, None
        self.errors: list[str] = []

    async def open(self, video: str = "1") -> None:
        if self.pg:
            await self.pg.close()
        self.pg = await self.browser.new_page(viewport={"width": 1920, "height": 1080})
        self.pg.on("console", lambda m: self.errors.append(f"{m.type}: {m.text[:200]}") if m.type == "error" else None)
        self.pg.on("pageerror", lambda e: self.errors.append(f"pageerror: {str(e)[:200]}"))
        await self.pg.goto(f"{self.url}/?video={video}&capture=1")
        await self.pg.wait_for_function("window.ssCapture && document.getElementById('curtain').hidden === true", timeout=300000)
        await self.ev("ssCapture.settle()")
        await self.ev("ssCapture.freeze()")

    async def ev(self, js: str, arg=None):
        return await self.pg.evaluate(js, arg) if arg is not None else await self.pg.evaluate(js)

    async def warm(self, cam: dict, n: int = 12) -> None:
        for _ in range(n):
            await self.ev("([c]) => { ssCapture.setCamera(c); return ssCapture.step(0); }", [cam])

    async def render(self, shot: str, frames_dir: Path, total_s: float, cam_at, per_frame=None) -> dict:
        d = frames_dir / shot
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        n = round(total_s * FPS)
        first = last = None
        fired = []
        for i in range(n):
            t = i / FPS
            if per_frame:
                await per_frame(t)
            cam = cam_at(t)
            r = await self.ev("([c]) => { ssCapture.setCamera(c); return ssCapture.step(1 / 30); }", [cam])
            if r["fired"]:
                fired.append(round(t, 3))
            await self.pg.screenshot(path=str(d / f"{i:05d}.png"))
            first = first or cam
            last = cam
        return {"frames": n, "duration_s": round(n / FPS, 3), "camera_start": first, "camera_end": last, "event_frames_s": fired}


def encode(frames: Path, mp4: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-profile:v", "high", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(mp4)], check=True)


# ------------------------------------------------------------------ shots
SITE = {"scenario": "data/scenarios/kandla_s0_X", "seed": 0, "demo_set": "v2b"}


async def s03(c: Console, fd: Path) -> list[dict]:
    """two bearings from S2 on trk:5433 at t=470 s; close dolly onto S2; bearings only (first tool result)"""
    tid = "k00X-direct_path-00064"
    await c.open()
    s2 = (await c.ev("ssCapture.sensors()"))["S2"]
    a = pose([s2[0] - 60, s2[1] + 520, 0], 2600, 340, -32)
    b = pose([s2[0] - 40, s2[1] + 260, 0], 1000, 345, -25)
    await c.ev("ssCapture.simT(470)")
    await c.warm(a)
    sched = await c.ev("([t]) => ssCapture.play(t, 'sft')", [tid])
    times = [0.0] + [NEVER] * (len(sched) - 1)
    times[1:5] = [1.70, 1.70, 1.85, 1.85]               # assistant, df_hypotheses call + result (both bearings), ledger
    await c.ev("([t, ts]) => ssCapture.play(t, 'sft', {times: ts})", [tid, times])
    info = await c.render("S03_problem_bearings", fd, 2 * HANDLE + 2, move(a, b, 2.0))
    st = await c.ev("ssCapture.state()")
    return [{"id": "S03_problem_bearings", "task_id": tid, "who": "sft", **SITE, "sim_t_s": 470, **info,
             "content": "first df_hypotheses result of the recorded run only: S2 bearings 005.7 deg and 313.7 deg on trk:5433 (both drawn at once, at 1.85 s); no answer",
             "labels_on_screen": sorted(set(st["labels"]))}]


async def multipath_layer(c: Console) -> dict:
    mp = await c.ev("ssCapture.multipath('S4', 'ism_2400', '5')")
    return mp


async def s04(c: Console, fd: Path) -> list[dict]:
    await c.open()
    mp = await multipath_layer(c)
    s4 = mp["pose"]
    a = pose([s4[0] - 260, s4[1] - 330, 0], 3300, 0, -89.5)
    b = pose([s4[0] - 180, s4[1] - 260, 0], 3150, 0, -89.5)
    total = 2 * HANDLE + 2
    await c.warm(a)
    info = await c.render("S04_problem_multipath", fd, total, drift(a, b, total))
    return [{"id": "S04_problem_multipath", "layer": "multipath S4, 2.4 GHz, drone 5 m (ui/kandla/data/multipath, scripts/multipath_layer.py)",
             **SITE, **info, "hotspots": mp["clusters"][:3], "content": "top-down, north up, slow constant drift over the whole clip; worst 5 % outlined, dashed ring = sensor reach"}]


async def s05b(c: Console, fd: Path) -> list[dict]:
    await c.open()
    mp = await multipath_layer(c)
    a = pose([60, -380, 0], 5200, 330, -33)
    b = pose([-40, -300, 0], 4950, 338, -35)
    total = 2 * HANDLE + 4
    await c.warm(a)
    info = await c.render("S05b_reveal_console", fd, total, drift(a, b, total))
    return [{"id": "S05b_reveal_console", "layer": "multipath S4, 2.4 GHz, drone 5 m", **SITE, **info, "hotspots": mp["clusters"][:1],
             "content": "wide of the port with the S4 multipath drape; constant slow drift over the whole clip (the R-pass crane should end on camera_start to match)"}]


async def diagnose(c: Console, fd: Path, shot: str, h0: float, h1: float, pitch: float, dist: float) -> dict:
    tid = "k00X-direct_path-00064"
    await c.open()
    sens = await c.ev("ssCapture.sensors()")
    s2 = sens["S2"]
    centre = [(s2[0] - 31.5) / 2 - 60, (s2[1] - 320.8) / 2 + 40, 0]          # between S2 and the recorded fix
    a, b = pose(centre, dist, h0, pitch), pose(centre, dist * 0.92, h1, pitch - 2)
    await c.ev("ssCapture.simT(470)")
    await c.warm(a)
    sched = await c.ev("([t]) => ssCapture.play(t, 'sft')", [tid])
    times = [2.3 + s["t"] * (4.6 / sched[-1]["t"]) for s in sched]
    times[0] = 0.0
    await c.ev("([t, ts]) => ssCapture.play(t, 'sft', {times: ts})", [tid, times])
    info = await c.render(shot, fd, 2 * HANDLE + 6, move(a, b, 6.0))
    st = await c.ev("ssCapture.state()")
    return {"id": shot, "task_id": tid, "who": "sft", **SITE, "sim_t_s": 470, **info,
            "schedule_s": {s["name"] or s["type"]: round(t, 2) for s, t in zip(sched, times) if s["type"] in ("tool_result", "answer")},
            "content": "recorded trained run: S2 + all-sensor bearings, ray-traced fit (rt_match, 150 positions), answer: S2 005.7 deg direct (orange), 313.7 deg rejected (dashed reflection colour), fix +-37 m",
            "note": "this recording has no predict_aoa step ('Predicting multipath at S2'); the agent separates the bearings with the ray-traced fit_candidates instead",
            "labels_on_screen": sorted(set(st["labels"]))}


async def s06(c: Console, fd: Path) -> list[dict]:
    return [await diagnose(c, fd, "S06_diagnose", 340, 25, -36, 2300),
            await diagnose(c, fd, "S06_diagnose_t2", 30, 345, -48, 2000)]


def ellipse_log(r: dict, task: dict) -> dict:
    steps = []
    for e in r["events"]:
        if e["type"] == "tool_result" and e["name"] in ("drone_state", "move_drone"):
            d = e["result"]["data"]
            u = d["estimate"]["uncertainty"]
            steps.append({"tool": e["name"], "wp": d.get("wp"), "a_m": u["a_m"], "b_m": u["b_m"], "theta_deg": u["theta_deg"],
                          "position": d["estimate"]["position"]})
    ans = next(e["answer"] for e in r["events"] if e["type"] == "answer")
    key = task["answer_key"]["position"]
    fp = ans["final_position"]
    return {"source": "data/demo_replays_v2b/k00X-ds_loiter-00097__sft.json", "kind": "semi-axes a/b of the estimate ellipse (~2 sigma)",
            "before_m": {"a_m": steps[0]["a_m"], "b_m": steps[0]["b_m"]}, "after_m": {"a_m": ans["uncertainty"]["a_m"], "b_m": ans["uncertainty"]["b_m"]},
            "per_step": steps, "final_error_vs_answer_key_m": round(math.hypot(fp["x_m"] - key["x_m"], fp["y_m"] - key["y_m"]), 1)}


async def locate(c: Console, fd: Path, shot: str, h0: float, h1: float, pitch: float, dist: float) -> dict:
    tid = "k00X-ds_loiter-00097"
    r = rec(tid)
    task = next(t for t in json.loads(TASKS.read_text(encoding="utf-8")) if t["task_id"] == tid)
    await c.open()
    centre = [-37.7, -324.5, 0]
    a, b = pose(centre, dist, h0, pitch), pose(centre, dist * 0.85, h1, pitch - 4)
    await c.ev("ssCapture.simT(450)")
    await c.warm(a)
    sched = await c.ev("([t]) => ssCapture.play(t, 'sft')", [tid])
    times = [2.2 + s["t"] * (5.4 / sched[-1]["t"]) for s in sched]
    times[0] = 0.0
    await c.ev("([t, ts]) => ssCapture.play(t, 'sft', {times: ts})", [tid, times])
    info = await c.render(shot, fd, 2 * HANDLE + 6, move(a, b, 6.0))
    return {"id": shot, "task_id": tid, "who": "sft", **SITE, **info,
            "schedule_s": [{"event": s["name"], "t": round(t, 2)} for s, t in zip(sched, times) if s["type"] == "tool_result" and s["name"] in ("drone_state", "move_drone", "submit")],
            "ellipse": ellipse_log(r, task),
            "content": "recorded trained run: DF drone D1 start, 3 moves (wp:05, wp:03, wp:01), estimate ellipse redrawn after each bearing, answer ellipse in orange",
            "note": "the console draws each DF-drone move as a jump to the waypoint plus a track segment (no in-flight animation)"}


async def s07(c: Console, fd: Path) -> list[dict]:
    return [await locate(c, fd, "S07_locate", 200, 240, -42, 1500),
            await locate(c, fd, "S07_locate_t2", 330, 15, -55, 1300)]


async def s08(c: Console, fd: Path) -> list[dict]:
    """same locked top-down framing; before: DF-accuracy map (5.8 GHz, 15 m, as the agent's query) and the question's
    TF-3 zone; after: the recorded run places the node"""
    tid = "k00X-suggest_one-00116"
    out = []
    cam = None
    for shot, upto in (("S08_place_before", 0), ("S08_place_after", None)):
        await c.open()
        await c.ev("ssCapture.rfmap('df_err', 'ism_5800', '15')")
        if cam is None:
            cam = await c.ev("([p]) => ssCapture.fit(p, {pitch: -89.5, heading: 0, pad: 0.16, minDist: 900})",
                             [[[-978.4, -387.3, 0], [-645.0, -10.2, 0], [770.4, 2.2, 22], [770.4, 2.2, 0]]])
        await c.warm(cam)
        sched = await c.ev("([t]) => ssCapture.play(t, 'sft')", [tid])
        if upto == 0:
            times = [0.0] + [NEVER] * (len(sched) - 1)
        else:
            times = [0.0] + [2.2 + s["t"] * (1.6 / sched[-1]["t"]) for s in sched[1:]]
        await c.ev("([t, ts]) => ssCapture.play(t, 'sft', {times: ts})", [tid, times])
        info = await c.render(shot, fd, 2 * HANDLE + 3, lambda t: cam)
        out.append({"id": shot, "task_id": tid, "who": "sft", **SITE, **info, "camera_locked": cam,
                    "layer": "RF map 'DF accuracy' (model ensemble), 5.8 GHz, drone 15 m (the height the agent queried)",
                    "content": "question's TF-3 restricted zone outlined (question acknowledgement), DF-accuracy drape" +
                               ("" if upto == 0 else "; recorded run: suggest_node -> nodecand:building_b026 (predicted gain 4.1), answer in orange with beacon"),
                    "note": "the recorded trained run has no weak_zones step and no after-coverage raster, so the weak zone does not visibly shrink; the drape is the site-wide DF-accuracy map"})
    return out


async def s09(c: Console, fd: Path) -> list[dict]:
    await c.open()
    mp = await multipath_layer(c)
    end = await c.ev("ssCapture.mpSpotView(0)")
    s4 = mp["pose"]
    a = pose([s4[0] - 120, s4[1] - 180, 0], 3400, 205.4, -58)
    b = pose(end["target"], end["dist"], end["heading"], end["pitch"])
    await c.warm(a)
    info = await c.render("S09_flyto_multipath", fd, 2 * HANDLE + 3, move(a, b, 3.0))
    return [{"id": "S09_flyto_multipath", "layer": "multipath S4, 2.4 GHz, drone 5 m", **SITE, **info, "hotspot": mp["clusters"][0],
             "content": "camera travels to hot spot 1 with the end pose of the layer panel's 'Fly to worst area' (flyToSpot), eased over 3 s (the console's own flight takes ~1 s)"}]


async def collateral(c: Console, fd: Path, shot: str, tid: str) -> dict:
    r = rec(tid)
    fp = next(e for e in r["events"] if e["type"] == "tool_result" and e["name"] == "effect_footprint")["result"]["data"]
    x0, y0, x1, y1 = fp["bbox"]
    await c.open()
    cam0 = await c.ev("([p]) => ssCapture.fit(p, {pitch: -55, heading: 0, pad: 0.06, minDist: 1500})", [[[x0, y0, 0], [x1, y1, 0]]])
    a = pose(cam0["target"], cam0["dist"] * 1.08, 350, -58)
    b = pose(cam0["target"], cam0["dist"] * 0.78, 5, -52)
    await c.warm(a)
    sched = await c.ev("([t]) => ssCapture.play(t, 'sft')", [tid])
    times = [0.0] + [0.6 + s["t"] * (1.2 / sched[-1]["t"]) for s in sched[1:]]
    await c.ev("([t, ts]) => ssCapture.play(t, 'sft', {times: ts})", [tid, times])
    info = await c.render(shot, fd, 2 * HANDLE + 3, move(a, b, 3.0))
    ans = next(e["answer"] for e in r["events"] if e["type"] == "answer")
    st = await c.ev("ssCapture.state()")
    return {"id": shot, "task_id": tid, "who": "sft", **SITE, **info, "violations": ans.get("violations"),
            "requires_human_confirm": ans.get("requires_human_confirm"), "labels_on_screen": sorted(set(st["labels"])),
            "note": "operator gate (hold A to acknowledge) opens in the side panel, which video mode hides; band and power labels are removed on the map"}


async def s10(c: Console, fd: Path) -> list[dict]:
    a = await collateral(c, fd, "S10_collateral_gate", "k00X-jam_links-00128")
    a["content"] = "recorded run jam_links (S2 jammer): footprint bands drawn and outlined as the answer; collateral_check found 0 own links or civil receivers above threshold"
    b = await collateral(c, fd, "S10_collateral_gate_alt", "k00X-jam_protected-00121")
    b["content"] = "alternate: recorded run jam_protected (S1 jammer): footprint plus the protected assets it would hit, marked in signal orange"
    return [a, b]


async def s11(c: Console, fd: Path) -> list[dict]:
    """replay: when trk:5433 first entered the TF-6 zone (answer t=364 s); timeline scrub 330 -> 380 s, camera follows"""
    tid = "k00X-first_entry-00138"
    await c.open(video="tl")
    sched = await c.ev("([t]) => ssCapture.play(t, 'sft')", [tid])
    times = [0.0] * len(sched)                                       # the recorded answer is on screen from frame 0
    await c.ev("([t, ts]) => ssCapture.play(t, 'sft', {times: ts})", [tid, times])
    await c.ev("ssCapture.step(0)")
    t0, t1 = 330.0, 380.0
    pts = {}
    for k in range(0, 61):
        tt = t0 + (t1 - t0) * k / 60
        pts[k] = await c.ev("([r, t]) => ssCapture.trackPos(r, t)", ["trk:5433", tt])

    def sim_at(t: float) -> float:
        return t0 + (t1 - t0) * ease((t - HANDLE) / 3.0)

    def track_at(st: float):
        k = (st - t0) / (t1 - t0) * 60
        i = max(0, min(59, int(k)))
        a, b = pts[i], pts[i + 1]
        f = k - i
        return [a[j] + (b[j] - a[j]) * f for j in range(3)] if a and b else (a or b)

    def cam_at(t: float) -> dict:
        p = track_at(sim_at(t))
        return pose([p[0] * 0.6 + (-219.3) * 0.4, p[1] * 0.6 + (-316.5) * 0.4, 0], 1050, 15, -50)

    async def per_frame(t: float) -> None:
        await c.ev("([s]) => ssCapture.simT(s)", [sim_at(t)])

    await c.warm(cam_at(0))
    info = await c.render("S11_replay_scrub", fd, 2 * HANDLE + 3, cam_at, per_frame)
    st = await c.ev("ssCapture.state()")
    return [{"id": "S11_replay_scrub", "task_id": tid, "who": "sft", **SITE, **info, "sim_t_range_s": [t0, t1], "answer_t_s": 364.0,
             "content": "recorded answer drawn from frame 0 (evt:00131: trk:5433 entered the TF-6 restricted zone at t 364 s, beacon + zone fill); the timeline scrubs 330 -> 380 s (eased) and the camera follows the track; timeline bar kept (?video=tl) for the timestamp",
             "labels_on_screen": sorted(set(st["labels"]))}]


async def s12(c: Console, fd: Path) -> list[dict]:
    tid = "k00X-direct_path-00064"
    cam = pose([-130, -700, 0], 2700, 0, -42)
    out = []
    for shot, who in (("S12_untrained", "scenescout"), ("S12_trained", "sft")):
        await c.open()
        await c.ev("ssCapture.simT(470)")
        await c.warm(cam)
        sched = await c.ev("([t, w]) => ssCapture.play(t, w)", [tid, who])
        times = [0.0] + [0.5 + s["t"] * (HANDLE + 3 + 1.0 - 0.5) / sched[-1]["t"] for s in sched[1:]]
        await c.ev("([t, w, ts]) => ssCapture.play(t, w, {times: ts})", [tid, who, times])
        info = await c.render(shot, fd, 2 * HANDLE + 3, lambda t: cam)
        s = rec(tid, who)["summary"]
        out.append({"id": shot, "half": "left" if who == "scenescout" else "right", "task_id": tid, "who": who, "model": s["label"],
                    "end_reason": s["end_reason"], "score": s["score"], "turns": s["turns"], **SITE, **info, "camera_locked": cam,
                    "content": "recorded run replayed over 0.5-6 s of the clip, same locked framing for both halves"})
    return out


SHOTS = {"S03": s03, "S04": s04, "S05b": s05b, "S06": s06, "S07": s07, "S08": s08, "S09": s09, "S10": s10, "S11": s11, "S12": s12}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8085")
    ap.add_argument("--shots", default=",".join(SHOTS))
    ap.add_argument("--frames-dir", default=str(REPO / "runs" / "video_frames"))
    ap.add_argument("--keep-frames", action="store_true")
    a = ap.parse_args()
    fd = Path(a.frames_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / "capture_log.json"
    log = json.loads(logf.read_text(encoding="utf-8")) if logf.exists() else {}
    log.setdefault("clips", {})
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
        c = Console(b, a.url)
        for k in a.shots.split(","):
            t = time.time()
            entries = await SHOTS[k](c, fd)
            for e in entries:
                mp4 = OUT / f"{e['id']}.mp4"
                encode(fd / e["id"], mp4)
                e["file"] = f"video/public/clips/{mp4.name}"
                e["console_errors"] = list(c.errors)
                if not a.keep_frames:
                    shutil.rmtree(fd / e["id"], ignore_errors=True)
                log["clips"][e["id"]] = e
                print(f"{e['id']}: {e['frames']} frames, {e['duration_s']} s", flush=True)
            c.errors.clear()
            print(f"  {k} took {time.time() - t:.0f} s", flush=True)
            log.update({"route": "C (live console, deterministic frame stepping)", "resolution": "1920x1080", "fps": FPS,
                        "handles_s": HANDLE, "codec": "H.264 high, CRF 12, yuv420p (ffmpeg libx264 preset slow)",
                        "console_url_flags": "?video=1&capture=1 (S11: ?video=tl&capture=1)",
                        "capture": "scripts/capture_console_clips.py; window.ssCapture in ui/kandla/console3.js: virtual clock, one rendered frame per step, Playwright screenshot per frame",
                        "public_safe": "header, side panel, camera toolbar, captions, legend hidden; map labels lose frequency, power, sensor-model and port/operator names; 'Simulation' badge kept",
                        "simulation_only": True})
            logf.write_text(json.dumps(log, indent=1), encoding="utf-8")
        await b.close()


if __name__ == "__main__":
    asyncio.run(main())
