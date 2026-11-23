"""Drones and swarms in the Blender world, flown on the simulation's own trajectories, plus the video shot list.

Trajectories come from data/site_kandla/truth/render/drones.json, exported offline through tasks.truth_access
(the only truth reader). Drone geometry is illustrative (generic ~0.8 m quadcopter). Positions between the 2 s
samples use centripetal Catmull-Rom so motion is smooth at 30-60 fps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

import bl_geometry as G

REPO = Path(__file__).resolve().parents[2]
TRAJ = REPO / "data" / "site_kandla" / "truth" / "render" / "drones.json"
FPS = 30


# ---------------------------------------------------------------------------------------------- trajectories
class Scenario:
    def __init__(self, path: Path = TRAJ):
        raw = json.loads(path.read_text())
        self.e = {k: {"kind": v["kind"], "swarm": v["swarm_id"], "t": np.asarray(v["t"], float),
                      "p": np.asarray(v["xyz"], float)} for k, v in raw.items()}

    def pos(self, eid: str, t: float):
        d = self.e[eid]
        T, P = d["t"], d["p"]
        if t < T[0] or t > T[-1]:
            return None
        i = int(np.clip(np.searchsorted(T, t) - 1, 0, len(T) - 2))
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(T) - 1)]
        u = (t - T[i]) / (T[i + 1] - T[i])
        return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                      + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)

    def vel(self, eid: str, t: float, dt: float = 0.5):
        a, b = self.pos(eid, t - dt), self.pos(eid, t + dt)
        if a is None or b is None:
            return np.zeros(3)
        return (b - a) / (2 * dt)

    def centroid(self, swarm: str, t: float):
        ps = [self.pos(k, t) for k, v in self.e.items() if v["swarm"] == swarm]
        ps = [p for p in ps if p is not None]
        return np.mean(ps, 0) if ps else None

    def swarm_vel(self, swarm: str, t: float, dt: float = 2.0):
        a, b = self.centroid(swarm, t - dt), self.centroid(swarm, t + dt)
        if a is None or b is None:
            return np.array([1.0, 0, 0])
        return (b - a) / (2 * dt)


# ---------------------------------------------------------------------------------------------- drone asset
def _mats():
    def pm(name, color, rough, metal=0.0, emit=0.0, alpha=1.0):
        m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        m.use_nodes = True
        p = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        p.inputs["Base Color"].default_value = (*color, 1)
        p.inputs["Roughness"].default_value = rough
        p.inputs["Metallic"].default_value = metal
        if emit:
            p.inputs["Emission Color"].default_value = (*color, 1)
            p.inputs["Emission Strength"].default_value = emit
        if alpha < 1:
            p.inputs["Alpha"].default_value = alpha
            if hasattr(m, "surface_render_method"):
                m.surface_render_method = "BLENDED"
        return m
    return {"drone_body": pm("drone_body", (0.04, 0.04, 0.045), 0.35, 0.2),
            "drone_white": pm("drone_white", (0.75, 0.76, 0.76), 0.35),
            "drone_arm": pm("drone_arm", (0.02, 0.02, 0.02), 0.5),
            "rotor_blur": pm("rotor_blur", (0.05, 0.05, 0.05), 0.4, alpha=0.22),
            "led_red": pm("led_red", (1.0, 0.05, 0.02), 0.3, emit=60.0),
            "led_green": pm("led_green", (0.05, 1.0, 0.1), 0.3, emit=60.0),
            "led_white": pm("led_white", (1.0, 1.0, 1.0), 0.3, emit=40.0),
            "lens": pm("lens", (0.01, 0.01, 0.02), 0.05, 0.8)}


def build_drone(name: str, body: str, coll) -> bpy.types.Object:
    M = _mats()
    mb = G.MB()
    mb.box((0, 0, 0), (0.30, 0.20, 0.11), body)
    mb.box((0.06, 0, 0.07), (0.16, 0.14, 0.04), body)
    R = 0.36
    for k in range(4):
        a = math.radians(45 + 90 * k)
        x, y = R * math.cos(a), R * math.sin(a)
        mb.tube((0.08 * math.cos(a), 0.06 * math.sin(a), 0.0), (x, y, 0.01), 0.016, 6, "drone_arm")
        mb.cylinder(x, y, 0.0, 0.055, 0.035, 12, "drone_arm")
        mb.tube((x * 0.85, y * 0.85, -0.01), (x * 0.85, y * 0.85, -0.17), 0.008, 4, "drone_arm")   # legs
    mb.box((0.17, 0, -0.08), (0.06, 0.06, 0.06), "drone_arm")                                  # gimbal
    mb.box((0.205, 0, -0.08), (0.012, 0.035, 0.035), "lens")
    mb.box((-0.16, 0.0, 0.02), (0.012, 0.02, 0.02), "led_white")
    mb.box((0.36 * math.cos(math.radians(45)), 0.36 * math.sin(math.radians(45)) + 0.02, -0.02), (0.02, 0.02, 0.02),
           "led_green")
    mb.box((0.36 * math.cos(math.radians(-45)), 0.36 * math.sin(math.radians(-45)) - 0.02, -0.02), (0.02, 0.02, 0.02),
           "led_red")
    ob = mb.build(name, M, coll, props={"fidelity": "illustrative", "source": "generic quadcopter, synthetic"})
    rotors = G.MB()
    for k in range(4):
        a = math.radians(45 + 90 * k)
        rotors.cylinder(R * math.cos(a), R * math.sin(a), 0.062, 0.066, 0.19, 24, "rotor_blur")
    rob = rotors.build(name + "_rotors", M, coll)
    rob.parent = ob
    return ob


# ---------------------------------------------------------------------------------------------- scene setup
def setup(S: Scenario):
    coll = bpy.data.collections.get("KW_drones") or bpy.data.collections.new("KW_drones")
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    for o in list(coll.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for eid, d in S.e.items():
        if d["kind"] == "decoy":
            continue
        body = "drone_white" if d["kind"] == "friendly_drone" else "drone_body"
        ob = build_drone(f"KW_drone_{eid}", body, coll)
        ob["emitter_id"] = eid
        ob["swarm_id"] = d["swarm"] or ""
        ob["kind"] = d["kind"]
        ob.rotation_mode = "XYZ"


def place_drones(S: Scenario, t: float, frame: float):
    for ob in bpy.data.collections["KW_drones"].objects:
        eid = ob.get("emitter_id")
        if not eid:
            continue
        p = S.pos(eid, t)
        hide = p is None
        ob.hide_render = hide
        ob.hide_viewport = hide
        for ch in ob.children:
            ch.hide_render = hide
            ch.hide_viewport = hide
        if hide:
            continue
        v = S.vel(eid, t)
        sp = float(np.linalg.norm(v[:2]))
        yaw = math.atan2(v[1], v[0]) if sp > 0.3 else ob.rotation_euler.z
        pitch = min(0.25, sp * 0.015)
        ob.location = Vector(p)
        ob.rotation_euler = (0.0, pitch, yaw)


# ---------------------------------------------------------------------------------------------- shot list
def _ease(u):
    return u * u * (3 - 2 * u)


def shots(S: Scenario):
    """Each shot: frames, scenario start time, camera(t, u) -> (eye, target, lens). u = 0..1 within the shot."""
    def establish(t, u):
        a = np.array([-2300.0, -2600.0, 1150.0])
        b = np.array([-900.0, -1500.0, 520.0])
        return a + (b - a) * _ease(u), np.array([500.0, 300.0, 0.0]), 24.0

    def chase1(t, u):
        hero = S.pos("swarm_1_m6", t)
        v = S.vel("swarm_1_m6", t, 2.0)
        d = v / (np.linalg.norm(v) + 1e-9)
        side = np.array([-d[1], d[0], 0.0])
        k = _ease(u)
        eye = hero - d * (9.0 + 14.0 * k) + side * (2.5 + 6.0 * k) + np.array([0, 0, 1.2 + 3.0 * k])
        return eye, hero + d * 30.0 + np.array([0, 0, -2.0 - 4.0 * k]), 32.0

    def tankers(t, u):
        c = S.centroid("swarm_1", t)
        eye = np.array([1100.0 - 25 * u, 150.0 + 20 * u, 9.0 + 3 * u])
        return eye, np.array([985.0, 265.0, 26.0]) * 0.85 + c * 0.15, 24.0

    def tankfarm(t, u):
        c = S.centroid("swarm_2", t)
        eye = c + np.array([-55.0 + 20 * u, -48.0, -6.0])
        return eye, c + np.array([25.0, 0.0, -14.0]), 30.0

    def node(t, u):
        s = np.array([942.7, 956.4, 25.0])
        ang = math.radians(140 + 60 * _ease(u))
        eye = s + np.array([6.5 * math.cos(ang), 6.5 * math.sin(ang), 0.6 - 0.8 * u])
        return eye, s + np.array([-6.5 * math.cos(ang), -6.5 * math.sin(ang), -3.0]), 30.0

    def finale(t, u):
        c = S.centroid("swarm_1", t)
        k = _ease(u) ** 1.3
        e0 = c + np.array([28.0, -36.0, 3.0])
        e1 = c + np.array([1300.0, 250.0, 480.0])
        tgt = c * (1 - k) + np.array([100.0, -250.0, 0.0]) * k
        return e0 + (e1 - e0) * k, tgt, 30.0 - 6 * u

    return [
        {"name": "establish", "sec": 8, "t0": 300.0, "cam": establish},
        {"name": "swarm1_chase", "sec": 12, "t0": 330.0, "cam": chase1},
        {"name": "swarm1_tankers", "sec": 10, "t0": 418.0, "cam": tankers},
        {"name": "swarm2_tankfarm", "sec": 10, "t0": 545.0, "cam": tankfarm},
        {"name": "df_node_S1", "sec": 8, "t0": 450.0, "cam": node},
        {"name": "finale", "sec": 10, "t0": 450.0, "cam": finale},
    ]


class Director:
    def __init__(self, S: Scenario):
        self.S = S
        self.shots = shots(S)
        f = 1
        for sh in self.shots:
            sh["f0"] = f
            sh["f1"] = f + sh["sec"] * FPS - 1
            f = sh["f1"] + 1
        self.last = f - 1
        cd = bpy.data.cameras.get("cam_video") or bpy.data.cameras.new("cam_video")
        cd.clip_start, cd.clip_end = 1.0, 45000
        self.cam = bpy.data.objects.get("cam_video") or bpy.data.objects.new("cam_video", cd)
        if self.cam.name not in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.link(self.cam)

    def shot_at(self, f):
        for sh in self.shots:
            if sh["f0"] <= f <= sh["f1"] + 0.999:
                return sh
        return self.shots[-1]

    def apply(self, scene):
        f = scene.frame_current + scene.frame_subframe
        sh = self.shot_at(f)
        u = (f - sh["f0"]) / max(1, sh["f1"] - sh["f0"])
        t = sh["t0"] + (f - sh["f0"]) / FPS
        place_drones(self.S, t, f)
        eye, tgt, lens = sh["cam"](t, min(max(u, 0.0), 1.0))
        self.cam.location = Vector(eye)
        self.cam.rotation_euler = (Vector(tgt) - Vector(eye)).to_track_quat("-Z", "Y").to_euler()
        self.cam.data.lens = lens
        scene.camera = self.cam


def hud(scene, cam) -> dict:
    """Screen-space boxes for every visible drone (for the C-UAS style overlay drawn after rendering)."""
    from bpy_extras.object_utils import world_to_camera_view
    out = []
    for ob in bpy.data.collections["KW_drones"].objects:
        eid = ob.get("emitter_id")
        if not eid or ob.hide_render:
            continue
        p = world_to_camera_view(scene, cam, ob.location)
        if p.z <= 0.5 or not (-0.05 < p.x < 1.05 and -0.05 < p.y < 1.05):
            continue
        out.append({"id": eid, "kind": ob.get("kind"), "swarm": ob.get("swarm_id"), "x": p.x, "y": 1 - p.y,
                    "range_m": round(p.z, 1), "alt_m": round(ob.location.z, 1)})
    return {"drones": out}


_DIRECTOR = None


def install(scene=None):
    """Create drones and register the frame handler (live session or headless render)."""
    global _DIRECTOR
    scene = scene or bpy.context.scene
    S = Scenario()
    setup(S)
    _DIRECTOR = Director(S)
    for h in list(bpy.app.handlers.frame_change_pre):
        if getattr(h, "__name__", "") == "_kw_frame":
            bpy.app.handlers.frame_change_pre.remove(h)

    def _kw_frame(scn, depsgraph=None):
        _DIRECTOR.apply(scn)
    bpy.app.handlers.frame_change_pre.append(_kw_frame)
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, _DIRECTOR.last
    scene.frame_set(1)
    return _DIRECTOR
