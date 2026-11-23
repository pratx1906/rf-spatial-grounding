"""Family D (SPEC 6.2 D): active search with the drone-borne DF node D1 (stateful, D-024).

A drone-search task carries a payload prepared OFFLINE by the task generator: the target track, the start pose,
a lattice of candidate waypoints (two rings around the target's track estimate at t0) and, per waypoint, the
bearing D1 would measure there. The episode replays exactly those measurements, so agent-side code never touches
the world. The belief is a 2D Gaussian (prior = the track estimate's ellipse) updated by an iterated EKF per
bearing. Flight time advances the episode clock at `speed_mps`.
"""

from __future__ import annotations

import math
from typing import Annotated

import numpy as np
from pydantic import Field

from sim.belief.rf_cache import ellipse
from tools.errors import ToolError
from tools.registry import ToolContext, ToolOutput, register
from tools.types import Strict, pt, r1


def _cov_from_ellipse(a_m: float, b_m: float, theta_deg: float) -> np.ndarray:
    """Inverse of sim.belief.rf_cache.ellipse: a, b are ~2-sigma semi-axes, theta = compass bearing of the major axis."""
    th = math.radians(theta_deg)
    u = np.array([math.sin(th), math.cos(th)])            # major axis in (x east, y north)
    v = np.array([u[1], -u[0]])
    return (a_m / 2) ** 2 * np.outer(u, u) + (b_m / 2) ** 2 * np.outer(v, v)


class DroneSim:
    def __init__(self, payload: dict):
        self.p = payload
        self.pos = np.array([payload["start"]["x_m"], payload["start"]["y_m"], payload["start"]["z_m"]], float)
        self.t = float(payload["t0_s"])
        pr = payload["prior"]
        self.mean = np.array([pr["x_m"], pr["y_m"]], float)
        self.cov = _cov_from_ellipse(pr["a_m"], pr["b_m"], pr["theta_deg"])
        self.sigma = math.radians(payload.get("sigma_deg", 3.0))
        self.wps = {w["id"]: w for w in payload["waypoints"]}
        self.moves: list[str] = []
        self.budget = int(payload.get("budget_moves", 4))

    def estimate(self) -> dict:
        a, b, th = ellipse(self.cov)
        return {"position": pt(*self.mean), "uncertainty": {"kind": "ellipse", "a_m": r1(a), "b_m": r1(b),
                                                            "theta_deg": r1(th)}}

    @staticmethod
    def _update(mean, cov, px, py, brg_rad, sigma, iters: int = 3):
        x = mean.copy()
        for _ in range(iters):
            dx, dy = x[0] - px, x[1] - py
            r2 = max(dx * dx + dy * dy, 1.0)
            H = np.array([[dy / r2, -dx / r2]])
            pred = math.atan2(dx, dy)
            innov = (brg_rad - pred + math.pi) % (2 * math.pi) - math.pi
            S = float((H @ cov @ H.T)[0, 0]) + sigma ** 2
            K = (cov @ H.T) / S
            x = mean + (K[:, 0] * (innov + float((H @ (x - mean))[0])))
        dx, dy = x[0] - px, x[1] - py
        r2 = max(dx * dx + dy * dy, 1.0)
        H = np.array([[dy / r2, -dx / r2]])
        S = float((H @ cov @ H.T)[0, 0]) + sigma ** 2
        K = (cov @ H.T) / S
        return x, (np.eye(2) - K @ H) @ cov

    def expected_after(self, wid: str) -> float:
        w = self.wps[wid]
        brg = math.atan2(self.mean[0] - w["x_m"], self.mean[1] - w["y_m"])
        _, c = self._update(self.mean, self.cov, w["x_m"], w["y_m"], brg, self.sigma, iters=1)
        return ellipse(c)[0]

    def move(self, wid: str) -> dict:
        w = self.wps[wid]
        d = float(np.linalg.norm(self.pos[:2] - np.array([w["x_m"], w["y_m"]])))
        self.t += d / self.p.get("speed_mps", 12.0)
        self.pos = np.array([w["x_m"], w["y_m"], w["z_m"]], float)
        self.moves.append(wid)
        if w.get("bearing_deg") is not None:
            self.mean, self.cov = self._update(self.mean, self.cov, w["x_m"], w["y_m"], math.radians(w["bearing_deg"]),
                                               self.sigma)
        return {"flown_m": r1(d), "measurement": {"bearing_deg": w.get("bearing_deg"), "snr_db": w.get("snr_db")}}


def _sim(ctx: ToolContext) -> DroneSim:
    sim = ctx.episode.get("drone")
    if sim is None:
        raise ToolError("NOT_PERMITTED", "no drone-borne DF platform is assigned to this question",
                        "Drone tools only work in drone-search tasks; use the fixed DF nodes instead.")
    return sim


class DroneStateArgs(Strict):
    pass


@register("drone_state", "D", "Pose, moves left, clock and current target estimate of the drone-borne DF (D1).",
          DroneStateArgs)
def drone_state(ctx: ToolContext, a: DroneStateArgs) -> ToolOutput:
    s = _sim(ctx)
    return ToolOutput({"sensor": "sensor:D1", "pose": pt(*s.pos), "t_s": r1(s.t), "target": s.p["target_track"],
                       "moves_used": len(s.moves), "moves_left": s.budget - len(s.moves),
                       "waypoints_flown": list(s.moves), "estimate": s.estimate()},
                      ["sensor:D1", s.p["target_track"]], {"estimate": "fused belief: track estimate + D1 bearings"})


class PlanWaypointsArgs(Strict):
    target_ref: str
    k: Annotated[int, Field(ge=1, le=8)] = 4


@register("plan_waypoints", "D", "Rank candidate waypoints by expected uncertainty after one bearing from there.",
          PlanWaypointsArgs)
def plan_waypoints(ctx: ToolContext, a: PlanWaypointsArgs) -> ToolOutput:
    s = _sim(ctx)
    tgt = a.target_ref.strip().split("@")[0]
    tgt = tgt if tgt.startswith("trk:") else f"trk:{tgt.replace('track:', '')}"
    if tgt != s.p["target_track"]:
        raise ToolError("INVALID_ARGS", f"the drone is tasked on {s.p['target_track']}",
                        f"Pass target_ref={s.p['target_track']}.")
    rows = []
    for wid, w in s.wps.items():
        if wid in s.moves:
            continue
        dist = float(np.hypot(w["x_m"] - s.pos[0], w["y_m"] - s.pos[1]))
        rows.append({"ref": wid, "position": pt(w["x_m"], w["y_m"], w["z_m"]), "expected_a_m": r1(s.expected_after(wid)),
                     "flight_s": r1(dist / s.p.get("speed_mps", 12.0))})
    rows.sort(key=lambda r: (r["expected_a_m"], r["flight_s"]))
    return ToolOutput({"target": s.p["target_track"], "current_a_m": s.estimate()["uncertainty"]["a_m"],
                       "waypoints": rows[: a.k], "moves_left": s.budget - len(s.moves)},
                      [r["ref"] for r in rows[: a.k]],
                      {"expected_a_m": "semi-major axis (~2 sigma) after one 3-deg bearing taken at the waypoint"})


class MoveDroneArgs(Strict):
    wp_ref: str


@register("move_drone", "D", "Fly D1 to a waypoint, take a bearing on the target and update the estimate.",
          MoveDroneArgs, stateful=True)
def move_drone(ctx: ToolContext, a: MoveDroneArgs) -> ToolOutput:
    s = _sim(ctx)
    wid = a.wp_ref.strip()
    if not wid.startswith("wp:"):
        wid = f"wp:{wid}"
    if wid not in s.wps:
        raise ToolError("UNKNOWN_REF", f"{a.wp_ref} is not a planned waypoint", "Use wp: refs from plan_waypoints.")
    if len(s.moves) >= s.budget:   # out of moves wins: the only useful next step is to submit
        raise ToolError("BUDGET_EXCEEDED", f"the drone has used all {s.budget} moves",
                        "Submit drone_search with the current estimate from drone_state.")
    if wid in s.moves:
        raise ToolError("INVALID_ARGS", f"{wid} was already flown; a second bearing from there adds no information",
                        "Choose another waypoint from plan_waypoints, or submit.")
    m = s.move(wid)
    return ToolOutput({"wp": wid, "pose": pt(*s.pos), "t_s": r1(s.t), **m, "estimate": s.estimate(),
                       "moves_left": s.budget - len(s.moves)}, [wid, "sensor:D1", s.p["target_track"]],
                      {"measurement": "bearing deg clockwise from north at D1 (drone-borne DF, ~3 deg)"})
