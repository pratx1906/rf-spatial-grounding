"""Tidal current field for the Kandla creek (numpy + PIL only; runs on Windows or WSL).

    python -m envbuild.blender.water_flow   -> data/site_kandla/blender/water_flow.npz

ILLUSTRATIVE, not a hydrodynamic model. The current is the potential flow through the connected creek (Laplace
equation, banks impermeable, upstream end phi = 1, gulf end phi = 0), i.e. it follows the channel and speeds up
where the creek narrows. The ebb tide is shown (flow towards the gulf); speeds are scaled so the channel median is
EBB_MEDIAN_MS (Kandla creek tidal streams are of the order of 1-2 knots; not measured here).
Wakes: obstacles (jetty footprints on piles, moored ships, the cargo-wharf quay face) are advected downstream along
the field (semi-Lagrangian), giving the foam/turbulence streaks behind them. Salt pans and other water bodies
not connected to the creek are still water.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
B = REPO / "data" / "site_kandla" / "blender"
EBB_MEDIAN_MS = 0.9
COARSE = 4                      # 5 m terrain grid -> 20 m solver grid
OUT_STEP_M = 10.0


def _bilinear(a: np.ndarray, fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
    h, w = a.shape
    fx = np.clip(fx, 0, w - 1.001)
    fy = np.clip(fy, 0, h - 1.001)
    i, j = fx.astype(int), fy.astype(int)
    u, v = fx - i, fy - j
    return ((1 - u) * (1 - v) * a[j, i] + u * (1 - v) * a[j, i + 1] + (1 - u) * v * a[j + 1, i]
            + u * v * a[j + 1, i + 1])


def _component(water: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Water cells 4-connected to `seed` (iterative constrained dilation)."""
    comp = seed & water
    while True:
        g = comp.copy()
        g[1:] |= comp[:-1]
        g[:-1] |= comp[1:]
        g[:, 1:] |= comp[:, :-1]
        g[:, :-1] |= comp[:, 1:]
        g &= water
        if g.sum() == comp.sum():
            return comp
        comp = g


def solve_potential(water: np.ndarray, fixed: np.ndarray, value: np.ndarray, iters: int = 6000) -> np.ndarray:
    """Laplace on `water` with Dirichlet cells `fixed` (= `value`), zero-flux banks. Matrix-free CG."""
    nbr = []
    for sh in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nbr.append((sh, water & np.roll(water, sh, (0, 1))))
    for (dy, dx), m in nbr:                    # no wrap-around through the array border
        if dy == 1: m[0, :] = False
        if dy == -1: m[-1, :] = False
        if dx == 1: m[:, 0] = False
        if dx == -1: m[:, -1] = False
    free = water & ~fixed

    def A(p):                                   # graph Laplacian restricted to free cells
        out = np.zeros_like(p)
        for (sh, m) in nbr:
            out += np.where(m, p - np.roll(p, sh, (0, 1)), 0.0)
        return np.where(free, out, 0.0)

    phi0 = np.where(fixed, value, 0.0)
    b = -A(phi0)
    b = np.where(free, b, 0.0)
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = float((r * r).sum())
    r0 = rr
    for k in range(iters):
        Ap = A(p)
        a = rr / float((p * Ap).sum())
        x += a * p
        r -= a * Ap
        rn = float((r * r).sum())
        if rn < 1e-12 * r0:
            break
        p = r + (rn / rr) * p
        rr = rn
    print(f"[water_flow] CG {k + 1} iterations, rel residual {np.sqrt(rn / r0):.1e}")
    return np.where(free, x, phi0)


def main() -> None:
    t = np.load(B / "terrain.npz")
    xs, ys, sd = t["x"], t["y"], t["sd"]
    bundle = json.loads((B / "scene_input.json").read_text())
    x0, x1, y0, y1 = bundle["tile"]
    sdc = sd[::COARSE, ::COARSE]
    xc, yc = xs[::COARSE], ys[::COARSE]
    g = float(xc[1] - xc[0])
    water = sdc < -2.0
    Y = np.broadcast_to(yc[:, None], water.shape)
    wy = Y[water]
    top, bot = wy.max(), wy.min()
    up = water & (Y > top - 50)
    down = water & (Y < bot + 50)
    creek = _component(water, up) & _component(water, down)
    fixed = creek & (up | down)
    phi = solve_potential(creek, fixed, np.where(up, 1.0, 0.0))
    # velocity = -grad(phi), one-sided at banks
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)
    for arr, ax in ((gx, 1), (gy, 0)):
        pp, pm = np.roll(phi, -1, ax), np.roll(phi, 1, ax)
        cp, cm = np.roll(creek, -1, ax), np.roll(creek, 1, ax)
        both = cp & cm
        arr[:] = np.where(both, (pp - pm) / (2 * g), np.where(cp, (pp - phi) / g, np.where(cm, (phi - pm) / g, 0)))
    vx, vy = -gx * creek, -gy * creek
    sp = np.hypot(vx, vy)
    channel = creek & (sdc < -40)
    k = EBB_MEDIAN_MS / float(np.median(sp[channel]))
    vx, vy = vx * k, vy * k
    sp = np.hypot(vx, vy)
    lim = 2.0 * EBB_MEDIAN_MS
    f = np.where(sp > lim, lim / np.maximum(sp, 1e-9), 1.0)
    # slower in the shallow band near the banks (bed friction)
    bank = np.clip(-sdc / 40.0, 0.0, 1.0) ** 0.5
    vx, vy = vx * f * bank, vy * f * bank

    # output grid: 10 m over the tile
    ox = np.arange(x0, x1 + 0.1, OUT_STEP_M)
    oy = np.arange(y0, y1 + 0.1, OUT_STEP_M)
    FX = (ox[None, :] - xc[0]) / g
    FY = (oy[:, None] - yc[0]) / g
    FX, FY = np.broadcast_to(FX, (len(oy), len(ox))), np.broadcast_to(FY, (len(oy), len(ox)))
    VX, VY = _bilinear(vx, FY, FX).astype(np.float32), _bilinear(vy, FY, FX).astype(np.float32)

    # obstacles: jetties on piles, moored ships, wharf quay face (water side)
    H, W = VX.shape
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    to_px = lambda P: [((p[0] - x0) / OUT_STEP_M, (p[1] - y0) / OUT_STEP_M) for p in P]
    for o in bundle["objects"]:
        if o["kind"] == "jetty" and o.get("outer"):
            if o.get("wharf") or o.get("area_m2", 0) > 100000:
                dr.line(to_px(o["outer"] + o["outer"][:1]), fill=110, width=2)     # quay face: weaker eddies
            else:
                dr.polygon(to_px(o["outer"]), fill=255)
    s2 = json.loads((B / "s2_structures.json").read_text())
    for sh in s2.get("ships", []):
        c = np.array(sh["centroid"])
        a = np.radians(sh["axis_deg"])
        u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        L, Bm = sh["length_m"] / 2, min(sh["beam_m"], 0.16 * sh["length_m"] + 4) / 2
        dr.polygon(to_px([c + u * L + v * Bm, c - u * L + v * Bm, c - u * L - v * Bm, c + u * L - v * Bm]), fill=255)
    O = np.asarray(im, np.float32) / 255.0
    wetc = _bilinear(creek.astype(np.float32), FY, FX) > 0.5
    O *= wetc
    # semi-Lagrangian advection of the obstacle mask downstream: W(p) = max(O(p), d * W(p - v dt))
    dt = 10.0
    Wk = O.copy()
    gy_, gx_ = np.mgrid[0:H, 0:W].astype(np.float32)
    bx, by = gx_ - VX * dt / OUT_STEP_M, gy_ - VY * dt / OUT_STEP_M
    for _ in range(45):
        Wk = np.maximum(O, 0.955 * _bilinear(Wk, by, bx))
    # widen slightly (turbulent spreading) with a small box blur
    pad = np.pad(Wk, 1, mode="edge")
    Wk = sum(pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    Wk = (Wk * wetc).astype(np.float32)

    np.savez_compressed(B / "water_flow.npz", vx=VX, vy=VY, wake=Wk, x0=x0, y0=y0, step=OUT_STEP_M,
                        note="ILLUSTRATIVE ebb-tide potential flow; speeds scaled, not measured")
    spo = np.hypot(VX, VY)
    print(f"[water_flow] grid {W}x{H} @ {OUT_STEP_M} m; speed median(channel) "
          f"{np.median(spo[spo > 0.05]):.2f} m/s, max {spo.max():.2f}; wake cells {(Wk > 0.2).sum()}")


if __name__ == "__main__":
    main()
