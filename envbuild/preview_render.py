"""Render preview images of the twin from preview.json (BEV map + oblique 3D). Needs only numpy + matplotlib.

    python -m envbuild.preview_render [--site data/site_kandla] [--t 300]
Writes <site>/preview/bev.png and oblique.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPoly  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SEA, LAND, INK = "#13222e", "#23302b", "#8fa6ad"
KIND_COLOR = {"tank": "#b9c2c6", "tank_farm": None, "jetty": "#9a8f7a", "building": "#5d6b66", "silo": "#7d8a85",
              "substation": "#c9a227", "ship": "#7a4a3a", "road": "#3b4743"}
GROUP_COLOR = {"G1": "#ff6b6b", "G2": "#ffd166", None: "#5fd3e8"}


def pos_at(track, t):
    pts = [p for p in track["pts"] if p[0] <= t]
    return pts[-1] if pts and t - pts[-1][0] <= 6 else None


def bev(d: dict, t: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 13), dpi=110)
    fig.patch.set_facecolor(SEA)
    ax.set_facecolor(SEA)
    for z in d["zones"]:
        if z["id"] == "kandla_creek":
            ax.add_patch(MplPoly(z["poly"], closed=True, fc="#1b3445", ec="none", zorder=1))
    for z in d["zones"]:
        if z["kind"] == "restricted":
            ax.add_patch(MplPoly(z["poly"], closed=True, fc="#c2477e22", ec="#c2477e", lw=1.2, ls="--", zorder=2))
        elif z["kind"] == "perimeter":
            ax.add_patch(MplPoly(z["poly"], closed=True, fc="none", ec=INK, lw=1, ls=":", zorder=2))
        elif z["id"] == "hero_zone":
            ax.add_patch(MplPoly(z["poly"], closed=True, fc="none", ec="#f2d16b", lw=1.5, zorder=2))
        elif z["id"] == "kict":
            ax.add_patch(MplPoly(z["poly"], closed=True, fc="#2c3d47", ec=INK, lw=0.8, zorder=1))
    for o in sorted(d["objects"], key=lambda o: o["kind"] != "road"):
        c = KIND_COLOR.get(o["kind"], "#5d6b66")
        if c is None:
            continue
        ec = "#e0a43a" if o["protected"] else "none"
        ax.add_patch(MplPoly(o["poly"], closed=True, fc=c, ec=ec, lw=0.6, zorder=3))
    for z in d["zones"]:
        if z["kind"] == "restricted" and "tank_farm" in z["id"]:
            p = np.array(z["poly"])
            ax.text(p[:, 0].min(), p[:, 1].max() + 30, z["name"].replace(" restricted zone", ""), color="#e58fb5",
                    fontsize=8, zorder=6)
    for tr in d["tracks"]:
        past = np.array([p for p in tr["pts"] if t - 150 <= p[0] <= t])
        col = GROUP_COLOR.get(tr["group"], "#5fd3e8")
        if len(past) > 1:
            ax.plot(past[:, 1], past[:, 2], color=col, lw=1.2, alpha=0.6, zorder=5)
        p = pos_at(tr, t)
        if p:
            ax.scatter([p[1]], [p[2]], s=28, color=col, edgecolor="black", lw=0.5, zorder=7)
    for s in d["sensors"]:
        if s["type"] == "fixed_df":
            ax.scatter([s["x_m"]], [s["y_m"]], marker="^", s=120, color="#dfe7e4", edgecolor="black", zorder=8)
            ax.text(s["x_m"] + 40, s["y_m"] + 40, s["id"], color="#dfe7e4", fontsize=10, weight="bold", zorder=8)
    hz = d["meta"]["hero_zone_m"]
    m = 2400
    ax.set_xlim(hz[0] - m + 1000, hz[2] + m - 1000)
    ax.set_ylim(hz[1] - m + 1000, hz[3] + m - 1000)
    ax.set_aspect("equal")
    ax.tick_params(colors=INK)
    for sp in ax.spines.values():
        sp.set_color(INK)
    ax.set_xlabel("east (m)", color=INK)
    ax.set_ylabel("north (m)", color=INK)
    fid = d["meta"]["fidelity"] or {}
    ax.set_title(f"Kandla Creek twin, BEV at t = {t:.0f} s   |   geometry: OSM, heights estimated, sensors and drones "
                 f"synthetic, RF: {d['meta']['rf_backend']}", color="#dfe7e4", fontsize=10)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=lab) for lab, c in
               (("swarm G1", GROUP_COLOR["G1"]), ("swarm G2", GROUP_COLOR["G2"]), ("single track", "#5fd3e8"))]
    handles += [plt.Line2D([], [], color="#c2477e", ls="--", label="restricted zone (proposed)"),
                plt.Line2D([], [], color="#e0a43a", lw=2, label="protected asset (proposed)"),
                plt.Line2D([], [], color="#dfe7e4", marker="^", ls="", label="DF sensor (synthetic)")]
    ax.legend(handles=handles, loc="lower left", facecolor="#182630", edgecolor=INK, labelcolor="#dfe7e4",
              fontsize=9)
    ax.text(0.99, 0.005, "(c) OpenStreetMap contributors", transform=ax.transAxes, ha="right", color=INK,
            fontsize=8)
    fig.tight_layout()
    fig.savefig(out, facecolor=SEA)
    plt.close(fig)


def prism(poly, h):
    p = np.array(poly)[:-1]
    faces = [[(x, y, h) for x, y in p]]
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        faces.append([(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], h), (a[0], a[1], h)])
    return faces


def oblique(d: dict, t: float, out: Path, azim: float = -60, elev: float = 28) -> None:
    hz = d["meta"]["hero_zone_m"]
    fig = plt.figure(figsize=(15, 10), dpi=110)
    fig.patch.set_facecolor(SEA)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.set_facecolor(SEA)
    xs = np.array([hz[0] - 300, hz[2] + 300])
    X, Y = np.meshgrid(xs, np.array([hz[1] - 300, hz[3] + 300]))
    ax.plot_surface(X, Y, np.zeros_like(X), color=LAND, alpha=1.0, zorder=0)
    faces, cols = [], []
    for o in d["objects"]:
        c = KIND_COLOR.get(o["kind"])
        if c is None or o["h"] < 1:
            continue
        p = np.array(o["poly"])
        if p[:, 0].max() < hz[0] - 300 or p[:, 0].min() > hz[2] + 300 or p[:, 1].max() < hz[1] - 300 \
                or p[:, 1].min() > hz[3] + 300:
            continue
        f = prism(o["poly"], o["h"] * 3)                      # 3x vertical exaggeration for readability
        faces += f
        cols += ["#e0a43a" if o["protected"] and o["kind"] != "tank" else c] * len(f)
    ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolors="#00000033", linewidths=0.2))
    for tr in d["tracks"]:
        col = GROUP_COLOR.get(tr["group"], "#5fd3e8")
        past = np.array([p for p in tr["pts"] if t - 150 <= p[0] <= t and hz[0] - 300 <= p[1] <= hz[2] + 300
                         and hz[1] - 300 <= p[2] <= hz[3] + 300])
        if len(past) > 1:
            ax.plot(past[:, 1], past[:, 2], past[:, 3] * 3, color=col, lw=1.4)
            ax.scatter([past[-1, 1]], [past[-1, 2]], [past[-1, 3] * 3], color=col, s=30, edgecolor="black")
    for s in d["sensors"]:
        if s["type"] == "fixed_df":
            ax.plot([s["x_m"]] * 2, [s["y_m"]] * 2, [0, s["z_m"] * 3], color="#dfe7e4", lw=2)
            ax.text(s["x_m"], s["y_m"], s["z_m"] * 3 + 40, s["id"], color="#dfe7e4", fontsize=9)
    ax.set_xlim(hz[0] - 300, hz[2] + 300)
    ax.set_ylim(hz[1] - 300, hz[3] + 300)
    ax.set_zlim(0, 500)
    ax.set_box_aspect((1, 1, 0.18))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(f"Hero zone, oblique view at t = {t:.0f} s (heights x3; tanks, jetties, ships, buildings from the twin)",
                 color="#dfe7e4", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SEA)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, default=REPO / "data" / "site_kandla")
    ap.add_argument("--t", type=float, default=330.0)
    a = ap.parse_args()
    d = json.loads((a.site / "preview" / "preview.json").read_text())
    bev(d, a.t, a.site / "preview" / "bev.png")
    oblique(d, a.t, a.site / "preview" / "oblique.png")
    print("wrote", a.site / "preview" / "bev.png", a.site / "preview" / "oblique.png")


if __name__ == "__main__":
    main()
