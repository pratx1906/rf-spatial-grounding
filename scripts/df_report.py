"""Render the AIR7311 DF study: maps over the port and comparison charts (numpy + matplotlib only).

    python scripts/df_report.py --sensor S4      -> runs/air7311_<sensor>/figs/*.png
Needs data/site_kandla/preview/preview.json (python -m envbuild.preview_export) for the port outline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Polygon as MplPoly  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BG, INK = "#13222e", "#8fa6ad"
BANDS = {"ism_2400": "2.4 GHz", "ism_5800": "5.8 GHz", "sub_ghz_868_915": "868/915 MHz"}
MATCHED = {"ism_5800": "A_5g8", "ism_2400": "B_2g4", "sub_ghz_868_915": "C_sub1g"}


def outline(ax, prev, extent):
    for o in prev["objects"]:
        if o["kind"] in ("tank", "jetty", "building", "silo", "ship", "substation"):
            ax.add_patch(MplPoly(o["poly"], closed=True, fc="none", ec="#dfe7e4", lw=0.4, alpha=0.7, zorder=4))
    for z in prev["zones"]:
        if z["id"] == "kandla_creek":
            ax.add_patch(MplPoly(z["poly"], closed=True, fc="none", ec="#5fd3e8", lw=0.8, ls="--", zorder=4))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.tick_params(colors=INK, labelsize=7)


def style(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(INK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", default="S4")
    a = ap.parse_args()
    run = REPO / "runs" / f"air7311_{a.sensor}"
    figs = run / "figs"
    figs.mkdir(exist_ok=True)
    summ = json.loads((run / "summary.json").read_text())
    prev = json.loads((REPO / "data" / "site_kandla" / "preview" / "preview.json").read_text())
    pose = summ["pose"]
    cache = REPO / "data" / "site_kandla" / "rf_cache" / f"air7311_{a.sensor}"
    grid = np.load(cache / "spec_diff" / "ism_2400.npz")
    xs, ys, hs = grid["xs"], grid["ys"], list(grid["heights"])
    ext = [xs[0], xs[-1], ys[0], ys[-1]]

    # 1) error maps + who-wins maps, matched array, full propagation, calibrated, at 15 m and 40 m
    cls_cmap = ListedColormap(["#7cc99a", "#e8735f", "#e0a43a"])
    for band, label in BANDS.items():
        name = f"{band}__spec_diff__{MATCHED[band]}__ic0"
        f = run / f"{name}.npz"
        if not f.exists():
            continue
        m = np.load(f)
        fig, axes = plt.subplots(2, 2, figsize=(14, 13), dpi=110)
        for col, hgt in enumerate((15.0, 40.0)):
            ih = hs.index(hgt)
            det = m["snr_db"][ih] >= 6.0
            err = np.where(det, m["err_music"][ih], np.nan)
            ax = axes[0, col]
            style(fig, ax)
            im = ax.imshow(np.clip(err, 0, 30), origin="lower", extent=ext, cmap="magma", vmin=0, vmax=30, zorder=2)
            outline(ax, prev, ext)
            ax.scatter([pose["x_m"]], [pose["y_m"]], marker="^", s=120, c="#5fd3e8", edgecolor="black", zorder=6)
            ax.set_title(f"{label}: MUSIC bearing error at {int(hgt)} m (deg, clipped at 30)", color="#dfe7e4",
                         fontsize=10)
            cb = fig.colorbar(im, ax=ax, fraction=0.035)
            cb.ax.tick_params(colors=INK)
            ax = axes[1, col]
            style(fig, ax)
            c = np.where(det, m["cls"][ih].astype(float), np.nan)
            c[c == 255] = np.nan
            ax.imshow(c, origin="lower", extent=ext, cmap=cls_cmap, vmin=0, vmax=2, zorder=2)
            outline(ax, prev, ext)
            ax.scatter([pose["x_m"]], [pose["y_m"]], marker="^", s=120, c="#5fd3e8", edgecolor="black", zorder=6)
            ax.set_title(f"{label}: strongest peak at {int(hgt)} m  (green direct, red ghost, amber ambiguous; "
                         f"blank undetected)", color="#dfe7e4", fontsize=9)
        fig.suptitle(f"AIR7311 at {a.sensor} (4 omnis, {MATCHED[band]} array, Sionna RT reflections + diffraction, "
                     f"calibrated)", color="#dfe7e4", fontsize=12)
        fig.tight_layout()
        fig.savefig(figs / f"map_{band}.png", facecolor=BG)
        plt.close(fig)

    # 2) array-size and propagation comparison: % bearings off by > 10 deg at 15 m
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=110)
    for ax, (band, label) in zip(axes, BANDS.items(), strict=True):
        style(fig, ax)
        arrays = ["A_5g8", "B_2g4", "C_sub1g"]
        setups = ["los", "spec", "spec_diff"]
        w = 0.25
        for j, setup in enumerate(setups):
            vals = []
            for arr in arrays:
                c = summ["combos"].get(f"{band}__{setup}__{arr}__ic0")
                vals.append(c["by_height"]["15m"].get("music_gt10_pct", np.nan) if c else np.nan)
            ax.bar(np.arange(3) + (j - 1) * w, vals, w, label=setup,
                   color=["#5fd3e8", "#e0a43a", "#c2477e"][j])
        sides = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())["antenna"][
            "candidate_sides_m"]
        ax.set_xticks(range(3), [f"{sides[k] * 100:.1f} cm" for k in arrays], color=INK)
        ax.set_title(f"{label}: % bearings > 10 deg off (15 m)", color="#dfe7e4", fontsize=10)
        ax.tick_params(colors=INK)
        ax.legend(facecolor=BG, labelcolor="#dfe7e4", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "array_vs_propagation.png", facecolor=BG)
    plt.close(fig)

    # 3) AIR7311 MUSIC vs Bartlett vs previous simple model, by height (matched array, full propagation)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=110)
    for ax, (band, label) in zip(axes, BANDS.items(), strict=True):
        style(fig, ax)
        c = summ["combos"].get(f"{band}__spec_diff__{MATCHED[band]}__ic0")
        if not c:
            continue
        hh = [k for k in c["by_height"] if "music_gt10_pct" in c["by_height"][k]]
        x = np.arange(len(hh))
        for j, (key, lab, col) in enumerate((("simple_model_gt10_pct", "previous simple model", "#7f9297"),
                                             ("bartlett_gt10_pct", "AIR7311 Bartlett", "#e0a43a"),
                                             ("music_gt10_pct", "AIR7311 MUSIC", "#5fd3e8"))):
            ax.bar(x + (j - 1) * 0.27, [c["by_height"][k][key] for k in hh], 0.27, label=lab, color=col)
        ax.set_xticks(x, hh, color=INK)
        ax.set_title(f"{label}: % bearings > 10 deg off", color="#dfe7e4", fontsize=10)
        ax.tick_params(colors=INK)
        ax.legend(facecolor=BG, labelcolor="#dfe7e4", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "models_by_height.png", facecolor=BG)
    plt.close(fig)
    print("figures ->", figs)


if __name__ == "__main__":
    main()
