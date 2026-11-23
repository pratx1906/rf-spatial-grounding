"""Export the belief-side twin (objects, zones, sensors, observed tracks) as JSON for preview renders.

    python -m envbuild.preview_export [--site data/site_kandla]  -> <site>/preview/preview.json
Belief side only (no truth/): what an operator's console would show.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.belief.site_view import SiteView

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--site", type=Path, default=REPO / "data" / "site_kandla")
    a = p.parse_args()
    s = SiteView(a.site)

    def ring(g):
        g = max(g.geoms, key=lambda x: x.area) if hasattr(g, "geoms") else g
        return [[round(x, 1), round(y, 1)] for x, y in g.exterior.coords]

    out = {
        "meta": {k: s.meta.get(k) for k in ("site", "extent_m", "hero_zone_m", "rf_backend", "fidelity", "frame")},
        "objects": [{"id": o.id, "name": o.name, "kind": o.kind, "h": o.height_m, "protected": o.protected,
                     "poly": ring(o.geom)} for o in s.objects.values()],
        "zones": [{"id": z.id, "name": z.name, "kind": z.kind, "poly": ring(z.geom)} for z in s.zones.values()],
        "sensors": [{"id": k, "type": v["type"], **v["pose"]} for k, v in s.sensors.items()],
        "tracks": [{"id": tid, "group": tr.get("group_id"), "conf": tr["confidence"],
                    "pts": [[float(t), float(x), float(y), float(z)] for t, x, y, z in
                            zip(s.points[tid]["t"], s.points[tid]["x"], s.points[tid]["y"], s.points[tid]["z"])]}
                   for tid, tr in s.tracks.items()],
    }
    d = a.site / "preview"
    d.mkdir(exist_ok=True)
    (d / "preview.json").write_text(json.dumps(out))
    print(f"{len(out['objects'])} objects, {len(out['tracks'])} tracks -> {d / 'preview.json'}")


if __name__ == "__main__":
    main()
