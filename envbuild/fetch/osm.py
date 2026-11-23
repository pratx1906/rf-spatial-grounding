"""E1 fetch: OpenStreetMap features for the Kandla Creek port area via the public Overpass API (ENV spec 3.1).

    python -m envbuild.fetch.osm [--bbox S W N E]

Writes data/raw/osm/<date>/kandla_osm.json plus manifest.json (URL, query, license, hash, retrieval time).
Raw data is never edited in place (ENV spec 3.1). License: ODbL; attribution "(c) OpenStreetMap contributors".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
OVERPASS = "https://overpass-api.de/api/interpreter"
# Port core of Deendayal Port (Kandla Creek); about 23.03 N, 70.23 E per public sources [VERIFY with imagery].
DEFAULT_BBOX = (22.985, 70.165, 23.060, 70.260)   # south, west, north, east

QUERY = """
[out:json][timeout:180];
(
  way["building"]({bbox});
  way["man_made"]({bbox});
  node["man_made"]({bbox});
  way["landuse"~"port|industrial|harbour|railway|commercial|military"]({bbox});
  relation["landuse"~"port|industrial|harbour"]({bbox});
  way["harbour"]({bbox});
  node["harbour"]({bbox});
  way["seamark:type"]({bbox});
  node["seamark:type"]({bbox});
  way["natural"~"water|coastline|wetland|bay|mud|beach"]({bbox});
  relation["natural"~"water|wetland|bay"]({bbox});
  way["waterway"]({bbox});
  way["railway"]({bbox});
  way["highway"~"primary|secondary|tertiary|trunk|motorway|service|unclassified"]({bbox});
  way["aeroway"]({bbox});
  node["amenity"]({bbox});
  way["amenity"]({bbox});
  way["power"]({bbox});
  node["power"~"tower|substation"]({bbox});
  way["barrier"="fence"]({bbox});
);
out body geom;
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("S", "W", "N", "E"))
    a = p.parse_args()
    bbox = ",".join(f"{v:.4f}" for v in a.bbox)
    query = QUERY.replace("{bbox}", bbox)
    t0 = time.time()
    r = requests.post(OVERPASS, data={"data": query}, timeout=300,
                      headers={"User-Agent": "SceneScout-envbuild/0.1 (research; contact via repo owner)"})
    r.raise_for_status()
    raw = r.content
    out = REPO / "data" / "raw" / "osm" / time.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    (out / "kandla_osm.json").write_bytes(raw)
    data = json.loads(raw)
    manifest = {
        "source": "OpenStreetMap via Overpass API", "url": OVERPASS, "bbox_swne": a.bbox, "query": query,
        "license": "ODbL 1.0", "attribution": "(c) OpenStreetMap contributors",
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sha256": hashlib.sha256(raw).hexdigest(),
        "elements": len(data.get("elements", [])), "seconds": round(time.time() - t0, 1),
        "osm_base_timestamp": data.get("osm3s", {}).get("timestamp_osm_base"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("elements", "seconds", "osm_base_timestamp")}), "->", out)


if __name__ == "__main__":
    main()
