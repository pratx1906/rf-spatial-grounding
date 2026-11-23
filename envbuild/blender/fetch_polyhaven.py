"""Download the CC0 Poly Haven assets used by the Kandla world (textures, HDRI, vegetation models).

    python envbuild/blender/fetch_polyhaven.py   -> data/assets/polyhaven/<id>/... + manifest.json

Poly Haven assets are CC0 (public domain): free for any use, no attribution required. We record source URL,
licence and sha256 per file anyway.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "assets" / "polyhaven"
UA = {"User-Agent": "SceneScout-twin/0.1 (internal research)"}

TEXTURES = {   # id -> resolution
    "dry_ground_01": "2k", "mud_cracked_dry_03": "2k", "brown_mud_02": "2k", "concrete_floor_worn_001": "2k",
    "tarred_gravel": "2k", "worn_asphalt": "2k", "coast_sand_01": "2k", "corrugated_iron_02": "2k",
    "worn_corrugated_iron": "2k", "white_rough_plaster": "2k", "concrete_wall_006": "2k",
    "painted_metal_shutter": "2k", "metal_plate_02": "2k", "rusty_metal_02": "2k", "gravel_concrete_03": "2k",
}
MAPS = {"Diffuse": "diff", "nor_gl": "nor", "Rough": "rough", "Displacement": "disp"}
HDRI = {"kloofendal_48d_partly_cloudy_puresky": "4k"}
MODELS = {"shrub_01": "1k", "shrub_02": "1k", "shrub_03": "1k", "shrub_04": "1k", "wild_rooibos_bush": "1k"}


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()


def _save(url, dst: Path, manifest: list, asset: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_bytes(_get(url))
    manifest.append({"asset": asset, "file": str(dst.relative_to(OUT)), "url": url, "license": "CC0 1.0",
                     "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})


def main() -> None:
    manifest: list = []
    for aid, res in TEXTURES.items():
        files = json.loads(_get(f"https://api.polyhaven.com/files/{aid}"))
        for key, short in MAPS.items():
            if key not in files:
                continue
            ent = files[key][res]
            fmt = "jpg" if "jpg" in ent else "png"
            _save(ent[fmt]["url"], OUT / aid / f"{short}.{fmt}", manifest, aid)
    for aid, res in HDRI.items():
        files = json.loads(_get(f"https://api.polyhaven.com/files/{aid}"))
        _save(files["hdri"][res]["hdr"]["url"], OUT / aid / f"{aid}_{res}.hdr", manifest, aid)
    for aid, res in MODELS.items():
        files = json.loads(_get(f"https://api.polyhaven.com/files/{aid}"))
        g = files["gltf"][res]["gltf"]
        _save(g["url"], OUT / aid / Path(g["url"]).name, manifest, aid)
        for rel, inc in g.get("include", {}).items():
            _save(inc["url"], OUT / aid / rel, manifest, aid)
    (OUT / "manifest.json").write_text(json.dumps({"source": "Poly Haven (polyhaven.com)", "license": "CC0 1.0",
                                                   "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                   "files": manifest}, indent=1))
    print(json.dumps({"files": len(manifest), "assets": len({m['asset'] for m in manifest}),
                      "MB": round(sum((OUT / m['file']).stat().st_size for m in manifest) / 1e6, 1)}))


if __name__ == "__main__":
    main()
