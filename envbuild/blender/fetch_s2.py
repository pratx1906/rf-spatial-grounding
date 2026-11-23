"""Sentinel-2 L2A true-colour drape for the Kandla world (10 m/pixel, open data).

    uv run --with numpy --with tifffile --with imagecodecs --with pillow python -m envbuild.blender.fetch_s2

Source: Copernicus Sentinel-2 L2A via the Element84 Earth Search STAC (AWS Open Data, sentinel-cogs bucket).
Licence: Copernicus open and free data policy; attribution "Contains modified Copernicus Sentinel data <year>".
Writes data/raw/sentinel2/<item>/{TCI.tif, item.json, manifest.json} and data/site_kandla/blender/s2_albedo.png
(ENU-aligned, north up, covering +-HALF_M around the site origin).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np

from envbuild.geo import ENU

REPO = Path(__file__).resolve().parents[2]
ITEM = "S2B_42QXL_20261001_0_L2A"          # 2026-10-01, cloud cover 0.016 %, no nodata over the site
STAC = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/"
HALF_M = 15000.0
PIX_M = 10.0


def utm_forward(lat, lon, zone):
    """WGS-84 geodetic -> UTM (northern hemisphere), Krueger series (sub-mm inside a zone)."""
    a, f = 6378137.0, 1 / 298.257223563
    k0 = 0.9996
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    al = [n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240]
    lam0 = math.radians(zone * 6 - 183)
    phi = np.radians(lat)
    lam = np.radians(lon) - lam0
    e = math.sqrt(f * (2 - f))
    t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t ** 2))
    E = eta + sum(al[j] * np.cos(2 * (j + 1) * xi) * np.sinh(2 * (j + 1) * eta) for j in range(3))
    N = xi + sum(al[j] * np.sin(2 * (j + 1) * xi) * np.cosh(2 * (j + 1) * eta) for j in range(3))
    return 500000.0 + k0 * A * E, k0 * A * N


def _origin():
    txt = (REPO / "data" / "site_kandla" / "meta.yaml").read_text().splitlines()
    lat = lon = None
    for ln in txt:
        s = ln.strip()
        if s.startswith("lat:") and lat is None:
            lat = float(s.split(":")[1])
        elif s.startswith("lon:") and lon is None:
            lon = float(s.split(":")[1])
    return lat, lon


def run() -> None:
    import tifffile
    from PIL import Image

    out = REPO / "data" / "raw" / "sentinel2" / ITEM
    out.mkdir(parents=True, exist_ok=True)
    item = json.loads(urllib.request.urlopen(STAC + ITEM, timeout=60).read())
    (out / "item.json").write_text(json.dumps(item, indent=1))
    href = item["assets"]["visual"]["href"]
    tif = out / "TCI.tif"
    if not tif.exists():
        t0 = time.time()
        urllib.request.urlretrieve(href, tif)
        print(f"downloaded {tif.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f} s")
    sha = hashlib.sha256(tif.read_bytes()).hexdigest()
    year = item["properties"]["datetime"][:4]
    (out / "manifest.json").write_text(json.dumps({
        "source": "Copernicus Sentinel-2 L2A, true colour (TCI, B04/B03/B02), 10 m",
        "item": ITEM, "datetime": item["properties"]["datetime"], "cloud_cover": item["properties"]["eo:cloud_cover"],
        "url": href, "stac": STAC + ITEM, "sha256": sha, "crs": f"EPSG:{item['properties']['proj:epsg']}",
        "license": "Copernicus Sentinel data: free, full and open (Commission Delegated Regulation (EU) 1159/2013)",
        "attribution": f"Contains modified Copernicus Sentinel data {year}",
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    img = tifffile.imread(tif)                     # (10980, 10980, 3) uint8
    tr = item["assets"]["visual"]["proj:transform"]
    lat0, lon0 = _origin()
    enu = ENU(lat0, lon0)
    n = int(2 * HALF_M / PIX_M)
    c = -HALF_M + PIX_M * (np.arange(n) + 0.5)
    X, Y = np.meshgrid(c, c[::-1])                 # row 0 = north
    lat, lon, _ = enu.to_geodetic(X.ravel(), Y.ravel(), 0.0)
    E, N = utm_forward(lat, lon, int(item["properties"]["proj:epsg"]) - 32600)
    col = (E - tr[2]) / tr[0] - 0.5
    row = (N - tr[5]) / tr[4] - 0.5
    i0 = np.clip(np.floor(col).astype(int), 0, img.shape[1] - 2)
    j0 = np.clip(np.floor(row).astype(int), 0, img.shape[0] - 2)
    u = (col - i0)[:, None]
    v = (row - j0)[:, None]
    f = img.astype(np.float32)
    s = ((1 - u) * (1 - v) * f[j0, i0] + u * (1 - v) * f[j0, i0 + 1] + (1 - u) * v * f[j0 + 1, i0]
         + u * v * f[j0 + 1, i0 + 1])
    rgb = np.clip(s, 0, 255).astype(np.uint8).reshape(n, n, 3)
    dst = REPO / "data" / "site_kandla" / "blender" / "s2_albedo.png"
    Image.fromarray(rgb).save(dst)
    meta = {"item": ITEM, "half_m": HALF_M, "pix_m": PIX_M, "size": n, "orientation": "row 0 = north, ENU",
            "attribution": f"Contains modified Copernicus Sentinel data {year}"}
    (dst.with_suffix(".json")).write_text(json.dumps(meta, indent=1))
    print(json.dumps({**meta, "mean_rgb": rgb.reshape(-1, 3).mean(0).round(1).tolist(), "out": str(dst)}))


if __name__ == "__main__":
    run()
