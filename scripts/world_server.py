"""SceneScout World: the photoreal interactive Kandla world (Cesium + Google Photorealistic 3D Tiles).

    python scripts/world_server.py [--port 8090]      (stdlib only; Windows or WSL)

This is a WORLD-side viewer: it shows the simulation as it is (drone truth trajectories from
data/site_kandla/truth/render/drones.json, exported offline via tasks.truth_access). It runs as its own process,
separate from the agent gateway, so nothing here can reach an agent episode. Listens on localhost only.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
UI = REPO / "ui" / "world"
SITE = REPO / "data" / "site_kandla"
TOKEN = REPO / "token.txt"
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("text/javascript", ".js")


def world_payload() -> dict:
    b = json.loads((SITE / "blender" / "scene_input.json").read_text())
    zones = json.loads((SITE / "zones.geojson").read_text())
    keep = ("tank", "tank_farm", "jetty", "building", "substation", "silo")
    return {
        "origin": b["origin"], "osm": b["osm"],
        "objects": [{k: o.get(k) for k in ("id", "name", "kind", "outer", "centroid", "height_m", "diameter_m")}
                    for o in b["objects"] if o["kind"] in keep],
        "sensors": b["sensors"],
        "zones": [{"id": f["properties"]["id"], "name": f["properties"].get("name"), "kind": f["properties"].get("kind"),
                   "ring": f["geometry"]["coordinates"][0]} for f in zones["features"]
                  if f["geometry"]["type"] == "Polygon"],
    }


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet; never logs query strings or tokens
        pass

    def _send(self, code, body: bytes, ctype="application/json", cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/cesium_token":
                tok = TOKEN.read_text().strip() if TOKEN.exists() else ""
                return self._send(200 if tok else 404, json.dumps({"token": tok}).encode())
            if path == "/api/world":
                return self._send(200, json.dumps(world_payload()).encode())
            if path == "/api/drones":
                f = SITE / "truth" / "render" / "drones.json"
                return self._send(200 if f.exists() else 404, f.read_bytes() if f.exists() else b"{}")
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (UI / rel).resolve()
            if UI.resolve() not in f.parents and f != UI.resolve() or not f.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            return self._send(200, f.read_bytes(), ctype, "no-cache")
        except Exception as e:  # noqa: BLE001
            return self._send(500, json.dumps({"error": type(e).__name__}).encode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"SceneScout World on http://localhost:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
