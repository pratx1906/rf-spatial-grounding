"""Render parity (Task B): capture the console's 3D world at the fixed views of docs/design/audit/shots_render/views.json
with every overlay, label and panel hidden, plus fps per view; optionally the console as the operator sees it with
the multipath layer on.

    uv run --no-project --with playwright python scripts/render_parity_shot.py --url http://localhost:8083/
        --out runs/render_parity/after [--before DIR] [--multipath]

--before DIR: serve every file in DIR in place of /ui/<name>.js or /world/<name> (copies taken before the change),
              so the old look can be captured and timed without touching the repo.
fps: rAF count over 4 s with vsync and the frame-rate limit off (uncapped), then a capped (vsync) run in a second
browser, per view, after everything (structures + vegetation) has loaded.
"""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

REPO = Path(__file__).resolve().parents[1]
VIEWS = REPO / "docs" / "design" / "audit" / "shots_render" / "views.json"
FPS_JS = """(ms) => new Promise((res) => { let n = 0; const t0 = performance.now();
  const f = () => { n++; if (performance.now() - t0 < ms) requestAnimationFrame(f); else res(n * 1000 / (performance.now() - t0)); };
  requestAnimationFrame(f); })"""
CLEAN_JS = """() => { const S = window.__scenescout, W = S.W, el = W.el;
  const st = document.createElement('style'); st.id = 'parity';
  st.textContent = 'body *{visibility:hidden !important} #cesium, #cesium canvas{visibility:visible !important}' +
    '#cesium{position:fixed !important; inset:0 !important; width:100vw !important; height:100vh !important; z-index:99999 !important}';
  document.head.appendChild(st);
  W.overlay.visible = false; W.labels.domElement.style.display = 'none';
  for (const b of W.beacons || []) { b.ring.visible = false; b.lamp.visible = false; }
  W.resize(); }"""
SET_CAM = """([p, t]) => { const W = window.__scenescout.W;
  W.anim = null; W.camera.fov = 40; W.camera.position.set(p[0], p[2], -p[1]); W.controls.target.set(t[0], t[2], -t[1]);
  W.camera.updateProjectionMatrix(); W.controls.update(); }"""
ARGS = ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"]


async def open_page(b, a, before):
    ctx = await b.new_context(viewport={"width": a.w, "height": a.h}, device_scale_factor=1)
    pg = await ctx.new_page()
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"[:300]))
    pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"[:300]))
    if before:
        for p in sorted(Path(before).iterdir()):
            url = f"**/ui/{p.name}" if p.suffix == ".js" else f"**/world/{p.name}"
            ct = {".js": "text/javascript", ".json": "application/json", ".glb": "model/gltf-binary"}.get(p.suffix, "application/octet-stream")

            def serve(body, ct):
                async def h(route, *_):
                    await route.fulfill(status=200, body=body, content_type=ct)
                return h
            await pg.route(url, serve(p.read_bytes(), ct))
    await pg.goto(a.url, wait_until="domcontentloaded", timeout=120000)
    try:
        await pg.wait_for_function("window.__scenescout && window.__ss3dT && window.__ss3dT.all", timeout=a.boot_s * 1000)
    except Exception:
        print(chr(10).join(logs[-20:]))
        raise
    await pg.wait_for_timeout(2000)
    return pg, logs


async def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    views = {k: v for k, v in json.loads(VIEWS.read_text()).items() if not k.startswith("_")}
    res = {"url": a.url, "before": bool(a.before), "w": a.w, "h": a.h, "views": {}}
    async with async_playwright() as p:
        b = await p.chromium.launch(args=ARGS + ["--disable-gpu-vsync", "--disable-frame-rate-limit"])
        pg, logs = await open_page(b, a, a.before)
        res["boot_ms"] = await pg.evaluate("() => window.__ss3dT")
        res["renderer"] = await pg.evaluate("() => { const g = window.__scenescout.W.renderer.getContext(); const d = g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : '?'; }")
        await pg.evaluate(CLEAN_JS)
        for name, (pos, tgt) in views.items():
            await pg.evaluate(SET_CAM, [pos, tgt])
            await pg.wait_for_timeout(1500)
            await pg.evaluate(SET_CAM, [pos, tgt])
            await pg.wait_for_timeout(600)
            fps = await pg.evaluate(FPS_JS, 4000)
            await pg.screenshot(path=str(out / f"{name}.png"))
            info = await pg.evaluate("() => { const i = window.__scenescout.W.renderer.info; return {calls: i.render.calls, tris: i.render.triangles, textures: i.memory.textures, geometries: i.memory.geometries}; }")
            res["views"][name] = {"fps_uncapped": round(fps, 1), **info}
            print(name, res["views"][name], flush=True)
        if a.multipath:
            # the operator's view: chrome back, overlays on, multipath layer for S4 2.4 GHz
            await pg.evaluate("() => { document.getElementById('parity').remove(); const W = window.__scenescout.W; W.overlay.visible = true; W.labels.domElement.style.display = ''; for (const b of W.beacons || []) { b.ring.visible = true; b.lamp.visible = true; } W.resize(); }")
            await pg.evaluate("() => { const c = document.getElementById('mpOn'); c.checked = true; c.dispatchEvent(new Event('change')); }")
            await pg.wait_for_timeout(2500)
            await pg.evaluate("() => { const W = window.__scenescout.W, m = (W.heatMeshes || {}).mp; if (!m) return; W.anim = null;"
                              " W.controls.target.copy(m.position); W.camera.position.set(m.position.x + 500, 1400, m.position.z + 1500); W.controls.update(); }")
            await pg.wait_for_timeout(2000)
            await pg.screenshot(path=str(out / "multipath_console.png"))
            res["multipath_heat"] = await pg.evaluate("() => !!(window.__scenescout.W.heatMeshes || {}).mp")
        res["console"] = [x for x in logs if x.startswith(("error", "pageerror", "warning"))]
        await b.close()
        b = await p.chromium.launch(args=ARGS)
        pg, _ = await open_page(b, a, a.before)
        await pg.evaluate(CLEAN_JS)
        for name, (pos, tgt) in views.items():
            await pg.evaluate(SET_CAM, [pos, tgt])
            await pg.wait_for_timeout(1500)
            res["views"][name]["fps_vsync"] = round(await pg.evaluate(FPS_JS, 3000), 1)
        await b.close()
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8083/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--before", default=None)
    ap.add_argument("--multipath", action="store_true")
    ap.add_argument("--boot-s", type=float, default=300)
    ap.add_argument("--w", type=int, default=1600)
    ap.add_argument("--h", type=int, default=900)
    asyncio.run(main(ap.parse_args()))
