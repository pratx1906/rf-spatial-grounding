"""Screenshot the SceneScout World page (headless Chromium with GPU) for visual checks.

    uv run --no-project --with playwright python scripts/world_shot.py --shot overview --t 330 --out runs/world/shot.png
"""

import argparse
import asyncio
import time

from playwright.async_api import async_playwright

ARGS = ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-webgpu"]


async def main(a):
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=ARGS)
        pg = await b.new_page(viewport={"width": a.w, "height": a.h})
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        await pg.goto(a.url, wait_until="domcontentloaded")
        await pg.wait_for_function("window.SceneScout && window.SceneScout.ready()", timeout=180000)
        await pg.evaluate(f"SceneScout.pause(); SceneScout.setQuality('{a.quality}'); SceneScout.setTime({a.t}); SceneScout.setShot('{a.shot}')")
        t0 = time.time()
        await asyncio.sleep(a.settle)
        while time.time() - t0 < a.timeout:
            if await pg.evaluate("SceneScout.tilesLoaded()"):
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(1.0)
        await pg.screenshot(path=a.out)
        print("saved", a.out, "tiles loaded:", await pg.evaluate("SceneScout.tilesLoaded()"),
              "gpu:", await pg.evaluate("(()=>{const c=document.createElement('canvas').getContext('webgl2');"
                                        "const d=c&&c.getExtension('WEBGL_debug_renderer_info');"
                                        "return d?c.getParameter(d.UNMASKED_RENDERER_WEBGL):'?'})()"))
        for line in logs[-8:]:
            print(line[:200])
        await b.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8090/")
    ap.add_argument("--shot", default="overview")
    ap.add_argument("--t", type=float, default=330)
    ap.add_argument("--quality", default="balanced")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--settle", type=float, default=4)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--out", default="runs/world/shot.png")
    asyncio.run(main(ap.parse_args()))
