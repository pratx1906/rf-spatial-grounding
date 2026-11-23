"""Capture views of the World page at given ENU eye/target (visual reference for approximating structures).

    uv run --no-project --with playwright python scripts/world_look.py --eye 1000 -4000 900 --tgt 1000 -3999 0 --out x.png
"""

import argparse
import asyncio
import time

from playwright.async_api import async_playwright


async def main(a):
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width": a.w, "height": a.h})
        await pg.goto("http://localhost:8090/")
        await pg.wait_for_function("window.SceneScout && window.SceneScout.ready()", timeout=180000)
        await pg.evaluate("SceneScout.pause(); SceneScout.setCapture(true); ['zones','sensors','drones','hud','twin']"
                          ".forEach(l => SceneScout.layer(l, false)); SceneScout.setQuality('cinematic')")
        await pg.evaluate(f"SceneScout.lookAt({list(a.eye)}, {list(a.tgt)})")
        t0 = time.time()
        await asyncio.sleep(3)
        while time.time() - t0 < 120 and not await pg.evaluate("SceneScout.tilesLoaded()"):
            await asyncio.sleep(0.5)
        await asyncio.sleep(1.5)
        await pg.screenshot(path=a.out)
        print("saved", a.out)
        await b.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye", type=float, nargs=3, required=True)
    ap.add_argument("--tgt", type=float, nargs=3, required=True)
    ap.add_argument("--w", type=int, default=1600)
    ap.add_argument("--h", type=int, default=1000)
    ap.add_argument("--out", default="runs/world/look.png")
    asyncio.run(main(ap.parse_args()))
