"""Drive the new console through the demo beats and screenshot each state (1920x1080).

    uv run --no-project --with playwright python scripts/console_demo_shot.py [--url http://localhost:8081] [--out runs/ui/demo]
           [--beats 1,2,5] [--recorded]

Beat numbers index data/demo_tasks_kandla.json (1-based). Prints browser console errors so regressions surface.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081")
    ap.add_argument("--out", default="runs/ui/demo")
    ap.add_argument("--beats", default="2,5")
    ap.add_argument("--recorded", action="store_true")
    ap.add_argument("--wall", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        pg.on("console", lambda m: print("CONSOLE", m.type, m.text[:240]) if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: print("PAGEERR", str(e)[:300]))
        await pg.goto(a.url)
        await pg.wait_for_function("document.getElementById('curtain').hidden === true", timeout=240000)
        await pg.wait_for_timeout(8000)
        if a.wall:
            await pg.keyboard.press("w")
        await pg.screenshot(path=str(out / "0_idle.png"))
        print("idle")
        for beat in [int(x) for x in a.beats.split(",")]:
            await pg.keyboard.press("Escape")
            await pg.keyboard.press("d")
            await pg.wait_for_timeout(300)
            if beat == 1:
                await pg.screenshot(path=str(out / "menu.png"))
            await pg.keyboard.press(f"{'Shift+' if a.recorded else ''}{beat}")
            await pg.wait_for_timeout(2500)
            await pg.screenshot(path=str(out / f"{beat}_working.png"))
            await pg.wait_for_function("window.__scenescout && window.__scenescout.finished === true", timeout=120000)
            await pg.wait_for_timeout(3500)
            await pg.screenshot(path=str(out / f"{beat}_answer.png"))
            verdict = await pg.inner_text("#verdict")
            print(f"beat {beat}:", " | ".join(verdict.split("\n")[:3]))
        await b.close()


asyncio.run(main())
