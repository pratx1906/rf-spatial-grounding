"""Drive the operator console with the scripted demo questions and screenshot each answer (visual check, M7).

    uv run --no-project --with playwright python scripts/console_shot.py --out runs/ui [--areas all]

Picks one demo chip per area (data/demo_tasks_kandla.json), runs it in scripted mode, waits for the answer card,
then saves a screenshot. Console errors are written next to the images.
"""

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

ARGS = ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"]
REPO = Path(__file__).resolve().parents[1]


async def main(a):
    demo = json.loads((REPO / "data" / "demo_tasks_kandla.json").read_text())
    picked, seen = [], set()
    for t in demo:
        if t["area"] not in seen:
            picked.append(t)
            seen.add(t["area"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=ARGS)
        pg = await b.new_page(viewport={"width": a.w, "height": a.h})
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        await pg.goto(a.url, wait_until="domcontentloaded")
        await pg.wait_for_function("window.__scenescout && window.__scenescout.v", timeout=180000)
        await asyncio.sleep(4)
        await pg.click("[data-mode=scripted]")
        for t in picked:
            await pg.evaluate("(id) => { window.__scenescout.taskId = id; }", t["task_id"])
            await pg.fill("#q", t["question"])
            await pg.click("#go")
            await pg.wait_for_function("window.__scenescout.finished === true", timeout=120000)
            await asyncio.sleep(a.settle)
            f = out / f"{t['area']}.png"
            await pg.screenshot(path=str(f))
            ans = await pg.inner_text("#answer")
            print(t["area"], "|", t["question"][:80], "|", ans.replace("\n", " / ")[:160])
        (out / "console_log.txt").write_text("\n".join(logs))
        print("console errors:", sum("error" in x.lower() for x in logs))
        await b.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/")
    ap.add_argument("--out", default="runs/ui")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--settle", type=float, default=5.0)
    asyncio.run(main(ap.parse_args()))
