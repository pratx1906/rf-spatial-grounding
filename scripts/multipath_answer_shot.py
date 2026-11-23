"""Browser check of the echo map inside DF answers (console3.js, ECHO): play recorded answers through the capture hook,
screenshot the answer, then check the fade, the layer toggles and the clear.

    uv run --no-project --with playwright python scripts/multipath_answer_shot.py --url http://localhost:8097/
        --out docs/design/audit/shots_multipath_answer [--tag after] [--before]

--before: only the screenshots, with the "Echo map in DF answers" layer off (the console as it was before the echo map).
Writes <out>/<tag>_<task>.png and <out>/<tag>_result.json.
"""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

ARGS = ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"]
DF_TASKS = ["k00X-direct_path-00064", "k00X-direct_path-00067"]
OTHER = "k00X-jam_links-00128"
ECHO_JS = """() => { const S = window.__scenescout, E = S.echo, m = (S.W.heatMeshes || {}).mpa;
  return E ? {key: E.key, sensor: E.sensor, want: E.want, op: +E.op.toFixed(3), mesh: !!m, visible: !!(m && m.visible),
              opacity: m ? +m.material.opacity.toFixed(3) : null,
              legend: [...document.querySelectorAll('#mapLegend li')].map((l) => l.textContent)} : null; }"""
TOGGLE_JS = """([sel, on]) => { const c = document.querySelector(sel); c.checked = on; c.dispatchEvent(new Event('change')); }"""


async def steps(pg, n):
    for _ in range(n):
        await pg.evaluate("() => window.ssCapture.step(1 / 30)")


async def play(pg, task):
    await pg.evaluate("([t]) => window.ssCapture.play(t, 'sft', {scale: 0.25})", [task])
    for _ in range(600):
        r = await pg.evaluate("() => window.ssCapture.step(1 / 30)")
        if r["finished"] and r["fired"] == 0:
            break
    await pg.keyboard.press("v")              # frame the answer (the console holds the view while a recording plays)
    await steps(pg, 75)                       # camera framing (1 s) and the fade (0.6 s)


async def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"url": a.url, "tag": a.tag, "checks": {}}
    async with async_playwright() as p:
        b = await p.chromium.launch(args=ARGS)
        ctx = await b.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
        pg = await ctx.new_page()
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"[:300]))
        pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"[:300]))
        await pg.goto(a.url.rstrip("/") + "/?capture=1", wait_until="domcontentloaded", timeout=120000)
        await pg.wait_for_function("window.__scenescout && window.ssCapture && window.__ss3dT && window.__ss3dT.all", timeout=300000)
        await pg.wait_for_timeout(1500)
        await pg.evaluate("() => window.ssCapture.freeze()")      # virtual clock: one rendered frame per step
        pg.set_default_timeout(180000)
        if a.before:
            await pg.evaluate(TOGGLE_JS, ["[data-layer=echo]", False])
        for task in DF_TASKS + [OTHER]:
            await play(pg, task)
            await pg.screenshot(path=str(out / f"{a.tag}_{task.split('-', 1)[1]}.png"))
            if a.before:
                continue
            c = res["checks"][task] = {"answer": await pg.evaluate(ECHO_JS)}
            if task == OTHER:
                continue
            await pg.evaluate(TOGGLE_JS, ["[data-layer=analysis]", False]); await steps(pg, 30)
            c["analysis_off"] = await pg.evaluate(ECHO_JS)
            await pg.evaluate(TOGGLE_JS, ["[data-layer=analysis]", True]); await steps(pg, 30)
            c["analysis_on"] = await pg.evaluate(ECHO_JS)
            await pg.evaluate(TOGGLE_JS, ["[data-layer=echo]", False]); await steps(pg, 30)
            c["echo_off"] = await pg.evaluate(ECHO_JS)
            await pg.evaluate(TOGGLE_JS, ["[data-layer=echo]", True]); await steps(pg, 30)
            c["echo_on"] = await pg.evaluate(ECHO_JS)
            await pg.evaluate("() => window.ssCapture.reset()"); await steps(pg, 6)
            c["clearing"] = await pg.evaluate(ECHO_JS)
            await steps(pg, 30)
            c["cleared"] = await pg.evaluate(ECHO_JS)
            if task == DF_TASKS[0]:
                await pg.screenshot(path=str(out / f"{a.tag}_cleared.png"))
        res["fps"] = await pg.evaluate("() => window.__fps")
        res["console"] = [x for x in logs if x.startswith(("error", "pageerror"))]
        await b.close()
    (out / f"{a.tag}_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8097/")
    ap.add_argument("--out", default="docs/design/audit/shots_multipath_answer")
    ap.add_argument("--tag", default="after")
    ap.add_argument("--before", action="store_true")
    asyncio.run(main(ap.parse_args()))
