"""Console UX gate (demo v2, S4): play the 6 demo beats from recordings, check the audit fixes, screenshot at 1280 and 1920.

    uv run --no-project --with playwright python scripts/console_s4_check.py [--url http://localhost:8096]
           [--out docs/design/audit/shots_demo_v2] [--sizes 1920x1080,1280x720]

Checks (printed as PASS/FAIL lines; exit 1 on any FAIL):
  * no console errors or page errors;
  * 60 fps (rAF counter window.__fps) idle and with an answer drawn;
  * P0-1: after hovering a step and leaving, no superseded estimate or rejected "other" bearing label is visible;
  * P0-2: no "OpenStreetMap" in the boot list or the fidelity badge text;
  * P1-1: the Now line and the elapsed counter change during a recorded run;
  * P1-3: the refusal beat reads "Refused" / "Not an action SceneScout can take";
  * P1-7 / S4.6: synthetic pilot_search, tdoa_fix, signal_detail, noise_floor and track_history events draw marks;
  * P1-9 / S4.7: at 1280x720 the gate fits and the verdict's explanation is not covered;
  * P1-4 / P1-5: with the model server offline, Ask is disabled and a typed question shows the offline block.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

BEATS = {1: "genuine_drone", 2: "direct_path", 3: "ds_hover", 4: "suggest_one", 5: "jam_links", 6: "actuation"}
FAILS: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


VISIBLE_LABELS = """() => Array.from(document.querySelectorAll('.wl')).filter((e) => {
  const cs = getComputedStyle(e); if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return false;
  const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; }).map((e) => ({t: e.textContent, c: e.className}))"""

SYNTH = """async () => {
  const S = window.__scenescout, A = S.api;
  const trk = S.tracks.find((t) => t.pts.length > 20);
  const mid = trk.pts[Math.floor(trk.pts.length / 2)], t = mid[0];
  const sens = S.site.sensors.filter((s) => s.type !== 'drone_df').slice(0, 4);
  A.newEpisode('Synthetic check: who is flying ' + trk.ref + ' and where exactly is it?');
  const ev = (o) => A.onEvent(o);
  const pause = (ms) => new Promise((r) => setTimeout(r, ms));
  ev({type: 'start', episode_id: 'synthetic'});
  const call = async (name, args, data, turn) => {
    ev({type: 'assistant', turn, gen_ms: 300, n_calls: 1});
    ev({type: 'tool_call', turn, name, arguments: args});
    await pause(250);
    ev({type: 'tool_result', turn, name, result: {ok: true, tool: name, data, refs: [], cost_ms: 12}});
    await pause(250);
  };
  const P = (x, y, z) => ({x_m: x, y_m: y, z_m: z});
  await call('pilot_search', {near_track_ref: trk.ref}, {drone: trk.ref, candidates: [
    {ref: 'trk:9001', pos: P(mid[1] + 600, mid[2] - 300, 2), p_controller: 0.72, distance_to_drone_m: 640, score: 0.8},
    {ref: 'trk:9002', pos: P(mid[1] - 500, mid[2] + 400, 2), p_controller: 0.31, distance_to_drone_m: 690, score: 0.5}]}, 1);
  await call('tdoa_fix', {track_ref: trk.ref, at_time_s: t}, {ref: trk.ref, t_s: t, pos: P(mid[1] + 40, mid[2] - 30, mid[3]),
    aoa_pos: P(mid[1], mid[2], mid[3]), shift_m: 50, range_diff_sigma_m: 45, sensors: sens.map((s) => s.ref), useful: true}, 2);
  await call('signal_detail', {track_ref: trk.ref, at_time_s: t}, {ref: trk.ref, n_bursts: 40, protocol_estimate: 'lora',
    freq_mhz: {median: 868.1}, bandwidth_khz: 500, hopping: false, distinct_channels: 1, snr_db: 14.2, sensors: sens.slice(0, 3).map((s) => s.ref)}, 3);
  await call('noise_floor', {sensor_ref: sens[0].ref, band: 'sub_ghz_868_915'}, {sensor: sens[0].ref, band: 'sub_ghz_868_915',
    noise_dbm_per_mhz: -103.2, rise_over_thermal_db: 7.8, clipping_pct: 0}, 4);
  await call('track_history', {track_ref: trk.ref}, {track: trk.ref, path_len_m: 2400, step_s: 10,
    points: trk.pts.filter((_, i) => i % 5 === 0).map((p) => ({t_s: p[0], pos: P(p[1], p[2], p[3])}))}, 5);
  return {steps: document.querySelectorAll('#steps li').length, labels: S.stepObjs.size};
}"""


async def fps(pg, n=3) -> int:
    vals = []
    for _ in range(n):
        await pg.wait_for_timeout(1100)
        vals.append(await pg.evaluate("window.__fps || 0"))
    return min(vals)


async def run_size(p, url: str, out: Path, w: int, h: int) -> None:
    tag = f"s4_{w}"
    b = await p.chromium.launch(args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
    pg = await b.new_page(viewport={"width": w, "height": h})
    errs: list[str] = []
    pg.on("console", lambda m: errs.append(f"console {m.text[:200]}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(f"pageerror {str(e)[:200]}"))
    await pg.goto(url)
    await pg.wait_for_function("document.getElementById('curtain').hidden === true", timeout=240000)
    await pg.wait_for_timeout(4000)
    boot = await pg.inner_text("#bootList")
    fid = await pg.inner_text("#fidelity")
    print(f"[{w}] boot list:", " | ".join(boot.split("\n")))
    check("OpenStreetMap" not in boot and "OpenStreetMap" not in fid, f"[{w}] P0-2 no 'OpenStreetMap' in boot list or badge")
    await pg.screenshot(path=str(out / f"{tag}_00_idle.png"))
    f_idle = await fps(pg)
    check(f_idle >= 55, f"[{w}] idle fps {f_idle}")
    llm_ok = await pg.evaluate("window.__scenescout.health.llm.ok")

    for beat, name in BEATS.items():
        await pg.keyboard.press("Escape")
        await pg.keyboard.press("Escape")
        await pg.keyboard.press("d")
        await pg.wait_for_timeout(300)
        await pg.keyboard.press(f"Shift+Digit{beat}")
        await pg.wait_for_timeout(700)
        n1 = (await pg.inner_text("#nowText"), await pg.inner_text("#nowClock"))
        await pg.wait_for_timeout(500)
        n2 = (await pg.inner_text("#nowText"), await pg.inner_text("#nowClock"))
        await pg.screenshot(path=str(out / f"{tag}_{beat}_{name}_working.png"))
        fin = await pg.evaluate("window.__scenescout.finished")
        check(fin or n1[1] != n2[1], f"[{w}] beat {beat} P1-1 counter runs ({n1[1]} -> {n2[1]}), now: {n2[0]!r}")
        await pg.wait_for_function("window.__scenescout && window.__scenescout.finished === true", timeout=120000)
        await pg.wait_for_timeout(2600)
        await pg.screenshot(path=str(out / f"{tag}_{beat}_{name}_answer.png"))
        verdict = await pg.inner_text("#verdict")
        print(f"[{w}] beat {beat}:", " | ".join(verdict.split("\n")[:4]))
        steps = await pg.inner_text("#steps")
        for bad in ("sensor:S", "rt_match", "plan: ", "of ? checked") + (("Cannot determine",) if beat == 6 else ()):
            check(bad not in verdict + steps, f"[{w}] beat {beat} P1-2 no machine text {bad!r}")
        check("argmax" not in verdict, f"[{w}] beat {beat} P1-2 raw model assumptions stay out of the verdict")
        if beat in (2, 3):
            await pg.hover("#steps li[data-step='1']")
            await pg.wait_for_timeout(400)
            await pg.mouse.move(10, h // 2)
            await pg.wait_for_timeout(500)
            labels = await pg.evaluate(VISIBLE_LABELS)
            if beat == 3:
                est = [x["t"] for x in labels if x["t"].startswith("estimate ±")]
                check(len(est) <= 1, f"[{w}] beat 3 P0-1 one live estimate label after hover ({est})")
            else:
                other = [x["t"] for x in labels if x["c"].strip() == "wl" and x["t"][:1] == "S" and x["t"].endswith("°")]
                check(not other, f"[{w}] beat 2 P0-1 no rejected 'other' bearing labels after hover ({other})")
            await pg.screenshot(path=str(out / f"{tag}_{beat}_{name}_after_hover.png"))
        if beat == 5:
            gate = await pg.query_selector("#gate")
            gb = await gate.bounding_box() if gate else None
            check(bool(gb) and gb["y"] + gb["height"] <= h, f"[{w}] beat 5 S4.7 gate fits in the viewport ({gb})")
            why = await pg.query_selector("#verdict .why")
            wb = await why.bounding_box()
            check(wb["y"] + wb["height"] <= gb["y"], f"[{w}] beat 5 P1-9 verdict explanation not covered by the gate")
            f_ans = await fps(pg)
            check(f_ans >= 55, f"[{w}] fps with the jam footprint drawn {f_ans}")
            await pg.keyboard.down("a")
            await pg.wait_for_timeout(1500)
            await pg.keyboard.up("a")
            await pg.screenshot(path=str(out / f"{tag}_{beat}_{name}_gate_ack.png"))
            res = await pg.inner_text("#gRes")
            check("not logged" in res or "audit log" in res, f"[{w}] gate copy is honest: {res!r}")
        if beat == 6:
            check("Refused" in verdict and "Not an action SceneScout can take" in verdict, f"[{w}] beat 6 P1-3 refusal reads as a refusal")

    await pg.keyboard.press("Escape")
    await pg.keyboard.press("Escape")
    r = await pg.evaluate(SYNTH)
    await pg.wait_for_timeout(1500)
    labels = [x["t"] for x in await pg.evaluate(VISIBLE_LABELS)]
    steps = await pg.inner_text("#steps")
    print(f"[{w}] synthetic tools:", r, "|", " | ".join(steps.split("\n")[:12]))
    check(any(t.startswith("pilot? p 0.72") for t in labels), f"[{w}] P1-7 pilot marker drawn")
    check(any(t.startswith("AoA+TDOA fix") for t in labels), f"[{w}] P1-7 TDOA fix drawn")
    check(any("dB over thermal" in t for t in labels), f"[{w}] S4.6 noise layer drawn")
    check(any(t.startswith("LoRa, 868.1 MHz") for t in labels), f"[{w}] S4.6 signal layer drawn")
    check("pilot_search" not in steps and "tdoa_fix" not in steps, f"[{w}] P1-7 tool names are labelled")
    await pg.screenshot(path=str(out / f"{tag}_07_signal_tools.png"))

    # S5.1 gate wiring: a scripted-expert run streamed by the gateway registers its answer hash; the console then
    # records the decision in two steps (review, confirm) through POST /api/gate
    await pg.keyboard.press("Escape")
    await pg.keyboard.press("Escape")
    await pg.evaluate("""() => new Promise((done) => {
      const S = window.__scenescout, t = S.demo.find((d) => d.template === 'jam_links');
      S.api.newEpisode(t.question); S.live = {who: 'scripted-expert', recorded: false, taskId: t.task_id};
      const es = new EventSource('/api/ask?mode=scripted&task_id=' + encodeURIComponent(t.task_id));
      es.onmessage = (m) => { const ev = JSON.parse(m.data); S.api.onEvent(ev); if (ev.type === 'done') { es.close(); done(); } };
      es.onerror = () => { es.close(); done(); };
    })""")
    await pg.wait_for_timeout(1500)
    if await pg.is_visible("#gate"):
        await pg.keyboard.down("a")
        await pg.wait_for_timeout(1500)
        await pg.keyboard.up("a")
        await pg.wait_for_timeout(1600)
        await pg.screenshot(path=str(out / f"{tag}_09_gate_review.png"))
        await pg.keyboard.press("Enter")
        await pg.wait_for_timeout(1500)
        res = await pg.inner_text("#gRes")
        await pg.screenshot(path=str(out / f"{tag}_10_gate_logged.png"))
        check("written to the audit log" in res, f"[{w}] S5.1 gate decision recorded server-side: {res!r}")
    else:
        check(False, f"[{w}] S5.1 gate opened on the scripted collateral run")

    await pg.keyboard.press("Escape")
    await pg.keyboard.press("Escape")
    if not llm_ok:
        go = await pg.query_selector("#go")
        check(await go.is_disabled() and (await go.inner_text()) == "Model offline", f"[{w}] P1-5 Ask disabled while the model is offline")
        await pg.fill("#q", "Which drones are inside TF-2 right now?")
        await pg.focus("#q")
        await pg.keyboard.press("Enter")
        await pg.wait_for_timeout(1200)
        v = await pg.inner_text("#verdict")
        check("Model server offline" in v and "APIConnectionError" not in v, f"[{w}] P1-4 offline block: {v.splitlines()[:2]}")
        await pg.screenshot(path=str(out / f"{tag}_08_offline_ask.png"))
    else:
        print(f"[{w}] model server up: offline checks skipped")
    check(not errs, f"[{w}] no console errors ({errs[:3]})")
    await b.close()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8096")
    ap.add_argument("--out", default="docs/design/audit/shots_demo_v2")
    ap.add_argument("--sizes", default="1920x1080,1280x720")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        for sz in a.sizes.split(","):
            w, h = (int(x) for x in sz.split("x"))
            await run_size(p, a.url, out, w, h)
    print(f"{len(FAILS)} FAIL" if FAILS else "ALL PASS")
    sys.exit(1 if FAILS else 0)


asyncio.run(main())
