"""Console gate for the coloured answer map, live drones and the free camera (no LLM needed).

    uv run --no-project --with playwright python scripts/console_colour_check.py [--url http://localhost:8098]
           [--events DIR] [--out docs/design/audit/shots_demo_v2] [--sizes 1920x1080,1280x720]

Plays the 6 recorded demo beats (D, Shift+digit), then free-form answers of every type through the same render path:
scripted-expert event streams saved as JSON in --events (one {"question", "events"} file per family; made with the
gateway's run_expert on the demo site), plus synthetic signal-tool answers. For each answer it checks that the map
carries answer geometry (answer layer objects, at least one beacon, a callout unless it is an abstention), that the
world grade keeps colour, and that the legend lists only drawn layers. Then: live drones (toggle, play, click to select),
fly mode near a jetty, a saved view recalled, orbit about the cursor, the "View held" hint and V. Screenshots go to
<out>/colour_<width>_<name>.png. Prints PASS/FAIL lines; exit 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

BEATS = {1: "genuine_drone", 2: "direct_path", 3: "ds_hover", 4: "suggest_one", 5: "jam_links", 6: "actuation"}
FAILS: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


STATE = """() => { const S = window.__scenescout; return {
  ans: S.ansObjs.length, beacons: S.ansObjs.filter((o) => o.userData.beacon).length,
  callouts: Array.from(document.querySelectorAll('.wl.callout')).filter((e) => e.style.display !== 'none' && e.style.opacity !== '0').map((e) => e.textContent),
  legend: Array.from(document.querySelectorAll('#mapLegend li span')).map((e) => e.textContent), legendShown: !document.getElementById('mapLegend').hidden,
  grade: S.W.gradePass.uniforms.amount.value, fps: window.__fps || 0, verdict: document.getElementById('verdict').innerText.split('\\n').slice(0, 3).join(' | ') }; }"""

PLAY = """async ([q, events]) => {
  const S = window.__scenescout, A = S.api;
  A.newEpisode(q); S.live = {who: 'scripted-expert', recorded: false, taskId: null};
  for (const ev of events) { A.onEvent(ev); await new Promise((r) => setTimeout(r, ev.type === 'tool_result' ? 220 : 40)); }
  if (!events.some((e) => e.type === 'done')) A.onEvent({type: 'done'});
}"""

SYNTH = """async () => {
  const S = window.__scenescout;
  const trk = S.tracks.find((t) => t.pts.length > 20);
  const mid = trk.pts[Math.floor(trk.pts.length / 2)], t = mid[0];
  const sens = S.site.sensors.filter((s) => s.type !== 'drone_df').slice(0, 4);
  const P = (x, y, z) => ({x_m: x, y_m: y, z_m: z});
  const ok = (name, data, refs) => ({type: 'tool_result', turn: 1, name, result: {ok: true, tool: name, data, refs, cost_ms: 12}});
  const call = (name, args) => ({type: 'tool_call', turn: 1, name, arguments: args});
  const ev = [{type: 'start', episode_id: 'synthetic'}, {type: 'assistant', turn: 1, gen_ms: 300, n_calls: 1},
    call('pilot_search', {near_track_ref: trk.ref}),
    ok('pilot_search', {drone: trk.ref, candidates: [
      {ref: 'trk:9001', pos: P(mid[1] + 600, mid[2] - 300, 2), p_controller: 0.72, distance_to_drone_m: 640, score: 0.8},
      {ref: 'trk:9002', pos: P(mid[1] - 500, mid[2] + 400, 2), p_controller: 0.31, distance_to_drone_m: 690, score: 0.5}]}, ['trk:9001', 'trk:9002']),
    call('tdoa_fix', {track_ref: trk.ref, at_time_s: t}),
    ok('tdoa_fix', {ref: trk.ref, t_s: t, pos: P(mid[1] + 40, mid[2] - 30, mid[3]), aoa_pos: P(mid[1], mid[2], mid[3]), shift_m: 50,
      range_diff_sigma_m: 45, sensors: sens.map((s) => s.ref), useful: true}, [trk.ref]),
    call('signal_detail', {track_ref: trk.ref, at_time_s: t}),
    ok('signal_detail', {ref: trk.ref, n_bursts: 40, protocol_estimate: 'lora', freq_mhz: {median: 868.1}, bandwidth_khz: 500,
      hopping: false, distinct_channels: 1, snr_db: 14.2, sensors: sens.slice(0, 3).map((s) => s.ref)}, [trk.ref]),
    {type: 'answer', answer: {intent: 'operator_query', track_refs: [trk.ref + '@' + t], region: null, boolean: null, protocol: 'lora',
      band: 'sub_ghz_868_915', evidence_refs: [trk.ref, 'trk:9001'], confidence: 0.7, assumptions: [], requires_human_confirm: false},
     verifier: {ok: true}, status: 'verified', end_reason: 'submitted', turns: 3},
    {type: 'done'}];
  return [`Synthetic: what protocol is ${trk.ref} using at t=${t} s, and where is its pilot?`, ev];
}"""

CIRCLE = """async () => {
  const S = window.__scenescout, j = S.objs.find((o) => o.name === 'Jetty J2') || S.objs[0];
  const ev = [{type: 'start', episode_id: 'synthetic2'},
    {type: 'tool_call', turn: 1, name: 'track_search', arguments: {t: 300}},
    {type: 'tool_result', turn: 1, name: 'track_search', result: {ok: true, data: {items: []}, refs: [], cost_ms: 9}},
    {type: 'answer', answer: {intent: 'operator_query', track_refs: [], boolean: false, region: {type: 'circle', center: {x_m: j.enu[0], y_m: j.enu[1]}, radius_m: 300},
      evidence_refs: [j.ref], confidence: 0.8, assumptions: [], requires_human_confirm: false}, verifier: {ok: true}, status: 'verified', end_reason: 'submitted', turns: 2},
    {type: 'done'}];
  return ['Synthetic: is any drone within 300 m of Jetty J2 at t=300 s?', ev];
}"""


async def fps(pg, n=3) -> int:
    vals = []
    for _ in range(n):
        await pg.wait_for_timeout(1100)
        vals.append(await pg.evaluate("window.__fps || 0"))
    return min(vals)


async def judge(pg, w: int, name: str, abstain: bool = False) -> None:
    st = await pg.evaluate(STATE)
    print(f"[{w}] {name}: {st['verdict']} || callouts {st['callouts']} || legend {st['legend']}")
    check(st["grade"] < 1.01, f"[{w}] {name} world grade stays in colour (amount {st['grade']:.2f}; shader caps at 15%)")
    if abstain:
        check(st["ans"] == 0, f"[{w}] {name} abstention draws no answer marks ({st['ans']})")
    else:
        check(st["ans"] > 0 and st["beacons"] > 0, f"[{w}] {name} answer layer drawn ({st['ans']} objs, {st['beacons']} beacons)")
        check(len(st["callouts"]) > 0, f"[{w}] {name} callout carries the answer ({st['callouts'][:2]})")
        check(st["legendShown"] and "Answer" in " ".join(st["legend"]), f"[{w}] {name} legend shown with the answer row")


async def run_size(p, url: str, out: Path, evdir: Path | None, w: int, h: int) -> None:
    tag = f"colour_{w}"
    b = await p.chromium.launch(args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
    pg = await b.new_page(viewport={"width": w, "height": h})
    errs: list[str] = []
    pg.on("console", lambda m: errs.append(f"console {m.text[:200]}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(f"pageerror {str(e)[:200]} {(e.stack or '')[:300]}"))
    pg.on("response", lambda r: errs.append(f"http {r.status} {r.url}") if r.status >= 400 else None)
    await pg.goto(url)
    await pg.wait_for_function("document.getElementById('curtain').hidden === true", timeout=240000)
    await pg.wait_for_timeout(5000)
    f_idle = await fps(pg)
    check(f_idle >= 55, f"[{w}] idle fps {f_idle}")

    async def shot(name):
        await pg.screenshot(path=str(out / f"{tag}_{name}.png"))

    async def esc():
        await pg.keyboard.press("Escape")
        await pg.keyboard.press("Escape")

    for beat, name in BEATS.items():
        await esc()
        await pg.keyboard.press("d")
        await pg.wait_for_timeout(300)
        await pg.keyboard.press(f"Shift+Digit{beat}")
        await pg.wait_for_function("window.__scenescout.finished === false", timeout=20000)
        await pg.wait_for_function("window.__scenescout && window.__scenescout.finished === true", timeout=120000)
        await pg.wait_for_timeout(2600)
        await shot(f"beat{beat}_{name}")
        if beat == 5:
            await pg.keyboard.press("Escape")        # gate open: Esc defers (gate keys keep their meaning)
        await judge(pg, w, f"beat {beat} {name}", abstain=(beat == 6))
        if beat == 3:
            f_ans = await fps(pg)
            check(f_ans >= 55, f"[{w}] fps with an answer drawn {f_ans}")
        if beat == 2:                                 # P0-1 still holds: hover a step, leave, no rejected bearing labels
            await pg.hover("#steps li[data-step='1']")
            await pg.wait_for_timeout(400)
            await pg.mouse.move(10, h // 2)
            await pg.wait_for_timeout(500)
            n = await pg.evaluate("""() => Array.from(document.querySelectorAll('.wl')).filter((e) => e.className.trim() === 'wl'
               && e.style.display !== 'none' && e.textContent.startsWith('S') && e.textContent.endsWith('°') && getComputedStyle(e).opacity !== '0').length""")
            check(n == 0, f"[{w}] beat 2 hover isolation leaves no 'other' bearing labels ({n})")
        if beat == 4:                                 # operator moves the camera: auto framing holds, V frames
            box = await pg.query_selector("#cesium canvas")
            bb = await box.bounding_box()
            await pg.mouse.move(bb["x"] + bb["width"] * 0.5, bb["y"] + bb["height"] * 0.5)
            await pg.mouse.down()
            await pg.mouse.move(bb["x"] + bb["width"] * 0.5 - 220, bb["y"] + bb["height"] * 0.5 + 60, steps=8)
            await pg.mouse.up()
            await pg.wait_for_timeout(500)
            held = await pg.is_visible("#heldHint")
            check(held, f"[{w}] 'View held' hint after the operator moves the camera")
            await shot("held_hint")
            await pg.keyboard.press("v")
            await pg.wait_for_timeout(1500)
            check(not await pg.is_visible("#heldHint"), f"[{w}] V frames the answer and clears the hint")

    plays = []
    if evdir:
        for f in sorted(evdir.glob("*.json")):
            d = json.loads(f.read_text())
            for e in d["events"]:          # not registered with this gateway's gate: the decision stays console-only
                e.pop("answer_hash", None)
            plays.append((f.stem, d["question"], d["events"]))
    for name, q, events in plays:
        await esc()
        await pg.evaluate(PLAY, [q, events])
        await pg.wait_for_timeout(2600)
        a = next((e.get("answer") for e in events if e.get("type") == "answer"), None) or {}
        await shot(f"free_{name}")
        await judge(pg, w, f"free-form {name}", abstain=a.get("intent") == "abstain")
    for nm, js in (("signal_pilot", SYNTH), ("circle_none", CIRCLE)):
        await esc()
        q, ev = await pg.evaluate(js)
        await pg.evaluate(PLAY, [q, ev])
        await pg.wait_for_timeout(2600)
        await shot(f"free_{nm}")
        await judge(pg, w, f"synthetic {nm}")

    # live drones
    await esc()
    await pg.click(".views [data-view=overview]")
    await pg.wait_for_timeout(1300)
    await pg.keyboard.press("u")
    await pg.evaluate("() => { const S = window.__scenescout; S.W.simT = 300; }")
    await pg.keyboard.press("Space")
    await pg.wait_for_timeout(2500)
    f_dr = await fps(pg)
    check(f_dr >= 55, f"[{w}] fps with live drones flying {f_dr}")
    await pg.keyboard.press("Space")
    st = await pg.evaluate("() => ({n: window.__scenescout.D.body.count, legend: Array.from(document.querySelectorAll('#mapLegend li span')).map((e) => e.textContent)})")
    check(st["n"] > 0, f"[{w}] drones drawn ({st['n']}); legend {st['legend']}")
    await shot("drones_overview")
    pos = await pg.evaluate("""() => { const S = window.__scenescout, D = S.D, m = new D._m.constructor(), v = D._p.clone();
      D.body.getMatrixAt(0, m); v.setFromMatrixPosition(m); const ref = D.tracks[D.idx[0]].ref;
      const e = [v.x, -v.z, v.y]; S.W.flyTo(e, {dist: 260, pitch: -20}); return ref; }""")
    await pg.wait_for_timeout(1600)
    xy = await pg.evaluate("""() => { const S = window.__scenescout, D = S.D, m = new D._m.constructor(), v = D._p.clone();
      D.body.getMatrixAt(0, m); v.setFromMatrixPosition(m).project(S.W.camera); const r = S.W.renderer.domElement.getBoundingClientRect();
      return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height]; }""")
    await pg.mouse.click(xy[0], xy[1])
    await pg.wait_for_timeout(600)
    sel = await pg.evaluate("window.__scenescout.highlighted.slice()")
    check(pos in sel, f"[{w}] clicking a drone selects its track ({pos} in {sel})")
    await shot("drones_close")
    await pg.keyboard.press("u")

    # fly mode at a low oblique angle near Jetty J1
    await esc()
    await pg.evaluate("""() => { const S = window.__scenescout, j = S.objs.find((o) => o.name === 'Jetty J1');
      S.W.flyTo([j.enu[0], j.enu[1], 0], {dist: 380, pitch: -9, heading: 300, dur: 0}); }""")
    await pg.wait_for_timeout(600)
    await pg.keyboard.press("f")
    fly = await pg.evaluate("window.__scenescout.cam.flying")
    check(fly, f"[{w}] F enters fly mode")
    y0 = await pg.evaluate("window.__scenescout.W.camera.position.toArray()")
    await pg.keyboard.down("w")
    await pg.keyboard.down("q")
    await pg.wait_for_timeout(900)
    await pg.keyboard.up("w")
    await pg.keyboard.up("q")
    await pg.wait_for_timeout(500)
    y1 = await pg.evaluate("window.__scenescout.W.camera.position.toArray()")
    demo_open = await pg.is_visible("#demo")
    wall = await pg.evaluate("document.documentElement.classList.contains('wall')")
    check(y1 != y0 and y1[1] < y0[1] and not demo_open and not wall, f"[{w}] fly keys move the camera ({[round(v) for v in y0]} -> {[round(v) for v in y1]}) and do not trigger W/D/Q")
    await shot("fly_jetty")
    await pg.keyboard.press("Escape")
    await pg.wait_for_timeout(300)
    check(not await pg.evaluate("window.__scenescout.cam.flying"), f"[{w}] Esc leaves fly mode")

    # saved view: custom oblique, save to 1, go elsewhere, recall
    await pg.evaluate("""() => { const S = window.__scenescout, t = S.objs.find((o) => o.kind === 'tank_farm');
      S.W.flyTo([t.enu[0], t.enu[1], 0], {dist: 900, pitch: -14, heading: 70, dur: 0}); }""")
    await pg.wait_for_timeout(400)
    await pg.keyboard.press("Shift+Digit1")
    saved = await pg.evaluate("window.__scenescout.W.camera.position.toArray()")
    await pg.click(".views [data-view=overview]")
    await pg.wait_for_timeout(1400)
    await pg.keyboard.press("Digit1")
    await pg.wait_for_timeout(1600)
    got = await pg.evaluate("window.__scenescout.W.camera.position.toArray()")
    check(max(abs(a - b) for a, b in zip(saved, got)) < 2, f"[{w}] saved view 1 recalled ({[round(v) for v in got]})")
    await shot("saved_view")

    # orbit about the cursor: right-drag turns the view, no NaN, camera stays above ground
    c0 = await pg.evaluate("window.__scenescout.W.camera.position.toArray()")
    await pg.mouse.move(w * 0.35, h * 0.55)
    await pg.mouse.down(button="right")
    await pg.mouse.move(w * 0.35 + 180, h * 0.55 - 60, steps=10)
    await pg.mouse.up(button="right")
    await pg.wait_for_timeout(800)
    c1 = await pg.evaluate("window.__scenescout.W.camera.position.toArray()")
    check(all(v == v for v in c1) and c1 != c0 and c1[1] > 2, f"[{w}] right-drag orbits about the cursor point ({[round(v) for v in c1]})")
    check(not errs, f"[{w}] no console errors ({errs[:3]})")
    await b.close()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8098")
    ap.add_argument("--events", default=None)
    ap.add_argument("--out", default="docs/design/audit/shots_demo_v2")
    ap.add_argument("--sizes", default="1920x1080,1280x720")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        for sz in a.sizes.split(","):
            w, h = (int(x) for x in sz.split("x"))
            await run_size(p, a.url, out, Path(a.events) if a.events else None, w, h)
    print(f"{len(FAILS)} FAIL" if FAILS else "ALL PASS")
    sys.exit(1 if FAILS else 0)


asyncio.run(main())
