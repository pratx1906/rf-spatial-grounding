/* SceneScout console (docs/design/FINAL.md). Cesium world + exported Blender structures (world.js), the agent's
   live work as numbered steps whose marks are stamped in 3D, a plain-language verdict, an honest confidence ruler,
   the operator gate, and a recorded comparison with the untrained model. Belief-side data only. */
"use strict";
const C = window.Cesium;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const INF = Number.POSITIVE_INFINITY;

const T = {void: "#000000", panel: "#121212", rule: "#2a2a2a", ash: "#9a958c", bone: "#ece8e1", sodium: "#f2a33a",
           signal: "#ff5a1f", friendly: "#7db0cc"};
const col = (hex, a = 1) => C.Color.fromCssColorString(hex).withAlpha(a);
const FONT = (px, w = 400) => `${w} ${px}px B612, sans-serif`;
const RAY_LEN_M = 3000;
const REVIEW_MARK = 0.6;
const MAX_TURNS = 8;

const TOOL_LABEL = {
  site_resolve: "Find the place", site_describe: "Describe the place", region_contents: "List what is in the area",
  spatial_relation: "Measure distances", inside_zone: "Check zone membership", track_search: "Search tracks",
  track_detail: "Read track details", df_hypotheses: "Read DF bearings", predict_aoa: "Predict multipath from the RF model",
  geometry_check: "Check bearing geometry", fit_candidates: "Fit emitter position", prop_query: "Query propagation",
  coverage_query: "Query DF coverage", weak_zones: "Find weak coverage", suggest_node: "Place a new DF node",
  model_trust: "Check RF model trust", drone_state: "Read DF drone state", plan_waypoints: "Plan drone waypoints",
  move_drone: "Fly the DF drone", effect_footprint: "Predict jamming footprint", collateral_check: "Check collateral",
  event_search: "Search events", track_history: "Read track history", sql_readonly: "Query the event log",
  declare_intent: "State the plan", ask_user: "Ask the operator", submit: "Submit the answer",
};
const ABSTAIN = {
  no_sensor_coverage: "No sensor covers that place or time, so there is nothing to base an answer on.",
  entity_not_found: "Nothing with that name exists in the port twin.",
  ambiguous_unresolved: "The question stayed ambiguous after checking; it needs more detail.",
  outside_tool_scope: "This is outside what SceneScout does. No tool in SceneScout can act on a drone or switch on an effector.",
  insufficient_evidence: "The evidence is not strong enough to give an answer.",
};
const END_REASON = {
  max_turns: "No answer within 8 turns", format_failed: "The answer did not follow the format",
  verifier_failed: "The answer cited evidence it never retrieved", timeout: "Ran out of time", submitted: "Answered",
};

const S = {
  v: null, site: null, meta: null, tracks: [], start: null, dur: 600, enuM: null, grid: null,
  ds: {}, recs: new Map(), rays: new Map(), sensors: new Map(), groups: new Map(), highlighted: [],
  demo: [], replays: {}, es: null, pending: [], finished: true, t0: 0, taskId: null,
  step: 0, stepEnts: new Map(), ep: null, userCam: false, camMoves: 0, prevView: null, gateOpen: false, tick: null,
};

/* ------------------------------------------------------------------ geometry */
function enuToLonLat(e, n) {
  const p = C.Matrix4.multiplyByPoint(S.enuM, new C.Cartesian3(e, n, 0), new C.Cartesian3());
  const c = C.Cartographic.fromCartesian(p);
  return [C.Math.toDegrees(c.longitude), C.Math.toDegrees(c.latitude)];
}
function lonLatToEnu(lon, lat) {
  const inv = C.Matrix4.inverseTransformation(S.enuM, new C.Matrix4());
  const p = C.Matrix4.multiplyByPoint(inv, C.Cartesian3.fromDegrees(lon, lat, 0), new C.Cartesian3());
  return [p.x, p.y];
}
function groundAt(lon, lat) {
  const g = S.grid;
  if (!g) return 0;
  const fx = Math.min(Math.max((lon - g.lon0) / g.dlon, 0), g.n - 1.0001);
  const fy = Math.min(Math.max((lat - g.lat0) / g.dlat, 0), g.n - 1.0001);
  const i = Math.floor(fx), j = Math.floor(fy), u = fx - i, w = fy - j;
  const h = (a, b) => g.h[b * g.n + a];
  return (1 - u) * (1 - w) * h(i, j) + u * (1 - w) * h(i + 1, j) + (1 - u) * w * h(i, j + 1) + u * w * h(i + 1, j + 1);
}
async function buildGroundGrid(terrain) {
  const ring = S.site.meta.extent || S.site.objects[0].polygon;
  const lons = ring.map((p) => p[0]), lats = ring.map((p) => p[1]);
  const n = 15, lon0 = Math.min(...lons), lat0 = Math.min(...lats);
  const dlon = (Math.max(...lons) - lon0) / (n - 1), dlat = (Math.max(...lats) - lat0) / (n - 1);
  const cartos = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) cartos.push(C.Cartographic.fromDegrees(lon0 + i * dlon, lat0 + j * dlat));
  const sens = S.site.sensors.map((s) => C.Cartographic.fromDegrees(s.lon, s.lat));
  let ok = false;
  if (terrain && !(terrain instanceof C.EllipsoidTerrainProvider)) {
    try { await C.sampleTerrainMostDetailed(terrain, cartos.concat(sens)); ok = true; } catch (e) { console.warn("terrain sampling", e); }
  }
  S.grid = {n, lon0, lat0, dlon, dlat, h: cartos.map((c) => (ok && Number.isFinite(c.height) ? c.height : 0))};
  S.site.sensors.forEach((s, k) => { s.ground = ok && Number.isFinite(sens[k].height) ? sens[k].height : groundAt(s.lon, s.lat); });
  S.groundOk = ok;
}
function posENU(x, y, z = 0) {
  const [lon, lat] = enuToLonLat(x, y);
  return C.Cartesian3.fromDegrees(lon, lat, groundAt(lon, lat) + z);
}
const secs = (time) => C.JulianDate.secondsDifference(time, S.start);
const jd = (t) => C.JulianDate.addSeconds(S.start, t, new C.JulianDate());
const now = () => secs(S.v.clock.currentTime);
function setSimTime(t) { S.v.clock.currentTime = jd(Math.max(0, Math.min(S.dur, t))); }
function idxAt(ts, t) {
  let lo = 0, hi = ts.length - 1;
  if (t <= ts[0]) return 0;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ts[m] <= t) lo = m; else hi = m - 1; }
  return lo;
}
const baseRef = (ref) => String(ref).split("@")[0];
const BAND = {sub_ghz_868_915: "868/915 MHz", ism_2g4: "2.4 GHz", ism_5g8: "5.8 GHz"};
const band = (b) => BAND[b] || String(b || "").replace(/_/g, " ");
const fmtM = (m) => (m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`);
const fmtDeg = (d) => `${(((d % 360) + 360) % 360).toFixed(1).padStart(5, "0")}°`;
const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];
function nearestPlace(x, y) {
  let best = null, bd = INF;
  for (const o of S.site.objects) {
    if (!["jetty", "tank_farm", "substation", "silo", "building"].includes(o.kind) || /synthetic/.test(o.provenance?.name_source || "")) continue;
    const [ox, oy] = lonLatToEnu(o.center[0], o.center[1]);
    const d = Math.hypot(x - ox, y - oy);
    if (d < bd) { bd = d; best = {o, dx: x - ox, dy: y - oy, d}; }
  }
  if (!best) return `${Math.round(x)} m E, ${Math.round(y)} m N of the site origin`;
  if (best.d < 60) return `at ${best.o.name}`;
  const dir = COMPASS[Math.round(((Math.atan2(best.dx, best.dy) * 180 / Math.PI + 360) % 360) / 45) % 8];
  return `${fmtM(best.d)} ${dir} of ${best.o.name}`;
}

/* ------------------------------------------------------------------ symbols (canvas billboards) */
const SYM = {};
function sym(kind, color, size = 22) {
  const key = `${kind}|${color}|${size}`;
  if (SYM[key]) return SYM[key];
  const c = document.createElement("canvas");
  c.width = c.height = size * 2;
  const g = c.getContext("2d");
  g.scale(2, 2);
  const m = size / 2, r = size / 2 - 3;
  g.lineWidth = 2;
  const path = () => {
    g.beginPath();
    if (kind === "unknown") g.arc(m, m, r, 0, Math.PI * 2);                          // open circle: unknown air track
    else if (kind === "hostile") { g.moveTo(m, m - r); g.lineTo(m + r, m); g.lineTo(m, m + r); g.lineTo(m - r, m); g.closePath(); }
    else if (kind === "friendly") g.rect(m - r, m - r * 0.7, r * 2, r * 1.4);           // rectangle: own asset
    else if (kind === "fix") { g.arc(m, m, r * 0.45, 0, Math.PI * 2); }
    else if (kind === "victim") { g.moveTo(m, m - r); g.lineTo(m + r, m); g.lineTo(m, m + r); g.lineTo(m - r, m); g.closePath(); }
  };
  g.strokeStyle = "#000"; g.lineWidth = 5; path(); g.stroke();
  g.strokeStyle = color; g.lineWidth = 2; path(); g.stroke();
  if (kind === "fix" || kind === "victim") { g.fillStyle = color; path(); g.fill(); }
  if (kind === "unknown") { g.fillStyle = color; g.beginPath(); g.arc(m, m, 2.2, 0, Math.PI * 2); g.fill(); }
  SYM[key] = c;
  return c;
}
function label(text, color = T.bone, px = 13, extra = {}) {
  return Object.assign({text, font: FONT(px), fillColor: col(color), showBackground: true, backgroundColor: col(T.void, 0.85),
    backgroundPadding: new C.Cartesian2(6, 4), disableDepthTestDistance: INF, horizontalOrigin: C.HorizontalOrigin.LEFT,
    verticalOrigin: C.VerticalOrigin.CENTER, pixelOffset: new C.Cartesian2(14, 0)}, extra);
}
function cased(color, a = 1) {
  return new C.PolylineOutlineMaterialProperty({color: col(color, a), outlineColor: col(T.void, 0.9), outlineWidth: 1});
}
function dashed(color, a = 1, len = 14) {
  return new C.PolylineDashMaterialProperty({color: col(color, a), gapColor: col(T.void, 0.35), dashLength: len});
}

/* ------------------------------------------------------------------ boot */
async function getJSON(u) {
  const r = await fetch(u);
  if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
  return r.json();
}
function bootLine(k, v, fault = false) {
  const li = document.createElement("li");
  if (fault) li.className = "fault";
  li.innerHTML = `<span>${esc(k)}</span><span>${esc(v)}</span>`;
  $("#bootList").appendChild(li);
}
function sysCell(id, value, state = "ok") {
  const c = $(id);
  c.dataset.state = state;
  c.querySelector(".v").textContent = value;
}
const wait = (ms) => new Promise((r) => setTimeout(r, window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : ms));

function modelName(m) {
  if (/sft/i.test(m)) return "SceneScout 4B, trained (SFT v1)";
  if (/rl/i.test(m)) return "SceneScout 4B, trained (RL)";
  return "4B base, untrained";
}

async function boot() {
  const [meta, site, trk, reps] = await Promise.all([getJSON("/api/meta"), getJSON("/api/site"), getJSON("/api/tracks?step_s=2"),
                                                      getJSON("/api/replays").catch(() => ({}))]);
  S.meta = meta; S.site = site; S.tracks = trk.tracks; S.demo = meta.demo || []; S.replays = reps;
  S.dur = site.meta.duration_s || meta.duration_s || 600;

  const served = (meta.llm.models || []).includes(meta.model);
  sysCell("#sysModel", modelName(meta.model), served ? "ok" : "fault");
  bootLine("Model", served ? `${modelName(meta.model)}, loaded` : `${meta.model} not served`, !served);
  await wait(160);
  sysCell("#sysServer", meta.llm.ok ? "ready" : "offline", meta.llm.ok ? "ok" : "fault");
  bootLine("Model server", meta.llm.ok ? "ready (vLLM)" : "offline: demo recordings still play", !meta.llm.ok);
  await wait(160);
  const fixed = site.sensors.filter((s) => s.type !== "drone_df").length;
  sysCell("#sysSensors", `${fixed} AIR7311 + DF drone`);
  bootLine("DF sensors", `${fixed} fixed AIR7311 nodes and a DF drone (synthetic sites)`);
  await wait(160);
  sysCell("#sysRf", /air7311/i.test(site.meta.rf_backend || "") ? "Sionna RT, AIR7311" : (site.meta.rf_backend || "unknown"));
  bootLine("RF model", site.meta.rf_backend || "unknown");
  await wait(160);
  sysCell("#sysData", `seed 0, ${String(site.meta.data_version || meta.data_version).slice(0, 8)}`);
  const cnt = site.meta.counts || {};
  bootLine("Port twin", `${site.objects.length} objects, ${site.zones.length} zones (OpenStreetMap)`);
  void cnt;

  let token = null;
  try { const r = await fetch("/api/cesium_token"); if (r.ok) token = (await r.json()).token; } catch (_) { /* offline */ }
  const opts = {baseLayerPicker: false, geocoder: false, homeButton: false, sceneModePicker: false, navigationHelpButton: false,
                fullscreenButton: false, infoBox: false, selectionIndicator: false, animation: false, timeline: false,
                requestRenderMode: false, shouldAnimate: false, msaaSamples: 4};
  let terrain = null;
  if (token) {
    C.Ion.defaultAccessToken = token;
    try { terrain = await C.createWorldTerrainAsync(); } catch (e) { console.warn("terrain", e); }
  } else {
    opts.baseLayer = new C.ImageryLayer(new C.OpenStreetMapImageryProvider({url: "https://tile.openstreetmap.org/"}));
  }
  opts.terrainProvider = terrain || new C.EllipsoidTerrainProvider();
  bootLine("Imagery and terrain", token ? "Cesium ion world terrain and aerial imagery" : "no ion token: OpenStreetMap tiles, flat", !token);
  const v = S.v = new C.Viewer("cesium", opts);
  v.scene.backgroundColor = col(T.void);
  const o = site.meta.origin;
  S.enuM = C.Transforms.eastNorthUpToFixedFrame(C.Cartesian3.fromDegrees(o.lon, o.lat, 0));
  S.start = C.JulianDate.fromIso8601(site.meta.t0_iso || meta.t0_iso || "2026-01-01T00:00:00Z");
  const clk = v.clock;
  clk.startTime = S.start.clone(); clk.stopTime = jd(S.dur); clk.currentTime = S.start.clone();
  clk.clockRange = C.ClockRange.CLAMPED; clk.multiplier = 10; clk.shouldAnimate = false;
  // light the port at the local afternoon of the scenario day, independent of the sim clock's playback
  v.scene.light = new C.SunLight();

  for (const k of ["zones", "objects", "sensors", "tracks", "rays", "overlay", "answer"]) {
    S.ds[k] = new C.CustomDataSource(k);
    await v.dataSources.add(S.ds[k]);
  }
  await document.fonts.load(FONT(13)).catch(() => null);
  await buildGroundGrid(v.terrainProvider);
  drawZones(); drawObjects(); drawSensors(); drawTracks();

  const li = document.createElement("li");
  li.innerHTML = `<span>Port structures</span><span id="bootWorld">loading</span>`;
  $("#bootList").appendChild(li);
  const w = await window.SSWorld.init(v, site, {
    tint: "#3b362e",
    onProgress: (n, total) => { $("#bootWorld").textContent = `${n} of ${total} groups`; sysCell("#sysWorld", `${n} / ${total}`, "wait"); },
  });
  $("#bootWorld").textContent = w.tiles ? `${w.tiles} groups from the Blender export` : "not exported: run envbuild/blender/export_web.py";
  sysCell("#sysWorld", w.tiles ? `${w.tiles} groups` : "imagery only", w.tiles ? "ok" : "fault");
  // the scene clock drives Cesium's sun; keep the port in late-afternoon light whatever the sim time is
  v.clock.onTick.addEventListener(onTick);
  sunAt();

  fidelity(); wireUI(); timelineTicks(); demoMenu(); view("overview", true);
  await wait(500);
  $("#curtain").classList.add("gone");
  setTimeout(() => ($("#curtain").hidden = true), 1000);
  if (!meta.llm.ok) banner("Model server offline.", "Live questions will fail. Demo recordings still play (D).");
  window.__scenescout = S;
  setInterval(clockTick, 1000); clockTick();
}

function sunAt() {
  // Cesium lights from the clock's time; use a fixed, flattering local time (15:30 IST) for the light only
  const light = S.v.scene.light;
  const day = C.JulianDate.fromIso8601("2026-10-18T10:00:00Z");
  if (light instanceof C.SunLight) {
    S.v.scene.light = new C.DirectionalLight({direction: sunDir(day), intensity: 2.2});
  }
}
function sunDir(time) {
  const icrf = C.Transforms.computeIcrfToFixedMatrix(time) || C.Matrix3.IDENTITY;
  const sun = C.Simon1994PlanetaryPositions.computeSunPositionInEarthInertialFrame(time, new C.Cartesian3());
  const fixed = C.Matrix3.multiplyByVector(icrf, sun, new C.Cartesian3());
  return C.Cartesian3.normalize(C.Cartesian3.negate(fixed, fixed), new C.Cartesian3());
}

function fidelity() {
  const meta = S.site.meta, f = meta.fidelity || {};
  const geom = /openstreetmap/i.test(f.geometry || "") ? "OpenStreetMap" : (f.geometry || "unknown");
  $("#fidText").textContent = `Geometry: ${geom}, heights estimated; sensors and drones synthetic; simulation`;
  const rows = Object.entries(f).map(([k, val]) => `<dt>${esc(k)}</dt><dd>${esc(val)}</dd>`);
  rows.push(`<dt>structures</dt><dd>Blender export of the OSM twin (illustrative detail; ML footprints omitted)</dd>`);
  rows.push(`<dt>terrain</dt><dd>${S.groundOk ? "Cesium World Terrain" : "flat ellipsoid"}</dd>`);
  rows.push(`<dt>attribution</dt><dd>${esc((meta.attribution || "").replace("(c)", "©"))}</dd>`);
  $("#fidMore").innerHTML = rows.join("");
}

/* ------------------------------------------------------------------ twin layers */
function drawObjects() {
  // Structures come from the Blender export; objects stay as near-invisible pick targets plus a few names.
  const ents = S.ds.objects.entities;
  ents.suspendEvents();
  for (const o of S.site.objects) {
    const pos = C.Cartesian3.fromDegreesArray(o.polygon.flat());
    const rec = {kind: "obj", data: o, positions: pos};
    rec.fill = new C.ColorMaterialProperty(col(T.bone, 0.004));
    const ent = {id: o.ref, name: o.name, position: C.Cartesian3.fromDegrees(o.center[0], o.center[1], (o.height_m || 0) + 8),
                 polygon: {hierarchy: pos, material: rec.fill, classificationType: C.ClassificationType.BOTH}};
    if (o.kind === "jetty") {
      ent.label = {text: o.name, font: FONT(12), fillColor: col(T.ash), style: C.LabelStyle.FILL_AND_OUTLINE,
                   outlineColor: col(T.void), outlineWidth: 3, heightReference: C.HeightReference.RELATIVE_TO_GROUND,
                   disableDepthTestDistance: INF, distanceDisplayCondition: new C.DistanceDisplayCondition(0, 6000)};
    }
    rec.ent = ents.add(ent);
    S.recs.set(o.ref, rec);
  }
  ents.resumeEvents();
}
function drawZones() {
  for (const z of S.site.zones) {
    if (!["restricted", "perimeter", "aviation"].includes(z.kind) && z.ref !== "zone:hero_zone") {
      S.recs.set(z.ref, {kind: "zone", data: z, positions: C.Cartesian3.fromDegreesArray(z.polygon.flat())});
      continue;
    }
    const pos = C.Cartesian3.fromDegreesArray(z.polygon.flat());
    const rec = {kind: "zone", data: z, positions: pos};
    const lineCol = z.kind === "restricted" ? T.ash : T.ash;
    rec.line = z.kind === "restricted" ? new C.ColorMaterialProperty(col(lineCol, 0.75)) : dashed(lineCol, 0.6, 20);
    rec.ent = S.ds.zones.entities.add({
      id: z.ref, name: z.name, position: C.Cartesian3.fromDegrees(z.center[0], z.center[1], 30),
      polyline: {positions: pos, clampToGround: true, width: z.kind === "restricted" ? 1.5 : 1.5, material: rec.line},
      label: {text: `${z.name}${/proposed|provisional/i.test(z.name) ? "" : z.kind === "restricted" ? " (proposed)" : ""}`,
              font: FONT(12), fillColor: col(T.ash), style: C.LabelStyle.FILL_AND_OUTLINE, outlineColor: col(T.void), outlineWidth: 3,
              heightReference: C.HeightReference.RELATIVE_TO_GROUND, disableDepthTestDistance: INF,
              distanceDisplayCondition: z.kind === "restricted" ? new C.DistanceDisplayCondition(0, 2400) : new C.DistanceDisplayCondition(2500, 16000)},
    });
    S.recs.set(z.ref, rec);
  }
}
function drawSensors() {
  for (const s of S.site.sensors) {
    S.sensors.set(s.ref, s);
    const hg = s.ground ?? 0;
    const top = C.Cartesian3.fromDegrees(s.lon, s.lat, hg + (s.z_m || 0));
    const drone = s.type === "drone_df";
    const rec = {kind: "sensor", data: s, positions: [top]};
    rec.ent = S.ds.sensors.entities.add({
      id: s.ref, name: s.id, position: top,
      billboard: {image: sym("friendly", T.friendly, 20), disableDepthTestDistance: INF},
      label: label(drone ? `${s.id} DF drone` : s.id, T.friendly, 13),
      polyline: drone ? undefined : {positions: [C.Cartesian3.fromDegrees(s.lon, s.lat, hg), top], width: 2, material: col(T.friendly, 0.8)},
    });
    S.recs.set(s.ref, rec);
  }
}
function drawTracks() {
  const groupIds = [...new Set(S.tracks.map((t) => t.group).filter(Boolean))].sort();
  groupIds.forEach((g) => S.groups.set(g, {members: []}));
  const ents = S.ds.tracks.entities;
  for (const tr of S.tracks) {
    if (!tr.pts.length) continue;
    const prop = new C.SampledPositionProperty();
    tr.ts = tr.pts.map((p) => p[0]);
    const times = tr.pts.map((p) => jd(p[0]));
    tr.ground = tr.pts.map((p) => groundAt(p[1], p[2]));
    prop.addSamples(times, tr.pts.map((p, k) => C.Cartesian3.fromDegrees(p[1], p[2], tr.ground[k] + p[3])));
    const avail = new C.TimeIntervalCollection([new C.TimeInterval({start: times[0], stop: times[times.length - 1]})]);
    tr.avail = [tr.ts[0], tr.ts[tr.ts.length - 1]];
    const rec = {kind: "trk", data: tr};
    rec.ent = ents.add({
      id: tr.ref, name: tr.ref, availability: avail, position: prop,
      billboard: {image: sym("unknown", T.bone, 20), disableDepthTestDistance: INF},
      label: label(tr.ref, T.bone, 12, {distanceDisplayCondition: new C.DistanceDisplayCondition(0, tr.group ? 2200 : 9000)}),
      path: {leadTime: 0, trailTime: tr.group ? 50 : 100, width: 1.5, resolution: 2, material: col(T.bone, tr.group ? 0.25 : 0.35)},
    });
    rec.tether = ents.add({id: `${tr.ref}#tether`, availability: avail, polyline: {
      positions: new C.CallbackProperty((time) => tetherPositions(tr, rec, time), false), width: 1, arcType: C.ArcType.NONE,
      material: col(T.bone, 0.3)}});
    S.recs.set(tr.ref, rec);
    if (tr.group) S.groups.get(tr.group).members.push(rec);
  }
  for (const [g, info] of S.groups) {
    const starts = info.members.map((m) => m.data.avail[0]), stops = info.members.map((m) => m.data.avail[1]);
    const avail = new C.TimeIntervalCollection([new C.TimeInterval({start: jd(Math.min(...starts)), stop: jd(Math.max(...stops))})]);
    const PosProp = C.CallbackPositionProperty || C.CallbackProperty;
    info.ent = ents.add({id: `grp:${g}`, name: `Swarm ${g}`, availability: avail,
      position: new PosProp((time, result) => groupCentroid(info, time, result), false),
      label: label(`Swarm ${g}, ${info.members.length} tracks`, T.bone, 13, {pixelOffset: new C.Cartesian2(0, -34),
        horizontalOrigin: C.HorizontalOrigin.CENTER})});
    S.recs.set(`grp:${g}`, {kind: "grp", data: {group: g, members: info.members.map((m) => m.data.ref)}, ent: info.ent});
  }
}
function tetherPositions(tr, rec, time) {
  const c = rec.ent.position.getValue(time);
  if (!c) return [];
  const carto = C.Cartographic.fromCartesian(c);
  return [c, C.Cartesian3.fromRadians(carto.longitude, carto.latitude, tr.ground[idxAt(tr.ts, secs(time))])];
}
function groupCentroid(info, time, result) {
  let n = 0, x = 0, y = 0, z = 0;
  for (const m of info.members) { const p = m.ent.position.getValue(time); if (p) { x += p.x; y += p.y; z += p.z; n++; } }
  if (!n) return undefined;
  const carto = C.Cartographic.fromCartesian(new C.Cartesian3(x / n, y / n, z / n));
  carto.height += 25;
  return C.Cartographic.toCartesian(carto, C.Ellipsoid.WGS84, result);
}

/* ------------------------------------------------------------------ focus + camera */
function setFocus(ref, on, color = T.sodium) {
  const rec = S.recs.get(ref);
  if (!rec) return;
  if (rec.kind === "trk") {
    rec.ent.billboard.image = sym("unknown", on ? color : T.bone, on ? 26 : 20);
    rec.ent.label.fillColor = col(on ? color : T.bone);
    rec.ent.label.distanceDisplayCondition = new C.DistanceDisplayCondition(0, on || !rec.data.group ? 12000 : 2200);
    rec.tether.polyline.material = col(on ? color : T.bone, on ? 0.8 : 0.3);
  } else if (rec.kind === "zone" && rec.line) {
    rec.ent.polyline.material = on ? cased(color) : rec.line;
    rec.ent.polyline.width = on ? 3 : 1.5;
    rec.ent.label.fillColor = col(on ? color : T.ash);
  } else if (rec.kind === "obj") {
    if (on) rec.outline = S.ds.answer.entities.add({polyline: {positions: rec.positions, clampToGround: true, width: 3, material: col(color)}});
    else if (rec.outline) { S.ds.answer.entities.remove(rec.outline); rec.outline = null; }
  } else if (rec.kind === "sensor") {
    rec.ent.label.fillColor = col(on ? T.bone : T.friendly);
  }
}
function clearHighlight() {
  const was = S.highlighted; S.highlighted = [];
  for (const r of was) setFocus(r, false);
  if (S.ep && !S.finished) recedeTracks(true);
}

function recPoints(rec) {
  if (rec.kind === "trk" || rec.kind === "grp") { const p = rec.ent.position.getValue(S.v.clock.currentTime); return p ? [p] : []; }
  if (rec.kind === "ray") return rec.positions;
  return rec.positions || [];
}
function camCaption(text) {
  $("#camText").textContent = text;
}
function flyToPoints(points, {pitch = -0.62, minRange = 600, heading, caption, auto = true} = {}) {
  if (!points.length) return;
  if (auto) {
    if (S.userCam || S.camMoves >= 2) return;          // at most establish + settle; a user drag cancels both
    if (!S.prevView) S.prevView = saveView();
    S.camMoves += 1;
    $("#camBack").hidden = false;
  }
  S.v.trackedEntity = undefined;
  const bs = C.BoundingSphere.fromPoints(points);
  const range = Math.max(bs.radius * 2.8, minRange);
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  S.v.camera.flyToBoundingSphere(bs, {offset: new C.HeadingPitchRange(heading ?? S.v.camera.heading, pitch, range),
                                      duration: reduce ? 0 : 1.2, easingFunction: C.EasingFunction.CUBIC_IN_OUT});
  if (caption) camCaption(caption);
}
function saveView() {
  const c = S.v.camera;
  return {destination: c.positionWC.clone(), orientation: {heading: c.heading, pitch: c.pitch, roll: c.roll}, caption: $("#camText").textContent};
}
function focusRefs(refs, {fly = true, color = T.sodium, caption, auto = false} = {}) {
  clearHighlight();
  let tSet = null;
  const recs = [];
  for (const r of refs.slice(0, 40)) {
    const [b, at] = String(r).split("@");
    if (b.startsWith("hyp:")) { const ray = S.rays.get(b) || S.rays.get(r); if (ray) recs.push(ray); continue; }
    const rec = S.recs.get(b);
    if (!rec) continue;
    if (at != null && b.startsWith("trk:")) tSet = Number(at);
    setFocus(b, true, color); S.highlighted.push(b); recs.push(rec);
  }
  const trks = recs.filter((r) => r.kind === "trk");
  if (tSet != null) setSimTime(tSet);
  else if (trks.length && !trks.some((r) => r.ent.isAvailable(S.v.clock.currentTime))) setSimTime((trks[0].data.avail[0] + trks[0].data.avail[1]) / 2);
  if (fly && recs.length) flyToPoints(recs.slice(0, 25).flatMap(recPoints), {caption, auto});
  return recs.length;
}
function view(name, instant = false) {
  const v = S.v;
  v.trackedEntity = undefined;
  $$(".views [data-view]").forEach((b) => b.classList.toggle("on", b.dataset.view === name));
  const pts = (objs) => objs.flatMap((o) => o.polygon).map((p) => C.Cartesian3.fromDegrees(p[0], p[1], 0));
  const dur = instant ? 0 : 1.6;
  const fly = (bs, h, p, k, cap) => { v.camera.flyToBoundingSphere(bs, {offset: new C.HeadingPitchRange(C.Math.toRadians(h), C.Math.toRadians(p), bs.radius * k), duration: dur}); camCaption(cap); };
  if (name === "overview") fly(C.BoundingSphere.fromPoints(pts(S.site.objects.filter((o) => ["tank", "jetty", "tank_farm"].includes(o.kind)))), 330, -34, 1.6, "Port overview");
  else if (name === "jetties") fly(C.BoundingSphere.fromPoints(pts(S.site.objects.filter((o) => /^Jetty J[1-5]$/.test(o.name)))), 250, -26, 1.5, "Oil jetties J1 to J5");
  else if (name === "tankfarms") fly(C.BoundingSphere.fromPoints(pts(S.site.objects.filter((o) => o.kind === "tank_farm"))), 20, -36, 1.7, "Tank farms TF-1 to TF-8");
  else if (name === "top") {
    const bs = C.BoundingSphere.fromPoints(pts(S.site.objects.filter((o) => o.kind !== "road")));
    v.camera.flyToBoundingSphere(bs, {offset: new C.HeadingPitchRange(0, -C.Math.PI_OVER_TWO, bs.radius * 2.6), duration: dur});
    camCaption("Top-down, north up");
  }
  S.prevView = null; $("#camBack").hidden = true;
}

/* ------------------------------------------------------------------ DF rays */
function sensorAnchor(sensorRef, posENU) {
  const s = S.sensors.get(sensorRef);
  if (s) return {lon: s.lon, lat: s.lat, h: (s.ground ?? groundAt(s.lon, s.lat)) + (s.z_m || 0)};
  const [lon, lat] = enuToLonLat(posENU.x_m, posENU.y_m);
  return {lon, lat, h: groundAt(lon, lat) + (posENU.z_m || 10)};
}
function rayEnd(anchor, bearingDeg, len = RAY_LEN_M) {
  const o = C.Cartesian3.fromDegrees(anchor.lon, anchor.lat, anchor.h);
  const m = C.Transforms.eastNorthUpToFixedFrame(o, undefined, new C.Matrix4());
  const b = C.Math.toRadians(bearingDeg);
  return [o, C.Matrix4.multiplyByPoint(m, new C.Cartesian3(len * Math.sin(b), len * Math.cos(b), 0), new C.Cartesian3())];
}
function drawRays(sensors, {t, track, step}) {
  if (t != null) setSimTime(t);
  for (const s of sensors) {
    for (const h of s.hyps) {
      if (S.rays.has(h.ref)) continue;
      const positions = rayEnd(s.anchor, h.bearing_deg);
      const rec = {kind: "ray", data: {...h, sensor: s.sensor, track, t}, positions};
      rec.ent = S.ds.rays.entities.add({
        id: `ray:${h.ref}`, name: h.ref, position: positions[1],
        polyline: {positions, width: 2, arcType: C.ArcType.NONE, material: cased(T.bone, 0.75)},
        label: label(`${s.sensor.replace("sensor:", "")} ${fmtDeg(h.bearing_deg)}`, T.bone, 12,
                     {distanceDisplayCondition: new C.DistanceDisplayCondition(0, 9000)}),
      });
      rec.step = step;
      tagStep(step, rec.ent);
      S.rays.set(h.ref, rec); S.rays.set(baseRef(h.ref), rec);
      S.ep.bearings.set(baseRef(h.ref), {deg: h.bearing_deg, sensor: s.sensor.replace("sensor:", ""), snr: h.snr_db});
    }
  }
}
function styleRay(rec, how) {
  const e = rec.ent;
  if (how === "chosen") { e.polyline.material = cased(T.sodium); e.polyline.width = 4; e.label.fillColor = col(T.sodium); }
  else if (how === "rejected") { e.polyline.material = dashed(T.ash, 0.95, 16); e.polyline.width = 2.5; e.label.fillColor = col(T.ash); }
  else { e.polyline.material = col(T.bone, 0.18); e.polyline.width = 1.5; e.label.show = false; }
}

/* ------------------------------------------------------------------ overlays with step stamps */
function tagStep(step, ent) {
  if (!S.stepEnts.has(step)) S.stepEnts.set(step, []);
  S.stepEnts.get(step).push(ent);
  const li = $(`#steps li[data-step="${step}"]`);
  if (li) li.classList.add("hasgeo");
}
function stamp(step, position) {
  const e = S.ds.overlay.entities.add({position, label: {text: String(step), font: FONT(12, 700), fillColor: col(T.void),
    showBackground: true, backgroundColor: col(T.bone, 0.95), backgroundPadding: new C.Cartesian2(5, 2), disableDepthTestDistance: INF,
    pixelOffset: new C.Cartesian2(-14, -14), horizontalOrigin: C.HorizontalOrigin.CENTER}});
  tagStep(step, e);
  return e;
}
function ov(step, ent) { const e = S.ds.overlay.entities.add(ent); tagStep(step, e); return e; }
function ovPoint(step, x, y, z, color, text, kind = "fix", size = 18) {
  return ov(step, {position: posENU(x, y, z), billboard: {image: sym(kind, color, size), disableDepthTestDistance: INF},
                   label: text ? label(text, color, 12) : undefined});
}
function ovRing(step, ring, color, alpha) {
  const flat = ring.flatMap(([x, y]) => enuToLonLat(x, y));
  return ov(step, {polygon: {hierarchy: C.Cartesian3.fromDegreesArray(flat), material: col(color, alpha), classificationType: C.ClassificationType.TERRAIN},
                   polyline: {positions: C.Cartesian3.fromDegreesArray(flat), clampToGround: true, width: 1.5, material: col(color, 0.9)}});
}
function ellipsePts(p, u, z = 0) {
  const pts = [];
  const th = C.Math.toRadians(u.theta_deg || 0);
  for (let k = 0; k <= 48; k++) {
    const g = (k / 48) * 2 * Math.PI, a = u.a_m * Math.cos(g), b = u.b_m * Math.sin(g);
    pts.push(posENU(p.x_m + a * Math.sin(th) + b * Math.cos(th), p.y_m + a * Math.cos(th) - b * Math.sin(th), z));
  }
  return pts;
}
function ovFix(step, p, u, color, text) {
  // fix point at altitude, a stalk to the ground and the 1-sigma ellipse on the ground
  const z = p.z_m || 0;
  const ents = [ovPoint(step, p.x_m, p.y_m, z, color, text, "fix", 20)];
  if (z > 2) ents.push(ov(step, {polyline: {positions: [posENU(p.x_m, p.y_m, 0), posENU(p.x_m, p.y_m, z)], width: 1.5, arcType: C.ArcType.NONE, material: col(color, 0.8)}}));
  if (u && u.a_m) ents.push(ov(step, {polyline: {positions: ellipsePts(p, u, 0.5), width: 2, arcType: C.ArcType.NONE, material: cased(color)}}));
  return ents;
}
function recolor(ents, color) {
  for (const e of ents) {
    if (e.billboard) { const k = /victim/.test(e._sym || "") ? "victim" : "fix"; e.billboard.image = sym(k, color, 20); }
    if (e.label) e.label.fillColor = col(color);
    if (e.polyline) e.polyline.material = e.polyline.width?.getValue?.() > 1.8 ? cased(color) : col(color, 0.8);
  }
}

function drawToolOverlay(ev, step) {
  const d = (ev.result && ev.result.data) || {}, ui = ev.ui || {};
  const E = S.ep;
  switch (ev.name) {
    case "df_hypotheses": {
      const sensors = (d.sensors || []).map((s) => ({sensor: s.sensor, anchor: sensorAnchor(s.sensor, s.sensor_pos || {x_m: 0, y_m: 0}),
                                                     hyps: s.hyps || []}));
      drawRays(sensors, {t: d.t_s, track: d.track, step});
      for (const s of sensors) stamp(step, C.Cartesian3.fromDegrees(s.anchor.lon, s.anchor.lat, s.anchor.h + 20));
      if (d.track) { E.track = d.track; focusRefs([d.track], {fly: false, color: T.bone}); }
      const pts = [...new Set(S.rays.values())].flatMap((r) => r.positions);
      flyToPoints(pts, {minRange: 2200, pitch: -0.75, caption: `Bearings on ${d.track || "the track"} at t ${d.t_s} s`});
      return;
    }
    case "predict_aoa": {
      const anc = sensorAnchor(d.sensor, {x_m: 0, y_m: 0});
      for (const pk of d.peaks || []) {
        const [o, end] = rayEnd(anc, pk.bearing_deg, RAY_LEN_M * 0.6);
        ov(step, {position: end, polyline: {positions: [o, end], width: 1.5, arcType: C.ArcType.NONE, material: dashed(T.bone, 0.6, 6)},
                  label: label(`predicted ${pk.kind} ${fmtDeg(pk.bearing_deg)}`, T.bone, 11, {distanceDisplayCondition: new C.DistanceDisplayCondition(0, 7000)})});
        if (pk.via_ref && ui.vias && ui.vias[pk.via_ref] && d.point) {
          const [vx, vy] = ui.vias[pk.via_ref];
          E.reflectors.push({ref: pk.via_ref, x: vx, y: vy});
          ov(step, {polyline: {positions: [C.Cartesian3.fromDegrees(anc.lon, anc.lat, anc.h), posENU(vx, vy, 15), posENU(d.point.x_m, d.point.y_m, d.point.z_m || 60)],
                               width: 1.5, arcType: C.ArcType.NONE, material: dashed(T.ash, 0.9, 4)}});
        }
      }
      stamp(step, C.Cartesian3.fromDegrees(anc.lon, anc.lat, anc.h + 40));
      return;
    }
    case "fit_candidates": {
      const cands = d.candidates || [];
      E.fits = cands;
      cands.forEach((c, i) => {
        const res = c.residual_deg_rms != null ? `${c.residual_deg_rms}° rms` : `${c.residual_m_rms} m rms`;
        const ents = i === 0 ? ovFix(step, c.position, c.uncertainty, T.bone, `${c.ref || "best fit"}, ${res}`)
                             : [ovPoint(step, c.position.x_m, c.position.y_m, c.position.z_m || 0, T.ash, c.ref || null, "fix", 12)];
        if (i === 0) { E.fitEnts = ents; stamp(step, posENU(c.position.x_m, c.position.y_m, (c.position.z_m || 0) + 30)); }
      });
      return;
    }
    case "weak_zones": {
      for (const z of ui.zones || []) for (const r of z.rings) ovRing(step, r, T.bone, 0.16);
      const zs = d.zones || [];
      zs.slice(0, 6).forEach((z) => ovPoint(step, z.centroid.x_m, z.centroid.y_m, 0, T.bone, `weak, ${(z.area_m2 / 1e4).toFixed(1)} ha`, "fix", 10));
      if (zs[0]) stamp(step, posENU(zs[0].centroid.x_m, zs[0].centroid.y_m, 40));
      if (zs.length) flyToPoints(zs.map((z) => posENU(z.centroid.x_m, z.centroid.y_m)), {minRange: 1800, pitch: -1.1, caption: "Weak DF coverage"});
      return;
    }
    case "suggest_node": {
      E.nodes = d.candidates || [];
      const pts = [];
      E.nodeEnts = [];
      for (const c of E.nodes) {
        const p = c.position;
        E.nodeEnts.push(ov(step, {polyline: {positions: [posENU(p.x_m, p.y_m, 0), posENU(p.x_m, p.y_m, (p.z_m || 20) + 10)], width: 3, arcType: C.ArcType.NONE, material: cased(T.bone)}}));
        E.nodeEnts.push(ovPoint(step, p.x_m, p.y_m, (p.z_m || 20) + 10, T.bone, `${c.ref.replace("nodecand:", "candidate ")}, gain ${c.predicted_gain}`, "fix", 14));
        pts.push(posENU(p.x_m, p.y_m));
      }
      if (E.nodes[0]) stamp(step, posENU(E.nodes[0].position.x_m, E.nodes[0].position.y_m, 60));
      if (pts.length) flyToPoints(pts, {minRange: 1600, caption: "Candidate DF node sites"});
      return;
    }
    case "effect_footprint": {
      E.fp = {band: d.band, eirp: d.eirp_dbm, effector: ev.args?.effector || d.effector, threshold: ev.args?.threshold_dbm ?? d.threshold_dbm};
      E.fpEnts = (ui.rings || []).map((r) => ovRing(step, r, T.bone, 0.06));     // layered light fills: inner bands read darker
      if (ui.effector) { const fe = ovPoint(step, ui.effector[0], ui.effector[1], 20, T.bone, `proposed ${band(d.band)}, ${d.eirp_dbm} dBm, not active`, "fix", 12);
                         fe.label.horizontalOrigin = C.HorizontalOrigin.RIGHT; fe.label.pixelOffset = new C.Cartesian2(-14, 22);
                         stamp(step, posENU(ui.effector[0], ui.effector[1], 60)); }
      if (d.bbox) flyToPoints([posENU(d.bbox[0], d.bbox[1]), posENU(d.bbox[2], d.bbox[3])], {pitch: -0.95, caption: "Predicted jamming footprint"});
      return;
    }
    case "collateral_check": {
      E.victims = ui.victims || [];
      E.victimEnts = [];
      const thr = Number(E.fp?.threshold ?? -90);
      E.victims.forEach((v, i) => {
        if (!v.pos) return;
        const margin = Number.isFinite(v.level_dbm) ? ` (${(v.level_dbm - thr).toFixed(0)} dB over)` : "";
        const e = ovPoint(step, v.pos[0], v.pos[1], 25, T.bone, `${v.ref.split(":")[1]}, ${v.level_dbm} dBm${margin}`, "victim", 20);
        e.label.pixelOffset = new C.Cartesian2(18, -26 - 26 * i);   // stagger so labels never stack on one point
        e._sym = "victim";
        E.victimEnts.push(e);
        ov(step, {polyline: {positions: [posENU(v.pos[0], v.pos[1], 25), posENU(v.pos[0], v.pos[1], 0)], width: 1, arcType: C.ArcType.NONE, material: col(T.bone, 0.5)}});
      });
      if (E.victims[0] && E.victims[0].pos) stamp(step, posENU(E.victims[0].pos[0], E.victims[0].pos[1], 60));
      return;
    }
    case "plan_waypoints":
      for (const w of d.waypoints || []) ovPoint(step, w.position.x_m, w.position.y_m, w.position.z_m || 80, T.ash, null, "fix", 10);
      return;
    case "drone_state":
    case "move_drone": {
      const p = d.pose, e = d.estimate;
      E.drone = E.drone || [];
      if (p) {
        E.drone.push(posENU(p.x_m, p.y_m, p.z_m || 80));
        if (ev.name === "move_drone") E.moves += 1;
        ovPoint(step, p.x_m, p.y_m, p.z_m || 80, T.friendly, ev.name === "move_drone" ? `D1 at ${d.wp}, bearing ${d.measurement ? fmtDeg(d.measurement.bearing_deg) : "?"}` : "D1 start", "friendly", 18);
        if (E.drone.length > 1) E.pathEnts.push(ov(step, {polyline: {positions: E.drone.slice(-2), width: 2.5, arcType: C.ArcType.NONE, material: cased(T.friendly)}}));
        stamp(step, posENU(p.x_m, p.y_m, (p.z_m || 80) + 30));
      }
      if (e) {
        for (const old of E.estEnts || []) {            // earlier estimates stay as faint ash rings; only the latest is labelled
          if (old.label) old.label.show = false;
          if (old.billboard) old.billboard.show = false;
          if (old.polyline) old.polyline.material = col(T.ash, 0.45);
        }
        E.estEnts = ovFix(step, e.position, e.uncertainty, T.bone, `estimate ±${Math.round(e.uncertainty.a_m)} m`);
      }
      if (p && e) flyToPoints([posENU(p.x_m, p.y_m), posENU(e.position.x_m, e.position.y_m)], {minRange: 1400, caption: "DF drone search"});
      return;
    }
    case "event_search": eventTicks(d.items || []); break;
    default: break;
  }
  const refs = ((ev.result && ev.result.refs) || []).filter((r) => /^(obj|zone|trk|sensor):/.test(r));
  if (refs.length && !["submit", "declare_intent"].includes(ev.name)) focusRefs(refs, {color: T.bone, auto: true});
}

function isolateStep(step) {
  for (const [k, ents] of S.stepEnts) for (const e of ents) e.show = step == null || k === step;
}

/* ------------------------------------------------------------------ the agent stream */
function recedeTracks(on) {
  for (const rec of S.recs.values()) {
    if (rec.kind !== "trk" || S.highlighted.includes(rec.data.ref)) continue;
    rec.ent.billboard.image = sym("unknown", on ? T.ash : T.bone, on ? 12 : 20);
    rec.ent.label.show = !on;
    rec.ent.path.show = !on;
    rec.tether.show = !on;
  }
  for (const [, info] of S.groups) if (info.ent) info.ent.label.show = !on;
}

function newEpisode(question) {
  S.epId = (S.epId || 0) + 1;
  if (S.es) { S.es.close(); S.es = null; }
  S.ds.overlay.entities.removeAll(); S.ds.rays.entities.removeAll(); S.ds.answer.entities.removeAll();
  S.rays.clear(); S.stepEnts.clear(); clearHighlight(); closeGate();
  $("#tlEvents").innerHTML = "";
  S.step = 0; S.pending = []; S.finished = false; S.t0 = performance.now(); S.userCam = false; S.camMoves = 0;
  S.ep = {bearings: new Map(), reflectors: [], fits: [], victims: [], moves: 0, drone: null, pathEnts: [], turn: 0, question};
  $("#idle").hidden = true; $("#question").hidden = false; $("#verdict").hidden = true; $("#compare").hidden = true;
  $("#qText").textContent = question; $("#qMeta").textContent = "";
  camCaption("View held from before the question");
  $("#steps").innerHTML = ""; $("#stepsBox").hidden = false;
  $("#working").hidden = false; $("#go").disabled = true;
  window.SSWorld.setAnalysis(true);
  recedeTracks(true);
  clearInterval(S.tick);
  S.tick = setInterval(() => {
    if (S.live && S.live.recorded) return;
    const s = ((performance.now() - S.t0) / 1000).toFixed(0);
    $("#workText").textContent = `Working, turn ${Math.max(1, S.ep.turn)} of ${MAX_TURNS}, ${s} s`;
  }, 250);
}

function gateBlocks() {
  if (!S.gateOpen) return false;
  banner("Decision pending.", "Hold A to acknowledge, R to dispute or Esc to defer the open assessment first.");
  $("#gate").scrollIntoView({block: "nearest"});
  return true;
}

function ask(opts = {}) {
  if (gateBlocks()) return;
  const q = (opts.question ?? $("#q").value).trim();
  if (!q) { $("#q").focus(); return; }
  hideDemo();
  newEpisode(q);
  S.live = {who: S.meta.model, recorded: false, taskId: S.taskId};
  const url = `/api/ask?mode=model&q=${encodeURIComponent(q)}`;
  const es = S.es = new EventSource(url);
  es.onmessage = (m) => { let ev; try { ev = JSON.parse(m.data); } catch (_) { return; } onEvent(ev); };
  es.onerror = () => {
    es.close(); S.es = null;
    if (!S.finished) failure("Connection lost before an answer.", "The gateway stream closed. Check the model server and ask again.");
  };
}

async function playRecording(taskId, who = "sft") {
  if (gateBlocks()) return;
  hideDemo();
  let rec;
  try { rec = await getJSON(`/api/replay?task_id=${encodeURIComponent(taskId)}&who=${who}`); }
  catch (e) { banner("No recording.", `${e.message}. Run scripts/record_demo.py.`); return; }
  const q = rec.events.find((e) => e.type === "start")?.question || "";
  $("#q").value = q;
  newEpisode(q);
  S.live = {who, recorded: true, taskId, summary: rec.summary};
  $("#qMeta").textContent = `Recorded run by ${modelName(who)}, ${rec.summary.recorded_at.replace("T", " ")}, replayed at about 2x speed`;
  clearInterval(S.tick);
  $("#workText").textContent = "Replaying a recording";
  const token = S.epId;
  for (const ev of rec.events) {
    if (S.epId !== token) return;                     // a new question or Esc ended this replay
    onEvent(ev);
    await wait(ev.type === "tool_result" ? 420 : ev.type === "assistant" ? 260 : 60);
  }
}

function addStep(html, cls = "") {
  const li = document.createElement("li");
  li.className = cls; li.innerHTML = html;
  $("#steps").appendChild(li);
  li.scrollIntoView({block: "nearest"});
  return li;
}

function onEvent(ev) {
  switch (ev.type) {
    case "start":
      S.episode = ev.episode_id;
      break;
    case "assistant":
      S.ep.turn = ev.turn;
      break;
    case "tool_call": {
      S.step += 1;
      const n = S.step;
      const li = addStep(`<span class="n">${n}</span><span class="nm" title="${esc(ev.name)}">${esc(TOOL_LABEL[ev.name] || ev.name)}</span><span class="ms"></span>`, "run");
      li.dataset.step = String(n);
      li.onmouseenter = () => { if (S.stepEnts.has(n)) { li.classList.add("iso"); isolateStep(n); } };
      li.onmouseleave = () => { li.classList.remove("iso"); isolateStep(null); };
      S.pending.push({name: ev.name, li, n, args: ev.arguments});
      break;
    }
    case "tool_result": onToolResult(ev); break;
    case "answer": renderAnswer(ev); break;
    case "error": failure("The agent stopped with an error.", ev.message); break;
    case "done":
      S.finished = true; $("#go").disabled = false; $("#working").hidden = true; clearInterval(S.tick);
      if (S.es) { S.es.close(); S.es = null; }
      break;
    default: break;
  }
}

function onToolResult(ev) {
  const res = ev.result || {};
  const k = S.pending.findIndex((p) => p.name === ev.name);
  const p = k >= 0 ? S.pending.splice(k, 1)[0] : null;
  const li = p ? p.li : addStep(`<span class="n">${++S.step}</span><span class="nm">${esc(TOOL_LABEL[ev.name] || ev.name)}</span><span class="ms"></span>`);
  const n = p ? p.n : S.step;
  li.classList.remove("run");
  li.querySelector(".ms").textContent = res.cost_ms != null ? `${res.cost_ms} ms` : "";
  if (res.ok) {
    li.insertAdjacentHTML("beforeend", `<span class="sum">${summarize(ev.name, res, p && p.args)}</span>`);
    drawToolOverlay({...ev, args: p && p.args}, n);
  } else {
    const e = res.error || {};
    li.insertAdjacentHTML("beforeend", `<span class="err"><b>${esc(e.code || "Error")}</b> ${esc(e.message || "")}${e.hint ? `. ${esc(e.hint)}` : ""}</span>`);
  }
  li.querySelectorAll("[data-ref]").forEach((b) => (b.onclick = () => handleRef(b.dataset.ref)));
}

function summarize(name, res, args) {
  const d = res.data || {};
  const refs = (res.refs || []).length;
  const tail = refs ? `, ${refs} refs` : "";
  switch (name) {
    case "df_hypotheses": {
      const nb = (d.sensors || []).reduce((a, s) => a + (s.hyps || []).length, 0);
      return `${esc(d.track || "")} at t ${d.t_s} s: <b>${nb} bearing${nb === 1 ? "" : "s"}</b> from ${(d.sensors || []).length} sensor${(d.sensors || []).length === 1 ? "" : "s"}`;
    }
    case "fit_candidates": {
      const c = (d.candidates || [])[0];
      return `${esc(d.method || "fit")}, ${d.n_tested ?? "?"} positions tested: <b>${(d.candidates || []).length} candidate${(d.candidates || []).length === 1 ? "" : "s"}</b>` +
             (c ? `, best ${c.residual_deg_rms != null ? `${c.residual_deg_rms}° rms` : `${c.residual_m_rms} m rms`}` : "");
    }
    case "effect_footprint": return `${esc(band(d.band))} at ${d.eirp_dbm} dBm EIRP from ${esc(args?.effector || "")}, threshold ${args?.threshold_dbm ?? d.threshold_dbm} dBm`;
    case "collateral_check": return `<b>${(d.violations || []).length} affected</b> of ${d.n_checked ?? "?"} checked; operator confirmation required`;
    case "suggest_node": return `<b>${(d.candidates || []).length} candidate site${(d.candidates || []).length === 1 ? "" : "s"}</b>${tail}`;
    case "weak_zones": return `<b>${(d.zones || []).length} weak zone${(d.zones || []).length === 1 ? "" : "s"}</b>${tail}`;
    case "move_drone": return `flew to ${esc(d.wp)}, uncertainty now <b>±${Math.round(d.estimate?.uncertainty?.a_m ?? 0)} m</b>`;
    case "drone_state": return `start uncertainty <b>±${Math.round(d.estimate?.uncertainty?.a_m ?? 0)} m</b>, ${d.moves_left ?? "?"} moves`;
    case "plan_waypoints": return `${(d.waypoints || []).length} waypoints ranked by expected uncertainty`;
    case "track_search": return `<b>${(d.items || []).length} track${(d.items || []).length === 1 ? "" : "s"}</b>${tail}`;
    case "track_detail": return `${esc(d.ref || d.track || "")}: ${esc(d.klass || "")}${d.confidence != null ? `, confidence ${d.confidence}` : ""}`;
    case "event_search": return `<b>${(d.items || []).length} event${(d.items || []).length === 1 ? "" : "s"}</b>`;
    case "site_resolve": return `${(d.matches || d.candidates || []).length || refs} match${((d.matches || d.candidates || []).length || refs) === 1 ? "" : "es"}`;
    case "declare_intent": return `plan: ${esc(d.intent || args?.intent || "")}`;
    case "submit": return "answer passed the citation check";
    default: {
      const bits = [];
      for (const [k2, v] of Object.entries(d)) {
        if (Array.isArray(v)) bits.push(`${k2.replace(/_/g, " ")} ${v.length}`);
        else if (typeof v === "number" || typeof v === "boolean") bits.push(`${k2.replace(/_/g, " ")} ${v}`);
        if (bits.length >= 3) break;
      }
      return esc(bits.join(", ")) + tail;
    }
  }
}

/* ------------------------------------------------------------------ the answer */
function swatch(kind) {
  const s = {
    chosen: `<line x1="0" y1="5" x2="30" y2="5" stroke="#000" stroke-width="5"/><line x1="0" y1="5" x2="30" y2="5" stroke="${T.sodium}" stroke-width="3"/>`,
    rejected: `<line x1="0" y1="5" x2="30" y2="5" stroke="${T.ash}" stroke-width="2" stroke-dasharray="6 4"/>`,
    reflect: `<line x1="0" y1="5" x2="30" y2="5" stroke="${T.ash}" stroke-width="2" stroke-dasharray="2 3"/>`,
    fix: `<ellipse cx="15" cy="5" rx="12" ry="4" fill="none" stroke="${T.sodium}" stroke-width="2"/><circle cx="15" cy="5" r="2.5" fill="${T.sodium}"/>`,
    track: `<circle cx="15" cy="5" r="4" fill="none" stroke="${T.sodium}" stroke-width="2"/>`,
    victim: `<path d="M15 0 L20 5 L15 10 L10 5 Z" fill="${T.signal}"/>`,
    area: `<rect x="1" y="1" width="28" height="8" fill="rgba(242,163,58,.25)" stroke="${T.sodium}" stroke-width="1.5"/>`,
    node: `<line x1="15" y1="10" x2="15" y2="1" stroke="${T.sodium}" stroke-width="3"/>`,
    own: `<rect x="10" y="1" width="10" height="8" fill="none" stroke="${T.friendly}" stroke-width="2"/>`,
    event: `<line x1="15" y1="0" x2="15" y2="10" stroke="${T.sodium}" stroke-width="3"/>`,
    none: "",
  }[kind] || "";
  return `<span class="sw"><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true">${s}</svg></span>`;
}
function refBtn(ref, cls = "") { return `<button type="button" class="ref ${cls}" data-ref="${esc(ref)}">${esc(ref)}</button>`; }
function evRow(kind, ref, role, cls = "") { return `<li>${swatch(kind)}${refBtn(ref, cls)}<span class="role">${esc(role)}</span></li>`; }

function verdictFor(a, ev) {
  const E = S.ep;
  const ctxIntent = {operator_query: "Operator query", df_ambiguity: "DF ambiguity", drone_search: "DF drone search", siting: "Sensor siting",
                     collateral: "Collateral check", replay: "Replay", abstain: "No answer possible"}[a.intent] || a.intent;
  let head = "", why = "", rows = [];
  if (a.intent === "abstain") {
    head = "Cannot determine";
    why = ABSTAIN[a.reason] || a.reason;
  } else if (a.intent === "operator_query") {
    const trs = a.track_refs || [];
    head = a.boolean === true ? "Yes" : a.boolean === false ? "No" : `${trs.length} track${trs.length === 1 ? "" : "s"}`;
    const desc = trs.slice(0, 4).map((r) => {
      const t = S.recs.get(baseRef(r))?.data;
      return t ? `${esc(baseRef(r))} is classed ${esc(t.klass || "unknown")}${t.confidence != null ? ` (track confidence ${t.confidence})` : ""}` : esc(r);
    });
    why = desc.length ? `${desc.join("; ")}.` : a.boolean === false ? "No track matches." : "";
    rows = trs.map((r) => evRow("track", r, "answer", "ok"));
    if (a.region?.type === "ref") rows.push(evRow("area", a.region.ref, "area", "ok"));
  } else if (a.intent === "df_ambiguity") {
    const ch = a.chosen_hyp_refs.map((r) => E.bearings.get(baseRef(r))).filter(Boolean);
    const rj = (a.rejected_hyp_refs || []).map((r) => E.bearings.get(baseRef(r))).filter(Boolean);
    head = ch.length ? `Bearing ${fmtDeg(ch[0].deg)} is the direct path` : `${a.chosen_hyp_refs.length} bearing chosen as direct`;
    why = rj.length ? `${rj.map((b) => `${fmtDeg(b.deg)} from ${b.sensor}`).join(", ")} ${rj.length === 1 ? "is" : "are"} rejected as ${rj.length === 1 ? "a reflection" : "reflections"}.` : "";
    if (a.position) why += ` The fix is ${nearestPlace(a.position.x_m, a.position.y_m)}${a.position.z_m ? `, ${Math.round(a.position.z_m)} m up` : ""}.`;
    rows = a.chosen_hyp_refs.map((r) => evRow("chosen", r, "direct path", "ok"))
      .concat((a.rejected_hyp_refs || []).map((r) => evRow("rejected", r, "rejected", "rej")));
    if (E.reflectors.length) rows.push(evRow("reflect", E.reflectors[0].ref, "predicted reflector"));
    if (a.position) rows.push(evRow("fix", (E.fits[0] && E.fits[0].ref) || "fix", "fix and 1σ ellipse", "ok"));
  } else if (a.intent === "drone_search") {
    const p = a.final_position;
    head = `Emitter fixed to ±${Math.round(a.uncertainty.a_m)} m`;
    why = `${nearestPlace(p.x_m, p.y_m)}${p.z_m ? `, ${Math.round(p.z_m)} m up` : ""}, after ${E.moves || (a.waypoints_used || []).length} drone move${(E.moves || 0) === 1 ? "" : "s"}.`;
    rows = [evRow("fix", "final fix", "1σ ellipse", "ok")].concat((a.waypoints_used || []).map((w) => evRow("own", w, "waypoint flown")));
  } else if (a.intent === "siting") {
    const nodes = a.node_candidates || [];
    const c = E.nodes && E.nodes.find((n) => nodes.includes(n.ref));
    head = nodes.length ? "Add one DF node here" : `${(a.zones || []).length} weak zone${(a.zones || []).length === 1 ? "" : "s"}`;
    why = c ? `${nearestPlace(c.position.x_m, c.position.y_m)}; predicted coverage gain ${c.predicted_gain}.` : "";
    rows = nodes.map((r) => evRow("node", r, "recommended site", "ok"));
  } else if (a.intent === "collateral") {
    const nv = (a.violations || []).length;
    const fp = E.fp || {};
    head = nv ? `Jamming would hit ${nv} of our own link${nv === 1 ? "" : "s"}` : "No own links or civil comms would be hit";
    const lv = (E.victims || []).filter((v) => (a.violations || []).includes(v.ref));
    why = nv ? `${lv.map((v) => `${v.ref.split(":")[1]} at ${v.level_dbm} dBm`).join(", ")} ${lv.length === 1 ? "is" : "are"} above the ${fp.threshold ?? "-90"} dBm threshold in the predicted ${band(fp.band)} footprint from ${(fp.effector || "").replace("sensor:", "")}.`
             : `The predicted ${band(fp.band)} footprint stays below ${fp.threshold ?? "-90"} dBm at every friendly link and civil receiver checked.`;
    rows = (a.violations || []).map((r) => evRow("victim", r, "above threshold", "act"));
    if (a.fp_ref) rows.push(evRow("area", a.fp_ref, "predicted footprint", "ok"));
  } else if (a.intent === "replay") {
    head = a.time_s != null ? `At t ${a.time_s} s` : `${(a.evt_refs || []).length} event${(a.evt_refs || []).length === 1 ? "" : "s"}`;
    why = (a.evt_refs || []).length ? `From the event log: ${a.evt_refs.slice(0, 3).map(esc).join(", ")}.` : "";
    rows = (a.evt_refs || []).map((r) => evRow("event", r, "event", "ok"));
  }
  for (const r of a.evidence_refs || []) if (!rows.some((x) => x.includes(`data-ref="${esc(r)}"`))) rows.push(evRow("none", r, "cited"));
  return {ctx: ctxIntent, head, why, rows};
}

function renderAnswer(ev) {
  const a = ev.answer, v = ev.verifier;
  clearInterval(S.tick);
  $("#working").hidden = true;
  const wall = S.live && S.live.recorded ? S.live.summary.wall_s : ((performance.now() - S.t0) / 1000).toFixed(1);
  const when = new Date().toLocaleTimeString("en-GB", {hour: "2-digit", minute: "2-digit"});
  $("#qMeta").textContent = S.live && S.live.recorded
    ? `Recorded run by ${modelName(S.live.who)}, ${S.live.summary.recorded_at.replace("T", " ")}, answered in ${wall} s`
    : `Asked ${when}, answered in ${wall} s by ${modelName(S.meta.model)}`;
  const box = $("#verdict");
  if (!a) {
    failure(END_REASON[ev.end_reason] || "No answer", ev.end_reason === "max_turns"
      ? "The agent used all 8 turns without submitting. Try a more specific question."
      : "The model's final answer could not be parsed or checked.");
    return;
  }
  const V = verdictFor(a, ev);
  const pct = Math.round((a.confidence || 0) * 100);
  const low = (a.confidence || 0) < REVIEW_MARK;
  const vok = v && v.ok;
  box.className = `verdict ${a.intent === "abstain" ? "abstain" : ""}`;
  box.innerHTML = `<p class="ctx">${esc(V.ctx)}</p><p class="head">${esc(V.head)}</p>${V.why ? `<p class="why">${V.why}</p>` : ""}` +
    (a.intent === "abstain" ? "" :
    `<div class="conf"><div class="row"><span>Confidence, model-stated, not calibrated</span><span class="val">${pct}%</span></div>
       <div class="bar" aria-hidden="true"><i class="base"></i><i class="fill ${low ? "low" : ""}" style="width:${pct}%"></i>
         <i class="mk" style="left:${REVIEW_MARK * 100}%"><span>review mark</span></i></div>
       <div class="note ${low ? "review" : ""}">${low ? "Below the review mark: check the evidence on the map before acting." : "Above the review mark."}</div></div>`) +
    (V.rows.length ? `<ul class="evidence">${V.rows.join("")}</ul>` : "") +
    (a.assumptions?.length ? `<p class="why" style="margin-top:12px;color:var(--ash)">Assumed: ${a.assumptions.map(esc).join("; ")}.</p>` : "") +
    `<div class="vline"><span>${vok ? "<b>Citations checked</b>" : `<span class="bad">Citation check failed: ${esc(v?.code || "not run")}</span>`}</span>
       <span>${ev.turns} turn${ev.turns === 1 ? "" : "s"}, ${S.step} tool call${S.step === 1 ? "" : "s"}</span>
       <span>${esc(END_REASON[ev.end_reason] || ev.end_reason || "")}</span></div>`;
  box.hidden = false;
  box.querySelectorAll("[data-ref]").forEach((b) => (b.onclick = () => handleRef(b.dataset.ref)));
  if (a.intent === "abstain") window.SSWorld.setAnalysis(false);
  if (vok) paintAnswer(a);
  else box.querySelectorAll(".ref.ok, .ref.act").forEach((b) => b.classList.remove("ok", "act"));
  if (a.requires_human_confirm && vok) openGate(a, V);
  else if (a.requires_human_confirm) box.insertAdjacentHTML("beforeend", `<p class="why" style="margin-top:12px">The citation check failed, so no operator decision is requested. Ask again.</p>`);
  compareBlock();
}

function paintAnswer(a) {
  const E = S.ep;
  S.camMoves = Math.min(S.camMoves, 1);                // the settle move is always allowed after establishing
  if (a.intent === "df_ambiguity") {
    const chosen = new Set(a.chosen_hyp_refs.map(baseRef)), rejected = new Set((a.rejected_hyp_refs || []).map(baseRef));
    const seen = new Set();
    for (const rec of S.rays.values()) {
      if (seen.has(rec)) continue; seen.add(rec);
      const r = baseRef(rec.data.ref);
      styleRay(rec, chosen.has(r) ? "chosen" : rejected.has(r) ? "rejected" : "other");
    }
    if (E.fitEnts) recolor(E.fitEnts, T.sodium);
    if (a.position && !E.fitEnts) ovFix(99, a.position, a.uncertainty, T.sodium, "fix");
    // frame the sensor and the fix (plus the first 800 m of each decided ray), not the full 3 km rays
    const pts = [];
    for (const r of seen) {
      if (!(chosen.has(baseRef(r.data.ref)) || rejected.has(baseRef(r.data.ref)))) continue;
      const [o, end] = r.positions;
      pts.push(o, C.Cartesian3.lerp(o, end, 800 / RAY_LEN_M, new C.Cartesian3()));
    }
    if (a.position) pts.push(posENU(a.position.x_m, a.position.y_m, a.position.z_m || 0));
    else if (E.fits[0]) { const f = E.fits[0].position; pts.push(posENU(f.x_m, f.y_m, f.z_m || 0)); }
    const tr = E.track && S.recs.get(baseRef(E.track));
    if (tr) { const p = tr.ent.position.getValue(S.v.clock.currentTime); if (p) pts.push(p); }
    if (E.track) setFocus(E.track, true, T.sodium), S.highlighted.push(E.track);
    flyToPoints(pts, {minRange: 1400, pitch: -0.5, caption: "Chosen and rejected bearings, with the fix"});
  } else if (a.intent === "drone_search") {
    if (E.estEnts) recolor(E.estEnts, T.sodium); else ovFix(99, a.final_position, a.uncertainty, T.sodium, "final fix");
    flyToPoints([posENU(a.final_position.x_m, a.final_position.y_m, a.final_position.z_m || 0)], {minRange: 900, caption: "Located emitter"});
  } else if (a.intent === "siting") {
    if (E.nodeEnts) recolor(E.nodeEnts, T.sodium);
  } else if (a.intent === "collateral") {
    for (const e of E.fpEnts || []) { if (e.polygon) e.polygon.material = col(T.sodium, 0.07); if (e.polyline) e.polyline.material = col(T.sodium, 0.75); }
    const vs = new Set(a.violations || []);
    (E.victimEnts || []).forEach((e, i) => { if (vs.has(E.victims[i]?.ref)) { e.billboard.image = sym("victim", T.signal, 22); e.label.fillColor = col(T.signal); } });
    if (vs.size) focusRefs([...vs], {fly: false, color: T.signal});
  } else if (a.intent === "operator_query") {
    const refs = [...(a.track_refs || [])];
    if (a.region?.type === "ref") refs.push(a.region.ref);
    if (refs.length) focusRefs(refs, {color: T.sodium, auto: true, caption: "Answered tracks"});
  } else if (a.intent === "replay") {
    if (a.time_s != null) setSimTime(a.time_s);
  }
}

function failure(head, why) {
  clearInterval(S.tick);
  S.finished = true; $("#go").disabled = false; $("#working").hidden = true;
  const box = $("#verdict");
  box.className = "verdict fail";
  box.innerHTML = `<p class="ctx">No answer</p><p class="head">${esc(head)}</p><p class="why">${esc(why)}</p>`;
  box.hidden = false;
  compareBlock();
}

/* ------------------------------------------------------------------ operator gate */
function openGate(a, V) {
  const g = $("#gate");
  const fp = S.ep.fp || {};
  S.gateOpen = true;
  g.innerHTML = `<p class="who" id="gateTitle">Agent recommends, operator decides</p>
    <p class="rec">${esc(V.head)}.</p>
    <p class="fine">This is a predicted assessment of a ${esc(band(fp.band))} jammer at ${esc(String(fp.eirp ?? ""))} dBm from ${esc((fp.effector || "").replace("sensor:", ""))}.
      No tool in SceneScout can switch on an effector; your decision is recorded against episode ${esc(S.episode || "")}.</p>
    <div class="btns">
      <button type="button" class="hold" id="gAck"><i></i><span>Hold <kbd>A</kbd> to acknowledge the assessment</span></button>
      <button type="button" id="gRej"><span><kbd>R</kbd> Dispute</span></button>
      <button type="button" id="gDef"><span><kbd>Esc</kbd> Defer</span></button>
    </div><div class="res" id="gRes" hidden></div>`;
  g.hidden = false;
  const ack = $("#gAck");
  ack.onpointerdown = () => holdStart();
  ack.onpointerup = ack.onpointerleave = () => holdCancel();
  $("#gRej").onclick = () => gateDecide("Disputed");
  $("#gDef").onclick = () => gateDecide("Deferred");
}
let holdRaf = 0, holdT0 = 0;
function holdStart() {
  if (!S.gateOpen) return;
  holdT0 = performance.now();
  const bar = $("#gAck i");
  const step = (t) => {
    const k = Math.min(1, (t - holdT0) / 1200);
    bar.style.width = `${k * 100}%`;
    if (k >= 1) { gateDecide("Acknowledged"); return; }
    holdRaf = requestAnimationFrame(step);
  };
  holdRaf = requestAnimationFrame(step);
}
function holdCancel() {
  cancelAnimationFrame(holdRaf);
  const bar = $("#gAck i");
  if (bar && S.gateOpen) bar.style.width = "0";
}
function gateDecide(what) {
  if (!S.gateOpen) return;
  S.gateOpen = false;
  cancelAnimationFrame(holdRaf);
  const r = $("#gRes");
  r.hidden = false;
  r.innerHTML = `<b>${esc(what)}</b> by the operator at ${new Date().toLocaleTimeString("en-GB")}. Assessment only: nothing was switched on.`;
  $$("#gate button").forEach((b) => (b.disabled = true));
}
function closeGate() { S.gateOpen = false; $("#gate").hidden = true; $("#gate").innerHTML = ""; }

/* ------------------------------------------------------------------ compare with the untrained model */
function compareBlock() {
  const box = $("#compare");
  const tid = S.live && S.live.taskId;
  const r = tid && S.replays[tid];
  if (!r || !r.scenescout) { box.hidden = true; return; }
  const base = r.scenescout, ours = S.live.recorded ? S.live.summary : r.sft;
  const out = (s) => (s.end_reason === "submitted" ? (s.score >= 0.95 ? "Correct answer" : s.score > 0 ? "Partly correct" : "Wrong answer")
                     : END_REASON[s.end_reason] || s.end_reason);
  box.innerHTML = `<h2>Same question, untrained model</h2>
    <div class="cols">
      <div class="col"><h3>4B base, untrained</h3><p class="out">${esc(out(base))}</p>
        <dl><dt>Turns</dt><dd>${base.turns} of 8</dd><dt>Time</dt><dd>${base.wall_s} s</dd><dt>Score</dt><dd>${base.score.toFixed(2)}</dd></dl></div>
      <div class="col"><h3>SceneScout SFT v1</h3><p class="out">${esc(out(ours || {end_reason: "?"}))}</p>
        <dl><dt>Turns</dt><dd>${ours ? ours.turns : "?"} of 8</dd><dt>Time</dt><dd>${ours ? ours.wall_s : "?"} s</dd><dt>Score</dt><dd>${ours ? ours.score.toFixed(2) : "?"}</dd></dl></div>
    </div>
    <p class="src">Recorded on this machine with scripts/record_demo.py, scored offline against the scenario truth. On the 603-question
      held-out test set: untrained 0.29, SceneScout 0.93.</p>
    <button type="button" class="btn toggle" id="playBase">Play the untrained run</button>`;
  box.hidden = false;
  $("#playBase").onclick = () => playRecording(tid, "scenescout");
}

/* ------------------------------------------------------------------ refs, info, picking */
function handleRef(ref) {
  const b = baseRef(ref);
  if (b.startsWith("hyp:")) {
    const rec = S.rays.get(ref) || S.rays.get(b);
    if (rec) { flyToPoints(rec.positions, {minRange: 1500, auto: false, caption: `Bearing ${b}`}); showInfo(rec); }
    return;
  }
  if (S.recs.has(b)) { focusRefs([ref], {color: T.sodium}); showInfo(S.recs.get(b)); }
}
function dl(rows) { return `<dl>${rows.filter(([, v]) => v != null && v !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`; }
function showInfo(rec) {
  const box = $("#info");
  if (!rec) { box.hidden = true; return; }
  const d = rec.data;
  let html = "";
  if (rec.kind === "obj") {
    const p = d.provenance || {};
    html = `<h3>${esc(d.name)}</h3><div class="sub">${esc(d.kind.replace("_", " "))}, ${esc(d.ref)}${d.protected ? ", protected" : ""}</div>` +
      dl([["Height", d.height_m != null ? `${d.height_m.toFixed(1)} m (${esc(p.height_source || "estimated")})` : "unknown"],
          ["Source", esc(p.source)], ["Name", esc(p.name_source)], ["OSM id", esc(p.osm_id)]]);
  } else if (rec.kind === "zone") {
    html = `<h3>${esc(d.name)}</h3><div class="sub">${esc(d.kind)} zone, ${esc(d.ref)}</div>` + dl([["Source", esc(d.provenance?.source)], ["Status", esc(d.provenance?.status)]]);
  } else if (rec.kind === "sensor") {
    html = `<h3>Sensor ${esc(d.id)}</h3><div class="sub">${esc(d.type)}, ${esc(d.placement || "synthetic")} site</div>` +
      dl([["Mast", `${d.z_m} m`], ["Bands", esc((d.bands || []).join(", "))], ["Position", `${d.lat.toFixed(5)}, ${d.lon.toFixed(5)}`]]);
  } else if (rec.kind === "trk") {
    const p = d.pts[idxAt(d.ts, now())];
    html = `<h3>${esc(d.ref)}</h3><div class="sub">${esc(d.klass)}, ${esc(d.band)}${d.group ? `, swarm ${esc(d.group)}` : ""}</div>` +
      dl([["Track confidence", d.confidence != null ? d.confidence.toFixed(2) : "?"], ["Seen", `t ${d.avail[0]} to ${d.avail[1]} s`],
          ["Now", `${p[3]} m up, ellipse ${p[4]} × ${p[5]} m`]]) +
      `<div class="acts"><button class="btn" id="infoFollow">Follow</button></div>`;
  } else if (rec.kind === "ray") {
    html = `<h3>DF bearing ${fmtDeg(d.bearing_deg)}</h3><div class="sub">${esc(d.ref)}</div>` +
      dl([["Sensor", esc(d.sensor)], ["SNR", d.snr_db != null ? `${d.snr_db} dB` : null], ["Track", esc(d.track)], ["t", d.t != null ? `${d.t} s` : null]]);
  } else if (rec.kind === "grp") {
    html = `<h3>Swarm ${esc(d.group)}</h3><div class="sub">${d.members.length} tracks grouped by the tracker</div>`;
  }
  box.innerHTML = `<button class="x" aria-label="Close">×</button>${html}`;
  box.hidden = false;
  box.querySelector(".x").onclick = () => (box.hidden = true);
  const f = $("#infoFollow");
  if (f) f.onclick = () => { S.v.trackedEntity = rec.ent; S.v.clock.shouldAnimate = true; setPlay(true); };
}
const PICK_ORDER = {trk: 0, grp: 1, sensor: 2, ray: 3, obj: 4, zone: 5};
function recForEntity(ent) {
  if (!ent) return null;
  const id = String(ent.id);
  if (id.startsWith("ray:")) return S.rays.get(id.slice(4));
  return S.recs.get(id.split("#")[0]) || null;
}
function onClick(ev) {
  const picks = S.v.scene.drillPick(ev.position, 8) || [];
  let best = null, rank = 99;
  for (const p of picks) {
    const ent = p && p.id;
    if (!(ent instanceof C.Entity)) continue;
    const rec = recForEntity(ent);
    if (rec && (PICK_ORDER[rec.kind] ?? 9) < rank) { best = rec; rank = PICK_ORDER[rec.kind] ?? 9; }
  }
  showInfo(best);
}

/* ------------------------------------------------------------------ time, clock, compass */
function timelineTicks() {
  const el = $("#tlTicks");
  let html = "";
  for (let t = 0; t <= S.dur; t += 30) {
    const maj = t % 60 === 0;
    html += `<i class="${maj ? "maj" : ""}" style="left:${(t / S.dur) * 100}%"></i>`;
    if (maj && t % 120 === 0) html += `<span style="left:${(t / S.dur) * 100}%">${t}</span>`;
  }
  el.innerHTML = html;
  $("#tl").setAttribute("aria-valuemax", String(S.dur));
}
function eventTicks(items) {
  $("#tlEvents").innerHTML = items.map((it) => `<b style="left:${(it.t_s / S.dur) * 100}%" title="${esc(`${it.type.replace("_", " ")} ${it.track_ref} ${(it.zone_ref || "").replace("zone:", "")} t ${it.t_s} s`)}" data-t="${it.t_s}"></b>`).join("");
  $$("#tlEvents b").forEach((b) => (b.onclick = (e) => { e.stopPropagation(); setSimTime(Number(b.dataset.t)); }));
}
let lastSec = -1;
function onTick(clock) {
  const t = secs(clock.currentTime);
  const s = Math.round(t);
  $("#tlHead").style.left = `${(t / S.dur) * 100}%`;
  if (s !== lastSec) {
    lastSec = s;
    $("#tlHeadText").textContent = `t ${s} s`;
    $("#sysSim .v").textContent = `t ${s} / ${S.dur} s${clock.shouldAnimate ? `, ${clock.multiplier}x` : ""}`;
    $("#tl").setAttribute("aria-valuenow", String(s));
  }
  if (clock.shouldAnimate && t >= S.dur - 0.01) setPlay(false);
  compass();
}
function compass() {
  const cam = S.v.camera;
  $("#northNeedle").setAttribute("transform", `rotate(${-C.Math.toDegrees(cam.heading)})`);
  const sc = S.v.scene, w = sc.canvas.clientWidth, h = sc.canvas.clientHeight;
  const a = cam.pickEllipsoid(new C.Cartesian2(w / 2, h - 60)), b = cam.pickEllipsoid(new C.Cartesian2(w / 2 + 100, h - 60));
  if (!a || !b) return;
  const mpp = C.Cartesian3.distance(a, b) / 100;
  const nice = [10, 20, 50, 100, 200, 250, 500, 1000, 2000, 5000].find((m) => m / mpp >= 60) || 5000;
  $("#scaleBar").style.width = `${Math.round(nice / mpp)}px`;
  $("#scaleText").textContent = fmtM(nice);
}
function clockTick() {
  $("#sysClock .v").textContent = new Date().toLocaleTimeString("en-GB", {timeZone: "Asia/Kolkata"});
}
function setPlay(on) {
  S.v.clock.shouldAnimate = on;
  $("#play").textContent = on ? "Pause" : "Play";
  $("#play").setAttribute("aria-label", on ? "Pause" : "Play");
}
function scrub(ev) {
  const r = $("#tl").getBoundingClientRect();
  setSimTime(((ev.clientX - r.left) / r.width) * S.dur);
}

/* ------------------------------------------------------------------ demo menu, banner, keys */
function demoMenu() {
  const list = $("#demoList");
  const BEAT = ["genuine_drone", "direct_path", "ds_hover", "suggest_one", "jam_links", "actuation"];
  list.innerHTML = S.demo.map((d, i) => {
    const rec = S.replays[d.task_id]?.sft;
    return `<li><button type="button" data-i="${i}"><span class="k">${i < 9 ? i + 1 : ""}</span>
      <span>${esc(d.question)}<span class="t">${BEAT.includes(d.template) ? "Demo script" : "Extra"}${rec ? `, recording ${rec.wall_s} s` : ""}</span></span></button></li>`;
  }).join("");
  $$("#demoList button").forEach((b) => (b.onclick = (e) => pickDemo(Number(b.dataset.i), e.shiftKey)));
}
function pickDemo(i, recorded = false) {
  const d = S.demo[i];
  if (!d) return;
  $("#q").value = d.question;
  S.taskId = d.task_id;
  if (recorded || !S.meta.llm.ok) playRecording(d.task_id, "sft");
  else ask({question: d.question});
}
function toggleDemo() {
  const m = $("#demo");
  m.hidden = !m.hidden;
  $("#demoBtn").setAttribute("aria-expanded", String(!m.hidden));
  if (!m.hidden) $("#demoList button")?.focus();
}
function hideDemo() { $("#demo").hidden = true; $("#demoBtn").setAttribute("aria-expanded", "false"); }
function banner(head, msg) {
  const b = $("#banner");
  b.innerHTML = `<b>${esc(head)}</b>${esc(msg)}`;
  b.hidden = false;
  setTimeout(() => (b.hidden = true), 9000);
}
function resetAnalysis() {
  S.epId = (S.epId || 0) + 1;
  if (S.es) { S.es.close(); S.es = null; }
  S.ds.overlay.entities.removeAll(); S.ds.rays.entities.removeAll(); S.ds.answer.entities.removeAll();
  S.rays.clear(); S.stepEnts.clear(); clearHighlight(); closeGate();
  window.SSWorld.setAnalysis(false);
  recedeTracks(false);
  $("#verdict").hidden = true; $("#question").hidden = true; $("#stepsBox").hidden = true; $("#compare").hidden = true;
  $("#working").hidden = true; $("#idle").hidden = false; clearInterval(S.tick);
  $("#go").disabled = false; S.finished = true;
}

function wireUI() {
  const h = S.v.screenSpaceEventHandler;
  h.setInputAction(onClick, C.ScreenSpaceEventType.LEFT_CLICK);
  h.removeInputAction(C.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
  for (const t of [C.ScreenSpaceEventType.LEFT_DOWN, C.ScreenSpaceEventType.RIGHT_DOWN, C.ScreenSpaceEventType.MIDDLE_DOWN, C.ScreenSpaceEventType.WHEEL]) {
    h.setInputAction(() => { S.userCam = true; }, t);
  }
  $$(".views [data-view]").forEach((b) => (b.onclick = () => view(b.dataset.view)));
  $("#layersBtn").onclick = () => { const l = $("#layers"); l.hidden = !l.hidden; $("#layersBtn").setAttribute("aria-expanded", String(!l.hidden)); };
  $$("[data-layer]").forEach((cb) => (cb.onchange = () => {
    const on = cb.checked, n = cb.dataset.layer;
    if (n === "structures") for (const m of Object.values(window.SSWorld.state.models)) m.show = on;
    else if (n === "analysis") { S.ds.overlay.show = on; S.ds.rays.show = on; S.ds.answer.show = on; }
    else if (S.ds[n]) S.ds[n].show = on;
  }));
  $("#camBack").onclick = () => {
    if (!S.prevView) return;
    S.v.camera.flyTo({destination: S.prevView.destination, orientation: S.prevView.orientation, duration: 1.0});
    camCaption(S.prevView.caption); S.prevView = null; $("#camBack").hidden = true;
  };
  const fid = $("#fidelity");
  const tog = () => { const m = $("#fidMore"); m.hidden = !m.hidden; fid.setAttribute("aria-expanded", String(!m.hidden)); };
  fid.onclick = tog; fid.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tog(); } };
  $("#askForm").onsubmit = (e) => { e.preventDefault(); ask(); };
  $("#q").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } };
  $("#q").oninput = () => { const d = S.demo.find((x) => x.question === $("#q").value.trim()); S.taskId = d ? d.task_id : null; };
  $("#demoBtn").onclick = toggleDemo;
  $("#play").onclick = () => setPlay(!S.v.clock.shouldAnimate);
  const tl = $("#tl");
  let drag = false;
  tl.onpointerdown = (e) => { drag = true; tl.setPointerCapture(e.pointerId); scrub(e); };
  tl.onpointermove = (e) => { if (drag) scrub(e); };
  tl.onpointerup = () => { drag = false; };
  tl.onkeydown = (e) => { if (e.key === "ArrowRight") setSimTime(now() + 10); if (e.key === "ArrowLeft") setSimTime(now() - 10); };
  wireSearch();

  document.addEventListener("keydown", (e) => {
    const typing = /^(TEXTAREA|INPUT)$/.test(e.target.tagName);
    const mod = e.ctrlKey || e.metaKey || e.altKey;
    if (S.gateOpen && !typing && !mod) {                // gate keys mean only gate actions while it is open
      if (e.key === "a" || e.key === "A") { if (!e.repeat) { holdStart(); $("#gAck").classList.add("on"); } e.preventDefault(); return; }
      if (e.key === "r" || e.key === "R") { gateDecide("Disputed"); e.preventDefault(); return; }
      if (e.key === "Escape") { gateDecide("Deferred"); e.preventDefault(); return; }
      return;
    }
    if (S.gateOpen && typing && e.key === "Escape") { e.target.blur(); return; }
    if (e.key === "Escape") {
      if (!$("#demo").hidden) { hideDemo(); return; }
      if (!$("#info").hidden) { $("#info").hidden = true; return; }
      if (typing) { e.target.blur(); return; }
      resetAnalysis(); return;
    }
    if (typing) return;
    const dig = /^Digit([1-9])$/.exec(e.code);       // e.code, so Shift+digit works on every keyboard layout
    if (!$("#demo").hidden && dig) { pickDemo(Number(dig[1]) - 1, e.shiftKey); e.preventDefault(); return; }
    if (e.key === "d" || e.key === "D") { toggleDemo(); e.preventDefault(); }
    else if (e.key === "w" || e.key === "W") { document.documentElement.classList.toggle("wall"); }
    else if (e.key === " ") { setPlay(!S.v.clock.shouldAnimate); e.preventDefault(); }
    else if (e.key === "/") { $("#layers").hidden = false; $("#search").focus(); e.preventDefault(); }
    else if (e.key === "c" || e.key === "C") { if (!$("#compare").hidden) $("#playBase")?.click(); }
  });
  document.addEventListener("keyup", (e) => { if ((e.key === "a" || e.key === "A") && S.gateOpen) holdCancel(); });
}

function wireSearch() {
  const inp = $("#search"), list = $("#searchList");
  const index = [];
  for (const [ref, rec] of S.recs) {
    if (!["obj", "zone", "sensor", "trk", "grp"].includes(rec.kind)) continue;
    const d = rec.data, name = d.name || d.ref || ref;
    index.push({ref, name, kind: d.kind || rec.kind, keys: [ref, name, d.id, ...(d.aliases || [])].filter(Boolean).map((s) => String(s).toLowerCase())});
  }
  const render = () => {
    const q = inp.value.trim().toLowerCase();
    if (!q) { list.hidden = true; return; }
    const exact = index.filter((x) => x.keys.includes(q));
    const hits = exact.concat(index.filter((x) => !exact.includes(x) && x.keys.some((k) => k.includes(q)))).slice(0, 8);
    list.innerHTML = hits.length ? hits.map((h, i) => `<li data-ref="${esc(h.ref)}" class="${i === 0 ? "act" : ""}"><span>${esc(h.name)}</span><small>${esc(String(h.kind).replace("_", " "))}</small></li>`).join("")
                                 : `<li><small>No match in the port twin</small></li>`;
    list.hidden = false;
    list.querySelectorAll("li[data-ref]").forEach((li) => (li.onmousedown = () => { handleRef(li.dataset.ref); list.hidden = true; }));
  };
  inp.oninput = render;
  inp.onkeydown = (e) => {
    if (e.key === "Enter") { const f = list.querySelector("li[data-ref]"); if (f) { handleRef(f.dataset.ref); list.hidden = true; } }
  };
  inp.onblur = () => setTimeout(() => (list.hidden = true), 150);
}

boot().catch((e) => {
  console.error(e);
  const li = document.createElement("li");
  li.className = "fault";
  li.innerHTML = `<span>Start-up</span><span>${esc(e.message)}</span>`;
  $("#bootList").appendChild(li);
});
