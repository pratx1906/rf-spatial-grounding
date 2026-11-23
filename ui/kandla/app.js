/* SceneScout operator console: CesiumJS port twin + agent trace (ENV SPEC 13). Belief-side data only. */
"use strict";

const C = window.Cesium;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const css = (hex, a = 1) => C.Color.fromCssColorString(hex).withAlpha(a);
const INF = Number.POSITIVE_INFINITY;
const RAY_LEN_M = 3000;

const COL = {
  paper: "#dfe7e4", dim: "#8297a0", asset: "#e0a43a", restricted: "#d14d8c", track: "#5fd3e8", focus: "#f2d16b",
  lowconf: "#9aa6ab", hero: "#f2d16b", region: "#8fa6ad", perimeter: "#dfe7e4", aviation: "#9fb4ff",
};
const GROUP_COLORS = ["#b48cff", "#7be08a", "#ff8f6b", "#6fb2ff", "#f7a8d8"];
const KIND_COLOR = {
  tank: "#c9d1d3", building: "#8d989c", silo: "#d8d2c4", jetty: "#b7ae9c", substation: "#9a8fc7", ship: "#5f7a8f",
  road: "#59656a", tank_farm: "#c9d1d3", warehouse: "#8d989c", crane: "#c7b26b", tower: "#c9d1d3",
};
const PROV = {
  osm: {color: "#7cc99a", label: "OSM geometry and height"},
  osm_estimated: {color: "#5fa8d3", label: "OSM geometry, estimated height"},
  derived: {color: "#b48cff", label: "Derived (clustered from OSM)"},
  synthetic: {color: "#e8735f", label: "Synthetic"},
  unknown: {color: "#8a969a", label: "Unknown source"},
};
const ABSTAIN = {
  no_sensor_coverage: "No sensor covers that place or time, so the agent cannot say.",
  entity_not_found: "The place or object in the question could not be found in the twin.",
  ambiguous_unresolved: "The question stayed ambiguous after checking; it needs more detail.",
  outside_tool_scope: "This is outside what the agent's tools can answer.",
  insufficient_evidence: "The evidence is not strong enough to give an answer.",
};
const INTENT_LABEL = {
  operator_query: "Operator query", df_ambiguity: "DF ambiguity", drone_search: "Drone search", siting: "Sensor siting",
  collateral: "Collateral check", replay: "Replay", abstain: "Abstained",
};
const FALLBACK_QUESTIONS = [
  "Is any drone inside the jetty restricted zone at t=300 s? List the tracks.",
  "Which tracks belong to the swarm near the tank farms at t=200 s?",
  "At t=190 s, sensor S3 reports 2 bearings for track trk:2146. Which one is the direct path?",
  "Which protected assets are within 300 m of trk:9719 at t=400 s?",
];

const S = {
  v: null, site: null, meta: null, tracks: [], start: null, dur: 600, enuM: null,
  ds: {}, recs: new Map(), rays: new Map(), highlighted: [], prov: false, demo: [], mode: "model", taskId: null,
  es: null, pending: [], grid: null, sensors: new Map(), groups: new Map(), osm: null, osmLoading: false,
  hudSec: -1, answerEnts: [],
};

/* ---------------------------------------------------------------- geometry helpers */
function enuToLonLat(e, n) {
  const p = C.Matrix4.multiplyByPoint(S.enuM, new C.Cartesian3(e, n, 0), new C.Cartesian3());
  const c = C.Cartographic.fromCartesian(p);
  return [C.Math.toDegrees(c.longitude), C.Math.toDegrees(c.latitude)];
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
    try { await C.sampleTerrainMostDetailed(terrain, cartos.concat(sens)); ok = true; } catch (e) { console.warn("terrain sampling failed", e); }
  }
  S.grid = {n, lon0, lat0, dlon, dlat, h: cartos.map((c) => (ok && Number.isFinite(c.height) ? c.height : 0))};
  S.site.sensors.forEach((s, k) => { s.ground = ok && Number.isFinite(sens[k].height) ? sens[k].height : groundAt(s.lon, s.lat); });
  S.groundOk = ok;
}
const secs = (time) => C.JulianDate.secondsDifference(time, S.start);
const jd = (t) => C.JulianDate.addSeconds(S.start, t, new C.JulianDate());
const now = () => secs(S.v.clock.currentTime);
const rr = () => S.v && S.v.scene.requestRender();
function setSimTime(t) {
  S.v.clock.currentTime = jd(Math.max(0, Math.min(S.dur, t)));
  rr();
}
function idxAt(ts, t) {
  let lo = 0, hi = ts.length - 1;
  if (t <= ts[0]) return 0;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ts[m] <= t) lo = m; else hi = m - 1; }
  return lo;
}

/* ---------------------------------------------------------------- boot */
async function getJSON(u) {
  const r = await fetch(u);
  if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
  return r.json();
}

async function boot() {
  const [meta, site, trk] = await Promise.all([getJSON("/api/meta"), getJSON("/api/site"), getJSON("/api/tracks?step_s=2")]);
  S.meta = meta; S.site = site; S.tracks = trk.tracks; S.demo = meta.demo || [];
  S.dur = site.meta.duration_s || meta.duration_s || 600;
  header();

  let token = null;
  try { const r = await fetch("/api/cesium_token"); if (r.ok) token = (await r.json()).token; } catch (_) { /* offline */ }
  let terrain = null;
  const opts = {
    baseLayerPicker: false, geocoder: false, homeButton: false, sceneModePicker: false, navigationHelpButton: false,
    fullscreenButton: false, infoBox: false, selectionIndicator: true, animation: true, timeline: true,
    requestRenderMode: true, shouldAnimate: false,
  };
  if (token) {
    C.Ion.defaultAccessToken = token;
    try { terrain = await C.createWorldTerrainAsync(); } catch (e) { console.warn("world terrain unavailable", e); }
  } else {
    opts.baseLayer = new C.ImageryLayer(new C.OpenStreetMapImageryProvider({url: "https://tile.openstreetmap.org/"}));
  }
  opts.terrainProvider = terrain || new C.EllipsoidTerrainProvider();
  S.token = !!token;
  const v = S.v = new C.Viewer("cesium", opts);
  v.scene.globe.enableLighting = false;
  v.scene.globe.depthTestAgainstTerrain = false;
  v.scene.highDynamicRange = false;
  const base = v.imageryLayers.get(0);
  if (base) { base.brightness = 0.78; base.saturation = 0.7; base.contrast = 1.05; }

  const o = site.meta.origin;
  S.enuM = C.Transforms.eastNorthUpToFixedFrame(C.Cartesian3.fromDegrees(o.lon, o.lat, 0));
  S.start = C.JulianDate.fromIso8601(site.meta.t0_iso || meta.t0_iso || "2026-01-01T00:00:00Z");
  const clk = v.clock;
  clk.startTime = S.start.clone(); clk.stopTime = jd(S.dur); clk.currentTime = S.start.clone();
  clk.clockRange = C.ClockRange.CLAMPED; clk.multiplier = 10; clk.shouldAnimate = false;
  v.timeline.makeLabel = (time) => `t+${Math.round(secs(time))} s`;
  v.timeline.zoomTo(clk.startTime, clk.stopTime);
  v.animation.viewModel.dateFormatter = () => "sim time";
  v.animation.viewModel.timeFormatter = (date) => `t = ${Math.round(secs(date))} s`;
  clk.onTick.addEventListener(hudTick);

  for (const k of ["zones", "objects", "sensors", "tracks", "rays", "overlay", "answer"]) {
    S.ds[k] = new C.CustomDataSource(k);
    await v.dataSources.add(S.ds[k]);
  }
  await buildGroundGrid(v.terrainProvider);
  drawZones(); drawObjects(); drawSensors(); drawTracks();
  fidelity(); legend(); wireUI(); preset("overview", true);
  $("#boot").hidden = true;
  if (!token) note("No Cesium ion token: showing OpenStreetMap tiles on a flat ellipsoid.");
  S.hudSec = -1; hudTick(clk);
  window.__scenescout = S;      // debugging handle (devtools); holds no secrets
  rr();
}

function note(msg) {
  const li = document.createElement("li");
  li.className = "card sys"; li.textContent = msg;
  const tr = $("#trace"); if (tr.querySelector(".empty")) tr.innerHTML = "";
  tr.appendChild(li);
}

function header() {
  const m = S.meta;
  $("#llm").innerHTML = `<span class="dot ${m.llm.ok ? "on" : "off"}"></span>Model server: ${m.llm.ok ? `ready (${esc((m.llm.models || []).join(", ") || m.model)})` : "offline, scripted mode only"}`;
  if (!/kandla/i.test(S.site.meta.site || "")) $("#siteTitle").textContent = `${S.site.meta.site || m.site} · counter-UAS operator console`;
  $("#dataver").textContent = `${S.site.meta.site || m.site} · data ${m.data_version}`;
  const policy = /rl/i.test(m.model) ? "RL" : /sft/i.test(m.model) ? "SFT" : "prompted";
  $("#llm").title = `policy: ${policy} (${m.model})`;
  $("#llm").insertAdjacentHTML("beforeend", ` · <b>${policy}</b>`);
  if (!m.llm.ok && S.demo.length) setMode("scripted"); else setMode("model");
  examples();
}

function fidelity() {
  const meta = S.site.meta, f = meta.fidelity || {};
  const geom = /openstreetmap/i.test(f.geometry || "") ? "OpenStreetMap" : (f.geometry || "unknown");
  const est = /estimated/i.test(f.geometry || "") ? ", heights estimated" : "";
  const syn = [/synthetic/i.test(f.sensors || "") && "sensors", /synthetic/i.test(f.emitters || "") && "drones"].filter(Boolean);
  const parts = [`Geometry: ${geom}${est}`];
  if (syn.length) parts.push(`${syn.join(" and ")} synthetic`.replace(/^./, (c) => c.toUpperCase()));
  parts.push(`RF: ${meta.rf_backend}`);
  if (!meta.origin.georeferenced) parts.push("not georeferenced (nominal origin)");
  $("#fidText").textContent = parts.join(" · ");
  $("#attr").textContent = meta.attribution ? meta.attribution.replace("(c)", "©") : "";
  $("#attr").hidden = !meta.attribution;
  const rows = Object.entries(f).map(([k, val]) => `<dt>${esc(k)}</dt><dd>${esc(val)}</dd>`);
  rows.push(`<dt>terrain</dt><dd>${S.groundOk ? "Cesium World Terrain (sampled)" : "flat ellipsoid"}</dd>`);
  rows.push(`<dt>imagery</dt><dd>${S.token ? "Cesium ion (Bing aerial)" : "OpenStreetMap tiles"}</dd>`);
  rows.push(`<dt>origin</dt><dd>${meta.origin.lat.toFixed(5)}, ${meta.origin.lon.toFixed(5)}</dd>`);
  $("#fidMore").innerHTML = rows.join("");
}

function legend() {
  const items = [
    ["sw", COL.asset, "Protected asset"], ["sw", COL.restricted, "Restricted zone"],
    ["sw line", COL.track, "Single drone"], ["sw dash", COL.lowconf, "Low confidence (< 0.6)"],
  ];
  for (const [g, c] of S.groups) items.push(["sw line", c.color, `Swarm ${g}`]);
  items.push(["sw line", COL.focus, "DF chosen"], ["sw dash", COL.restricted, "DF rejected"]);
  if (S.prov) for (const p of Object.values(PROV)) items.push(["sw", p.color, p.label]);
  $("#legend").innerHTML = items.map(([cls, c, t]) =>
    `<div><span class="${cls}" style="${cls === "sw" ? `background:${c}` : `border-color:${c}`}"></span>${esc(t)}</div>`).join("");
}

/* ---------------------------------------------------------------- twin */
function objColor(o) {
  if (S.prov) return css(PROV[o.provenance.class]?.color || PROV.unknown.color, 0.92);
  if (o.protected) return css(COL.asset, 0.95);
  return css(KIND_COLOR[o.kind] || KIND_COLOR.building, o.kind === "tank" ? 0.95 : 0.9);
}
const LABEL_KINDS = new Set(["jetty", "tank_farm", "ship", "substation", "silo"]);

function drawObjects() {
  const ents = S.ds.objects.entities;
  ents.suspendEvents();
  for (const o of S.site.objects) {
    const pos = C.Cartesian3.fromDegreesArray(o.polygon.flat());
    const h = o.height_m || 0;
    const rec = {kind: "obj", data: o, positions: pos};
    const ent = {id: o.ref, name: o.name, position: C.Cartesian3.fromDegrees(o.center[0], o.center[1], h + 6)};
    if (o.kind === "tank_farm") {
      rec.fill = new C.ColorMaterialProperty(objColor(o).withAlpha(0.1));
      rec.line = new C.ColorMaterialProperty(objColor(o).withAlpha(0.85));
      ent.polygon = {hierarchy: pos, material: rec.fill, classificationType: C.ClassificationType.TERRAIN};
      ent.polyline = {positions: pos, clampToGround: true, width: 2, material: rec.line};
    } else if (o.kind === "road" || h <= 0.5) {
      rec.fill = new C.ColorMaterialProperty(objColor(o).withAlpha(0.45));
      ent.polygon = {hierarchy: pos, material: rec.fill, classificationType: C.ClassificationType.TERRAIN};
    } else {
      rec.fill = new C.ColorMaterialProperty(objColor(o));
      ent.polygon = {
        hierarchy: pos, material: rec.fill, height: 0, heightReference: C.HeightReference.RELATIVE_TO_GROUND,
        extrudedHeight: h, extrudedHeightReference: C.HeightReference.RELATIVE_TO_GROUND,
      };
    }
    if (LABEL_KINDS.has(o.kind) || o.protected) {
      ent.label = {
        text: o.name, font: "500 12px 'IBM Plex Sans Condensed', sans-serif", fillColor: o.protected ? css(COL.asset) : css(COL.paper),
        outlineColor: css("#05090c"), outlineWidth: 3, style: C.LabelStyle.FILL_AND_OUTLINE,
        heightReference: C.HeightReference.RELATIVE_TO_GROUND, verticalOrigin: C.VerticalOrigin.BOTTOM,
        distanceDisplayCondition: new C.DistanceDisplayCondition(0, o.kind === "jetty" ? 7000 : o.kind === "tank_farm" ? 4500 : 3000),
        disableDepthTestDistance: INF, scaleByDistance: new C.NearFarScalar(800, 1.0, 9000, 0.75),
      };
    }
    rec.ent = ents.add(ent);
    S.recs.set(o.ref, rec);
  }
  ents.resumeEvents();
}

function restyleObj(rec, focus = false) {
  const o = rec.data;
  const c = focus ? css(COL.focus, 0.95) : objColor(o);
  if (o.kind === "tank_farm") { rec.fill.color = c.withAlpha(focus ? 0.25 : 0.1); rec.line.color = c.withAlpha(0.9); }
  else if (o.kind === "road" || !(o.height_m > 0.5)) rec.fill.color = c.withAlpha(0.45);
  else rec.fill.color = c;
}

function zoneStyle(z) {
  if (z.kind === "restricted") return {color: COL.restricted};
  if (z.kind === "perimeter") return {color: COL.perimeter};
  if (z.ref === "zone:hero_zone") return {color: COL.hero};
  if (z.kind === "aviation" || z.kind === "corridor") return {color: COL.aviation};
  return {color: COL.region};
}

function drawZones() {
  const ents = S.ds.zones.entities;
  for (const z of S.site.zones) {
    const pos = C.Cartesian3.fromDegreesArray(z.polygon.flat());
    const st = zoneStyle(z);
    const rec = {kind: "zone", data: z, positions: pos, color: st.color};
    const ent = {id: z.ref, name: z.name, position: C.Cartesian3.fromDegrees(z.center[0], z.center[1], 140)};
    if (z.kind === "restricted") {
      rec.fill = new C.ColorMaterialProperty(css(st.color, 0.12));
      rec.line = new C.ColorMaterialProperty(css(st.color, 0.9));
      ent.polygon = {
        hierarchy: pos, material: rec.fill, height: 0, heightReference: C.HeightReference.RELATIVE_TO_GROUND,
        extrudedHeight: 120, extrudedHeightReference: C.HeightReference.RELATIVE_TO_GROUND,
      };
      ent.polyline = {positions: pos, clampToGround: true, width: 2, material: rec.line};
    } else if (z.kind === "perimeter") {
      rec.line = new C.PolylineDashMaterialProperty({color: css(st.color, 0.85), dashLength: 22});
      ent.polyline = {positions: pos, clampToGround: true, width: 2.5, material: rec.line};
    } else {
      rec.line = new C.ColorMaterialProperty(css(st.color, z.ref === "zone:hero_zone" ? 0.75 : 0.5));
      ent.polyline = {positions: pos, clampToGround: true, width: z.ref === "zone:hero_zone" ? 2.5 : 1.5, material: rec.line};
    }
    ent.label = {
      text: z.name, font: "500 12px 'IBM Plex Sans Condensed', sans-serif", fillColor: css(st.color, 0.95),
      outlineColor: css("#05090c"), outlineWidth: 3, style: C.LabelStyle.FILL_AND_OUTLINE,
      heightReference: C.HeightReference.RELATIVE_TO_GROUND, disableDepthTestDistance: INF,
      distanceDisplayCondition: z.kind === "restricted" ? new C.DistanceDisplayCondition(0, 2600) : new C.DistanceDisplayCondition(1500, 14000),
    };
    rec.ent = ents.add(ent);
    S.recs.set(z.ref, rec);
  }
}

function restyleZone(rec, focus = false) {
  const c = focus ? COL.focus : rec.color;
  if (rec.fill) rec.fill.color = css(c, focus ? 0.22 : 0.12);
  if (rec.line instanceof C.PolylineDashMaterialProperty) rec.line.color = css(c, 0.9);
  else if (rec.line) rec.line.color = css(c, focus ? 0.95 : (rec.data.kind === "restricted" ? 0.9 : rec.data.ref === "zone:hero_zone" ? 0.75 : 0.5));
}

function drawSensors() {
  for (const s of S.site.sensors) {
    S.sensors.set(s.ref, s);
    const hg = s.ground ?? 0;
    const top = C.Cartesian3.fromDegrees(s.lon, s.lat, hg + (s.z_m || 0));
    const drone = s.type === "drone_df";
    const color = drone ? css("#9fd8ff") : css(COL.paper);
    const rec = {kind: "sensor", data: s, positions: [top], color};
    rec.ent = S.ds.sensors.entities.add({
      id: s.ref, name: s.id, position: top,
      point: {pixelSize: drone ? 9 : 10, color, outlineColor: css("#0b141b"), outlineWidth: 2, disableDepthTestDistance: INF},
      label: {
        text: drone ? `${s.id} (drone DF)` : s.id, font: "600 13px 'IBM Plex Sans Condensed', sans-serif", fillColor: color,
        pixelOffset: new C.Cartesian2(0, -16), showBackground: true, backgroundColor: css("#0b141b", 0.75),
        backgroundPadding: new C.Cartesian2(5, 3), disableDepthTestDistance: INF,
      },
      polyline: drone ? undefined : {positions: [C.Cartesian3.fromDegrees(s.lon, s.lat, hg), top], width: 2, material: css(COL.paper, 0.7)},
    });
    S.recs.set(s.ref, rec);
  }
}

/* ---------------------------------------------------------------- tracks */
function trackColor(tr) {
  if (tr.confidence != null && tr.confidence < 0.6) return COL.lowconf;
  if (tr.group) return S.groups.get(tr.group).color;
  return COL.track;
}

function drawTracks() {
  const groupIds = [...new Set(S.tracks.map((t) => t.group).filter(Boolean))].sort();
  groupIds.forEach((g, i) => S.groups.set(g, {color: GROUP_COLORS[i % GROUP_COLORS.length], members: []}));
  const ents = S.ds.tracks.entities;
  for (const tr of S.tracks) {
    if (!tr.pts.length) continue;
    const low = tr.confidence != null && tr.confidence < 0.6;
    const color = css(trackColor(tr));
    const prop = new C.SampledPositionProperty();
    tr.ts = tr.pts.map((p) => p[0]);
    const times = tr.pts.map((p) => jd(p[0]));
    tr.ground = tr.pts.map((p) => groundAt(p[1], p[2]));
    prop.addSamples(times, tr.pts.map((p, k) => C.Cartesian3.fromDegrees(p[1], p[2], tr.ground[k] + p[3])));
    const avail = new C.TimeIntervalCollection([new C.TimeInterval({start: times[0], stop: times[times.length - 1]})]);
    tr.avail = [tr.ts[0], tr.ts[tr.ts.length - 1]];
    const rec = {kind: "trk", data: tr, color, low};
    rec.ent = ents.add({
      id: tr.ref, name: tr.ref, availability: avail, position: prop,
      point: {pixelSize: 9, color, outlineColor: css("#0b141b"), outlineWidth: 2, disableDepthTestDistance: INF},
      label: {
        text: tr.ref, font: "12px 'IBM Plex Mono', monospace", fillColor: color, pixelOffset: new C.Cartesian2(10, -10),
        horizontalOrigin: C.HorizontalOrigin.LEFT, showBackground: true, backgroundColor: css("#0b141b", 0.7),
        backgroundPadding: new C.Cartesian2(4, 2), disableDepthTestDistance: INF,
        distanceDisplayCondition: new C.DistanceDisplayCondition(0, tr.group ? 2600 : 12000),
      },
      path: {
        leadTime: 0, trailTime: tr.group ? 60 : 120, width: low || tr.group ? 1.5 : 2, resolution: 2,
        material: low ? new C.PolylineDashMaterialProperty({color: css(COL.lowconf, 0.8), dashLength: 12}) : color.withAlpha(tr.group ? 0.45 : 0.6),
      },
      viewFrom: new C.Cartesian3(-400, -700, 450),
    });
    rec.ell = ents.add({
      id: `${tr.ref}#ellipse`, availability: avail,
      polyline: {
        positions: new C.CallbackProperty((time) => ellipsePositions(tr, rec, time), false), width: 1.5, arcType: C.ArcType.NONE,
        material: low ? new C.PolylineDashMaterialProperty({color: css(COL.lowconf, 0.85), dashLength: 8}) : color.withAlpha(0.85),
      },
    });
    rec.tether = ents.add({
      id: `${tr.ref}#tether`, availability: avail,
      polyline: {positions: new C.CallbackProperty((time) => tetherPositions(tr, rec, time), false), width: 1, arcType: C.ArcType.NONE,
                 material: color.withAlpha(0.3)},
    });
    S.recs.set(tr.ref, rec);
    if (tr.group) S.groups.get(tr.group).members.push(rec);
  }
  for (const [g, info] of S.groups) {
    const starts = info.members.map((m) => m.data.avail[0]), stops = info.members.map((m) => m.data.avail[1]);
    const avail = new C.TimeIntervalCollection([new C.TimeInterval({start: jd(Math.min(...starts)), stop: jd(Math.max(...stops))})]);
    const PosProp = C.CallbackPositionProperty || C.CallbackProperty;
    const position = new PosProp((time, result) => groupCentroid(info, time, result), false);
    info.ent = ents.add({
      id: `grp:${g}`, name: `Swarm ${g}`, availability: avail, position,
      point: {pixelSize: 5, color: css(info.color, 0.5), disableDepthTestDistance: INF},
      label: {
        text: `${g} · swarm of ${info.members.length}`, font: "600 13px 'IBM Plex Sans Condensed', sans-serif",
        fillColor: css(info.color), showBackground: true, backgroundColor: css("#0b141b", 0.8),
        backgroundPadding: new C.Cartesian2(6, 3), pixelOffset: new C.Cartesian2(0, -30), disableDepthTestDistance: INF,
      },
      viewFrom: new C.Cartesian3(-900, -1500, 1000),
    });
    S.recs.set(`grp:${g}`, {kind: "grp", data: {group: g, members: info.members.map((m) => m.data.ref)}, ent: info.ent});
  }
  legend();
}

const _m4 = new C.Matrix4();
function ellipsePositions(tr, rec, time) {
  const c = rec.ent.position.getValue(time);
  if (!c) return [];
  const p = tr.pts[idxAt(tr.ts, secs(time))];
  const a = Math.max(p[4], p[5], 2), b = Math.max(Math.min(p[4], p[5]), 2);
  const th = C.Math.toRadians(p[4] >= p[5] ? p[6] : p[6] + 90);
  const st = Math.sin(th), ct = Math.cos(th);
  const m = C.Transforms.eastNorthUpToFixedFrame(c, undefined, _m4);
  const out = [];
  for (let k = 0; k <= 40; k++) {
    const g = (k / 40) * 2 * Math.PI, ca = a * Math.cos(g), sb = b * Math.sin(g);
    out.push(C.Matrix4.multiplyByPoint(m, new C.Cartesian3(ca * st + sb * ct, ca * ct - sb * st, 0), new C.Cartesian3()));
  }
  return out;
}
function tetherPositions(tr, rec, time) {
  const c = rec.ent.position.getValue(time);
  if (!c) return [];
  const i = idxAt(tr.ts, secs(time));
  const carto = C.Cartographic.fromCartesian(c);
  return [c, C.Cartesian3.fromRadians(carto.longitude, carto.latitude, tr.ground[i])];
}
function groupCentroid(info, time, result) {
  let n = 0, x = 0, y = 0, z = 0;
  for (const m of info.members) {
    const p = m.ent.position.getValue(time);
    if (p) { x += p.x; y += p.y; z += p.z; n++; }
  }
  if (!n) return undefined;
  const c = new C.Cartesian3(x / n, y / n, z / n);
  const carto = C.Cartographic.fromCartesian(c);
  carto.height += 25;
  return C.Cartographic.toCartesian(carto, C.Ellipsoid.WGS84, result);
}

function restyleTrack(rec, focus = false) {
  rec.ent.point.pixelSize = focus ? 14 : 9;
  rec.ent.point.outlineColor = focus ? css(COL.focus) : css("#0b141b");
  rec.ent.label.fillColor = focus ? css(COL.focus) : rec.color;
  rec.ent.label.distanceDisplayCondition = new C.DistanceDisplayCondition(0, focus || !rec.data.group ? 12000 : 2600);
}

/* ---------------------------------------------------------------- highlight + camera */
function setFocus(ref, on) {
  const rec = S.recs.get(ref);
  if (!rec) return;
  if (rec.kind === "obj") restyleObj(rec, on);
  else if (rec.kind === "zone") restyleZone(rec, on);
  else if (rec.kind === "trk") restyleTrack(rec, on);
  else if (rec.kind === "sensor") rec.ent.point.color = on ? css(COL.focus) : rec.color;
}
function clearHighlight() {
  for (const r of S.highlighted) setFocus(r, false);
  S.highlighted = [];
}
function baseRef(ref) { return String(ref).split("@")[0]; }

function recPoints(rec) {
  if (rec.kind === "trk") {
    const p = rec.ent.position.getValue(S.v.clock.currentTime);
    return p ? [p] : [];
  }
  if (rec.kind === "grp") { const p = rec.ent.position.getValue(S.v.clock.currentTime); return p ? [p] : []; }
  if (rec.kind === "ray") return rec.positions;
  const h = rec.data.height_m || 0;
  return (rec.positions || []).map((p) => {
    const c = C.Cartographic.fromCartesian(p);
    return C.Cartesian3.fromRadians(c.longitude, c.latitude, groundAt(C.Math.toDegrees(c.longitude), C.Math.toDegrees(c.latitude)) + h);
  });
}

function flyToPoints(points, {pitch = -0.7, minRange = 450, heading} = {}) {
  if (!points.length) return;
  S.v.trackedEntity = undefined;
  const bs = C.BoundingSphere.fromPoints(points);
  const range = Math.max(bs.radius * 3.0, minRange);
  S.v.camera.flyToBoundingSphere(bs, {offset: new C.HeadingPitchRange(heading ?? S.v.camera.heading, pitch, range), duration: 1.3});
  rr();
}

/** Highlight refs (obj:/zone:/trk:/sensor:, optional @t) and fly to them. hyp: refs are handled by the rays layer. */
function focusRefs(refs, {fly = true, select = false} = {}) {
  clearHighlight();
  let tSet = null;
  const recs = [];
  for (const r of refs.slice(0, 40)) {
    const [b, at] = String(r).split("@");
    const kind = b.split(":")[0];
    if (kind === "hyp") { const ray = S.rays.get(b) || S.rays.get(r); if (ray) recs.push(ray); continue; }
    const rec = S.recs.get(b);
    if (!rec) continue;
    if (at != null && (kind === "trk" || kind === "evt")) tSet = Number(at);
    setFocus(b, true); S.highlighted.push(b); recs.push(rec);
  }
  const trks = recs.filter((r) => r.kind === "trk");
  if (tSet != null) setSimTime(tSet);
  else if (trks.length && !trks.some((r) => r.ent.isAvailable(S.v.clock.currentTime))) {
    const a = trks[0].data.avail; setSimTime((a[0] + a[1]) / 2);
  }
  if (select && recs.length === 1 && recs[0].ent) S.v.selectedEntity = recs[0].ent;
  if (fly && recs.length) flyToPoints(recs.slice(0, 25).flatMap(recPoints));
  rr();
  return recs.length;
}

function preset(name, instant = false) {
  const v = S.v, meta = S.site.meta;
  v.trackedEntity = undefined;
  $$(".presets button").forEach((b) => b.classList.toggle("on", b.dataset.preset === name));
  const ringPts = (ring) => ring.map((p) => C.Cartesian3.fromDegrees(p[0], p[1], 0));
  const hero = meta.hero_zone || S.site.zones.find((z) => z.ref === "zone:hero_zone")?.polygon;
  const dur = instant ? 0 : 1.6;
  if (name === "overview") {
    const pts = ringPts(S.site.objects.map((o) => o.center));
    const bs = C.BoundingSphere.fromPoints(pts);
    v.camera.flyToBoundingSphere(bs, {offset: new C.HeadingPitchRange(0, C.Math.toRadians(-48), bs.radius * 2.3), duration: dur});
  } else if (name === "hero" && hero) {
    const lons = hero.map((p) => p[0]), lats = hero.map((p) => p[1]);
    v.camera.flyTo({destination: C.Rectangle.fromDegrees(Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)),
                    orientation: {heading: 0, pitch: -C.Math.PI_OVER_TWO, roll: 0}, duration: dur});
  } else if (name === "oblique" && hero) {
    const bs = C.BoundingSphere.fromPoints(ringPts(hero));
    v.camera.flyToBoundingSphere(bs, {offset: new C.HeadingPitchRange(C.Math.toRadians(35), C.Math.toRadians(-28), bs.radius * 1.9), duration: dur});
  } else if (name === "jetties" || name === "tankfarms") {
    const kind = name === "jetties" ? "jetty" : "tank_farm";
    const pts = ringPts(S.site.objects.filter((o) => o.kind === kind).flatMap((o) => o.polygon));
    if (pts.length) {
      const bs = C.BoundingSphere.fromPoints(pts);
      v.camera.flyToBoundingSphere(bs, {offset: new C.HeadingPitchRange(C.Math.toRadians(name === "jetties" ? 300 : 20), C.Math.toRadians(-38), Math.max(bs.radius * 2.4, 900)), duration: dur});
    }
  } else if (name === "swarm") {
    const g = S.groups.get("G1") || [...S.groups.values()][0];
    if (!g) { note("No swarm group in the observed tracks."); return; }
    if (!g.ent.isAvailable(v.clock.currentTime)) setSimTime(g.members[0].data.avail[0] + 10);
    v.trackedEntity = g.ent;
    v.clock.shouldAnimate = true;
  }
  rr();
}

/* ---------------------------------------------------------------- DF rays */
function clearRays() {
  S.ds.rays.entities.removeAll();
  S.rays.clear();
  rr();
}

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

/** sensors: [{sensor, anchor, track_est_deg, hyps:[{ref, bearing_deg, snr_db}]}] */
function drawRays(sensors, {t, track, replace = true, only = null} = {}) {
  if (replace) clearRays();
  if (t != null) setSimTime(t);
  const ents = S.ds.rays.entities;
  for (const s of sensors) {
    if (s.track_est_deg != null) {
      ents.add({id: `est:${s.sensor}:${track}:${t}`, polyline: {positions: rayEnd(s.anchor, s.track_est_deg, RAY_LEN_M * 0.85),
        width: 1, arcType: C.ArcType.NONE, material: new C.PolylineDashMaterialProperty({color: css(COL.paper, 0.35), dashLength: 6})}});
    }
    for (const h of s.hyps) {
      if (only && !only.has(baseRef(h.ref)) && !only.has(h.ref)) continue;
      const positions = rayEnd(s.anchor, h.bearing_deg);
      const mat = new C.ColorMaterialProperty(css(COL.paper, 0.8));
      const rec = {kind: "ray", data: {...h, sensor: s.sensor, track, t}, positions, mat};
      rec.ent = ents.add({
        id: `ray:${h.ref}`, name: h.ref, position: positions[1],
        polyline: {positions, width: 2.5, arcType: C.ArcType.NONE, material: mat},
        label: {text: `${s.sensor.replace("sensor:", "")} ${h.bearing_deg.toFixed(1)}°`, font: "11px 'IBM Plex Mono', monospace",
                fillColor: css(COL.paper, 0.9), showBackground: true, backgroundColor: css("#0b141b", 0.7),
                disableDepthTestDistance: INF, distanceDisplayCondition: new C.DistanceDisplayCondition(0, 9000)},
      });
      S.rays.set(h.ref, rec); S.rays.set(baseRef(h.ref), rec);
    }
  }
  rr();
}

function raysFromDfResult(data) {
  const sensors = (data.sensors || []).map((s) => ({
    sensor: s.sensor, anchor: sensorAnchor(s.sensor, s.sensor_pos || {x_m: 0, y_m: 0}),
    track_est_deg: s.bearing_to_track_estimate_deg, hyps: s.hyps || [],
  }));
  drawRays(sensors, {t: data.t_s, track: data.track});
}

async function showBearings(trackRef, t, onlyHyp = null) {
  const q = new URLSearchParams();
  if (t != null) q.set("t", String(t));
  if (trackRef) q.set("track", trackRef);
  try {
    const d = await getJSON(`/api/bearings?${q}`);
    if (!d.sensors.length) { note(`No DF bearings near t=${Math.round(t)} s${trackRef ? ` for ${trackRef}` : ""}.`); return; }
    const sensors = d.sensors.map((s) => ({
      sensor: s.ref, anchor: {lon: s.lon, lat: s.lat, h: (S.sensors.get(s.ref)?.ground ?? groundAt(s.lon, s.lat)) + (s.z_m || 0)},
      hyps: s.bearings.map((b) => ({ref: b.hyp, bearing_deg: b.bearing_deg, snr_db: b.snr_db, track: b.track})),
    }));
    drawRays(sensors, {t: d.t_s, track: trackRef, only: onlyHyp ? new Set([onlyHyp, baseRef(onlyHyp)]) : null});
    if (!$("[data-layer=rays]").checked) { $("[data-layer=rays]").checked = true; S.ds.rays.show = true; }
  } catch (e) { note(`Bearings unavailable: ${e.message}`); }
}

function styleRay(rec, how) {
  const e = rec.ent;
  if (how === "chosen") { e.polyline.material = css(COL.focus, 1); e.polyline.width = 4; e.label.fillColor = css(COL.focus); }
  else if (how === "rejected") {
    e.polyline.material = new C.PolylineDashMaterialProperty({color: css(COL.restricted, 0.95), dashLength: 16});
    e.polyline.width = 2.5; e.label.fillColor = css(COL.restricted);
  } else { e.polyline.material = css(COL.paper, 0.25); e.polyline.width = 1.5; }
}

function applyDfVerdict(ans) {
  const chosen = new Set((ans.chosen_hyp_refs || []).flatMap((r) => [r, baseRef(r)]));
  const rejected = new Set((ans.rejected_hyp_refs || []).flatMap((r) => [r, baseRef(r)]));
  const seen = new Set();
  for (const [k, rec] of S.rays) {
    if (seen.has(rec)) continue;
    seen.add(rec);
    const r = rec.data.ref;
    styleRay(rec, chosen.has(r) || chosen.has(baseRef(r)) ? "chosen" : rejected.has(r) || rejected.has(baseRef(r)) ? "rejected" : "other");
    void k;
  }
  rr();
}

/* ---------------------------------------------------------------- answer overlays */
function clearAnswerOverlays() { S.ds.answer.entities.removeAll(); rr(); }

function drawAnswerPoint(p, ell, label) {
  if (!p) return;
  const [lon, lat] = enuToLonLat(p.x_m, p.y_m);
  const hg = groundAt(lon, lat), z = p.z_m || 0;
  const c = C.Cartesian3.fromDegrees(lon, lat, hg + z);
  S.ds.answer.entities.add({
    id: `ans:${label}`, position: c,
    point: {pixelSize: 12, color: css(COL.focus), outlineColor: css("#0b141b"), outlineWidth: 2, disableDepthTestDistance: INF},
    label: {text: label, font: "600 12px 'IBM Plex Sans Condensed', sans-serif", fillColor: css(COL.focus), pixelOffset: new C.Cartesian2(0, -18),
            showBackground: true, backgroundColor: css("#0b141b", 0.8), disableDepthTestDistance: INF},
  });
  if (ell && ell.a_m) {
    const tr = {pts: [[0, lon, lat, z, ell.a_m, ell.b_m, ell.theta_deg]], ts: [0]};
    const fake = {ent: {position: new C.ConstantPositionProperty(c)}};
    S.ds.answer.entities.add({polyline: {positions: ellipsePositions(tr, fake, S.v.clock.currentTime), width: 2, arcType: C.ArcType.NONE, material: css(COL.focus, 0.9)}});
  }
  flyToPoints([c], {minRange: 900});
}

function drawAnswerRegion(region) {
  if (!region) return;
  if (region.type === "ref") { focusRefs([region.ref]); return; }
  let ring = [];
  if (region.type === "circle") {
    for (let k = 0; k <= 48; k++) {
      const g = (k / 48) * 2 * Math.PI;
      ring.push(enuToLonLat(region.center.x_m + region.radius_m * Math.sin(g), region.center.y_m + region.radius_m * Math.cos(g)));
    }
  } else if (region.type === "polygon") {
    ring = region.points.map((p) => enuToLonLat(p.x_m, p.y_m)); ring.push(ring[0]);
  }
  if (ring.length) {
    const pos = C.Cartesian3.fromDegreesArray(ring.flat());
    S.ds.answer.entities.add({polyline: {positions: pos, clampToGround: true, width: 3, material: css(COL.focus, 0.9)}});
    flyToPoints(pos);
  }
}

/* ---------------------------------------------------------------- info card */
function provPill(p) {
  if (!p) return "";
  const cls = p.class === "synthetic" ? "syn" : p.class === "osm_estimated" ? "est" : "";
  return `<span class="pill ${cls}">${esc(PROV[p.class]?.label || p.source || "unknown")}</span>`;
}
function dl(rows) {
  return `<dl>${rows.filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
}
function refChip(ref) {
  const b = baseRef(ref), kind = b.split(":")[0];
  const live = S.recs.has(b) || kind === "hyp";
  return `<button type="button" class="ref ${kind} ${live ? "" : "inert"}" data-ref="${esc(ref)}" title="${live ? "Show on map" : "No map entity"}">${esc(ref)}</button>`;
}

function showInfo(rec) {
  const box = $("#info");
  if (!rec) { box.hidden = true; return; }
  const d = rec.data;
  let html = "";
  if (rec.kind === "obj") {
    const p = d.provenance || {};
    html = `<h4>${esc(d.name)}</h4><div class="sub">${esc(d.kind.replace("_", " "))} · ${refChip(d.ref)} ${d.protected ? '<span class="pill prot">protected</span>' : ""}</div>` +
      dl([["Aliases", esc((d.aliases || []).join(", "))], ["Height", d.height_m != null ? `${d.height_m.toFixed(1)} m` : "unknown"],
          ["Material", esc(d.material)], ["Protected", d.protected ? `yes${p.protected_status ? ` (${esc(p.protected_status)})` : ""}` : "no"],
          ["Provenance", provPill(p)], ["Source", esc(p.source)], ["Status", esc(p.status)], ["Name source", esc(p.name_source)],
          ["Height source", esc(p.height_source)], ["Height conf.", esc(p.height_confidence)], ["OSM id", esc(p.osm_id)],
          ["Part of", d.parent ? refChip(`obj:${d.parent}`) : null]]);
  } else if (rec.kind === "zone") {
    html = `<h4>${esc(d.name)}</h4><div class="sub">${esc(d.kind.replace("_", " "))} zone · ${refChip(d.ref)}</div>` +
      dl([["Aliases", esc((d.aliases || []).join(", "))], ["Source", esc(d.provenance?.source)], ["Status", esc(d.provenance?.status)]]);
  } else if (rec.kind === "sensor") {
    html = `<h4>Sensor ${esc(d.id)}</h4><div class="sub">${esc(d.type)} · ${refChip(d.ref)} <span class="pill syn">${esc(d.placement || "synthetic")}</span></div>` +
      dl([["Mount", esc(d.mount)], ["Mast height", `${d.z_m} m above ground`], ["Bands", esc((d.bands || []).join(", "))],
          ["Position", `${d.lat.toFixed(5)}, ${d.lon.toFixed(5)}`]]) +
      `<div class="acts"><button class="btn" data-act="sensorRays">DF bearings now</button></div>`;
  } else if (rec.kind === "trk") {
    const t = now(), i = idxAt(d.ts, t), p = d.pts[i];
    const probs = Object.entries(d.klass_probs || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(", ");
    html = `<h4>${esc(d.ref)}</h4><div class="sub">${esc(d.klass)} · ${esc(d.band)}${d.group ? ` · swarm ${esc(d.group)}` : ""}${rec.low ? ' · <span class="pill">low confidence</span>' : ""}</div>` +
      dl([["Confidence", d.confidence != null ? d.confidence.toFixed(2) : "?"], ["Class", esc(probs)],
          ["Frequency", d.freq_mhz ? `${d.freq_mhz} MHz` : null], ["Seen", `t = ${d.avail[0]}–${d.avail[1]} s`],
          ["At t", `${p[0]} s: alt ${p[3]} m, ellipse ${p[4]} × ${p[5]} m @ ${p[6]}°`]]) +
      `<div class="acts"><button class="btn" data-act="trkRays">DF bearings at t</button><button class="btn" data-act="follow">Follow</button></div>`;
  } else if (rec.kind === "ray") {
    html = `<h4>DF bearing</h4><div class="sub">${refChip(d.ref)}</div>` +
      dl([["Sensor", refChip(d.sensor)], ["Bearing", `${d.bearing_deg}°`], ["SNR", d.snr_db != null ? `${d.snr_db} dB` : null],
          ["Track", d.track ? refChip(d.track) : null], ["t", d.t != null ? `${d.t} s` : null]]);
  } else if (rec.kind === "grp") {
    html = `<h4>Swarm ${esc(d.group)}</h4><div class="sub">${d.members.length} tracks grouped by the tracker</div>` +
      `<div class="refs">${d.members.map(refChip).join(" ")}</div>`;
  }
  box.innerHTML = `<button class="x" aria-label="Close">×</button>${html}`;
  box.hidden = false;
  box.querySelector(".x").onclick = () => { S.v.selectedEntity = undefined; };
  box.querySelectorAll("[data-ref]").forEach((b) => (b.onclick = () => handleRef(b.dataset.ref)));
  const act = (a, fn) => { const b = box.querySelector(`[data-act=${a}]`); if (b) b.onclick = fn; };
  act("trkRays", () => showBearings(d.ref, now()));
  act("sensorRays", async () => {
    const r = await getJSON(`/api/bearings?t=${now()}`).catch(() => null);
    if (!r || !r.sensors.length) { note("No DF bearings near this time."); return; }
    const s = r.sensors.find((x) => x.ref === d.ref);
    if (!s) { note(`${d.ref} reports no bearings near t=${Math.round(now())} s.`); return; }
    drawRays([{sensor: s.ref, anchor: {lon: s.lon, lat: s.lat, h: (d.ground ?? 0) + s.z_m},
               hyps: s.bearings.map((b) => ({ref: b.hyp, bearing_deg: b.bearing_deg, snr_db: b.snr_db}))}], {t: r.t_s});
  });
  act("follow", () => { S.v.trackedEntity = rec.ent; S.v.clock.shouldAnimate = true; rr(); });
}

function recForEntity(ent) {
  if (!ent) return null;
  const id = String(ent.id);
  if (id.startsWith("ray:")) return S.rays.get(id.slice(4));
  return S.recs.get(id.split("#")[0]) || null;
}

/* ---------------------------------------------------------------- picking */
const PICK_ORDER = {trk: 0, grp: 1, sensor: 2, ray: 3, obj: 4, zone: 5};
function onClick(ev) {
  const picks = S.v.scene.drillPick(ev.position, 8) || [];
  let best = null, bestRank = 99;
  for (const p of picks) {
    const ent = p && p.id;
    if (!(ent instanceof C.Entity)) continue;
    const rec = recForEntity(ent);
    if (!rec) continue;
    const rank = PICK_ORDER[rec.kind] ?? 9;
    if (rank < bestRank) { best = rec; bestRank = rank; }
  }
  S.v.selectedEntity = best ? best.ent : undefined;
}

function handleRef(ref) {
  const b = baseRef(ref), kind = b.split(":")[0];
  if (kind === "hyp") {
    const at = String(ref).split("@")[1];
    const rec = S.rays.get(ref) || S.rays.get(b);
    if (rec) { if (at) setSimTime(Number(at)); flyToPoints(rec.positions, {minRange: 1500}); S.v.selectedEntity = rec.ent; }
    else if (at) showBearings(null, Number(at), ref);
    return;
  }
  if (S.recs.has(b)) {
    focusRefs([ref], {select: true});
  }
}

/* ---------------------------------------------------------------- UI wiring */
function hudTick(clock) {
  const t = Math.round(secs(clock.currentTime));
  if (t === S.hudSec && clock.shouldAnimate === S.hudAnim) return;
  S.hudSec = t; S.hudAnim = clock.shouldAnimate;
  const active = S.tracks.filter((tr) => tr.avail && t >= tr.avail[0] && t <= tr.avail[1]).length;
  $("#hud").innerHTML = `t = <b>${t}</b> / ${S.dur} s · ${active} tracks · ${clock.shouldAnimate ? `playing ×${clock.multiplier}` : "paused"}`;
}

function setMode(m) {
  if (m === "scripted" && !S.demo.length) m = "model";
  S.mode = m;
  $$(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  const sb = $(".seg [data-mode=scripted]");
  sb.disabled = !S.demo.length;
  sb.title = S.demo.length ? "Replay the scripted expert on a demo question" : "No demo questions for this site yet";
  $("#agentBadge").textContent = m === "model" ? `Agent: model (${S.meta?.model || "scenescout"}, prompted)` : "Agent: scripted expert";
  $("#modeNote").textContent = m === "scripted" ? "Scripted mode replays the expert on the selected example question."
    : S.meta && !S.meta.llm.ok ? "The model server is offline; start it or use scripted mode." : "";
}

function examples() {
  const box = $("#examples");
  if (S.demo.length) {
    box.innerHTML = S.demo.slice(0, 14).map((d) =>
      `<button type="button" class="chip" data-task="${esc(d.task_id)}" title="${esc(d.question)}">${esc(d.question.length > 70 ? d.question.slice(0, 68) + "…" : d.question)}<small>${esc(d.template)}</small></button>`).join("");
  } else {
    box.innerHTML = FALLBACK_QUESTIONS.map((q) => `<button type="button" class="chip" data-q="${esc(q)}">${esc(q)}</button>`).join("") +
      `<div class="note">Scripted demo questions for this site are not generated yet; these run in model mode.</div>`;
  }
  box.querySelectorAll(".chip").forEach((c) => (c.onclick = () => {
    box.querySelectorAll(".chip").forEach((x) => x.classList.remove("sel"));
    c.classList.add("sel");
    if (c.dataset.task) {
      const d = S.demo.find((x) => x.task_id === c.dataset.task);
      $("#q").value = d.question; S.taskId = d.task_id;
    } else { $("#q").value = c.dataset.q; S.taskId = null; }
  }));
}

function wireUI() {
  S.v.screenSpaceEventHandler.setInputAction(onClick, C.ScreenSpaceEventType.LEFT_CLICK);
  S.v.screenSpaceEventHandler.removeInputAction(C.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
  S.v.selectedEntityChanged.addEventListener((ent) => showInfo(recForEntity(ent)));
  $$(".presets button").forEach((b) => (b.onclick = () => preset(b.dataset.preset)));
  $$(".seg button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  $$("[data-layer]").forEach((cb) => (cb.onchange = () => layer(cb.dataset.layer, cb.checked)));
  const fid = $("#fidelity");
  const tog = () => { const m = $("#fidMore"); m.hidden = !m.hidden; fid.setAttribute("aria-expanded", String(!m.hidden)); };
  fid.onclick = tog; fid.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tog(); } };
  $("#askForm").onsubmit = (e) => { e.preventDefault(); ask(); };
  $("#q").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } };
  $("#q").oninput = () => {
    const d = S.demo.find((x) => x.question === $("#q").value.trim());
    S.taskId = d ? d.task_id : null;
  };
  wireSearch();
}

async function layer(name, on) {
  if (S.ds[name]) S.ds[name].show = on;
  if (name === "zones" || name === "objects" || name === "sensors" || name === "tracks" || name === "rays") S.ds[name].show = on;
  if (name === "rays") S.ds.answer.show = on;
  if (name === "prov") {
    S.prov = on;
    for (const rec of S.recs.values()) if (rec.kind === "obj") restyleObj(rec, S.highlighted.includes(rec.data.ref));
    legend();
  }
  if (name === "osm") {
    if (on && !S.osm && !S.osmLoading) {
      if (!S.token) { note("Cesium OSM buildings need an ion token."); return; }
      S.osmLoading = true;
      try { S.osm = await C.createOsmBuildingsAsync(); S.v.scene.primitives.add(S.osm); }
      catch (e) { note(`OSM buildings unavailable: ${e.message}`); }
      S.osmLoading = false;
    }
    if (S.osm) S.osm.show = $("[data-layer=osm]").checked;
  }
  if (name === "low") {
    S.v.resolutionScale = on ? 0.6 : 1.0;
    S.v.scene.postProcessStages.fxaa.enabled = !on;
    S.v.scene.globe.maximumScreenSpaceError = on ? 4 : 2;
    for (const rec of S.recs.values()) if (rec.kind === "trk") rec.ent.path.show = !on;
  }
  rr();
}

function wireSearch() {
  const inp = $("#search"), list = $("#searchList");
  const index = [];
  for (const [ref, rec] of S.recs) {
    if (!["obj", "zone", "sensor", "trk", "grp"].includes(rec.kind)) continue;
    const d = rec.data;
    const name = d.name || d.ref || ref;
    index.push({ref, name, kind: d.kind || rec.kind, keys: [ref, name, ...(d.aliases || [])].map((s) => String(s).toLowerCase())});
  }
  const render = () => {
    const q = inp.value.trim().toLowerCase();
    if (!q) { list.hidden = true; return; }
    const exact = index.filter((x) => x.keys.includes(q));
    const hits = exact.concat(index.filter((x) => !exact.includes(x) && x.keys.some((k) => k.includes(q)))).slice(0, 10);
    list.innerHTML = hits.length ? hits.map((h, i) =>
      `<li data-ref="${esc(h.ref)}" class="${i === 0 ? "act" : ""}"><span>${esc(h.name)}</span><small>${esc(h.kind)}${exact.length > 1 && exact.includes(h) ? " · same name" : ""}</small></li>`).join("")
      : `<li><small>No match</small></li>`;
    if (exact.length > 1) list.insertAdjacentHTML("afterbegin", `<li><small>${exact.length} entities share this name or alias</small></li>`);
    list.hidden = false;
    list.querySelectorAll("li[data-ref]").forEach((li) => (li.onmousedown = () => { handleRef(li.dataset.ref); list.hidden = true; }));
  };
  inp.oninput = render;
  inp.onkeydown = (e) => {
    if (e.key === "Enter") { const f = list.querySelector("li[data-ref]"); if (f) { handleRef(f.dataset.ref); list.hidden = true; } }
    if (e.key === "Escape") list.hidden = true;
  };
  inp.onblur = () => setTimeout(() => (list.hidden = true), 150);
}

/* ---------------------------------------------------------------- agent stream */
function busy(on) { $("#go").disabled = on; $("#go").textContent = on ? "Running…" : "Ask"; }

function ask() {
  const q = $("#q").value.trim();
  if (S.es) { S.es.close(); S.es = null; }
  let url;
  if (S.mode === "scripted") {
    if (!S.taskId) { note("Scripted mode: pick one of the example questions first."); return; }
    url = `/api/ask?mode=scripted&task_id=${encodeURIComponent(S.taskId)}`;
  } else {
    if (!q) { $("#q").focus(); return; }
    url = `/api/ask?mode=model&q=${encodeURIComponent(q)}`;
  }
  $("#trace").innerHTML = ""; $("#answer").hidden = true; $("#traceMeta").textContent = "";
  S.pending = []; S.finished = false; S.t0 = performance.now();
  clearRays(); clearAnswerOverlays(); clearHighlight(); clearOverlay();
  busy(true);
  const es = S.es = new EventSource(url);
  es.onmessage = (m) => { let ev; try { ev = JSON.parse(m.data); } catch (_) { return; } onEvent(ev); };
  es.onerror = () => {
    es.close(); S.es = null;
    if (!S.finished) { addCard("err", `<div class="hd"><span class="nm">stream</span></div><div class="error">The connection closed before an answer (server error or bad request).</div>`); busy(false); }
  };
}

function addCard(cls, html) {
  const li = document.createElement("li");
  li.className = `card ${cls}`; li.innerHTML = html;
  $("#trace").appendChild(li);
  li.querySelectorAll("[data-ref]").forEach((b) => (b.onclick = () => handleRef(b.dataset.ref)));
  li.scrollIntoView({block: "nearest"});
  return li;
}

function shortArgs(a) {
  if (!a || typeof a !== "object") return "";
  let s = JSON.stringify(a);
  if (s === "{}") return "";
  return s.length > 110 ? s.slice(0, 108) + "…" : s;
}

function summarize(name, res) {
  if (!res) return "";
  if (!res.ok) return "";
  const d = res.data || {};
  if (name === "declare_intent") return `Intent: ${esc(d.intent)}`;
  if (name === "ask_user") return `Operator reply: ${esc(d.reply)}`;
  if (name === "submit") return "Answer accepted by the verifier";
  const bits = [];
  for (const [k, v] of Object.entries(d)) {
    if (Array.isArray(v)) bits.push(`${k}: ${v.length}`);
    else if (typeof v === "boolean") bits.push(`${k}: ${v ? "yes" : "no"}`);
    else if (typeof v === "number") bits.push(`${k}: ${v}`);
    else if (typeof v === "string" && v.length <= 40) bits.push(`${k}: ${v}`);
    if (bits.length >= 5) break;
  }
  if (res.refs?.length) bits.push(`${res.refs.length} refs`);
  return esc(bits.join(" · "));
}

function onEvent(ev) {
  switch (ev.type) {
    case "start":
      addCard("sys", `Episode ${esc(ev.episode_id)} · ${esc(ev.mode)} · data ${esc(ev.data_version)}<br><span style="color:var(--paper)">${esc(ev.question)}</span>`);
      break;
    case "assistant":
      addCard("turn", `Turn ${ev.turn} · model ${ev.gen_ms} ms · ${ev.tokens} tokens · ${ev.n_calls} call${ev.n_calls === 1 ? "" : "s"}${ev.text ? `<p>${esc(ev.text)}</p>` : ""}`);
      break;
    case "tool_call": {
      const li = addCard("pending", `<div class="hd"><span class="nm">${esc(ev.name)}</span><span class="lat">running</span></div><div class="args">${esc(shortArgs(ev.arguments))}</div>`);
      S.pending.push({name: ev.name, li});
      break;
    }
    case "tool_result": onToolResult(ev); break;
    case "ledger": {
      const last = $("#trace").lastElementChild;
      if (last) last.insertAdjacentHTML("beforeend", `<details><summary>Evidence ledger</summary><pre>${esc(ev.delta)}</pre></details>`);
      break;
    }
    case "answer": renderAnswer(ev); break;
    case "error": addCard("err", `<div class="hd"><span class="nm">error</span></div><div class="error">${esc(ev.message)}</div>`); break;
    case "done":
      S.finished = true; busy(false);
      if (S.es) { S.es.close(); S.es = null; }
      if (S.t0) $("#traceMeta").textContent = `${((performance.now() - S.t0) / 1000).toFixed(1)} s total`;
      break;
    default: break;
  }
}

function onToolResult(ev) {
  const res = ev.result || {};
  const k = S.pending.findIndex((p) => p.name === ev.name);
  let li;
  if (k >= 0) { li = S.pending[k].li; S.pending.splice(k, 1); }
  else li = addCard("pending", `<div class="hd"><span class="nm">${esc(ev.name)}</span><span class="lat"></span></div>`);
  li.className = `card ${res.ok ? "ok" : "err"}`;
  li.querySelector(".lat").textContent = res.cost_ms != null ? `${res.cost_ms} ms` : res.ok ? "ok" : "failed";
  let extra = "";
  if (res.ok) {
    extra += `<div class="sum">${summarize(ev.name, res)}</div>`;
    const refs = (res.refs || []).filter((r) => !String(r).startsWith("hyp:") || ev.name !== "df_hypotheses");
    if (refs.length) extra += `<div class="refs">${refs.slice(0, 8).map(refChip).join("")}${refs.length > 8 ? `<span class="note">+${refs.length - 8} more</span>` : ""}</div>`;
  } else {
    const e = res.error || {};
    const msg = typeof e === "string" ? e : `${e.code ? `<b>${esc(e.code)}</b>: ` : ""}${esc(e.message || "")}`;
    extra += `<div class="error">${msg}</div>`;
    if (e.hint) extra += `<div class="hint">Hint: ${esc(e.hint)}</div>`;
    if (e.candidates) extra += `<div class="refs">${[].concat(e.candidates).slice(0, 6).map((c) => typeof c === "string" ? refChip(c) : `<span class="note">${esc(JSON.stringify(c))}</span>`).join("")}</div>`;
    if (e.fields) extra += `<pre>${esc(JSON.stringify(e.fields, null, 1))}</pre>`;
  }
  li.insertAdjacentHTML("beforeend", extra);
  li.querySelectorAll("[data-ref]").forEach((b) => (b.onclick = () => handleRef(b.dataset.ref)));
  if (!res.ok) return;
  if (drawToolOverlay(ev)) return;
  if (ev.name === "df_hypotheses" && res.data) {
    raysFromDfResult(res.data);
    const pts = [...S.rays.values()].flatMap((r) => r.positions);
    focusRefs([res.data.track].filter(Boolean), {fly: false});
    flyToPoints(pts, {minRange: 2500, pitch: -0.9});
    return;
  }
  const refs = (res.refs || []).filter((r) => /^(obj|zone|trk|sensor):/.test(r));
  if (refs.length) focusRefs(refs);
}

function renderAnswer(ev) {
  const box = $("#answer");
  const a = ev.answer;
  const v = ev.verifier;
  let html = "";
  if (!a) {
    html = `<div class="intent">No answer</div><div class="verdict abstain">No answer submitted<small>Episode ended: ${esc(ev.end_reason || "unknown")}</small></div>`;
  } else {
    html += `<div class="intent">${esc(INTENT_LABEL[a.intent] || a.intent)}</div>`;
    if (a.requires_human_confirm) html += `<div class="confirm">Needs operator confirmation</div>`;
    let verdict = "", vcls = "";
    if (a.intent === "abstain") { verdict = `Cannot determine<small>${esc(ABSTAIN[a.reason] || a.reason)}</small>`; vcls = "abstain"; }
    else if (a.intent === "operator_query") {
      if (a.boolean === true) { verdict = "Yes"; vcls = "yes"; }
      else if (a.boolean === false) { verdict = "No"; vcls = "no"; }
      else verdict = `${(a.track_refs || []).length} track${(a.track_refs || []).length === 1 ? "" : "s"}`;
    } else if (a.intent === "df_ambiguity") {
      verdict = `${a.chosen_hyp_refs.length} bearing${a.chosen_hyp_refs.length === 1 ? "" : "s"} chosen<small>${a.rejected_hyp_refs.length} rejected as non-direct paths</small>`;
    } else if (a.intent === "drone_search") {
      verdict = `Drone located<small>${a.final_position.x_m.toFixed(0)} m E, ${a.final_position.y_m.toFixed(0)} m N of origin · ±${a.uncertainty.a_m} m</small>`;
    } else if (a.intent === "collateral") {
      verdict = `${a.violations.length} asset${a.violations.length === 1 ? "" : "s"} or link${a.violations.length === 1 ? "" : "s"} affected<small>assessment only: nothing has been switched on</small>`;
    } else if (a.intent === "replay") {
      verdict = a.time_s != null ? `t = ${a.time_s} s` : `${a.evt_refs.length} events`;
    } else if (a.intent === "siting") {
      const nz = (a.zones || []).length, nn = (a.node_candidates || []).length;
      verdict = nz ? `${nz} weak zone${nz === 1 ? "" : "s"}${nn ? ` · ${nn} candidate site${nn === 1 ? "" : "s"}` : ""}`
                   : `${nn} candidate site${nn === 1 ? "" : "s"}`;
    }
    html += `<div class="verdict ${vcls}">${verdict}</div>`;
    const pct = Math.round((a.confidence || 0) * 100);
    html += `<div class="conf"><span>Confidence</span><div class="bar"><i style="width:${pct}%"></i></div><span>${pct}%</span></div>`;
    const groups = [];
    if (a.track_refs?.length) groups.push(["Tracks", a.track_refs]);
    if (a.chosen_hyp_refs?.length) groups.push(["Chosen bearings", a.chosen_hyp_refs]);
    if (a.rejected_hyp_refs?.length) groups.push(["Rejected bearings", a.rejected_hyp_refs]);
    if (a.violations?.length) groups.push(["Affected assets", a.violations]);
    if (a.evt_refs?.length) groups.push(["Events", a.evt_refs]);
    if (a.node_candidates?.length) groups.push(["Candidate sites", a.node_candidates]);
    for (const [h, refs] of groups) html += `<h5>${h}</h5><div class="refs">${refs.map(refChip).join("")}</div>`;
    if (a.assumptions?.length) html += `<h5>Assumptions</h5><ul>${a.assumptions.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>`;
    if (a.evidence_refs?.length) html += `<h5>Evidence</h5><div class="refs">${a.evidence_refs.map(refChip).join("")}</div>`;
  }
  const vok = v ? (v.ok ? `<span class="ok">Verifier: grounded</span>` : `<span class="bad">Verifier: ${esc(v.code || "failed")}${v.message ? `, ${esc(v.message)}` : ""}</span>`) : "Verifier: not run";
  html += `<div class="vline">${vok} · ${ev.turns} turn${ev.turns === 1 ? "" : "s"} · ended: ${esc((ev.end_reason || "").replace("_", " "))}</div>`;
  box.innerHTML = html; box.hidden = false;
  const tw = $(".trace-wrap"); tw.scrollTop = tw.scrollHeight;
  box.querySelectorAll("[data-ref]").forEach((b) => (b.onclick = () => handleRef(b.dataset.ref)));
  if (!a) return;
  if (a.intent === "df_ambiguity") {
    applyDfVerdict(a);
    if (a.position) drawAnswerPoint(a.position, a.uncertainty, "DF fix");
  } else if (a.intent === "drone_search") {
    drawAnswerPoint(a.final_position, a.uncertainty, "Located drone");
  } else if (a.intent === "operator_query") {
    const refs = [...(a.track_refs || [])];
    if (a.region?.type === "ref") refs.push(a.region.ref);
    if (refs.length) focusRefs(refs);
    if (a.region && a.region.type !== "ref") drawAnswerRegion(a.region);
  } else if (a.intent === "collateral") {
    if (a.violations?.length) focusRefs(a.violations);
    confirmGate(box, a);
  } else if (a.intent === "replay") {
    if (a.time_s != null) setSimTime(a.time_s);
  }
}

/* ---------------------------------------------------------------- tool overlays (SPEC 13.2) */
const OV = {direct: "#7be08a", reflection: "#ff9d4d", diffraction: "#b48cff", scatter: "#9aa6ab", weak: "#e8735f",
            node: "#5fd3e8", fp: "#ff9d4d", victim: "#ff5a5a", drone: "#f2d16b", wp: "#8fa6ad"};

function clearOverlay() { S.ds.overlay.entities.removeAll(); S.drone = null; $("#evtl").hidden = true; rr(); }

function posENU(x, y, z = 0) {
  const [lon, lat] = enuToLonLat(x, y);
  return C.Cartesian3.fromDegrees(lon, lat, groundAt(lon, lat) + z);
}

function ovPoint(x, y, z, color, label, size = 9) {
  return S.ds.overlay.entities.add({position: posENU(x, y, z),
    point: {pixelSize: size, color: css(color), outlineColor: css("#0b141b"), outlineWidth: 2, disableDepthTestDistance: INF},
    label: label ? {text: label, font: "11px 'IBM Plex Mono', monospace", fillColor: css(color), pixelOffset: new C.Cartesian2(0, -16),
                    showBackground: true, backgroundColor: css("#0b141b", 0.75), disableDepthTestDistance: INF,
                    distanceDisplayCondition: new C.DistanceDisplayCondition(0, 12000)} : undefined});
}

function ovRing(ring, color, alpha) {
  const flat = ring.flatMap(([x, y]) => enuToLonLat(x, y));
  return S.ds.overlay.entities.add({
    polygon: {hierarchy: C.Cartesian3.fromDegreesArray(flat), material: css(color, alpha), classificationType: C.ClassificationType.TERRAIN},
    polyline: {positions: C.Cartesian3.fromDegreesArray(flat), clampToGround: true, width: 2, material: css(color, 0.95)}});
}

function ovLine(pts, color, width = 2, dashed = false) {
  return S.ds.overlay.entities.add({polyline: {positions: pts, width, arcType: C.ArcType.NONE,
    material: dashed ? new C.PolylineDashMaterialProperty({color: css(color, 0.9), dashLength: 10}) : css(color, 0.9)}});
}

function ovEllipse(p, u, color) {
  if (!u || !u.a_m) return;
  const pts = [];
  const th = C.Math.toRadians(u.theta_deg || 0);
  for (let k = 0; k <= 40; k++) {
    const g = (k / 40) * 2 * Math.PI, a = u.a_m * Math.cos(g), b = u.b_m * Math.sin(g);
    pts.push(posENU(p.x_m + a * Math.sin(th) + b * Math.cos(th), p.y_m + a * Math.cos(th) - b * Math.sin(th), p.z_m || 0));
  }
  ovLine(pts, color, 2);
}

function drawToolOverlay(ev) {
  const d = ev.result.data || {}, ui = ev.ui || {};
  S.ds.overlay.show = true;
  switch (ev.name) {
    case "predict_aoa": {
      const anc = sensorAnchor(d.sensor, {x_m: 0, y_m: 0});
      for (const pk of d.peaks || []) {
        const col = OV[pk.kind] || OV.scatter;
        const [o, end] = rayEnd(anc, pk.bearing_deg, RAY_LEN_M * 0.7);
        const via = pk.via_ref ? " via " + pk.via_ref.replace("obj:", "") : "";
        S.ds.overlay.entities.add({position: end,
          polyline: {positions: [o, end], width: 2, arcType: C.ArcType.NONE,
                     material: new C.PolylineDashMaterialProperty({color: css(col, 0.95), dashLength: 6})},
          label: {text: pk.kind + via + " " + pk.rel_power_db + " dB", font: "10px 'IBM Plex Mono', monospace",
                  fillColor: css(col), showBackground: true, backgroundColor: css("#0b141b", 0.7),
                  disableDepthTestDistance: INF, distanceDisplayCondition: new C.DistanceDisplayCondition(0, 9000)}});
        if (pk.via_ref && ui.vias && ui.vias[pk.via_ref]) {     // reflection geometry: sensor -> reflector -> emitter
          const [vx, vy] = ui.vias[pk.via_ref];
          ovLine([C.Cartesian3.fromDegrees(anc.lon, anc.lat, anc.h), posENU(vx, vy, 15),
                  posENU(d.point.x_m, d.point.y_m, d.point.z_m || 60)], col, 1.5, true);
        }
      }
      ovPoint(d.point.x_m, d.point.y_m, d.point.z_m || 60, OV.drone, "test point", 8);
      return true;
    }
    case "fit_candidates": {
      const cands = d.candidates || [];
      cands.forEach((c, i) => {
        const col = i === 0 ? OV.drone : OV.wp;
        const r = c.residual_deg_rms != null ? c.residual_deg_rms + " deg rms" : c.residual_m_rms + " m rms";
        ovPoint(c.position.x_m, c.position.y_m, c.position.z_m || 0, col, i === 0 ? "best fit · " + r : null, i === 0 ? 11 : 6);
        if (i === 0) ovEllipse(c.position, c.uncertainty, col);
      });
      const top = cands[0];
      if (top) {
        const verdict = new Map(top.hyps.map((h) => [baseRef(h.ref), h.explained_as]));
        for (const [, rec] of S.rays) {
          const ex = verdict.get(baseRef(rec.data.ref));
          if (ex) styleRay(rec, ex === "direct" ? "chosen" : "rejected");
        }
        flyToPoints([posENU(top.position.x_m, top.position.y_m)], {minRange: 1500});
      }
      return true;
    }
    case "weak_zones": {
      for (const z of ui.zones || []) for (const r of z.rings) ovRing(r, OV.weak, 0.35);
      const zs = d.zones || [];
      zs.forEach((z) => ovPoint(z.centroid.x_m, z.centroid.y_m, 0, OV.weak,
        "weak · worst " + z.worst + " · " + (z.area_m2 / 1e4).toFixed(1) + " ha", 5));
      if (zs.length) flyToPoints(zs.map((z) => posENU(z.centroid.x_m, z.centroid.y_m)), {minRange: 1800, pitch: -1.2});
      return true;
    }
    case "suggest_node": {
      const pts = [];
      for (const c of d.candidates || []) {
        const p = c.position;
        ovLine([posENU(p.x_m, p.y_m, 0), posENU(p.x_m, p.y_m, (p.z_m || 20) + 10)], OV.node, 4);
        ovPoint(p.x_m, p.y_m, (p.z_m || 20) + 10, OV.node, "new node @ " + c.ref.replace("nodecand:", "") + " · gain " + c.predicted_gain, 10);
        pts.push(posENU(p.x_m, p.y_m));
      }
      if (pts.length) flyToPoints(pts, {minRange: 1500});
      return true;
    }
    case "effect_footprint": {
      for (const r of ui.rings || []) ovRing(r, OV.fp, 0.22);
      if (ui.effector) ovPoint(ui.effector[0], ui.effector[1], 20, OV.fp, "effector " + d.eirp_dbm + " dBm · " + d.band, 12);
      if (d.bbox) flyToPoints([posENU(d.bbox[0], d.bbox[1]), posENU(d.bbox[2], d.bbox[3])], {pitch: -1.2});
      return true;
    }
    case "collateral_check":
      for (const v of ui.victims || []) {
        if (!v.pos) continue;
        ovPoint(v.pos[0], v.pos[1], 25, OV.victim, v.ref.split(":")[1] + " · " + v.level_dbm + " dBm · " + v.severity,
                v.severity === "high" ? 13 : 10);
      }
      return true;
    case "plan_waypoints":
      for (const w of d.waypoints || []) ovPoint(w.position.x_m, w.position.y_m, w.position.z_m || 80, OV.wp, w.ref + " → " + w.expected_a_m + " m", 6);
      return true;
    case "drone_state":
    case "move_drone": {
      const p = d.pose, e = d.estimate;
      S.drone = S.drone || [];
      if (p) {
        S.drone.push(posENU(p.x_m, p.y_m, p.z_m || 80));
        const lab = ev.name === "move_drone" ? "D1 @ " + d.wp + " · bearing " + (d.measurement ? d.measurement.bearing_deg : "?") + "°" : "D1 start";
        ovPoint(p.x_m, p.y_m, p.z_m || 80, OV.drone, lab, 9);
        if (S.drone.length > 1) ovLine(S.drone.slice(-2), OV.drone, 3);
      }
      if (e) ovEllipse(e.position, e.uncertainty, ev.name === "move_drone" ? OV.drone : OV.wp);
      if (p && e) flyToPoints([posENU(p.x_m, p.y_m), posENU(e.position.x_m, e.position.y_m)], {minRange: 1200});
      return true;
    }
    case "event_search":
      eventTimeline(d.items || []);
      return false;            // also highlight the tracks and zones it returned
    default:
      return false;
  }
}

function eventTimeline(items) {
  const el = $("#evtl");
  if (!items.length) { el.hidden = true; return; }
  const W = 100 / Math.max(1, S.dur);
  const dots = items.map((it) => '<button class="evt ' + esc(it.type) + '" style="left:' + (it.t_s * W).toFixed(2) + '%" data-t="' +
    it.t_s + '" data-trk="' + esc(it.track_ref) + '" title="' + esc(it.type + " " + it.track_ref + " " + (it.zone_ref || "") + " t=" + it.t_s + " s") + '"></button>').join("");
  const rows = items.map((it) => '<button class="evtrow" data-t="' + it.t_s + '" data-trk="' + esc(it.track_ref) + '"><b>t=' +
    it.t_s + "s</b> " + esc(it.type.replace("_", " ")) + " " + esc(it.track_ref) + " " + esc((it.zone_ref || "").replace("zone:", "")) + "</button>").join("");
  el.innerHTML = '<div class="evtl-bar">' + dots + '</div><div class="evtl-list">' + rows + "</div>";
  el.hidden = false;
  el.querySelectorAll("[data-t]").forEach((b) => (b.onclick = () => { setSimTime(Number(b.dataset.t)); focusRefs([b.dataset.trk]); }));
}

function confirmGate(box, a) {
  const div = document.createElement("div");
  div.className = "gate";
  div.innerHTML = '<div class="gate-h">Agent recommends, operator decides</div>' +
    "<p>This is a predicted assessment only. No tool can switch on any effector.</p>" +
    '<div class="gate-b"><button data-d="ack">Acknowledge assessment</button><button data-d="reject">Reject</button></div>' +
    '<div class="gate-r" hidden></div>';
  box.appendChild(div);
  div.querySelectorAll("button").forEach((b) => (b.onclick = () => {
    const r = div.querySelector(".gate-r");
    r.hidden = false;
    r.textContent = (b.dataset.d === "ack" ? "Acknowledged" : "Rejected") + " by the operator at " +
      new Date().toLocaleTimeString() + " (logged in this browser only).";
    div.querySelectorAll("button").forEach((x) => (x.disabled = true));
  }));
  void a;
}

boot().catch((e) => {
  console.error(e);
  $("#boot").textContent = `Could not load the console: ${e.message}`;
});
