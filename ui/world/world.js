/* SceneScout World: photoreal Kandla (Google Photorealistic 3D Tiles via Cesium ion) with the simulated drones,
   swarms, DF sensors and zones in the site ENU frame. World-side viewer (served by scripts/world_server.py). */
"use strict";

const C = window.Cesium;
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const HOSTILE = "#ffb224", FRIENDLY = "#60e28c", SYN = "#e8735f", OSM = "#7cc99a";
const GOOGLE_3D = 2275207, BING = 2;

const QUALITY = {   // world-brief starting values; tuned on the reference machine
  performance: {res: 0.7, tiles: 24, globe: 4, msaa: 1},
  balanced: {res: 0.85, tiles: 12, globe: 2, msaa: 4},
  cinematic: {res: 1.0, tiles: 4, globe: 1, msaa: 8},
};
const SHOT_TEXT = {
  overview: ["Deendayal Port, Kandla Creek", "Gulf of Kutch, Gujarat"],
  jetties: ["Oil jetties", "berths J1-J5 on Kandla Creek"],
  tankfarms: ["Tank farms", "263 storage tanks in 8 farms (OSM)"],
  chase1: ["Swarm SW-1", "8 UAS, V formation, inbound from the north-east"],
  chase2: ["Swarm SW-2", "5 UAS, line abreast, westbound over the tank farms"],
  nodeS1: ["DF node S1, jetty head", "AIR7311, 4 coherent channels, 2x2 omni array (synthetic site)"],
  free: ["Free flight", "mouse: rotate, pan, zoom"],
};

const S = {
  viewer: null, photoreal: null, enuM: null, h0: 0, world: null, drones: {}, ents: {}, t: 300, playing: true,
  speed: 4, shot: "overview", camEye: null, camTgt: null, frameTimes: [], last: performance.now(), layers: {},
};

/* ------------------------------------------------------------------------------------------- frame helpers */
function enu(x, y, z = 0) {
  return C.Matrix4.multiplyByPoint(S.enuM, new C.Cartesian3(x, y, z + S.h0), new C.Cartesian3());
}
function setEnuOrigin(h0) {
  const o = S.world.origin;
  S.h0 = h0;
  S.enuM = C.Transforms.eastNorthUpToFixedFrame(C.Cartesian3.fromDegrees(o.lon, o.lat, 0));
}
const v3 = (a) => new C.Cartesian3(a[0], a[1], a[2]);

/* ------------------------------------------------------------------------------------------- trajectories */
function catmull(T, P, t) {
  if (t < T[0] || t > T[T.length - 1]) return null;
  let i = 0, lo = 0, hi = T.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
  i = Math.min(lo, T.length - 2);
  const p0 = P[Math.max(i - 1, 0)], p1 = P[i], p2 = P[i + 1], p3 = P[Math.min(i + 2, T.length - 1)];
  const u = (t - T[i]) / (T[i + 1] - T[i]);
  return [0, 1, 2].map((k) => 0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * u + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * u * u
    + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * u * u * u));
}
const pos = (id, t) => { const d = S.drones[id]; return d ? catmull(d.t, d.xyz, t) : null; };
function vel(id, t, dt = 1.0) {
  const a = pos(id, t - dt), b = pos(id, t + dt);
  return a && b ? [(b[0] - a[0]) / (2 * dt), (b[1] - a[1]) / (2 * dt), (b[2] - a[2]) / (2 * dt)] : [0, 0, 0];
}
function centroid(swarm, t) {
  const ps = Object.keys(S.drones).filter((k) => S.drones[k].swarm_id === swarm).map((k) => pos(k, t)).filter(Boolean);
  if (!ps.length) return null;
  return [0, 1, 2].map((k) => ps.reduce((s, p) => s + p[k], 0) / ps.length);
}
function swarmVel(swarm, t) {
  const a = centroid(swarm, t - 2), b = centroid(swarm, t + 2);
  return a && b ? [(b[0] - a[0]) / 4, (b[1] - a[1]) / 4, (b[2] - a[2]) / 4] : [1, 0, 0];
}

/* ------------------------------------------------------------------------------------------- layers */
function addZones() {
  const ents = [];
  for (const z of S.world.zones) {
    if (!(z.kind === "restricted" || z.id === "perimeter")) continue;
    const ring = z.ring.map((p) => enu(p[0], p[1], 0));
    const col = z.kind === "restricted" ? C.Color.fromCssColorString(SYN) : C.Color.fromCssColorString("#dfe7e4");
    ents.push(S.viewer.entities.add({
      name: z.name,
      polygon: {hierarchy: ring, material: col.withAlpha(z.kind === "restricted" ? 0.07 : 0.0),
                classificationType: C.ClassificationType.BOTH},
      polyline: {positions: ring.concat([ring[0]]), width: 2, clampToGround: true,
                 material: new C.PolylineDashMaterialProperty({color: col.withAlpha(0.85), dashLength: 18}),
                 classificationType: C.ClassificationType.BOTH},
    }));
  }
  S.layers.zones = ents;
}
function addTwin() {
  const ents = [];
  for (const o of S.world.objects) {
    if (!o.outer || o.kind === "tank_farm") continue;
    const ring = o.outer.map((p) => enu(p[0], p[1], 0));
    ents.push(S.viewer.entities.add({
      name: o.name, show: false,
      polyline: {positions: ring.concat([ring[0]]), width: 1.5, clampToGround: true,
                 material: C.Color.fromCssColorString(OSM).withAlpha(0.9), classificationType: C.ClassificationType.BOTH},
    }));
  }
  S.layers.twin = ents;
}
function addSensors() {
  const ents = [];
  for (const s of S.world.sensors) {
    if (s.type !== "fixed_df") continue;
    const [x, y, z] = s.xyz;
    const base = Math.max(0, z - 6);
    ents.push(S.viewer.entities.add({
      name: `DF sensor ${s.id}`,
      polyline: {positions: [enu(x, y, base), enu(x, y, z)], width: 4, material: C.Color.fromCssColorString("#cfd6d9")},
    }));
    ents.push(S.viewer.entities.add({
      position: enu(x, y, z + 0.6),
      point: {pixelSize: 9, color: C.Color.fromCssColorString(SYN), outlineColor: C.Color.BLACK, outlineWidth: 1,
              disableDepthTestDistance: 2000},
      label: {text: `${s.id}  AIR7311`, font: "500 13px IBM Plex Mono", fillColor: C.Color.WHITE,
              showBackground: true, backgroundColor: C.Color.fromCssColorString("#0b1013").withAlpha(0.7),
              pixelOffset: new C.Cartesian2(0, -18), distanceDisplayCondition: new C.DistanceDisplayCondition(0, 6000),
              disableDepthTestDistance: 2000},
    }));
  }
  S.layers.sensors = ents;
}
function addDrones() {
  const ents = [];
  for (const [id, d] of Object.entries(S.drones)) {
    if (d.kind === "decoy") continue;
    const friendly = d.kind === "friendly_drone";
    const positionCb = (time, result) => {
      const p = pos(id, S.t);
      return p ? C.Matrix4.multiplyByPoint(S.enuM, new C.Cartesian3(p[0], p[1], p[2] + S.h0), result || new C.Cartesian3()) : undefined;
    };
    const posProp = C.CallbackPositionProperty ? new C.CallbackPositionProperty(positionCb, false) : new C.CallbackProperty(positionCb, false);
    const ori = new C.CallbackProperty(() => {
      const p = pos(id, S.t);
      if (!p) return undefined;
      const v = vel(id, S.t);
      const sp = Math.hypot(v[0], v[1]);
      const yaw = sp > 0.3 ? Math.atan2(v[1], v[0]) : 0;
      const hpr = new C.HeadingPitchRoll(-yaw, -Math.min(0.25, sp * 0.015), 0);
      return C.Transforms.headingPitchRollQuaternion(enu(p[0], p[1], p[2]), hpr);
    }, false);
    const e = S.viewer.entities.add({
      id: `drone:${id}`, name: friendly ? "Friendly patrol" : (d.swarm_id ? `${d.swarm_id} member` : "UAS"),
      position: posProp, orientation: ori,
      model: {uri: friendly ? "assets/drone_friendly.glb" : "assets/drone.glb", minimumPixelSize: 30, maximumScale: 400,
              silhouetteColor: C.Color.fromCssColorString(friendly ? FRIENDLY : HOSTILE).withAlpha(0.55), silhouetteSize: 1.0},
    });
    e._meta = {id, friendly, swarm: d.swarm_id};
    ents.push(e);
  }
  S.layers.drones = ents;
}

/* ------------------------------------------------------------------------------------------- camera */
function look(eye, tgt) {
  const E = enu(...eye), T = enu(...tgt);
  const dir = C.Cartesian3.normalize(C.Cartesian3.subtract(T, E, new C.Cartesian3()), new C.Cartesian3());
  const upG = C.Ellipsoid.WGS84.geodeticSurfaceNormal(E, new C.Cartesian3());
  const right = C.Cartesian3.normalize(C.Cartesian3.cross(dir, upG, new C.Cartesian3()), new C.Cartesian3());
  const up = C.Cartesian3.cross(right, dir, new C.Cartesian3());
  return {destination: E, orientation: {direction: dir, up}};
}
const STATIC = {
  overview: [[-2300, -2600, 1150], [450, 250, 0]],
  jetties: [[1650, -650, 160], [930, 150, 10]],
  tankfarms: [[-900, -1150, 260], [-150, -200, 0]],
};
function setShot(name) {
  S.shot = name;
  $$(".shots button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.shot === name)));
  const [a, b] = SHOT_TEXT[name] || ["", ""];
  $("h1").textContent = a;
  $("#shotLine").textContent = b;
  const ctl = S.viewer.scene.screenSpaceCameraController;
  ctl.enableInputs = name === "free" || name in STATIC;
  S.camEye = null;
  const active = {chase1: [140, 600, 330], chase2: [270, 600, 540]}[name];   // jump into the swarm's flight window
  if (active && (S.t < active[0] || S.t > active[1])) S.t = active[2];
  if (name in STATIC) {
    const v = look(...STATIC[name]);
    S.viewer.camera.flyTo({...v, duration: 2.5});
  }
}
function lerp(a, b, k) { return a.map((v, i) => v + (b[i] - v) * k); }
function dynamicCamera() {
  let eye, tgt;
  if (S.shot === "chase1" || S.shot === "chase2") {
    const sw = S.shot === "chase1" ? "swarm_1" : "swarm_2";
    const hero = S.shot === "chase1" ? "swarm_1_m6" : "swarm_2_m3";
    const p = pos(hero, S.t) || centroid(sw, S.t);
    if (!p) return;
    let v = vel(hero, S.t, 2.0);
    if (Math.hypot(v[0], v[1]) < 0.5) v = swarmVel(sw, S.t);
    const n = Math.hypot(v[0], v[1]) || 1, d = [v[0] / n, v[1] / n, 0], side = [-d[1], d[0], 0];
    eye = [p[0] - d[0] * 16 + side[0] * 4, p[1] - d[1] * 16 + side[1] * 4, p[2] + 2.5];
    tgt = [p[0] + d[0] * 40, p[1] + d[1] * 40, p[2] - 6];
  } else if (S.shot === "nodeS1") {
    const s = S.world.sensors.find((x) => x.id === "S1").xyz;
    const a = (performance.now() / 1000) * 0.12;
    eye = [s[0] + 60 * Math.cos(a), s[1] + 60 * Math.sin(a), s[2] + 14];
    tgt = [s[0], s[1], s[2] - 8];
  } else return;
  S.camEye = S.camEye ? lerp(S.camEye, eye, 0.12) : eye;
  S.camTgt = S.camTgt ? lerp(S.camTgt, tgt, 0.12) : tgt;
  S.viewer.camera.setView(look(S.camEye, S.camTgt));
}

/* ------------------------------------------------------------------------------------------- HUD */
function drawHud() {
  const cv = $("#hud"), dpr = window.devicePixelRatio || 1;
  if (cv.width !== innerWidth * dpr) { cv.width = innerWidth * dpr; cv.height = innerHeight * dpr; }
  const g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  if (!S.layers.hudOn || !S.layers.dronesOn) return;
  const camPos = S.viewer.camera.positionWC;
  g.lineWidth = 2 * dpr;
  g.font = `${12 * dpr}px "IBM Plex Mono", monospace`;
  for (const e of S.layers.drones || []) {
    const p = e.position.getValue(S.viewer.clock.currentTime);
    if (!p) continue;
    const w = C.SceneTransforms.worldToWindowCoordinates(S.viewer.scene, p);
    if (!w) continue;
    const r3 = C.Cartesian3.distance(p, camPos);
    if (r3 > 6000) continue;
    const x = w.x * dpr, y = w.y * dpr, r = Math.max(10, Math.min(70, 1500 / r3)) * dpr, k = r * 0.45;
    const col = e._meta.friendly ? FRIENDLY : HOSTILE;
    g.strokeStyle = col; g.fillStyle = col;
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
      g.beginPath(); g.moveTo(x + sx * r - sx * k, y + sy * r); g.lineTo(x + sx * r, y + sy * r); g.lineTo(x + sx * r, y + sy * r - sy * k); g.stroke();
    }
    if (r3 < 1500) {
      const m = e._meta;
      const lab = m.friendly ? "FRIENDLY" : m.swarm ? `${m.swarm.replace("swarm_", "SW-")}.${m.id.split("_m").pop()}` : `UAS-${m.id.split("_").pop().toUpperCase()}`;
      const alt = pos(m.id, S.t)[2];
      g.fillText(`${lab}  ALT ${alt.toFixed(0)} m  RNG ${r3.toFixed(0)} m`, x - r, y - r - 6 * dpr);
    }
  }
}

/* ------------------------------------------------------------------------------------------- quality + stats */
function setQuality(name) {
  const q = QUALITY[name], v = S.viewer;
  v.useBrowserRecommendedResolution = false;
  v.resolutionScale = q.res * (window.devicePixelRatio || 1);
  v.scene.globe.maximumScreenSpaceError = q.globe;
  if (S.photoreal) S.photoreal.maximumScreenSpaceError = q.tiles;
  try { v.scene.msaaSamples = q.msaa; } catch (e) { /* WebGL1 */ }
  $$(".presets button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.q === name)));
  S.quality = name;
}
function stats() {
  const now = performance.now(), dt = now - S.last;
  S.last = now;
  S.frameTimes.push(dt);
  if (S.frameTimes.length > 240) S.frameTimes.shift();
  if (S.frameTimes.length % 15 !== 0) return;
  const s = [...S.frameTimes].sort((a, b) => a - b), pct = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  $("#fps").textContent = `${(1000 / pct(0.5)).toFixed(0)} fps`;
  $("#ft").textContent = `p50 ${pct(0.5).toFixed(1)} ms, p95 ${pct(0.95).toFixed(1)} ms`;
  const c = S.viewer.scene.canvas;
  $("#res").textContent = `${c.width}x${c.height}`;
  $("#tiles").textContent = S.photoreal ? (S.photoreal.tilesLoaded ? "tiles: loaded" : "tiles: streaming") : "tiles: --";
}
function clockText(t) { return `T+${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`; }

/* ------------------------------------------------------------------------------------------- boot */
async function boot() {
  const [tok, world, drones] = await Promise.all([
    fetch("/api/cesium_token").then((r) => r.json()), fetch("/api/world").then((r) => r.json()),
    fetch("/api/drones").then((r) => r.json())]);
  C.Ion.defaultAccessToken = tok.token;
  S.world = world;
  S.drones = drones;
  const v = new C.Viewer("cesium", {
    animation: false, timeline: false, baseLayerPicker: false, geocoder: false, homeButton: false, sceneModePicker: false,
    navigationHelpButton: false, fullscreenButton: false, infoBox: false, selectionIndicator: false,
    baseLayer: C.ImageryLayer.fromProviderAsync(C.IonImageryProvider.fromAssetId(BING)),
    terrain: C.Terrain.fromWorldTerrain(), msaaSamples: 4,
    contextOptions: {webgl: {preserveDrawingBuffer: true}},
  });
  S.viewer = v;
  v.scene.globe.show = false;
  v.scene.fog.enabled = true;
  v.scene.skyAtmosphere.show = true;
  try {
    S.photoreal = await C.Cesium3DTileset.fromIonAssetId(GOOGLE_3D, {maximumScreenSpaceError: 12, showCreditsOnScreen: true});
    v.scene.primitives.add(S.photoreal);
  } catch (e) {
    console.warn("Google photoreal tiles unavailable; falling back to aerial imagery and terrain", e);
    v.scene.globe.show = true;
    $('[data-layer="photoreal"]').checked = false;
    $('[data-layer="aerial"]').checked = true;
  }
  setEnuOrigin(0);
  v.camera.setView(look(...STATIC.overview));
  // ground height at the site origin from the photoreal mesh (fallback: terrain), so ENU z = 0 sits on the quay
  try {
    await new Promise((res) => { const f = () => (S.photoreal && !S.photoreal.tilesLoaded ? setTimeout(f, 300) : res()); f(); });
    const o = world.origin;
    const [c] = await v.scene.sampleHeightMostDetailed([C.Cartographic.fromDegrees(o.lon, o.lat)]);
    if (c && Number.isFinite(c.height)) setEnuOrigin(c.height);
  } catch (e) { console.warn("height sampling failed; using ellipsoid 0", e); }
  addZones(); addTwin(); addSensors(); addDrones();
  S.layers.hudOn = true; S.layers.dronesOn = true;
  setQuality("balanced");
  setShot("overview");
  v.scene.preRender.addEventListener(() => {
    const now = performance.now();
    if (S.playing && !S.capture) {
      S.t += ((now - (S.tick || now)) / 1000) * S.speed;
      if (S.t > 600) S.t = 0;
    }
    S.tick = now;
    $("#time").value = S.t;
    $("#clock").textContent = clockText(S.t);
    dynamicCamera();
  });
  v.scene.postRender.addEventListener(() => { drawHud(); stats(); });
  wire();
  $("#boot").classList.add("done");
}

function wire() {
  $$(".presets button").forEach((b) => b.addEventListener("click", () => setQuality(b.dataset.q)));
  $$(".shots button").forEach((b) => b.addEventListener("click", () => setShot(b.dataset.shot)));
  $("#play").addEventListener("click", () => { S.playing = !S.playing; $("#play").textContent = S.playing ? "Pause" : "Play"; });
  $("#speed").addEventListener("change", (e) => { S.speed = Number(e.target.value); });
  $("#time").addEventListener("input", (e) => { S.t = Number(e.target.value); });
  $$(".layers input").forEach((i) => i.addEventListener("change", () => {
    const on = i.checked, k = i.dataset.layer, sc = S.viewer.scene;
    if (k === "photoreal" && S.photoreal) { S.photoreal.show = on; sc.globe.show = !on || $('[data-layer="aerial"]').checked; }
    else if (k === "aerial") sc.globe.show = on || !S.photoreal || !S.photoreal.show;
    else if (k === "hud") S.layers.hudOn = on;
    else { (S.layers[k] || []).forEach((e) => { e.show = on; }); if (k === "drones") S.layers.dronesOn = on; }
  }));
}

/* capture hooks for the frame-exact 4K recorder (scripts/world_capture.py) */
window.SceneScout = {
  setTime(t) { S.t = t; }, setShot, pause(on = true) { S.playing = !on; }, setQuality, state: () => ({t: S.t, shot: S.shot}),
  setCapture(on) { S.capture = on; document.body.classList.toggle("capture", on); },
  tilesLoaded: () => !S.photoreal || S.photoreal.tilesLoaded,
  debug: () => S,
  lookAt(eye, tgt) { setShot("free"); S.viewer.camera.setView(look(eye, tgt)); },
  layer(name, on) { const i = document.querySelector(`[data-layer="${name}"]`); if (i) { i.checked = on; i.dispatchEvent(new Event("change")); } },
  ready: () => Boolean(S.layers.drones),
};

boot().catch((e) => { console.error(e); $("#boot").textContent = `Could not start: ${e.message}`; });
