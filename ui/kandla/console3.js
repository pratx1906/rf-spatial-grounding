/* SceneScout console on the three.js engine (ss3d.js): the Blender world rendered live, the agent's live work as
   numbered steps whose marks are stamped in 3D, a plain-language verdict, an honest confidence ruler, the operator
   gate and the recorded comparison with the untrained model (docs/design/FINAL.md). Belief-side data only. */
import {World} from "/ui/ss3d.js";
import {C, LEGEND, Beacons, area, frame} from "/ui/marks.js";
import {Drones} from "/ui/drones.js";
import {FreeCam} from "/ui/camera.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const T = {void: "#000000", ash: "#9a958c", bone: "#ece8e1", sodium: "#f2a33a", signal: "#ff5a1f", friendly: "#7db0cc"};
const RAY_LEN_M = 3000, REVIEW_MARK = 0.6, MAX_TURNS = 8;

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
  pilot_search: "Search for pilots", tdoa_fix: "Fix by AoA and TDOA", signal_detail: "Read signal detail",
  noise_floor: "Measure noise floor",
};
const shortRef = (r) => String(r ?? "").replace(/^(sensor|trk|zone|obj|fe|wp|nodecand|cand|hyp|fp|evt):/, "");
const tRef = (r) => { const [b, at] = String(r ?? "").split("@"); return at != null ? `${b} at t ${at} s` : b; };
/* the "Now" line: what the agent is doing, present tense, from the tool call's arguments */
const TOOL_NOW = {
  site_resolve: (a) => `Finding "${a.query || "the place"}" in the port model`,
  site_describe: (a) => `Describing ${shortRef(a.ref) || "the place"}`,
  region_contents: () => "Listing what is in the area",
  spatial_relation: () => "Measuring distances", inside_zone: () => "Checking zone membership",
  track_search: () => "Searching the track picture", track_detail: (a) => `Reading track ${tRef(a.track_ref) || ""}`.trim(),
  df_hypotheses: (a) => `Reading DF bearings on ${a.track_ref || "the track"}${a.at_time_s != null ? ` at t ${a.at_time_s} s` : ""}`,
  predict_aoa: (a) => `Predicting multipath at ${shortRef(a.sensor_ref) || "the sensor"} from the RF model`,
  geometry_check: () => "Checking the bearing geometry",
  fit_candidates: (a) => `Fitting the emitter position (${a.method === "rt_match" ? "ray-traced" : "least squares"})`,
  prop_query: (a) => `Querying propagation from ${shortRef(a.sensor_ref) || "the sensor"}`,
  coverage_query: () => "Querying DF coverage", weak_zones: () => "Finding weak DF coverage",
  suggest_node: () => "Searching roofs and jetties for a new DF node", model_trust: () => "Checking how far the RF model can be trusted",
  drone_state: () => "Reading the DF drone's state", plan_waypoints: () => "Planning DF drone waypoints",
  move_drone: (a) => `Flying the DF drone to ${shortRef(a.wp_ref) || "the next waypoint"}`,
  effect_footprint: (a) => `Predicting the jamming footprint from ${shortRef(typeof a.effector === "string" ? a.effector : "") || "the effector"}`,
  collateral_check: () => "Checking what the footprint would hit", event_search: () => "Searching the event log",
  track_history: (a) => `Reading the history of ${a.track_ref || "the track"}`, sql_readonly: () => "Querying the event log",
  declare_intent: () => "Stating the plan", ask_user: () => "Asking the operator", submit: () => "Writing the answer",
  pilot_search: (a) => `Searching for pilots near ${a.near_track_ref || "the drone"}`,
  tdoa_fix: (a) => `Fixing ${a.track_ref || "the track"} by AoA and TDOA${a.at_time_s != null ? ` at t ${a.at_time_s} s` : ""}`,
  signal_detail: (a) => `Reading the signal of ${a.track_ref || "the track"}`,
  noise_floor: (a) => `Measuring the noise floor at ${shortRef(a.sensor_ref) || "the sensor"}`,
};
const nowFor = (name, args) => { try { return (TOOL_NOW[name] || (() => TOOL_LABEL[name] || "Running a tool"))(args || {}); } catch (_) { return TOOL_LABEL[name] || "Running a tool"; } };
const INTENT_PLAN = {
  operator_query: "answer from the track picture", df_ambiguity: "separate the direct path from reflections",
  drone_search: "fly the DF drone to fix the emitter", siting: "find weak coverage and a new node site",
  collateral: "predict a jamming footprint and what it would hit", replay: "read the event log",
  compound: "answer in several steps", out_of_scope: "refuse, outside SceneScout's scope",
};
const FIT_METHOD = {rt_match: "Ray-traced fit", triangulate: "Least-squares fit", ls: "Least-squares fit"};
/* model assumptions with an operator meaning; anything else stays in the work trace only */
const ASSUME = [
  [/genuine\s*=\s*argmax drone.*confidence\s*>=\s*0\.6/i, "Treated as a real UAV because the drone class is most likely and track confidence is at least 0.6."],
  [/drone with confidence\s*>=\s*0\.6/i, "Counted only tracks classed as drones with track confidence of at least 0.6."],
  [/line-of-sight proxy/i, "New-node coverage is estimated from line of sight, not a full ray trace."],
];
const ERR_COPY = {
  llm_unreachable: ["Model server not reachable", "SceneScout could not reach the language model at {t}. Nothing was assessed."],
  llm_timeout: ["Model server did not answer in time", "The language model stopped responding at {t}. Nothing was assessed."],
  internal: ["The assessment stopped on a server fault", "The gateway hit an internal fault at {t}. Nothing was assessed."],
  wallclock: ["Ran out of time", "The question used its whole time budget at {t} without a checked answer."],
  format_retry: ["The model's answer could not be read", "The model did not produce a readable answer at {t}. Nothing was assessed."],
  busy: ["A question is still running", "This console already has a question running at {t}. Stop it (Esc) or wait for its answer."],
};
function errorCopy(ev) {
  const t = new Date().toLocaleTimeString("en-GB", {hour: "2-digit", minute: "2-digit"});
  const m = String(ev.message || "");
  const code = ev.code || (/APIConnection|Connection ?(error|refused)|ECONNREFUSED/i.test(m) ? "llm_unreachable"
                          : /Timeout/i.test(m) ? "llm_timeout" : "internal");
  const [head, why] = ERR_COPY[code] || ERR_COPY.internal;
  return [head, why.replace("{t}", t)];
}
const ABSTAIN = {
  no_sensor_coverage: "No sensor covers that place or time, so there is nothing to base an answer on.",
  entity_not_found: "Nothing with that name exists in the port twin.",
  ambiguous_unresolved: "The question stayed ambiguous after checking; it needs more detail.",
  outside_tool_scope: "SceneScout assesses; it cannot spoof, jam or take over a drone. Effector actions stay with the operator under the port's rules of engagement.",
  insufficient_evidence: "The evidence is not strong enough to give an answer.",
};
const END_REASON = {
  max_turns: "No answer within 8 turns", format_failed: "The answer did not follow the format",
  verifier_failed: "The answer cited evidence it never retrieved", timeout: "Ran out of time", submitted: "Answered",
  wallclock: "Ran out of time", llm_error: "The model server failed",
};
const BAND = {sub_ghz_868_915: "868/915 MHz", ism_2400: "2.4 GHz", ism_5800: "5.8 GHz", ism_2g4: "2.4 GHz", ism_5g8: "5.8 GHz"};
const band = (b) => BAND[b] || String(b || "").replace(/_/g, " ");

const S = {
  W: null, site: null, meta: null, tracks: [], dur: 600, origin: null,
  recs: new Map(), rays: new Map(), sensors: new Map(), groups: new Map(), highlighted: [],
  demo: [], replays: {}, es: null, pending: [], finished: true, t0: 0, taskId: null,
  step: 0, stepObjs: new Map(), ep: null, camMoves: 0, prevView: null, gateOpen: false, tick: null, epId: 0,
  ansObjs: [], legend: new Set(), follow: true, drones: false,
};

/* ------------------------------------------------------------------ geodesy (site ENU, WGS84 local) */
function lonLatToEnu(lon, lat) {
  const a = 6378137, e2 = 0.00669437999014, o = S.origin;
  const p = (o.lat * Math.PI) / 180, s = Math.sin(p);
  const N = a / Math.sqrt(1 - e2 * s * s), M = (a * (1 - e2)) / Math.pow(1 - e2 * s * s, 1.5);
  return [((lon - o.lon) * Math.PI / 180) * N * Math.cos(p), ((lat - o.lat) * Math.PI / 180) * M];
}
const ringEnu = (poly) => poly.map(([lon, lat]) => lonLatToEnu(lon, lat));
const baseRef = (ref) => String(ref).split("@")[0];
const fmtM = (m) => (m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`);
const fmtDeg = (d) => `${(((d % 360) + 360) % 360).toFixed(1).padStart(5, "0")}°`;
const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];
function inRing(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function ringDist(x, y, o) {
  if (o.ring.length > 2 && inRing(x, y, o.ring)) return 0;
  let d = Math.hypot(x - o.enu[0], y - o.enu[1]);
  for (let i = 1; i < o.ring.length; i++) {
    const [ax, ay] = o.ring[i - 1], [bx, by] = o.ring[i], vx = bx - ax, vy = by - ay, L = vx * vx + vy * vy || 1;
    const k = Math.max(0, Math.min(1, ((x - ax) * vx + (y - ay) * vy) / L));
    d = Math.min(d, Math.hypot(x - ax - k * vx, y - ay - k * vy));
  }
  return d;
}
const dirOf = (dx, dy) => COMPASS[Math.round(((Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360) / 45) % 8];
const PLACE_KINDS = ["jetty", "tank_farm", "substation", "silo", "building", "ship", "tank"];
/* a place sentence for a site ENU point: "on Jetty J1, 64 m south of sensor S1" or "320 m north-east of TF-2".
   Synthetic labels are used only when close (< 300 m) and marked "(label)"; sensors are landmarks too. */
function nearestPlace(x, y) {
  let on = null, near = null;
  for (const o of S.objs) {
    if (!PLACE_KINDS.includes(o.kind) || !o.name) continue;
    const d = ringDist(x, y, o), synth = o.synthetic;
    if (d === 0 && !["tank", "ship"].includes(o.kind) && (!on || o.area < on.o.area)) on = {o, d};
    if (synth && (d > 300 || o.kind === "ship")) continue;           // synthetic ships move; never a landmark
    if (!near || d < near.d) near = {o, d};
  }
  let sen = null;
  for (const s of S.site.sensors) {
    if (s.type === "drone_df" || !s.enu) continue;
    const d = Math.hypot(x - s.enu[0], y - s.enu[1]);
    if (!sen || d < sen.d) sen = {s, d};
  }
  const nm = (o) => `${o.name}${o.synthetic ? " (label)" : ""}`;
  const onTxt = on ? `on ${nm(on.o)}` : "";
  if (sen && sen.d < 300 && (!near || sen.d < near.d || on)) {
    const rel = sen.d < 15 ? `at sensor ${sen.s.id}` : `${fmtM(sen.d)} ${dirOf(x - sen.s.enu[0], y - sen.s.enu[1])} of sensor ${sen.s.id}`;
    return onTxt ? `${onTxt}, ${rel}` : rel;
  }
  if (on) return onTxt;
  if (!near) return `${Math.round(x)} m east, ${Math.round(y)} m north of the site origin`;
  if (near.d < 40) return `at ${nm(near.o)}`;
  return `${fmtM(near.d)} ${dirOf(x - near.o.enu[0], y - near.o.enu[1])} of ${nm(near.o)}`;
}
const wait = (ms, real = false) => new Promise((r) => setTimeout(r, !real && window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : ms));

const fmtDbm = (v) => (Number.isFinite(Number(v)) ? `${Number(v) < 0 ? "−" : ""}${Math.abs(Number(v))} dBm` : "? dBm");
const PROTO = {ocusync_like: "OcuSync-like", wifi: "Wi-Fi", lora: "LoRa", gfsk: "GFSK", fm_video: "FM video", unknown: "unknown protocol"};
/* Douglas-Peucker in metres: footprint outlines read as zones, not pixel traces */
function simplifyRing(r, tol) {
  if (!r || r.length < 5) return r || [];
  const keep = new Uint8Array(r.length); keep[0] = keep[r.length - 1] = 1;
  let far = 1, fd = -1;                                    // closed ring: split at the vertex farthest from the start
  for (let k = 1; k < r.length - 1; k++) { const dd = Math.hypot(r[k][0] - r[0][0], r[k][1] - r[0][1]); if (dd > fd) { fd = dd; far = k; } }
  keep[far] = 1;
  const st = [[0, far], [far, r.length - 1]];
  while (st.length) {
    const [i, j] = st.pop(); const [ax, ay] = r[i], [bx, by] = r[j];
    const vx = bx - ax, vy = by - ay, L = Math.hypot(vx, vy) || 1;
    let dm = 0, km = -1;
    for (let k = i + 1; k < j; k++) { const dd = Math.abs((r[k][0] - ax) * vy - (r[k][1] - ay) * vx) / L; if (dd > dm) { dm = dd; km = k; } }
    if (dm > tol && km > 0) { keep[km] = 1; st.push([i, km], [km, j]); }
  }
  return r.filter((_, k) => keep[k]);
}
/* TDOA: the hyperbola branch through point p with foci f1, f2 (2-D), clipped to +-half metres of arc from p */
function hyperbolaBranch(f1, f2, p, half = 2500) {
  const d1 = Math.hypot(p[0] - f1[0], p[1] - f1[1]), d2 = Math.hypot(p[0] - f2[0], p[1] - f2[1]);
  const c = Math.hypot(f2[0] - f1[0], f2[1] - f1[1]) / 2, a = (d1 - d2) / 2;
  if (c < 1 || Math.abs(a) >= c - 1e-6) return [];
  const b = Math.sqrt(c * c - a * a), cx = (f1[0] + f2[0]) / 2, cy = (f1[1] + f2[1]) / 2;
  const ux = (f2[0] - f1[0]) / (2 * c), uy = (f2[1] - f1[1]) / (2 * c);       // axis from f1 to f2
  const toW = (x, y) => [cx + x * ux - y * uy, cy + x * uy + y * ux];
  const lx = (p[0] - cx) * ux + (p[1] - cy) * uy, ly = -(p[0] - cx) * uy + (p[1] - cy) * ux;
  const sgn = a >= 0 ? 1 : -1;                                                  // d1 > d2: the branch nearer f2
  const t0 = Math.asinh(ly / b), out = [];
  for (let k = -40; k <= 40; k++) {
    const t = t0 + (k / 40) * Math.asinh(half / b);
    out.push(toW(sgn * Math.abs(a) * Math.cosh(t), b * Math.sinh(t)));
  }
  void lx;
  return out;
}
/* evidence refs as operator words; the raw ref stays as a small clickable suffix */
function refLabel(ref) {
  const r = String(ref ?? ""), b = baseRef(r), at = r.includes("@") ? r.split("@")[1] : null;
  const i = b.indexOf(":"), k = i > 0 ? b.slice(0, i) : "", id = i > 0 ? b.slice(i + 1) : b;
  const objName = (oid) => S.recs.get(`obj:${oid}`)?.data?.name;
  switch (k) {
    case "hyp": { const bb = S.ep?.bearings.get(b); return bb ? `${bb.sensor} bearing ${fmtDeg(bb.deg)}` : "DF bearing"; }
    case "fe": { const n = S.ep?.names.get(b); return n ? `${n}` : `Own link ${id}`; }
    case "cand": return id === "0" ? "Best fit" : `Fit candidate ${Number(id) + 1}`;
    case "nodecand": { const n = objName(id); return n ? `Roof of ${n}` : `Site on ${id.replace(/_/g, " ")}`; }
    case "trk": return `Track ${id}${at != null ? ` at t ${at} s` : ""}`;
    case "sensor": return `Sensor ${id}`;
    case "zone": return S.recs.get(b)?.data?.name || `Zone ${id.replace(/_/g, " ")}`;
    case "obj": return objName(id) || id.replace(/_/g, " ");
    case "fp": return "Predicted jamming footprint";
    case "ras": { const z = S.ep?.weak?.get(b); const k = [...(S.ep?.weak?.keys() || [])].indexOf(b);
      return z ? `Weak ${z.metric === "detect_prob" ? "detection" : "DF"} zone ${k + 1}, ${(z.area_m2 / 1e4).toFixed(1)} ha` : "Weak coverage zone"; }
    case "wp": return `Waypoint ${id}`;
    case "evt": return `Event ${id}`;
    case "grp": return `Swarm ${id}`;
    default: return r;
  }
}

/* ------------------------------------------------------------------ symbols */
const SYM = {};
function sym(kind, color, size = 22) {
  const key = `${kind}|${color}|${size}`;
  if (SYM[key]) return SYM[key];
  const c = document.createElement("canvas");
  c.width = c.height = size * 2;
  const g = c.getContext("2d");
  g.scale(2, 2);
  const m = size / 2, r = size / 2 - 3;
  const path = () => {
    g.beginPath();
    if (kind === "unknown") g.arc(m, m, r, 0, Math.PI * 2);
    else if (kind === "friendly") g.rect(m - r, m - r * 0.7, r * 2, r * 1.4);
    else if (kind === "fix") g.arc(m, m, r * 0.45, 0, Math.PI * 2);
    else if (kind === "victim") { g.moveTo(m, m - r); g.lineTo(m + r, m); g.lineTo(m, m + r); g.lineTo(m - r, m); g.closePath(); }
    else if (kind === "pilot") g.rect(m - r * 0.6, m - r * 0.6, r * 1.2, r * 1.2);
    else if (kind === "cand") { g.moveTo(m, m - r); g.lineTo(m + r * 0.9, m + r * 0.7); g.lineTo(m - r * 0.9, m + r * 0.7); g.closePath(); }
  };
  g.strokeStyle = "#000"; g.lineWidth = 5; path(); g.stroke();
  g.strokeStyle = color; g.lineWidth = 2; path(); g.stroke();
  if (kind === "fix" || kind === "victim" || kind === "pilot" || kind === "cand") { g.fillStyle = color; path(); g.fill(); }
  if (kind === "unknown") { g.fillStyle = color; g.beginPath(); g.arc(m, m, 2.2, 0, Math.PI * 2); g.fill(); }
  SYM[key] = c;
  return c;
}
const CLS = {[T.sodium]: "ok", [T.ash]: "ash", [T.signal]: "act", [T.friendly]: "fr", [T.bone]: "", [C.reflect]: "refl", [C.gap]: "gap",
             [C.cand]: "cand", [C.jam]: "jam"};

/* ------------------------------------------------------------------ boot */
async function getJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.json(); }
function bootLine(k, v, fault = false, id = "") {
  const li = document.createElement("li");
  if (fault) li.className = "fault";
  li.innerHTML = `<span>${esc(k)}</span><span${id ? ` id="${id}"` : ""}>${esc(v)}</span>`;
  $("#bootList").appendChild(li);
}
function sysCell(id, value, state = "ok") { const c = $(id); c.dataset.state = state; c.querySelector(".v").textContent = value; }
// The demo model is SFT v1.1 (checkpoints/sft_v11, ledger Ruling S2.5b): live `sft` and the recordings (v1 and v2
// demo sets, both recorded with v1.1; each recording also carries its own summary.label, which wins) say so.
function modelName(m, recorded = false) {  // eslint-disable-line no-unused-vars
  if (/sft/i.test(m)) return "SceneScout 4B, trained (SFT v1.1)";
  if (/rl/i.test(m)) return "SceneScout 4B, trained (RL)";
  return "4B base, untrained";
}

const BLENDER_GEOMETRY = "Blender port model (kandla_world.blend): OSM footprints and Sentinel-2 imagery were inputs; heights modelled or estimated";
/* Objects for picking, focus outlines, framing and place names. Preferred: the Blender world's object list
   (/api/blender_objects, ENU rings, mesh tops). Fallback: the site catalogue the tools use (/api/site), which shares
   ids and footprints with the Blender build; the boot list says which one is in use. */
function normObjects(site, bl) {
  const area = (r) => Math.abs(r.reduce((a, p, i) => { const q = r[(i + 1) % r.length]; return a + p[0] * q[1] - q[0] * p[1]; }, 0)) / 2;
  const mk = (o) => ({...o, area: o.ring.length > 2 ? area(o.ring) : 0, synthetic: /synthetic/.test(o.name_source || "")});
  if (bl && Array.isArray(bl.objects) && bl.objects.length) {
    return bl.objects.filter((o) => Array.isArray(o.ring) && o.ring.length > 2).map((o) => mk({
      ref: o.ref || `obj:${o.id}`, id: o.id, name: o.name || "", kind: o.kind || "structure", ring: o.ring,
      enu: o.center || [o.ring.reduce((a, p) => a + p[0], 0) / o.ring.length, o.ring.reduce((a, p) => a + p[1], 0) / o.ring.length],
      top: o.top_m ?? null, base: o.base_m ?? 0, fidelity: o.fidelity || "", name_source: o.name_source || "",
      in_tools: o.in_tools !== false, aliases: o.aliases || [], src: "blender", collection: o.collection || ""}));
  }
  return site.objects.map((o) => mk({
    ref: o.ref, id: o.id, name: o.name, kind: o.kind, ring: ringEnu(o.polygon), enu: lonLatToEnu(o.center[0], o.center[1]),
    top: o.height_m ?? null, base: 0, fidelity: o.provenance?.height_source || "estimated", name_source: o.provenance?.name_source || "",
    in_tools: true, aliases: o.aliases || [], src: "catalogue", protected: o.protected}));
}

async function loadData() {
  for (let attempt = 1; ; attempt++) {
    try {
      /* optional endpoints are looked up in the gateway's OpenAPI list first, so a missing one is not a 404 in the console */
      const api = await getJSON("/openapi.json").then((o) => o.paths || {}).catch(() => ({}));
      S.apiPaths = api;
      return await Promise.all([getJSON("/api/meta"), getJSON("/api/site"), getJSON("/api/tracks?step_s=2&frame=enu3"),
                                getJSON("/api/replays").catch(() => ({})),
                                api["/api/blender_objects"] ? getJSON("/api/blender_objects").catch(() => null) : null]);
    } catch (e) {
      let li = $("#bootRetry");
      if (!li) { bootLine("Gateway", "", true, "bootRetry"); li = $("#bootRetry"); }
      for (let k = 5; k > 0; k--) { li.textContent = `not reachable (attempt ${attempt}); retrying in ${k} s`; await wait(1000, true); }
      li.textContent = "retrying now";
    }
  }
}

const BOOT_MS = {};                                   // boot timing (ms since navigation), read by scripts/demo_stress.py
const bootMark = (k) => { BOOT_MS[k] = Math.round(performance.now()); };
async function boot() {
  const [meta, site, trk, reps, bl] = await loadData();
  bootMark("data");
  if ($("#bootRetry")) { $("#bootRetry").textContent = "reachable"; $("#bootRetry").parentElement.classList.remove("fault"); }
  S.meta = meta; S.site = site; S.tracks = trk.tracks; S.demo = meta.demo || []; S.replays = reps;
  S.dur = site.meta.duration_s || 600;
  S.origin = site.meta.origin;
  for (const s of site.sensors) s.enu = [...lonLatToEnu(s.lon, s.lat), s.z_m || 10];
  S.objs = normObjects(site, bl);
  S.objSource = S.objs[0]?.src || "none";
  S.geometryText = (bl && bl.geometry) || BLENDER_GEOMETRY;
  S.health = {llm: meta.llm, model: meta.model, gw: true, fails: 0};
  sysModel();
  const served = modelServed();
  bootLine("Model", served ? `${modelName(meta.model)}, loaded` : `${modelName(meta.model)}: not loaded`, !served);
  if (served && untrained(meta.model)) bootLine("Model", "Untrained base model selected: set SCENESCOUT_SERVED_MODEL=sft", true);
  sysCell("#sysServer", meta.llm.ok ? "ready" : "offline", meta.llm.ok ? "ok" : "fault");
  bootLine("Model server", meta.llm.ok ? "ready (vLLM)" : "offline: demo recordings still play", !meta.llm.ok);
  const fixed = site.sensors.filter((s) => s.type !== "drone_df").length;
  sysCell("#sysSensors", `${fixed} AIR7311 + DF drone`);
  bootLine("DF sensors", `${fixed} fixed AIR7311 nodes and a DF drone (synthetic sites)`);
  sysCell("#sysRf", /air7311/i.test(site.meta.rf_backend || "") ? "Sionna RT, AIR7311" : (site.meta.rf_backend || "unknown"));
  bootLine("RF model", site.meta.rf_backend || "unknown");
  sysCell("#sysData", `seed 0, ${String(site.meta.data_version || meta.data_version).slice(0, 8)}`);
  bootLine("Port model", S.objSource === "blender"
    ? `Blender world (OSM footprints and imagery as input), ${S.objs.length} pickable objects, ${site.zones.length} zones`
    : `Blender world; picking uses the ${S.objs.length}-object tool catalogue (same footprints), ${site.zones.length} zones`);
  bootLine("Blender world", "loading", false, "bootWorld");

  const W = S.W = new World($("#cesium"));
  if (VIDEO) { const lab = W.label.bind(W); W.label = (text, ...rest) => lab(publicText(text), ...rest); }   // public-safe map labels
  const steps = {sky: "sky and light", ground: "terrain and Sentinel-2 ground", water: "creek water", structures: "port structures", vegetation: "vegetation"};
  let info;
  try {
    info = await W.init(site, (what, k) => {
      $("#bootWorld").textContent = `${steps[what] || what}${k ? ` ${Math.round(k * 100)}%` : ""}`;
      if (!S.worldLive) sysCell("#sysWorld", "loading", "wait");     // later progress is background streaming
    }, {deferRest: true});
  } catch (e) {
    $("#bootWorld").textContent = "3D engine files could not be loaded";
    $("#bootWorld").parentElement.classList.add("fault");
    throw e;
  }
  bootMark("world");
  $("#bootWorld").textContent = `${info.groups} structure groups, ${info.veg} vegetation kinds`;
  S.worldLive = true;
  sysCell("#sysWorld", "Blender world, live");
  W.dur = S.dur;
  W.on("tick", onTick);
  W.on("usercam", userMoved);
  W.on("stop", () => setPlay(false));
  drawZones(); drawObjects(); drawSensors(); drawTracks();
  S.B = new Beacons(W);
  S.D = new Drones(W, S.tracks, trackPos);
  S.cam = new FreeCam(W, {onUser: userMoved, onFly: (on) => {
    $("#flyBtn").classList.toggle("on", on); $("#flyBtn").setAttribute("aria-pressed", String(on));
    $("#flyHint").hidden = !on;
  }});
  bootMark("draw");
  fidelity(); wireUI(); timelineTicks(); demoMenu(); view("overview", true);
  bootMark("ui");
  /* S6 (S3 request): no fixed pauses (2 x 140 ms + 600 ms). Lift the curtain after the overview's first rendered
     frame (texture uploads + shadow map; ~2.4 s cold with the GPU busy training, less when idle); the remaining
     structures stream only after that (ss3d deferRest). Every boot line above is kept. */
  await new Promise((r) => requestAnimationFrame(() => r()));
  bootMark("waited");
  $("#curtain").classList.add("gone");
  setTimeout(() => ($("#curtain").hidden = true), 1000);
  bootMark("interactive");
  requestAnimationFrame(() => W.streamRest && W.streamRest());   // remaining structures + vegetation, in the background
  if (!meta.llm.ok) banner("Model server offline.", "Live questions are paused. Demo recordings still play (D, then Shift+number).");
  else if (untrained(meta.model)) banner("Untrained model selected.", "Live questions go to the base model. Set SCENESCOUT_SERVED_MODEL=sft on the gateway.");
  updateAsk();
  S.api = {onEvent, newEpisode};               // for scripts/console_s4_check.py (synthetic tool events)
  window.__scenescout = S; S.echo = ECHO; S.echoPick = echoPick;
  if (CAPTURE) window.ssCapture = captureApi();
  S.bootMs = BOOT_MS;
  setInterval(clockTick, 1000); clockTick();
  setInterval(pollHealth, 10000);
}

/* ------------------------------------------------------------------ health: model and gateway, every 10 s */
const untrained = (m) => modelName(m) === "4B base, untrained";
function modelServed() { const h = S.health; return !!(h.llm && h.llm.ok && (h.llm.models || []).includes(h.model)); }
function sysModel() {
  const h = S.health, served = modelServed();
  sysCell("#sysModel", modelName(h.model), served && !untrained(h.model) ? "ok" : "fault");
}
function liveOk() { return S.health.gw && !!(S.health.llm && S.health.llm.ok); }
function updateAsk() {
  const go = $("#go"), off = !liveOk();
  go.disabled = off || !S.finished;
  go.textContent = off ? "Model offline" : "Ask";
  go.title = off ? "The model server is offline. Recorded demo runs still play: press D, then Shift+number." : "";
  $("#stopBtn").hidden = S.finished;
}
async function pollHealth() {
  const h = S.health, was = {llm: !!(h.llm && h.llm.ok), gw: h.gw};
  try {
    let d = await getJSON("/api/health");
    if (!d.llm) d = await getJSON("/api/meta");
    h.fails = 0; h.gw = true; h.llm = d.llm; if (d.model) h.model = d.model;
  } catch (_) {
    h.fails += 1;
    if (h.fails >= 2) h.gw = false;
  }
  const t = new Date().toLocaleTimeString("en-GB", {hour: "2-digit", minute: "2-digit"});
  if (!h.gw) {
    for (const id of ["#sysModel", "#sysServer", "#sysSensors", "#sysRf", "#sysData"]) $(id).dataset.state = "wait";
    if (was.gw) gwCountdown();
  } else {
    if (!was.gw) { banner("Gateway back.", `Reconnected at ${t}.`, "ok"); for (const id of ["#sysSensors", "#sysRf", "#sysData"]) $(id).dataset.state = "ok"; }
    const ok = !!(h.llm && h.llm.ok);
    sysCell("#sysServer", ok ? "ready" : "offline", ok ? "ok" : "fault");
    sysModel();
    if (was.llm && !ok) banner("Model server lost.", `Lost at ${t}. Live questions are paused; recorded runs still play.`);
    else if (!was.llm && ok && was.gw) banner("Model server ready.", `Back at ${t}.`, "ok");
  }
  updateAsk();
}
async function gwCountdown() {
  for (let k = 10; k > 0 && !S.health.gw; k--) {
    banner("Gateway not reachable.", `Reconnecting in ${k} s.`, "", 1500);
    await wait(1000, true);
  }
}

function fidelity() {
  const f = {...(S.site.meta.fidelity || {})};
  f.geometry = S.geometryText;
  $("#fidText").textContent = "Geometry: Blender port model (heights modelled or estimated); sensors and drones synthetic; simulation";
  const rows = Object.entries(f).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`);
  rows.push(`<dt>picking</dt><dd>${S.objSource === "blender" ? "Blender world objects" : "tool catalogue objects (same ids and footprints as the Blender build); Blender object list not served yet"}</dd>`);
  rows.push("<dt>ground</dt><dd>Copernicus Sentinel-2 L2A, 10 m, 2026-10-01; 5 m terrain with estimated creek bed</dd>");
  rows.push(`<dt>attribution</dt><dd>${esc((S.site.meta.attribution || "").replace("(c)", "©"))}; contains modified Copernicus Sentinel data 2026</dd>`);
  $("#fidMore").innerHTML = rows.join("");
}

/* ------------------------------------------------------------------ twin layers */
const LAYER = {zones: [], sensors: [], tracks: [], objects: []};
function drawZones() {
  for (const z of S.site.zones) {
    const ring = ringEnu(z.polygon);
    const rec = {kind: "zone", data: z, ring};
    S.recs.set(z.ref, rec);
    if (!["restricted", "perimeter", "aviation"].includes(z.kind)) continue;
    rec.line = S.W.line(ring.map((p) => [p[0], p[1], 1.5]), {color: T.ash, width: 1.4, dashed: z.kind !== "restricted", opacity: 0.85, cased: false});
    const c = lonLatToEnu(z.center[0], z.center[1]);
    rec.label = S.W.label(`${z.name}${/proposed|provisional/i.test(z.name) ? "" : z.kind === "restricted" ? " (proposed)" : ""}`, [c[0], c[1], 20], "zone");
    LAYER.zones.push(rec.line, rec.label);
  }
}
function drawObjects() {
  for (const o of S.objs) {
    const rec = {kind: "obj", data: o, ring: o.ring};
    S.recs.set(o.ref, rec);
    if (o.kind === "jetty") {
      rec.label = S.W.label(o.name, [o.enu[0], o.enu[1], 14], "place");
      LAYER.objects.push(rec.label);
    }
    const pick = S.W.fill(rec.ring, "#000000", 0.0, (o.top || 0) + 0.5);
    pick.material.visible = false;
    S.W.addPick(pick, rec);
  }
}
function drawSensors() {
  for (const s of S.site.sensors) {
    const [x, y] = s.enu;
    S.sensors.set(s.ref, s);
    const drone = s.type === "drone_df";
    const rec = {kind: "sensor", data: s};
    rec.spr = S.W.sprite(sym("friendly", T.friendly, 20), s.enu, 20);
    rec.label = S.W.label(drone ? `${s.id} DF drone` : s.id, s.enu, "fr");
    rec.mast = drone ? null : S.W.line([[x, y, 0], s.enu], {color: T.friendly, width: 2, cased: false});
    S.W.addPick(rec.spr, rec);
    S.recs.set(s.ref, rec);
    LAYER.sensors.push(rec.spr, rec.label, rec.mast);
  }
}
function drawTracks() {
  for (const tr of S.tracks) {
    if (!tr.pts.length) continue;
    tr.ts = tr.pts.map((p) => p[0]);
    tr.avail = [tr.ts[0], tr.ts[tr.ts.length - 1]];
    const rec = {kind: "trk", data: tr};
    rec.spr = S.W.sprite(sym("unknown", T.bone, 20), [0, 0, 0], 20);
    rec.label = S.W.label(tr.ref, [0, 0, 0], tr.group ? "trk grp" : "trk");
    rec.trail = S.W.line([[0, 0, 0], [0, 0, 1]], {color: T.bone, width: 1.4, opacity: tr.group ? 0.3 : 0.45, cased: false});
    rec.tether = S.W.line([[0, 0, 0], [0, 0, 1]], {color: T.bone, width: 1, opacity: 0.35, cased: false});
    S.W.addPick(rec.spr, rec);
    S.recs.set(tr.ref, rec);
    if (tr.group) { if (!S.groups.has(tr.group)) S.groups.set(tr.group, {members: []}); S.groups.get(tr.group).members.push(rec); }
    LAYER.tracks.push(rec.spr, rec.label, rec.trail, rec.tether);
  }
  for (const [g, info] of S.groups) {
    info.label = S.W.label(`Swarm ${g}, ${info.members.length} tracks`, [0, 0, 0], "grp");
    S.recs.set(`grp:${g}`, {kind: "grp", data: {group: g, members: info.members.map((m) => m.data.ref)}, label: info.label});
    LAYER.tracks.push(info.label);
  }
}
function trackPos(tr, t) {
  if (t < tr.avail[0] || t > tr.avail[1]) return null;
  let lo = 0, hi = tr.ts.length - 1;
  while (lo < hi - 1) { const m = (lo + hi) >> 1; if (tr.ts[m] <= t) lo = m; else hi = m; }
  const a = tr.pts[lo], b = tr.pts[hi], k = hi === lo ? 0 : Math.min(1, Math.max(0, (t - a[0]) / (b[0] - a[0])));
  return [a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k, a[3] + (b[3] - a[3]) * k];
}
function updateTracks(t) {
  const P = (p) => ({x: p[0], y: p[2], z: -p[1]});
  for (const [g, info] of S.groups) info.c = [0, 0, 0, 0];
  for (const tr of S.tracks) {
    const rec = S.recs.get(tr.ref);
    if (!rec) continue;
    const p = trackPos(tr, t);
    const on = !!p && LAYER_ON.tracks;
    const showLbl = on && !rec.receded, dr = S.drones;          // drones layer: the quadcopter replaces dot and trail
    rec.spr.visible = on && !dr; rec.label.visible = showLbl; rec.trail.visible = on && !rec.receded && !dr; rec.tether.visible = on && !rec.receded && !dr;
    if (!p) continue;
    const txt = dr ? `${tr.ref}, ${Math.round(p[2])} m` : tr.ref;
    if (rec.label.element.textContent !== txt) rec.label.element.textContent = txt;
    rec.pos = p;
    const q = P(p);
    rec.spr.position.set(q.x, q.y, q.z);
    rec.label.position.set(q.x, q.y, q.z);
    rec.tether.children.forEach((l) => l.geometry.setPositions([q.x, q.y, q.z, q.x, 0.5, q.z]));
    const trail = [];
    for (let s = Math.max(tr.avail[0], t - (tr.group ? 50 : 100)); s <= t; s += 4) { const pp = trackPos(tr, s); if (pp) trail.push(pp[0], pp[2], -pp[1]); }
    trail.push(q.x, q.y, q.z);
    if (trail.length >= 6) rec.trail.children.forEach((l) => l.geometry.setPositions(trail));
    if (tr.group) { const c = S.groups.get(tr.group).c; c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; c[3]++; }
  }
  for (const [, info] of S.groups) {
    const c = info.c;
    info.label.visible = c[3] > 0 && LAYER_ON.tracks && !S.receded;
    if (c[3]) info.label.position.set(c[0] / c[3], c[2] / c[3] + 30, -c[1] / c[3]);
  }
}
const LAYER_ON = {tracks: true};

/* ------------------------------------------------------------------ focus + camera */
function setFocus(ref, on, color = T.sodium) {
  const rec = S.recs.get(ref);
  if (!rec) return;
  if (rec.kind === "trk") {
    S.W.setSprite(rec.spr, sym("unknown", on ? color : T.bone, on ? 26 : 20), on ? 26 : 20);
    rec.label.element.className = `wl trk ${on ? CLS[color] : ""}`;
    S.W.setLine(rec.tether, {color: on ? color : T.bone, opacity: on ? 0.85 : 0.35});
    if (on) rec.receded = false;
  } else if (rec.kind === "zone") {
    if (on) rec.focus = S.W.line(rec.ring.map((p) => [p[0], p[1], 2]), {color, width: 3});
    else { S.W.remove(rec.focus); rec.focus = null; }
    if (rec.label) rec.label.element.className = `wl zone ${on ? CLS[color] : ""}`;
  } else if (rec.kind === "obj") {
    if (on) rec.focus = S.W.line(rec.ring.map((p) => [p[0], p[1], (rec.data.top || 0) + 1]), {color, width: 3});
    else { S.W.remove(rec.focus); rec.focus = null; }
  } else if (rec.kind === "sensor") {
    rec.label.element.className = `wl ${on ? "" : "fr"}`;
  }
}
function clearHighlight() {
  const was = S.highlighted; S.highlighted = [];
  for (const r of was) setFocus(r, false);
  if (S.ep && !S.finished) recedeTracks(true);
}
function recPoints(rec) {
  if (rec.kind === "trk") return rec.pos ? [rec.pos] : [];
  if (rec.kind === "ray") return rec.pts;
  if (rec.kind === "sensor") return [rec.data.enu];
  if (rec.ring) return rec.ring.map((p) => [p[0], p[1], 0]);
  return [];
}
function camCaption(text) { $("#camText").textContent = text; }
/* Camera moves by the agent never fight the operator: with "Follow agent" off, or once the operator has moved the
   camera during this question, automatic framing stops and the "View held" hint offers V to frame the answer. */
function flyToPoints(points, {pitch = -38, minRange = 600, heading, caption, auto = true, pad = 0.18} = {}) {
  if (!points.length) return;
  if (S.ep && auto) S.ep.lastFrame = {points, pitch, minRange, heading, caption, pad};
  if (auto) {
    if (!S.follow || S.userCam || S.camMoves >= 2 || S.cam?.flying) { if (S.ep && (S.userCam || !S.follow)) heldHint(true); return; }
    if (!S.prevView) S.prevView = {...S.W.view(), caption: $("#camText").textContent};
    S.camMoves += 1;
    $("#camBack").hidden = false;
  }
  frame(S.W, points, {pitch, heading, pad, minDist: minRange});
  if (caption) camCaption(caption);
}
function userMoved() {
  S.userCam = true;
  if (S.ep) heldHint(true);
}
function heldHint(on) {
  const h = $("#heldHint");
  if (h) h.hidden = !(on && S.ep && (S.ep.ansFrame || S.ep.lastFrame));
}
/* V: frame the answer (or the agent's latest work) on request, whatever the follow setting */
function frameAnswer() {
  const f = S.ep && (S.ep.ansFrame || S.ep.lastFrame);
  if (!f) { banner("Nothing to frame.", "Ask a question first."); return; }
  if (S.cam?.flying) S.cam.exitFly();
  frame(S.W, f.points, {pitch: f.pitch, heading: f.heading, pad: f.pad ?? 0.18, minDist: f.minRange});
  if (f.caption) camCaption(f.caption);
  heldHint(false);
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
  else if (trks.length && !trks.some((r) => trackPos(r.data, S.W.simT))) setSimTime((trks[0].data.avail[0] + trks[0].data.avail[1]) / 2);
  updateTracks(S.W.simT);
  if (fly && recs.length) flyToPoints(recs.slice(0, 25).flatMap(recPoints), {caption, auto});
  return recs.length;
}
function bboxOf(objs) { return objs.flatMap((o) => o.ring.map((p) => [p[0], p[1], 0])); }
function view(name, instant = false) {
  $$(".views [data-view]").forEach((b) => b.classList.toggle("on", b.dataset.view === name));
  const objs = S.objs;
  const dur = instant ? 0 : 1.0;
  if (name === "overview") { S.W.fit(bboxOf(objs.filter((o) => ["tank", "jetty", "tank_farm"].includes(o.kind))), {pitch: -30, heading: 335, mul: 1.15, dur}); camCaption("Port overview"); }
  else if (name === "jetties") { S.W.fit(bboxOf(objs.filter((o) => /^Jetty J[1-5]$/.test(o.name))), {pitch: -22, heading: 245, mul: 1.0, dur}); camCaption("Oil jetties J1 to J5"); }
  else if (name === "tankfarms") { S.W.fit(bboxOf(objs.filter((o) => o.kind === "tank_farm")), {pitch: -32, heading: 20, mul: 0.95, dur}); camCaption("Tank farms TF-1 to TF-8"); }
  else if (name === "top") {
    const pts = [...bboxOf(objs.filter((o) => o.kind === "jetty")), ...S.site.sensors.filter((s) => s.enu).map((s) => [s.enu[0], s.enu[1], 0])];
    S.W.fit(pts, {pitch: -89, heading: 0, mul: 1.25, dur}); camCaption("Top-down, north up: all sensors and jetties");
  }
  S.prevView = null; $("#camBack").hidden = true;
}
function setSimTime(t) { S.W.simT = Math.max(0, Math.min(S.dur, t)); }
const now = () => S.W.simT;

/* ------------------------------------------------------------------ rays + overlays with step stamps */
function tag(step, ...objs) {
  if (!S.stepObjs.has(step)) S.stepObjs.set(step, []);
  S.stepObjs.get(step).push(...objs.filter(Boolean));
  const li = $(`#steps li[data-step="${step}"]`);
  if (li) li.classList.add("hasgeo");
  return objs[0];
}
const ovl = (step, pts, opt) => tag(step, S.W.line(pts, opt));
const ovLabel = (step, text, p, cls = "") => tag(step, S.W.label(text, p, cls));
function stamp(step, p) { return tag(step, S.W.label(String(step), [p[0], p[1], p[2] ?? 0], "stamp")); }
function ovPoint(step, p, color, text, kind = "fix", px = 18) {
  const s = tag(step, S.W.sprite(sym(kind, color, px), p, px));
  s.userData.kind = kind;
  const l = text ? ovLabel(step, text, p, CLS[color]) : null;
  return [s, l];
}
function ellipsePts(p, u, z = 0.8) {
  const pts = [], th = (u.theta_deg || 0) * Math.PI / 180;
  for (let k = 0; k <= 64; k++) {
    const g = (k / 64) * 2 * Math.PI, a = u.a_m * Math.cos(g), b = u.b_m * Math.sin(g);
    pts.push([p.x_m + a * Math.sin(th) + b * Math.cos(th), p.y_m + a * Math.cos(th) - b * Math.sin(th), z]);
  }
  return pts;
}
function ovFix(step, p, u, color, text) {
  const z = p.z_m || 0;
  const out = [...ovPoint(step, [p.x_m, p.y_m, z], color, text, "fix", 20)];
  if (z > 2) out.push(ovl(step, [[p.x_m, p.y_m, 0.5], [p.x_m, p.y_m, z]], {color, width: 1.5, cased: false}));
  if (u && u.a_m) out.push(ovl(step, ellipsePts(p, u), {color, width: 2.2}));
  return out.filter(Boolean);
}
function recolor(objs, color) {
  for (const o of objs || []) {
    if (!o) continue;
    if (o.isSprite) S.W.setSprite(o, sym(o.userData.kind || "fix", color, o.userData.px));
    else if (o.element) o.element.className = `wl ${CLS[color]}`;
    else if (o.userData?.top) S.W.setLine(o, {color});
  }
}
function rayEnd(s, bearing, L = RAY_LEN_M) {
  const b = bearing * Math.PI / 180;
  return [s[0] + L * Math.sin(b), s[1] + L * Math.cos(b), s[2]];
}
function sensorAnchor(ref, pos) {
  const s = S.sensors.get(ref);
  if (s) return s.enu;
  return [pos?.x_m || 0, pos?.y_m || 0, pos?.z_m || 10];
}
function drawRays(sensors, {t, track, step}) {
  if (t != null) setSimTime(t);
  for (const s of sensors) for (const h of s.hyps) {
    if (S.rays.has(h.ref)) continue;
    const e = rayEnd(s.anchor, h.bearing_deg);
    const rec = {kind: "ray", data: {...h, sensor: s.sensor, track, t}, pts: [s.anchor, e]};
    rec.line = ovl(step, rec.pts, {color: T.bone, width: 2, opacity: 0.8});
    const lp = [s.anchor[0] + (e[0] - s.anchor[0]) * 0.3, s.anchor[1] + (e[1] - s.anchor[1]) * 0.3, s.anchor[2]];
    rec.label = ovLabel(step, `${s.sensor.replace("sensor:", "")} ${fmtDeg(h.bearing_deg)}`, lp);
    S.rays.set(h.ref, rec); S.rays.set(baseRef(h.ref), rec);
    S.ep.bearings.set(baseRef(h.ref), {deg: h.bearing_deg, sensor: s.sensor.replace("sensor:", "")});
  }
}
function styleRay(rec, how) {
  if (how === "chosen") {
    S.W.setLine(rec.line, {color: T.sodium, width: 4.5, opacity: 1}); rec.label.element.className = "wl ok";
    rec.line.userData.ans = true; rec.line.children.forEach((c) => (c.renderOrder += 20));
  } else if (how === "rejected") {
    S.W.remove(rec.line);
    rec.line = S.W.line(rec.pts, {color: C.reflect, width: 3, dashed: true, dash: 16, gap: 12});
    rec.line.userData.ans = true;
    rec.label.element.className = "wl refl";
  } else {
    S.W.remove(rec.line);
    rec.line = S.W.line(rec.pts, {color: T.bone, width: 1.2, opacity: 0.22, cased: false});
    rec.label.visible = false;
    rec.label.userData.retired = true;          // the answer rejected it: hover and layer toggles must not bring it back
  }
}

function drawToolOverlay(ev, step) {
  const d = (ev.result && ev.result.data) || {}, ui = ev.ui || {};
  const E = S.ep;
  switch (ev.name) {
    case "df_hypotheses": {
      const sensors = (d.sensors || []).map((s) => ({sensor: s.sensor, anchor: sensorAnchor(s.sensor, s.sensor_pos), hyps: s.hyps || []}));
      drawRays(sensors, {t: d.t_s, track: d.track, step});
      for (const s of sensors) stamp(step, s.anchor);
      if (d.track) { E.track = d.track; focusRefs([d.track], {fly: false, color: T.bone}); }
      const pts = [...new Set(S.rays.values())].flatMap((r) => [r.pts[0], [(r.pts[0][0] + r.pts[1][0]) / 2, (r.pts[0][1] + r.pts[1][1]) / 2, 0]]);
      flyToPoints(pts, {minRange: 1800, pitch: -40, caption: `Bearings on ${d.track || "the track"} at t ${d.t_s} s`});
      return;
    }
    case "predict_aoa": {
      const anc = sensorAnchor(d.sensor);
      (E.aoa ||= []).push({sensor: d.sensor, band: ev.args?.band, z: d.point?.z_m});
      for (const pk of d.peaks || []) {
        const e = rayEnd(anc, pk.bearing_deg, RAY_LEN_M * 0.6);
        ovl(step, [anc, e], {color: T.bone, width: 1.5, dashed: true, dash: 6, gap: 6, opacity: 0.6});
        ovLabel(step, `predicted ${pk.kind} ${fmtDeg(pk.bearing_deg)}`, e);
        if (pk.via_ref && ui.vias && ui.vias[pk.via_ref] && d.point) {
          const [vx, vy] = ui.vias[pk.via_ref];
          E.reflectors.push({ref: pk.via_ref, x: vx, y: vy});
          (E.reflPaths ||= []).push(ovl(step, [anc, [vx, vy, 15], [d.point.x_m, d.point.y_m, d.point.z_m || 60]], {color: C.reflect, width: 1.5, dashed: true, dash: 4, gap: 4, opacity: 0.7}));
          legendAdd("reflect");
        }
      }
      stamp(step, anc);
      return;
    }
    case "fit_candidates": {
      const cands = d.candidates || [];
      E.fits = cands; if (d.band) E.fitBand = d.band;
      cands.forEach((c, i) => {
        const res = c.residual_deg_rms != null ? `${c.residual_deg_rms}° rms` : `${c.residual_m_rms} m rms`;
        if (i === 0) { E.fitEnts = ovFix(step, c.position, c.uncertainty, T.bone, `${c.ref === "cand:0" || !c.ref ? "best fit" : refLabel(c.ref)}, ${res}`);
                       stamp(step, [c.position.x_m, c.position.y_m, c.position.z_m || 0]); }
        else ovPoint(step, [c.position.x_m, c.position.y_m, c.position.z_m || 0], T.ash, null, "fix", 12);
      });
      return;
    }
    case "weak_zones": {
      E.weak = E.weak || new Map();
      const rings = new Map((ui.zones || []).map((z) => [z.ref, z.rings]));
      for (const z of ui.zones || []) for (const r of z.rings) {
        tag(step, area(S.W, r, C.gap, {alpha: 0.10, hatch: true, hatchAlpha: 0.32, order: 5}));
        ovl(step, r.map((p) => [p[0], p[1], 1.5]), {color: C.gap, width: 1.6, cased: false, opacity: 0.9});
      }
      const zs = d.zones || [];
      for (const z of zs) E.weak.set(z.ref, {...z, rings: rings.get(z.ref) || [], metric: d.metric, height: d.height_m});
      E.weakLbl = [];
      zs.slice(0, 6).forEach((z) => E.weakLbl.push(...ovPoint(step, [z.centroid.x_m, z.centroid.y_m, 0], C.gap, `weak, ${(z.area_m2 / 1e4).toFixed(1)} ha`, "fix", 10)));
      if (zs.length) legendAdd("gap");
      if (zs[0]) stamp(step, [zs[0].centroid.x_m, zs[0].centroid.y_m, 0]);
      if (zs.length) flyToPoints(zs.map((z) => [z.centroid.x_m, z.centroid.y_m, 0]), {minRange: 1600, pitch: -60, caption: "Weak DF coverage"});
      return;
    }
    case "suggest_node": {
      E.nodes = d.candidates || []; E.nodeEnts = []; E.nodeMap = new Map();
      const pts = [];
      for (const c of E.nodes) {
        const p = c.position, top = [p.x_m, p.y_m, (p.z_m || 20) + 10];
        const ents = [ovl(step, [[p.x_m, p.y_m, 0], top], {color: C.cand, width: 3}), ...ovPoint(step, top, C.cand, `${refLabel(c.ref)}, gain ${c.predicted_gain}`, "cand", 16)];
        ents.forEach((o) => { if (o) o.userData.kind = o.userData.kind || "cand"; });
        E.nodeEnts.push(...ents); E.nodeMap.set(c.ref, {c, ents});
        pts.push(top);
      }
      if (E.nodes.length) legendAdd("cand");
      if (E.nodes[0]) stamp(step, [E.nodes[0].position.x_m, E.nodes[0].position.y_m, (E.nodes[0].position.z_m || 20) + 10]);
      if (pts.length) flyToPoints(pts, {minRange: 1200, caption: "Candidate DF node sites"});
      return;
    }
    case "effect_footprint": {
      E.fp = {band: d.band, eirp: d.eirp_dbm, effector: ev.args?.effector || d.effector, threshold: ev.args?.threshold_dbm ?? d.threshold_dbm};
      E.fpEnts = []; E.fpFill = [];
      const thr = Number(E.fp.threshold ?? -90);
      const bands = (ui.bands && ui.bands.length) ? ui.bands : [{level_dbm: thr, rings: (ui.rings || []).map((r) => simplifyRing(r, 25))}];
      const W_BY_RANK = [1, 1.5, 2.5];                         // outer (threshold) thin, inner (strongest) heavy
      E.fpOuter = [];
      bands.forEach((bd, i) => {
        const rank = bands.length - 1 - i;                                  // every band filled, stronger = denser
        const alpha = bands.length > 1 ? [0.30, 0.20, 0.12][Math.min(i, 2)] : 0.16;
        let top = null;
        for (const r of bd.rings || []) {
          const f = tag(step, S.W.fill(r, C.jam, alpha * 0.6)); f.userData.alpha = alpha; E.fpFill.push(f); E.fpEnts.push(f);
          const l = ovl(step, r.map((p) => [p[0], p[1], 1.5]), {color: C.jam, width: bands.length > 1 ? W_BY_RANK[Math.min(rank, 2)] : 1.5, cased: false, opacity: 0.9});
          E.fpEnts.push(l);
          if (i === bands.length - 1) E.fpOuter.push(r);
          for (const p of r) if (!top || p[1] > top[1]) top = p;
        }
        if (top && bands.length > 1) E.fpEnts.push(ovLabel(step, `${fmtDbm(bd.level_dbm)}`, [top[0], top[1], 2], "band"));
      });
      legendAdd("jam");
      if (ui.effector) {
        E.fpEffector = ui.effector;
        ovPoint(step, [ui.effector[0], ui.effector[1], 20], T.bone, null, "fix", 12);
        ovLabel(step, `Proposed ${band(d.band)} jammer, ${d.eirp_dbm} dBm, not active`, [ui.effector[0], ui.effector[1], 20]);
        stamp(step, [ui.effector[0], ui.effector[1], 20]);
      }
      if (d.bbox) flyToPoints([[d.bbox[0], d.bbox[1], 0], [d.bbox[2], d.bbox[3], 0]], {pitch: -55, minRange: 2500, caption: "Predicted jamming footprint"});
      return;
    }
    case "collateral_check": {
      E.victims = ui.victims || []; E.victimEnts = [];
      const thr = Number(E.fp?.threshold ?? -90);
      for (const v of (d.violations || [])) if (v.victim) S.ep.names.set(v.ref, v.victim);
      E.victims.forEach((v) => {
        if (!v.pos) return;
        if (v.name) S.ep.names.set(v.ref, v.name);
        const lvl = v.note ? "transmitting node, receiver blanked"
          : `${fmtDbm(v.level_dbm)}${Number.isFinite(v.level_dbm) ? `, ${(v.level_dbm - thr).toFixed(0)} dB over` : ""}`;
        const [s, l] = ovPoint(step, [v.pos[0], v.pos[1], 25], T.bone, `${shortRef(v.ref)}, ${lvl}`, "victim", 20);
        ovl(step, [[v.pos[0], v.pos[1], 25], [v.pos[0], v.pos[1], 0.5]], {color: T.bone, width: 1, cased: false, opacity: 0.6});
        E.victimEnts.push([s, l, v.ref]);
      });
      if (E.victims[0] && E.victims[0].pos) stamp(step, [E.victims[0].pos[0], E.victims[0].pos[1], 25]);
      return;
    }
    case "plan_waypoints":
      for (const w of d.waypoints || []) ovPoint(step, [w.position.x_m, w.position.y_m, w.position.z_m || 80], T.ash, null, "fix", 10);
      return;
    case "drone_state":
    case "move_drone": {
      const p = d.pose, e = d.estimate;
      E.drone = E.drone || [];
      if (p) {
        const q = [p.x_m, p.y_m, p.z_m || 80];
        E.drone.push(q);
        if (ev.name === "move_drone") E.moves += 1;
        const [s] = ovPoint(step, q, T.friendly, ev.name === "move_drone" ? `D1 at ${d.wp}, bearing ${d.measurement ? fmtDeg(d.measurement.bearing_deg) : "?"}` : "D1 start", "friendly", 18);
        void s;
        if (E.drone.length > 1) ovl(step, E.drone.slice(-2), {color: T.friendly, width: 2.5});
        legendAdd("own");
        stamp(step, q);
      }
      if (e) {
        for (const old of E.estEnts || []) {
          if (old.isSprite || old.element) { old.visible = false; old.userData.retired = true; }      // superseded estimate
          else if (old.userData?.top) S.W.setLine(old, {color: T.ash, opacity: 0.45});
        }
        E.estEnts = ovFix(step, e.position, e.uncertainty, T.bone, `estimate ±${Math.round(e.uncertainty.a_m)} m`);
      }
      if (p && e) flyToPoints([[p.x_m, p.y_m, 0], [e.position.x_m, e.position.y_m, 0]], {minRange: 1100, caption: "DF drone search"});
      return;
    }
    case "pilot_search": {
      const dr = S.recs.get(baseRef(d.drone || ""));
      const dp = dr && (dr.pos || trackPos(dr.data, S.W.simT) || trackPos(dr.data, (dr.data.avail[0] + dr.data.avail[1]) / 2));
      E.pilots = E.pilots || [];
      (d.candidates || []).slice(0, 3).forEach((c, i) => {
        const p = [c.pos.x_m, c.pos.y_m, Math.max(2, c.pos.z_m || 2)], top = [p[0], p[1], p[2] + 18];
        const ents = [ovl(step, [[p[0], p[1], 0.5], top], {color: T.bone, width: 2, cased: false})];
        ents.push(...ovPoint(step, top, T.bone, `pilot? p ${Number(c.p_controller).toFixed(2)}${i === 0 ? ", best" : ""}`, "pilot", 18));
        if (dp) ents.push(ovl(step, [top, dp], {color: T.bone, width: 1.2, dashed: true, dash: 8, gap: 6, opacity: i === 0 ? 0.8 : 0.4}));
        E.pilots.push({ref: c.ref, ents});
      });
      const c0 = (d.candidates || [])[0];
      if (c0) {
        stamp(step, [c0.pos.x_m, c0.pos.y_m, Math.max(2, c0.pos.z_m || 2) + 18]);
        flyToPoints([[c0.pos.x_m, c0.pos.y_m, 0], ...(dp ? [dp] : [])], {minRange: 900, pitch: -45, caption: "Pilot candidates and the uplink to the drone"});
      }
      return;
    }
    case "tdoa_fix": {
      const p = d.pos, a0 = d.aoa_pos;
      const sens = (d.sensors || []).map((r) => S.sensors.get(r)).filter(Boolean);
      E.tdoaEnts = [];
      if (p && sens.length >= 2) {
        const f0 = sens[0].enu;
        for (const sk of sens.slice(1)) {
          const br = hyperbolaBranch([f0[0], f0[1]], [sk.enu[0], sk.enu[1]], [p.x_m, p.y_m], 2500);
          if (br.length > 1) E.tdoaEnts.push(ovl(step, br.map((q) => [q[0], q[1], 2]), {color: T.bone, width: 1.2, cased: false, opacity: 0.55}));
        }
      }
      if (a0) E.tdoaEnts.push(...ovPoint(step, [a0.x_m, a0.y_m, a0.z_m || 0], T.ash, null, "fix", 12));
      if (a0 && p) E.tdoaEnts.push(ovl(step, [[a0.x_m, a0.y_m, a0.z_m || 0], [p.x_m, p.y_m, p.z_m || 0]], {color: T.ash, width: 1.2, dashed: true, dash: 5, gap: 4}));
      if (p) {
        const sig = Number(d.range_diff_sigma_m) || 0;
        E.tdoaEnts.push(...ovFix(step, p, sig ? {a_m: sig, b_m: sig, theta_deg: 0} : null, T.bone, `AoA+TDOA fix, moved ${fmtM(d.shift_m || 0)}`));
        E.tdoaFix = [p.x_m, p.y_m, p.z_m || 0];
        stamp(step, [p.x_m, p.y_m, p.z_m || 0]);
        flyToPoints([[p.x_m, p.y_m, 0], ...sens.map((s2) => s2.enu)], {minRange: 1400, pitch: -55, caption: "TDOA hyperbolas and the hybrid fix"});
      }
      return;
    }
    case "signal_detail": {
      const tr = S.recs.get(baseRef(d.ref || ""));
      const at = ev.args?.at_time_s;
      const tp = tr && (at != null ? trackPos(tr.data, at) : null) || tr?.pos;
      if (tp) {
        for (const r of d.sensors || []) {
          const s2 = S.sensors.get(r);
          if (s2) ovl(step, [s2.enu, tp], {color: T.bone, width: 1, dashed: true, dash: 3, gap: 5, opacity: 0.55});
        }
        ovLabel(step, `${PROTO[d.protocol_estimate] || d.protocol_estimate}, ${d.freq_mhz?.median} MHz, SNR ${d.snr_db} dB`, tp);
        stamp(step, tp);
      }
      return;
    }
    case "noise_floor": {
      const s2 = S.sensors.get(d.sensor);
      if (s2) {
        const rise = Number(d.rise_over_thermal_db) || 0, ring = [];
        for (let k = 0; k <= 48; k++) { const g = (k / 48) * 2 * Math.PI; ring.push([s2.enu[0] + 120 * Math.sin(g), s2.enu[1] + 120 * Math.cos(g), 2]); }
        ovl(step, ring, {color: T.bone, width: rise >= 6 ? 2.5 : 1.2, cased: false, opacity: 0.8});
        ovLabel(step, `noise ${d.noise_dbm_per_mhz} dBm/MHz, ${rise >= 0 ? "+" : ""}${rise} dB over thermal`, [s2.enu[0], s2.enu[1] - 120, 2]);
        stamp(step, s2.enu);
      }
      return;
    }
    case "track_history": {
      const pts = (d.points || []).map((q) => [q.pos.x_m, q.pos.y_m, q.pos.z_m || 0]);
      if (pts.length > 1) {
        ovl(step, pts, {color: T.bone, width: 1.6, opacity: 0.85});
        for (const q of pts.filter((_, i) => i % 6 === 0)) ovPoint(step, q, T.bone, null, "fix", 7);
        stamp(step, pts[pts.length - 1]);
        flyToPoints(pts, {minRange: 900, pitch: -50, caption: `History of ${d.track || "the track"}`});
      }
      return;
    }
    case "event_search": eventTicks(d.items || []); E.events = E.events || new Map(); for (const it of d.items || []) E.events.set(it.ref, it); break;
    default: break;
  }
  const refs = ((ev.result && ev.result.refs) || []).filter((r) => /^(obj|zone|trk|sensor):/.test(r));
  if (refs.length && !["submit", "declare_intent"].includes(ev.name)) focusRefs(refs, {color: T.bone, auto: true});
}
/* hover isolation and the layer toggle never re-show what the agent's later steps or the answer retired; once the
   answer is in, the steps' marks recede (dimmed) under the answer layer, and a hovered step comes back to full strength */
const analysisOn = () => $("[data-layer=analysis]")?.checked !== false;
function isolateStep(step) {
  const on = analysisOn();
  for (const [k, objs] of S.stepObjs) for (const o of objs) {
    S.W.show(o, on && !o.userData.retired && (step == null || k === step));
    if (S.receding && !o.userData.ans) dim(o, step == null);
  }
  for (const o of S.ansObjs) { S.W.show(o, on && !o.userData.retired); if (o.userData.beacon) o.userData.dim = step != null; }
}
/* recede (or restore) one mark: lines, fills and sprites to a third of their opacity, labels to the dim style */
function dim(o, on) {
  if (!o) return;
  o.traverse?.((c) => {
    if (c.element) { c.element.classList.toggle("rec", on); return; }
    const m = c.material;
    if (!m || m.opacity == null) return;
    if (c.userData.op0 == null) c.userData.op0 = m.opacity;
    m.opacity = on ? c.userData.op0 * 0.33 : c.userData.op0;
  });
  if (o.element) o.element.classList.toggle("rec", on);
}
function recedeWork() {
  S.receding = true;
  for (const objs of S.stepObjs.values()) for (const o of objs) if (!o.userData.ans) dim(o, true);
  for (const rec of new Set(S.rays.values())) if (!rec.line.userData.ans) dim(rec.line, true);
}
function clearOverlays() {
  hideEcho();
  for (const objs of S.stepObjs.values()) for (const o of objs) S.W.remove(o);
  for (const rec of new Set(S.rays.values())) { S.W.remove(rec.line); S.W.remove(rec.label); }
  for (const o of S.ansObjs) S.W.remove(o);
  S.stepObjs.clear(); S.rays.clear(); S.ansObjs = []; S.receding = false;
  S.legend.clear(); renderLegend();
}
/* analysis legend: bottom-left of the map, only while an answer is being built or shown, only the layers drawn */
const LEGEND_ORDER = ["answer", "answerArea", "direct", "reflect", "echo", "los", "gap", "jam", "act", "cand", "own", "drone", "controller", "work"];
function legendAdd(...keys) { let ch = false; for (const k of keys) if (!S.legend.has(k)) { S.legend.add(k); ch = true; } if (ch) renderLegend(); }
function renderLegend() {
  const box = $("#mapLegend");
  if (!box) return;
  const keys = new Set(S.legend);
  if (S.drones) { keys.add("drone"); if (S.tracks.some((t) => t.klass !== "drone")) keys.add("controller"); }
  if (S.ep && S.step > 0) keys.add("work");
  const rows = LEGEND_ORDER.filter((k) => keys.has(k) && LEGEND[k]);
  box.hidden = !(S.ep || S.drones) || !rows.length;
  box.innerHTML = rows.map((k) => `<li>${LEGEND[k][1]}<span>${esc(LEGEND[k][0])}</span></li>`).join("");
}
function recedeTracks(on) {
  S.receded = on;
  for (const rec of S.recs.values()) {
    if (rec.kind !== "trk" || S.highlighted.includes(rec.data.ref)) continue;
    rec.receded = on;
    S.W.setSprite(rec.spr, sym("unknown", on ? T.ash : T.bone, on ? 12 : 20), on ? 12 : 20);
  }
}

/* ------------------------------------------------------------------ the agent stream */
/* 0-150 ms: acknowledge the question locally (not an agent claim, so Bone) */
function localAck(q) {
  const refs = new Set(), names = [];
  for (const m of q.matchAll(/\b(trk|zone|obj|sensor):[\w.-]+/gi)) { refs.add(m[0].replace(/[.,;]$/, "")); names.push(m[0]); }
  for (const m of q.matchAll(/\bsensor\s+(S[1-5])\b|\b(S[1-5])\b/g)) { const id = (m[1] || m[2]).toUpperCase(); refs.add(`sensor:${id}`); names.push(id); }
  for (const m of q.matchAll(/\b(TF-\d+|Jetty J\d+)\b/gi)) {
    const key = m[1].toLowerCase();
    const o = S.objs.find((x) => (x.name || "").toLowerCase() === key || (x.name || "").toLowerCase().endsWith(` ${key}`)
                                 || (x.aliases || []).some((al) => String(al).toLowerCase() === key));
    if (o) { refs.add(o.ref); names.push(m[1]); }
  }
  const list = [...refs].filter((r) => S.recs.has(r));
  if (!list.length) return;
  focusRefs(list, {fly: false, color: T.bone});
  const uniq = [...new Set(names)].slice(0, 3);
  camCaption(`Looking at ${uniq.length > 1 ? `${uniq.slice(0, -1).join(", ")} and ${uniq[uniq.length - 1]}` : uniq[0]}`);
}
function setNow(text) { S.nowHold = null; $("#nowText").textContent = text; }
function holdNow(text, ms, next) { $("#nowText").textContent = text; S.nowHold = {until: performance.now() + ms, next}; }
function turnBar(n) { $$("#turnBar i").forEach((c, i) => c.classList.toggle("on", i < n)); }
const plain = (html) => { const d = document.createElement("div"); d.innerHTML = html; return d.textContent; };
const elapsedS = () => (performance.now() - S.t0) / 1000;
function tickNow() {
  if (S.finished || !S.ep) return;
  const el = elapsedS();
  $("#nowClock").textContent = `${el.toFixed(1)} s`;
  for (const p of S.pending) if (p.li) p.li.querySelector(".ms").textContent = `${Math.round(performance.now() - p.t0)} ms`;
  if (S.nowHold && performance.now() > S.nowHold.until) setNow(S.nowHold.next);
  const typ = S.live?.taskId && S.replays[S.live.taskId]?.sft?.wall_s;
  const note = $("#nowNote");
  if (el > 8 && !S.live?.recorded) {
    note.hidden = false;
    $("#nowNoteText").textContent = `Still working: turn ${Math.max(1, S.ep.turn)} of ${MAX_TURNS}, ${el.toFixed(1)} s.${typ ? ` Typical for this question: ${typ} s.` : ""}`;
    if (el > 15 && !S.ep.offered) {
      S.ep.offered = true;
      $("#nowActs").innerHTML = (typ ? `<button type="button" class="btn" data-act="replay">Play the recorded run instead</button>` : "") +
        `<button type="button" class="btn" data-act="stop">Stop <kbd>Esc</kbd></button>`;
      wireActs($("#nowActs"));
    }
  } else note.hidden = true;
}
function wireActs(box) {
  box.querySelectorAll("[data-act]").forEach((b) => (b.onclick = () => {
    const tid = S.live?.taskId || S.taskId, q = S.ep?.question;
    if (b.dataset.act === "replay" && tid) playRecording(tid, "sft");
    else if (b.dataset.act === "ask" && q) ask({question: q});
    else if (b.dataset.act === "stop") resetAnalysis();
  }));
}
function newEpisode(question) {
  const dropped = S.ep && !S.finished ? S.step : -1;
  S.epId += 1;
  if (S.es) { S.es.close(); S.es = null; }
  clearOverlays(); clearHighlight(); closeGate();
  $("#layers").hidden = true; $("#layersBtn").setAttribute("aria-expanded", "false");
  $("#tlEvents").innerHTML = "";
  S.step = 0; S.pending = []; S.finished = false; S.t0 = performance.now(); S.userCam = false; S.W.userCam = false; S.camMoves = 0;
  S.ep = {bearings: new Map(), names: new Map(), reflectors: [], fits: [], victims: [], moves: 0, drone: null, turn: 0, tools: 0, question};
  heldHint(false); renderLegend();
  $("#idle").hidden = true; $("#question").hidden = false; $("#verdict").hidden = true; $("#compare").hidden = true;
  $("#qText").textContent = question; $("#qMeta").textContent = "";
  $("#qNote").hidden = dropped < 0;
  $("#qNote").textContent = dropped >= 0 ? `Previous question stopped at step ${dropped}; its answer was dropped.` : "";
  camCaption("View held from before the question");
  $("#steps").innerHTML = ""; $("#stepsBox").hidden = true;
  $("#working").hidden = false; $("#nowNote").hidden = true; $("#nowActs").innerHTML = "";
  setNow("Reading the question"); turnBar(0); $("#nowClock").textContent = "0.0 s";
  updateAsk();
  S.W.setAnalysis(true);
  recedeTracks(true);
  localAck(question);
  clearInterval(S.tick);
  S.tick = setInterval(tickNow, 100);
}
function gateBlocks() {
  if (!S.gateOpen) return false;
  banner("Decision pending.", "Hold A to acknowledge, R to dispute or Esc to defer the open assessment first.");
  return true;
}
function ask(opts = {}) {
  if (gateBlocks()) return;
  const q = (opts.question ?? $("#q").value).trim();
  if (!q) { $("#q").focus(); return; }
  hideDemo();
  newEpisode(q);
  S.live = {who: S.health.model, recorded: false, taskId: S.taskId};
  if (!liveOk()) {
    const t = new Date().toLocaleTimeString("en-GB", {hour: "2-digit", minute: "2-digit"});
    failure(S.health.gw ? "Model server offline" : "Gateway not reachable",
            `SceneScout cannot reach the language model at ${t}. Nothing was assessed.`, {retry: false});
    return;
  }
  const es = S.es = new EventSource(`/api/ask?mode=model&q=${encodeURIComponent(q)}`);
  es.onmessage = (m) => { let ev; try { ev = JSON.parse(m.data); } catch (_) { return; } onEvent(ev); };
  es.onerror = () => {
    es.close(); S.es = null;
    if (!S.finished) failure("Stream lost", `Stream lost after step ${S.step} at ${elapsedS().toFixed(1)} s. The answer was not received.`, {keepSteps: true});
  };
}
async function playRecording(taskId, who = "sft") {
  if (gateBlocks()) return;
  hideDemo();
  let rec;
  try { rec = await getJSON(`/api/replay?task_id=${encodeURIComponent(taskId)}&who=${who}`); }
  catch (e) { banner("No recording.", "This question has no recorded run on this machine (scripts/record_demo.py makes one)."); return; }
  startReplay(rec, taskId, who);
  const pace = replayPace(rec), token = S.epId;
  for (let i = 0; i < rec.events.length; i++) {
    if (pace[i] > 0) await wait(pace[i], true);
    if (S.epId !== token) return;
    onEvent(rec.events[i]);
  }
}
function startReplay(rec, taskId, who) {
  const q = rec.events.find((e) => e.type === "start")?.question || "";
  $("#q").value = q;
  S.taskId = taskId;
  newEpisode(q);
  S.live = {who, recorded: true, taskId, summary: rec.summary};
  $("#qMeta").innerHTML = `<span class="rec">Recorded run</span> by ${esc(rec.summary.label || modelName(who, true))}, ${esc(rec.summary.recorded_at.replace("T", " "))}, replayed at recorded speed`;
}
/* pace each event by its recorded duration (model generation, tool time), scaled to the recorded wall time: the wait in
   ms before each event (0 = immediately after the previous one) */
function replayPace(rec) {
  const dur = (e) => (e.type === "assistant" ? Number(e.gen_ms) || 0 : e.type === "tool_result" ? Number(e.result?.cost_ms) || 0 : 0);
  const sum = rec.events.reduce((a, e) => a + dur(e), 0) || 1;
  const k = Math.min(3, Math.max(0.5, (Number(rec.summary.wall_s) || sum / 1000) * 1000 / sum));
  return rec.events.map((ev) => { const ms = dur(ev) * k; return ms > 0 ? Math.max(ev.type === "tool_result" ? 150 : 40, ms) : 0; });
}
function addStep(html, cls = "") {
  const li = document.createElement("li");
  li.className = cls; li.innerHTML = html;
  $("#steps").appendChild(li);
  $("#stepsBox").hidden = false;
  li.scrollIntoView({block: "nearest"});
  return li;
}
const QUIET = new Set(["submit"]);                       // folded into the verdict line, not a step of its own
function onEvent(ev) {
  switch (ev.type) {
    case "start": S.episode = ev.episode_id; break;
    case "assistant":
      S.ep.turn = ev.turn; turnBar(ev.turn);
      if (!ev.n_calls) setNow("Writing the answer");
      break;
    case "tool_call": {
      setNow(nowFor(ev.name, ev.arguments));
      { const ta = ev.arguments || {}, tt = ta.at_time_s ?? ta.t_s ?? ta.t;          // the time the question is about
        if (tt != null && Number.isFinite(Number(tt))) S.ep.t = Number(tt); }
      if (QUIET.has(ev.name)) { S.pending.push({name: ev.name, li: null, n: 0, args: ev.arguments, t0: performance.now()}); break; }
      S.step += 1;
      if (ev.name !== "declare_intent") S.ep.tools += 1;
      const n = S.step;
      const li = addStep(`<span class="n">${n}</span><span class="nm" title="${esc(ev.name)}">${esc(TOOL_LABEL[ev.name] || ev.name.replace(/_/g, " "))}</span><span class="ms">0 ms</span>`, "run");
      li.dataset.step = String(n);
      li.onmouseenter = () => { if (S.stepObjs.has(n)) { li.classList.add("iso"); isolateStep(n); } };
      li.onmouseleave = () => { li.classList.remove("iso"); isolateStep(null); };
      S.pending.push({name: ev.name, li, n, args: ev.arguments, t0: performance.now()});
      break;
    }
    case "tool_result": onToolResult(ev); break;
    case "answer": renderAnswer(ev); break;
    case "error": { const [h, w] = errorCopy(ev); failure(h, w, {keepSteps: true}); break; }
    case "done":
      S.finished = true; $("#working").hidden = true; clearInterval(S.tick);
      if (S.es) { S.es.close(); S.es = null; }
      updateAsk();
      break;
    default: break;
  }
}
const ERR_CODE = {UNKNOWN_REF: "Unknown reference", INVALID_ARGS: "Request not understood", NOT_FOUND: "Nothing found",
                  OUT_OF_COVERAGE: "Outside coverage", TIMEOUT: "Tool timed out"};
function onToolResult(ev) {
  const res = ev.result || {};
  const k = S.pending.findIndex((p) => p.name === ev.name);
  const p = k >= 0 ? S.pending.splice(k, 1)[0] : null;
  const next = `Planning the next step (turn ${Math.min(MAX_TURNS, (S.ep.turn || 0) + 1)} of ${MAX_TURNS})`;
  if (QUIET.has(ev.name)) {
    if (!res.ok) {
      const e = res.error || {};
      addStep(`<span class="n"></span><span class="nm">Answer returned for correction</span><span class="ms"></span><span class="err"><b>${esc(ERR_CODE[e.code] || "Not accepted")}</b> ${esc(e.message || "")}</span>`);
    }
    setNow(res.ok ? "Checking the answer's citations" : next);
    return;
  }
  const li = p && p.li ? p.li : addStep(`<span class="n">${++S.step}</span><span class="nm">${esc(TOOL_LABEL[ev.name] || ev.name.replace(/_/g, " "))}</span><span class="ms"></span>`);
  const n = p && p.n ? p.n : S.step;
  li.classList.remove("run");
  li.querySelector(".ms").textContent = res.cost_ms != null ? `${res.cost_ms} ms` : "";
  if (res.ok) {
    const sum = summarize(ev.name, res, p && p.args);
    li.insertAdjacentHTML("beforeend", `<span class="sum">${sum}</span>`);
    drawToolOverlay({...ev, args: p && p.args}, n);
    holdNow(plain(sum), 600, next);
  } else {
    const e = res.error || {};
    li.insertAdjacentHTML("beforeend", `<span class="err"><b>${esc(ERR_CODE[e.code] || "Tool error")}</b> ${esc(e.message || "")}</span>`);
    holdNow(`${TOOL_LABEL[ev.name] || "The tool"} returned nothing usable`, 600, next);
  }
}
function summarize(name, res, args) {
  const d = res.data || {};
  const refs = (res.refs || []).length, tail = refs ? `, ${refs} refs` : "";
  const pl = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  switch (name) {
    case "df_hypotheses": { const nb = (d.sensors || []).reduce((a, s) => a + (s.hyps || []).length, 0);
      return `${esc(d.track || "")} at t ${d.t_s} s: <b>${pl(nb, "bearing")}</b> from ${pl((d.sensors || []).length, "sensor")}`; }
    case "fit_candidates": { const c = (d.candidates || [])[0];
      return `${esc(FIT_METHOD[d.method] || "Fit")}, ${d.n_tested != null ? `${d.n_tested} positions tested: ` : ""}<b>${pl((d.candidates || []).length, "candidate")}</b>` +
             (c ? `, best ${c.residual_deg_rms != null ? `${c.residual_deg_rms}° rms` : `${c.residual_m_rms} m rms`}` : ""); }
    case "effect_footprint": {
      const eff = typeof args?.effector === "string" ? shortRef(args.effector) : "the chosen point";
      return `${esc(band(d.band))}, ${d.eirp_dbm} dBm EIRP from ${esc(eff)}; threshold ${fmtDbm(args?.threshold_dbm ?? d.threshold_dbm)}`; }
    case "collateral_check": {
      const vs = d.violations || [], n = d.n_violations ?? vs.length;
      if (!n) return "Nothing above the threshold in the checked set";
      const KIND = {friendly_emitter: ["own link", "own links"], civil_comms: ["civil link", "civil links"], own_sensor: ["own DF sensor", "own DF sensors"],
                    protected_asset: ["protected asset", "protected assets"], aviation: ["aviation zone", "aviation zones"]};
      const c = {};
      for (const v of vs) c[v.kind] = (c[v.kind] || 0) + 1;
      const parts = Object.entries(c).map(([k2, m]) => `${m} ${(KIND[k2] || [k2, k2])[m === 1 ? 0 : 1]}`);
      return `<b>${parts.join(", ")}</b> above the threshold. Operator decision required`; }
    case "suggest_node": return `<b>${pl((d.candidates || []).length, "candidate site")}</b>`;
    case "weak_zones": return `<b>${pl((d.zones || []).length, "weak zone")}</b>`;
    case "move_drone": return `Flew to ${esc(shortRef(d.wp))}, uncertainty now <b>±${Math.round(d.estimate?.uncertainty?.a_m ?? 0)} m</b>`;
    case "drone_state": return `Start uncertainty <b>±${Math.round(d.estimate?.uncertainty?.a_m ?? 0)} m</b>${d.moves_left != null ? `, ${d.moves_left} moves left` : ""}`;
    case "plan_waypoints": return `${pl((d.waypoints || []).length, "waypoint")} ranked by expected uncertainty`;
    case "track_search": return `<b>${pl((d.items || []).length, "track")}</b> match`;
    case "track_detail": { const k2 = Object.entries(d.klass_probs || {}).sort((a, b) => b[1] - a[1])[0]?.[0];
      return `${esc(d.ref || "")}: ${esc(k2 ? `classed ${k2.replace(/_/g, " ")}` : "")}${d.confidence != null ? `, track confidence ${d.confidence}` : ""}`; }
    case "event_search": return `<b>${pl((d.items || []).length, "event")}</b>`;
    case "declare_intent": return `Plan: ${esc(INTENT_PLAN[d.intent || args?.intent] || String(d.intent || args?.intent || "").replace(/_/g, " "))}`;
    case "pilot_search": { const c = (d.candidates || [])[0];
      return c ? `<b>${pl(d.candidates.length, "pilot candidate")}</b>; best ${esc(c.ref)}, p(controller) ${c.p_controller}, ${fmtM(c.distance_to_drone_m)} from the drone`
               : "No pilot candidates within range"; }
    case "tdoa_fix": return `Fix moved <b>${fmtM(d.shift_m || 0)}</b> from the AoA estimate; range-difference 1σ ${fmtM(d.range_diff_sigma_m || 0)}${d.useful === false ? " (narrowband: TDOA adds little)" : ""}`;
    case "signal_detail": return `<b>${esc(PROTO[d.protocol_estimate] || d.protocol_estimate || "unknown")}</b>, ${d.freq_mhz?.median} MHz, ${d.bandwidth_khz} kHz, ${d.hopping ? `hopping over ${d.distinct_channels} channels` : "fixed channel"}, SNR ${d.snr_db} dB`;
    case "noise_floor": return `${esc(shortRef(d.sensor))}: <b>${d.noise_dbm_per_mhz} dBm/MHz</b>, ${d.rise_over_thermal_db} dB over thermal${d.clipping_pct ? `, clipping ${d.clipping_pct}%` : ""}`;
    case "track_history": return `${pl((d.points || []).length, "position")} over ${fmtM(d.path_len_m || 0)}`;
    case "site_resolve": { const it = d.candidates || d.items || []; return it.length ? `<b>${pl(it.length, "match")}</b>${it[0]?.name ? `, first ${esc(it[0].name)}` : ""}` : "No match in the port model"; }
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
    rejected: `<line x1="0" y1="5" x2="30" y2="5" stroke="${C.reflect}" stroke-width="2.5" stroke-dasharray="6 4"/>`,
    reflect: `<line x1="0" y1="5" x2="30" y2="5" stroke="${C.reflect}" stroke-width="2" stroke-dasharray="2 3"/>`,
    gap: `<rect x="1" y="1" width="28" height="8" fill="rgba(45,212,191,.3)" stroke="${C.answer}" stroke-width="1.5"/><line x1="4" y1="9" x2="12" y2="1" stroke="${C.gap}" stroke-width="2"/><line x1="14" y1="9" x2="22" y2="1" stroke="${C.gap}" stroke-width="2"/>`,
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
function refBtn(ref, cls = "") {
  const lbl = refLabel(ref);
  return `<button type="button" class="ref ${cls}" data-ref="${esc(ref)}"><span class="pri">${esc(lbl)}</span>${lbl !== String(ref) ? ` <span class="raw">${esc(ref)}</span>` : ""}</button>`;
}
const evRow = (kind, ref, role, cls = "") => `<li>${swatch(kind)}${refBtn(ref, cls)}<span class="role">${esc(role)}</span></li>`;

function verdictFor(a) {
  const E = S.ep;
  const ctx = {operator_query: "Operator query", df_ambiguity: "DF ambiguity", drone_search: "DF drone search", siting: "Sensor siting",
               collateral: "Collateral check", replay: "Replay", abstain: "No answer possible"}[a.intent] || String(a.intent).replace(/_/g, " ");
  let head = "", why = "", rows = [], next = "";
  let refused = false;
  if (a.intent === "abstain" && a.reason === "outside_tool_scope") {
    refused = true; head = "Not an action SceneScout can take"; why = ABSTAIN.outside_tool_scope;
  } else if (a.intent === "abstain") { head = "Cannot determine"; why = ABSTAIN[a.reason] || "The evidence does not support an answer."; }
  else if (a.intent === "operator_query") {
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
    if ((a.confidence || 0) < REVIEW_MARK) next = "Confirm with a second sensor before acting.";
    rows = a.chosen_hyp_refs.map((r) => evRow("chosen", r, "direct path", "ok")).concat((a.rejected_hyp_refs || []).map((r) => evRow("rejected", r, "rejected", "rej")));
    if (E.reflectors.length) rows.push(evRow("reflect", E.reflectors[0].ref, "predicted reflector"));
    if (E.fits[0]) rows.push(evRow("fix", E.fits[0].ref || "fix", "fix and 1σ ellipse", "ok"));
  } else if (a.intent === "drone_search") {
    const p = a.final_position;
    head = `Emitter fixed to ±${Math.round(a.uncertainty.a_m)} m`;
    const place = nearestPlace(p.x_m, p.y_m);
    const nm = E.moves || (a.waypoints_used || []).length;
    why = `The emitter is ${place}${p.z_m ? `, ${Math.round(p.z_m)} m up` : ""}, fixed after ${nm} drone move${nm === 1 ? "" : "s"}.`;
    next = "Hand the fix to the response team.";
    rows = [evRow("fix", "final fix", "1σ ellipse", "ok")].concat((a.waypoints_used || []).map((w) => evRow("own", w, "waypoint flown")));
  } else if (a.intent === "siting") {
    const nodes = a.node_candidates || [];
    const c = E.nodes && E.nodes.find((n) => nodes.includes(n.ref));
    head = nodes.length === 1 ? "Add one DF node here" : nodes.length ? `Add ${nodes.length} DF nodes here` : `${(a.zones || []).length} weak zone${(a.zones || []).length === 1 ? "" : "s"}`;
    why = c ? `The site is ${nearestPlace(c.position.x_m, c.position.y_m)}; predicted coverage gain ${c.predicted_gain}.` : "";
    rows = nodes.map((r) => evRow("node", r, "recommended site", "ok"))
      .concat((a.zones || []).filter((z) => z.type === "ref").map((z) => evRow("gap", z.ref, "weak zone", "ok")));
  } else if (a.intent === "collateral") {
    const nv = (a.violations || []).length, fp = E.fp || {};
    const KW = {friendly_emitter: ["own link", "own links"], civil_comms: ["civil link", "civil links"], own_sensor: ["own DF sensor", "own DF sensors"],
                protected_asset: ["protected asset", "protected assets"], aviation: ["aviation zone", "aviation zones"]};
    const kc = {};
    for (const v of E.victims || []) if ((a.violations || []).includes(v.ref)) kc[v.kind] = (kc[v.kind] || 0) + 1;
    const kparts = Object.entries(kc).map(([k2, m]) => `${m} ${(KW[k2] || [k2.replace(/_/g, " "), k2.replace(/_/g, " ")])[m === 1 ? 0 : 1]}`);
    head = nv ? (kparts.length ? `Jamming would hit ${kparts.join(" and ")}` : `Jamming would hit ${nv} of our own link${nv === 1 ? "" : "s"}`)
              : "No own links or civil comms would be hit";
    const lv = (E.victims || []).filter((v) => (a.violations || []).includes(v.ref));
    const eff = typeof fp.effector === "string" ? shortRef(fp.effector) : "the chosen point";
    why = nv ? `${lv.map((v) => `${shortRef(v.ref)} at ${fmtDbm(v.level_dbm)}`).join(", ")} ${lv.length === 1 ? "is" : "are"} above the ${fmtDbm(fp.threshold ?? -90)} threshold in the predicted ${band(fp.band)} footprint from ${eff}.`
             : `The predicted ${band(fp.band)} footprint stays below ${fmtDbm(fp.threshold ?? -90)} at every friendly link and civil receiver checked.`;
    next = "Decide below.";
    rows = (a.violations || []).map((r) => evRow("victim", r, "above threshold", "act"));
    if (a.fp_ref) rows.push(evRow("area", a.fp_ref, "predicted footprint", "ok"));
  } else if (a.intent === "replay") {
    head = a.time_s != null ? `At t ${a.time_s} s` : `${(a.evt_refs || []).length} event${(a.evt_refs || []).length === 1 ? "" : "s"}`;
    why = (a.evt_refs || []).length ? `From the event log: ${a.evt_refs.slice(0, 3).map((r) => esc(refLabel(r))).join(", ")}.` : "";
    rows = (a.evt_refs || []).map((r) => evRow("event", r, "event", "ok"));
  }
  for (const r of a.evidence_refs || []) if (!rows.some((x) => x.includes(`data-ref="${esc(r)}"`))) rows.push(evRow("none", r, "cited"));
  if (refused) return {ctx: "Refused", head, why, rows, refused};
  if (next) why += `${why ? " " : ""}<span class="next">Next: ${esc(next)}</span>`;
  return {ctx, head, why, rows, refused};
}

function renderAnswer(ev) {
  const a = ev.answer, v = ev.verifier;
  clearInterval(S.tick);
  $("#working").hidden = true;
  const wall = S.live && S.live.recorded ? S.live.summary.wall_s : elapsedS().toFixed(1);
  const when = new Date().toLocaleTimeString("en-GB", {hour: "2-digit", minute: "2-digit"});
  if (S.live && S.live.recorded) $("#qMeta").innerHTML = `<span class="rec">Recorded run</span> by ${esc(S.live.summary.label || modelName(S.live.who, true))}, ${esc(S.live.summary.recorded_at.replace("T", " "))}, answered in ${wall} s`;
  else $("#qMeta").textContent = `Asked ${when}, answered in ${wall} s by ${modelName(S.live?.who || S.health.model)}`;
  const box = $("#verdict");
  if (!a) {
    failure(END_REASON[ev.end_reason] || "No answer", ev.end_reason === "max_turns" ? "The agent used all 8 turns without submitting. Try a more specific question." : "The model's final answer could not be read or checked. Nothing was assessed.", {keepSteps: true});
    return;
  }
  const V = verdictFor(a), pct = Math.round((a.confidence || 0) * 100), low = (a.confidence || 0) < REVIEW_MARK;
  const flagged = ev.flagged === true || ev.status === "flagged" || !(v && v.ok), vok = !flagged;
  const abst = a.intent === "abstain";
  S.ep.answerHash = ev.answer_hash || null; S.ep.gateRequired = ev.gate_required;
  if (flagged && !V.refused) {
    const flags = (ev.flags && ev.flags.length ? ev.flags : [v?.code]).filter(Boolean);
    V.why = `The model answered "${esc(V.head)}", but the check found that ${esc(flags.map((c) => VERIFY[c] || "the answer could not be checked").join("; ") || "the answer could not be checked")}. Treat it as unconfirmed and ask again.`;
    V.ctx = `${V.ctx}, not verified`; V.head = "Answer not verified";
  }
  const assumed = (a.assumptions || []).map((t) => ASSUME.find(([re]) => re.test(t))?.[1]).filter(Boolean);
  box.className = `verdict ${V.refused ? "refused" : abst ? "abstain" : ""} ${vok && !abst ? "accepted" : ""} ${flagged && !V.refused ? "flagged" : ""}`;
  box.innerHTML = `<p class="ctx">${esc(V.ctx)}</p><p class="head" aria-live="polite">${esc(V.head)}</p>${V.why ? `<p class="why">${V.why}</p>` : ""}` +
    (abst || flagged ? "" :
    `<div class="conf"><div class="row"><span>Confidence, model-stated, not calibrated</span><span class="val">${pct}%</span></div>
       <div class="bar" aria-hidden="true"><i class="base"></i><i class="fill ${low ? "low" : ""}" style="width:${pct}%"></i>
         <i class="mk" style="left:${REVIEW_MARK * 100}%"></i></div>
       <div class="mkl"><span style="left:${REVIEW_MARK * 100}%">review mark</span></div>
       <div class="note ${low ? "review" : ""}">${low ? "Below the review mark: check the evidence on the map before acting." : "Above the review mark."}</div></div>`) +
    (V.rows.length ? `<ul class="evidence">${V.rows.join("")}</ul>` : "") +
    (assumed.length ? `<p class="assume">${assumed.map(esc).join(" ")}</p>` : "") +
    (V.refused ? `<div class="vline"><span>${ev.turns} turn${ev.turns === 1 ? "" : "s"}; no tool was asked to act</span></div>` :
    `<div class="vline"><span>${vok ? "<b>Citations checked</b>" : `<span class="bad">Citation check failed: ${esc(VERIFY[v?.code] || "the answer could not be checked")}</span>`}</span>
       <span>${ev.turns} turn${ev.turns === 1 ? "" : "s"}, ${S.ep.tools} tool call${S.ep.tools === 1 ? "" : "s"}</span>
       ${ev.end_reason && ev.end_reason !== "submitted" ? `<span>${esc(END_REASON[ev.end_reason] || "")}</span>` : ""}</div>`);
  box.hidden = false;
  if ((a.assumptions || []).length) addStep(`<span class="n"></span><span class="nm">Model's working assumptions</span><span class="ms"></span><span class="sum">${a.assumptions.map(esc).join("; ")}</span>`, "think");
  box.querySelectorAll("[data-ref]").forEach((b) => (b.onclick = () => handleRef(b.dataset.ref)));
  if (abst) { S.W.setAnalysis(false); clearHighlight(); recedeTracks(false); if (S.camMoves === 0) camCaption("View unchanged"); S.legend.clear(); renderLegend(); }
  if (vok && !abst) { paintAnswer(a, V); showEcho(a); }
  else box.querySelectorAll(".ref.ok, .ref.act").forEach((b) => b.classList.remove("ok", "act"));
  if (a.requires_human_confirm && vok) openGate(V, a, ev);
  else if (a.requires_human_confirm) box.insertAdjacentHTML("beforeend", `<p class="why" style="margin-top:12px">The citation check failed, so no operator decision is requested. Ask again.</p>`);
  compareBlock();
}
const VERIFY = {UNGROUNDED: "it cites evidence no tool returned", INCONSISTENT: "it contradicts the evidence it cites",
                INCOMPLETE_COLLATERAL: "it leaves out things the footprint would hit", RF_AMBIGUOUS: "the RF model is not trustworthy here and the answer does not say so",
                AMBIGUOUS_REFERENCE: "it names a place or track that is ambiguous", NOT_PERMITTED: "it skips the operator decision"};

/* ------------------------------------------------------------------ the answer layer on the map
   Every verified answer is drawn as its own layer above the agent's work (which recedes): filled translucent areas,
   a pulsing beacon at each key location, a callout with the answer's words, and a camera framing of all of it.
   Colours from marks.js C (answer sodium, act signal, reflect, gap, jam, cand, own); shape and dash carry each meaning
   as well. Abstentions and refusals draw nothing. */
function ansAdd(...objs) { for (const o of objs) if (o) { o.userData.ans = true; S.ansObjs.push(o); } return objs[0]; }
function aLine(pts, color, opt = {}) {
  const l = S.W.line(pts, {color, width: 3.5, ...opt});
  l.children.forEach((c) => (c.renderOrder += 20));
  return ansAdd(l);
}
const closeRing = (r, z = 2) => { const o = r.map((p) => [p[0], p[1], z]); if (o.length) o.push(o[0]); return o; };
function aArea(ring, color, opt = {}) { return ansAdd(area(S.W, ring, color, {order: 7, ...opt})); }
function aBeacon(p, color = C.answer) { return ansAdd(S.B.add(p, color)); }
function aCallout(text, p, cls = "") { return ansAdd(S.W.label(text, p, `callout ${cls}`.trim())); }
function circleRing(cx, cy, r, n = 64) { const o = []; for (let k = 0; k < n; k++) { const g = (k / n) * 2 * Math.PI; o.push([cx + r * Math.sin(g), cy + r * Math.cos(g)]); } return o; }
function ringCentroid(r) { let x = 0, y = 0; for (const p of r) { x += p[0]; y += p[1]; } return [x / r.length, y / r.length]; }
/* rings (site ENU) of an answer region: a zone or object ref, a weak-zone raster ref, a circle or a polygon */
function regionRings(reg) {
  if (!reg) return [];
  if (reg.type === "circle") return [circleRing(reg.center.x_m, reg.center.y_m, reg.radius_m)];
  if (reg.type === "polygon") return [reg.points.map((p) => [p.x_m, p.y_m])];
  const b = baseRef(reg.ref || "");
  if (b.startsWith("ras:")) return S.ep.weak?.get(b)?.rings || [];
  const rec = S.recs.get(b);
  return rec && rec.ring && rec.ring.length > 2 ? [rec.ring] : [];
}
function refPoint(ref, t) {
  const b = baseRef(ref), rec = S.recs.get(b);
  if (!rec) return null;
  if (rec.kind === "trk") { const at = String(ref).includes("@") ? Number(String(ref).split("@")[1]) : t; return trackPos(rec.data, at ?? S.W.simT) || rec.pos || null; }
  if (rec.kind === "sensor") return rec.data.enu;
  if (rec.ring && rec.ring.length > 2) { const c = ringCentroid(rec.ring); return [c[0], c[1], (rec.data.top || 0) + 2]; }
  return null;
}
const KLASS_WORD = {drone: "drone", rc_controller: "controller", wifi_device: "Wi-Fi device", unknown: "unknown"};
const EVT_WORD = {zone_enter: "entered", zone_exit: "left", loiter: "loitered in", band_change: "changed band", detection_start: "first detected"};

function paintAnswer(a, V) {
  const E = S.ep, pts = [], W = S.W;
  S.camMoves = Math.min(S.camMoves, 1);
  const ground = (p) => [p[0], p[1], 0];
  let camOpt = {pitch: -48, minRange: 700, caption: "The answer"};
  const zoneArea = (rings, {fill = C.answer, hatch = false, alpha = 0.22} = {}) => {
    for (const r of rings) {
      aArea(r, fill, {alpha, hatch, hatchAlpha: 0.75});
      aLine(closeRing(r), C.answer, {width: 3.5});
      pts.push(...r.map((p) => [p[0], p[1], 0]));
    }
    if (rings.length) legendAdd(fill === C.answer ? "answerArea" : "gap");
  };
  const t = (() => {
    for (const r of [...(a.track_refs || []), ...(a.evidence_refs || [])]) if (String(r).includes("@")) return Number(String(r).split("@")[1]);
    return a.time_s ?? E.t ?? null;
  })();
  if (a.intent === "operator_query") {
    if (t != null) setSimTime(t);
    updateTracks(W.simT);
    const trs = a.track_refs || [];
    const rings = regionRings(a.region);
    zoneArea(rings);
    const extra = [a.protocol ? (PROTO[a.protocol] || a.protocol) : null, a.band ? band(a.band) : null].filter(Boolean).join(", ");
    const yn = a.boolean === true ? "Yes" : a.boolean === false ? "No" : "";
    const sens = (a.evidence_refs || []).map(baseRef).filter((r) => r.startsWith("sensor:")).map((r) => S.sensors.get(r)).filter(Boolean);
    trs.forEach((r, i) => {
      const rec = S.recs.get(baseRef(r)), p = refPoint(r, t);
      if (!rec || !p) return;
      aBeacon(p); aLine([[p[0], p[1], 0.5], p], C.answer, {width: 2.5});
      if (sens.length === 1 && trs.length === 1 && a.boolean != null) {              // line-of-sight question
        aLine([sens[0].enu, p], C.answer, {width: 2.5, dashed: a.boolean === false, dash: 10, gap: 8});
        legendAdd("los"); pts.push(sens[0].enu);
      }
      if (i < 4) {
        const k = KLASS_WORD[rec.data.klass] || rec.data.klass || "track";
        const lead = i === 0 ? [yn, extra].filter(Boolean).join(", ") : "";
        aCallout(`${lead ? `${lead}: ` : ""}${baseRef(r)}, ${k}, ${Math.round(p[2])} m up${t != null ? `, t ${Math.round(t)} s` : ""}`, p);
      }
      pts.push(p, ground(p));
    });
    if (trs.length > 4 && pts.length) aCallout(`and ${trs.length - 4} more tracks`, pts[pts.length - 1]);
    if (trs.length) focusRefs(trs.map((r) => (t != null && !String(r).includes("@") ? `${baseRef(r)}@${t}` : r)), {fly: false, color: T.sodium});
    if (!trs.length) {
      const c = rings[0] ? ringCentroid(rings[0]) : null;
      const at = c || (() => { for (const r of a.evidence_refs || []) { const q = refPoint(r, t); if (q) return q; } return null; })();
      if (at) {
        aBeacon(at);
        const none = a.boolean == null && /^0 tracks$/.test(V.head);
        aCallout(`${none ? "No track matches" : a.boolean === false ? "No: no drone there" : yn || V.head}${extra ? `, ${extra}` : ""}`, [at[0], at[1], 4]);
        pts.push([at[0], at[1], 0]);
      }
    }
    camOpt = {pitch: -50, minRange: 600, caption: "Answered tracks"};
  } else if (a.intent === "df_ambiguity") {
    const chosen = new Set(a.chosen_hyp_refs.map(baseRef)), rejected = new Set((a.rejected_hyp_refs || []).map(baseRef));
    for (const rec of new Set(S.rays.values())) {
      const r = baseRef(rec.data.ref);
      const how = chosen.has(r) ? "chosen" : rejected.has(r) ? "rejected" : "other";
      styleRay(rec, how);
      if (how !== "other") { const [o, e] = rec.pts; pts.push(o, [o[0] + (e[0] - o[0]) * 0.3, o[1] + (e[1] - o[1]) * 0.3, 0]); }
      if (how === "chosen") {
        const [o, e] = rec.pts;
        aCallout(`Direct path: ${rec.data.sensor.replace("sensor:", "")} ${fmtDeg(rec.data.bearing_deg)}`, [o[0] + (e[0] - o[0]) * 0.22, o[1] + (e[1] - o[1]) * 0.22, o[2]]);
      }
    }
    legendAdd("direct"); if (rejected.size) legendAdd("reflect");
    for (const l of E.reflPaths || []) { S.W.setLine(l, {color: C.reflect, opacity: 0.95}); l.userData.ans = true; }
    if (E.fitEnts) { recolor(E.fitEnts, T.sodium); for (const o of E.fitEnts) if (o.element) { o.visible = false; o.userData.retired = true; } }
    const f = a.position || E.fits[0]?.position, u = a.uncertainty || E.fits[0]?.uncertainty;
    if (f) {
      const p = [f.x_m, f.y_m, f.z_m || 0];
      if (u && u.a_m) aArea(ellipsePts(f, u).map((q) => [q[0], q[1]]), C.answer, {alpha: 0.28});
      aBeacon(p);
      aCallout(`Emitter fix ${nearestPlace(f.x_m, f.y_m)}${u && u.a_m ? `, ±${Math.round(u.a_m)} m` : ""}`, p);
      pts.push(p, ground(p));
    }
    const tr = E.track && S.recs.get(baseRef(E.track));
    if (tr && tr.pos) pts.push(tr.pos);
    if (E.track) { setFocus(baseRef(E.track), true, T.sodium); S.highlighted.push(baseRef(E.track)); }
    legendAdd("answer");
    camOpt = {pitch: -34, minRange: 900, caption: "Chosen and rejected bearings, with the fix"};
  } else if (a.intent === "drone_search") {
    const f = a.final_position, p = [f.x_m, f.y_m, f.z_m || 0];
    if (E.estEnts) { const live = E.estEnts.filter((o) => !o.userData?.retired); recolor(live, T.sodium); for (const o of live) if (o.element) { o.visible = false; o.userData.retired = true; } }
    aArea(ellipsePts(f, a.uncertainty).map((q) => [q[0], q[1]]), C.answer, {alpha: 0.3});
    aLine(ellipsePts(f, a.uncertainty, 1.5), C.answer, {width: 3});
    aBeacon(p);
    aCallout(`Emitter located ${nearestPlace(f.x_m, f.y_m)}, ±${Math.round(a.uncertainty.a_m)} m`, p);
    pts.push(p, ...(E.drone || []));
    legendAdd("answer", "own");
    camOpt = {pitch: -42, minRange: 700, caption: "Located emitter"};
  } else if (a.intent === "siting") {
    const nodes = a.node_candidates || [];
    for (const [ref, {c, ents}] of E.nodeMap || []) {
      const p = c.position, top = [p.x_m, p.y_m, (p.z_m || 20) + 10];
      if (nodes.includes(ref)) {
        recolor(ents, T.sodium);
        for (const o of ents) { if (!o) continue; o.userData.ans = true; if (o.userData.top) S.W.setLine(o, {width: 4.5}); if (o.element) { o.visible = false; o.userData.retired = true; } }
        aBeacon(top);
        aCallout(`New DF node: ${refLabel(ref)}, coverage gain ${c.predicted_gain}`, top);
      }
      pts.push(top, [p.x_m, p.y_m, 0]);
    }
    const zs = a.zones || [];
    if (zs.length) for (const o of E.weakLbl || []) if (o) { o.visible = false; o.userData.retired = true; }   // the callouts say it now
    zs.forEach((reg, i) => {
      const rings = regionRings(reg);
      zoneArea(rings, {fill: C.gap, hatch: true, alpha: 0.3});
      const wz = reg.ref && E.weak?.get(baseRef(reg.ref));
      const c = wz ? [wz.centroid.x_m, wz.centroid.y_m] : rings[0] ? ringCentroid(rings[0]) : null;
      if (!c) return;
      aBeacon(c);
      if (i < 4) {
        const what = wz?.metric === "detect_prob" ? "Weak detection zone" : "Weak DF zone";
        const ha = wz ? (wz.area_m2 / 1e4).toFixed(1) : null;
        aCallout(`${what}${ha ? `, ${ha} ha` : ""}${wz?.worst != null && wz.metric === "df_err_p90_deg" ? `, bearing error up to ${Math.round(wz.worst)}°` : ""}`, [c[0], c[1], 4]);
      }
    });
    if (nodes.length) legendAdd("answer");
    camOpt = {pitch: zs.length ? -62 : -45, minRange: 700, caption: zs.length ? "Weak DF coverage" : "Recommended DF node site"};
  } else if (a.intent === "collateral") {
    for (const o of E.fpEnts || []) {
      o.userData.ans = true;
      if (o.isMesh) { o.material.opacity = o.userData.alpha ?? 0.14; o.renderOrder = 6; }
      else if (o.element) o.element.className = "wl band jam";
    }
    for (const r of E.fpOuter || []) { aLine(closeRing(r), C.answer, {width: 3}); pts.push(...r.map((q) => [q[0], q[1], 0])); }
    const vs = new Set(a.violations || []);
    let nc = 0;
    for (const [s, l, ref] of E.victimEnts || []) {
      if (!vs.has(ref)) continue;
      S.W.setSprite(s, sym("victim", T.signal, 24), 24); s.userData.ans = true;
      if (l) { l.visible = false; l.userData.retired = true; }
      const v = (E.victims || []).find((x) => x.ref === ref);
      if (!v || !v.pos) continue;
      aBeacon(v.pos, C.act);
      const thr = Number(E.fp?.threshold ?? -90);
      const lvl = v.note ? "receiver blanked" : `${fmtDbm(v.level_dbm)}${Number.isFinite(v.level_dbm) ? `, ${(v.level_dbm - thr).toFixed(0)} dB over` : ""}`;
      if (nc++ < 4) aCallout(`Would be hit: ${refLabel(ref)}, ${lvl}`, [v.pos[0], v.pos[1], 25], "act");
      pts.push([v.pos[0], v.pos[1], 0]);
    }
    const eff = E.fpEffector;
    if (eff) {
      aCallout(vs.size ? V.head : "No own link or civil receiver above the threshold", [eff[0], eff[1], 20], vs.size ? "act" : "");
      if (!vs.size) aBeacon(eff);
      pts.push([eff[0], eff[1], 0]);
    }
    legendAdd("answerArea", "jam"); if (vs.size) legendAdd("act");
    camOpt = {pitch: -55, minRange: 1200, caption: "Predicted jamming footprint and what it would hit"};
  } else if (a.intent === "replay") {
    if (a.time_s != null) setSimTime(a.time_s);
    updateTracks(W.simT);
    const evs = (a.evt_refs || []).map((r) => E.events?.get(baseRef(r))).filter(Boolean);
    const trks = new Set();
    evs.forEach((it, i) => {
      const p = [it.pos.x_m, it.pos.y_m, it.pos.z_m || 0];
      aBeacon(p);
      const z = it.zone_ref ? refLabel(it.zone_ref) : "";
      if (i < 4) aCallout(`${it.track_ref} ${EVT_WORD[it.type] || it.type.replace(/_/g, " ")}${z && /zone/.test(it.type) ? ` ${z}` : ""} at t ${Math.round(it.t_s)} s`, [p[0], p[1], 6]);
      if (it.zone_ref) zoneArea(regionRings({type: "ref", ref: it.zone_ref}), {alpha: 0.12});
      trks.add(`${it.track_ref}@${it.t_s}`);
      pts.push(p);
    });
    if (!evs.length) {
      for (const r of a.evidence_refs || []) {
        const b = baseRef(r), rec = S.recs.get(b);
        if (!rec) continue;
        if (rec.kind === "trk") {
          const tt = Math.min(a.time_s ?? t ?? rec.data.avail[1], rec.data.avail[1]), p = trackPos(rec.data, tt);
          if (!p) continue;
          aBeacon(p); aLine([[p[0], p[1], 0.5], p], C.answer, {width: 2.5});
          aCallout(`${b} at t ${Math.round(tt)} s, ${Math.round(p[2])} m up`, p);
          trks.add(`${b}@${tt}`); pts.push(p, ground(p));
        } else if (rec.ring && rec.ring.length > 2) {
          zoneArea([rec.ring], {alpha: 0.16});
          const c = ringCentroid(rec.ring);
          aCallout(/^0 events$/.test(V.head) ? "No event in this window" : V.head, [c[0], c[1], 4]);
          aBeacon(c);
        }
      }
    }
    if (trks.size) focusRefs([...trks], {fly: false, color: T.sodium});
    legendAdd("answer");
    camOpt = {pitch: -50, minRange: 700, caption: a.time_s != null ? `The answer at t ${Math.round(a.time_s)} s` : "Events from the log"};
  }
  /* signal-tool evidence cited by any answer: pilots and the hybrid fix */
  const cited = new Set([...(a.evidence_refs || []), ...(a.track_refs || [])].map(baseRef));
  for (const pc of E.pilots || []) if (cited.has(baseRef(pc.ref))) {
    recolor(pc.ents, T.sodium);
    const sp = pc.ents.find((o) => o?.isSprite);
    if (sp) { const p = [sp.position.x, -sp.position.z, sp.position.y]; aBeacon(p); aCallout(`Pilot candidate ${baseRef(pc.ref)}`, p); pts.push(ground(p)); }
    for (const o of pc.ents) if (o?.element) { o.visible = false; o.userData.retired = true; }
  }
  const trkCited = [...cited].some((r) => r.startsWith("trk:"));
  if (E.tdoaEnts && trkCited) recolor(E.tdoaEnts.filter((o) => o && (!o.isSprite || o.userData.px > 12)), T.sodium);
  if (E.tdoaFix && trkCited) { aBeacon(E.tdoaFix); pts.push(ground(E.tdoaFix)); }
  if (!S.ansObjs.length) {                                        // last resort: the cited places, so no answer is invisible
    for (const r of a.evidence_refs || []) {
      const p = refPoint(r, t);
      if (!p) continue;
      aBeacon(p); pts.push(ground(p));
      if (pts.length === 1) aCallout(V.head, p);
    }
  }
  if (!S.ansObjs.length) console.info("SceneScout: answer has no drawable geometry", a.intent);
  else legendAdd("answer");
  recedeWork();
  renderLegend();
  if (pts.length) {
    E.ansFrame = {points: pts, ...camOpt, pad: 0.2};
    flyToPoints(pts, {...camOpt, pad: 0.2});
  }
}

function failure(head, why, {keepSteps = false, retry = true} = {}) {
  clearInterval(S.tick);
  S.finished = true; $("#working").hidden = true;
  if (S.es) { S.es.close(); S.es = null; }
  S.W.setAnalysis(false); recedeTracks(false);
  if (!keepSteps || S.step === 0) { $("#stepsBox").hidden = true; clearHighlight(); }
  if (S.camMoves === 0) camCaption("View unchanged");
  const tid = S.live?.taskId || S.taskId, hasRec = !!(tid && S.replays[tid]?.sft) && !S.live?.recorded;
  const acts = (hasRec ? `<button type="button" class="btn" data-act="replay">Play the recorded run</button>` : "") +
               (retry && liveOk() ? `<button type="button" class="btn" data-act="ask">Ask again</button>` : "");
  const box = $("#verdict");
  box.className = "verdict fail";
  box.innerHTML = `<p class="ctx">No answer</p><p class="head">${esc(head)}</p><p class="why">${esc(why)}</p>` +
    (!liveOk() ? `<p class="why">Recorded runs of the demo questions still play: press <kbd>D</kbd>, then <kbd>Shift</kbd>+number.</p>` : "") +
    (acts ? `<div class="acts">${acts}</div>` : "");
  box.hidden = false;
  wireActs(box);
  updateAsk();
  compareBlock();
}

/* ------------------------------------------------------------------ operator gate */
const OPERATOR_ID = "console-operator";
const GATE_DECISION = {Acknowledged: "acknowledge", Disputed: "dispute", Deferred: "defer"};
const GATE_ERR = {unknown_episode: "the gateway has no record of this answer", hash_mismatch: "the answer changed since it was shown",
                  expired: "the decision window has closed", already_decided: "a decision is already recorded",
                  token_expired: "the review expired", too_soon: "confirmed too early", no_gate: "this answer needs no decision"};
function openGate(V, a, ev = {}) {
  const g = $("#gate"), fp = S.ep.fp || {};
  S.gateOpen = true;
  S.gate = {api: !!(S.apiPaths && S.apiPaths["/api/gate"] && ev.answer_hash && ev.gate_required && !S.live?.recorded),
            hash: ev.answer_hash, pending: null};
  const eff = typeof fp.effector === "string" ? shortRef(fp.effector) : "the chosen point";
  const vs = (a?.violations || []).map(shortRef);
  const list = vs.length > 1 ? `${vs.slice(0, -1).join(", ")} and ${vs[vs.length - 1]}` : vs[0];
  const rec = vs.length ? `Recommendation: do not jam ${band(fp.band)} from ${eff} at ${fp.eirp ?? "?"} dBm.`
                        : `No own link or civil receiver is predicted above ${fmtDbm(fp.threshold ?? -90)}.`;
  const where = S.gate.api ? "written to the audit log in two steps: decide, then confirm"
              : S.live?.recorded ? "shown on this console only; decisions on recorded runs are not logged"
              : "shown on this console only; it is not written to an audit log";
  g.innerHTML = `<p class="who" id="gateTitle">Agent recommends, operator decides</p>
    <p class="rec">${esc(rec)}</p>
    ${vs.length ? `<p class="sub">It would disrupt ${esc(list)}.</p>` : ""}
    <p class="safe">No tool in SceneScout can switch on an effector.</p>
    <details class="fine"><summary>Details</summary>
      <p>A predicted assessment of a ${esc(band(fp.band))} jammer at ${esc(String(fp.eirp ?? ""))} dBm from ${esc(eff)}, simulation model.
      Your decision is ${where}.</p></details>
    <div class="btns">
      <button type="button" class="hold" id="gAck"><i></i><span>Hold <kbd>A</kbd> to acknowledge</span></button>
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
  if (!S.gateOpen || S.gate?.pending) return;
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
function holdCancel() { cancelAnimationFrame(holdRaf); const bar = $("#gAck i"); if (bar && S.gateOpen && !S.gate?.pending) bar.style.width = "0"; }
const nowHMS = () => new Date().toLocaleTimeString("en-GB");
function gateDone(html) {
  S.gateOpen = false;
  const r = $("#gRes");
  r.hidden = false; r.innerHTML = html;
  $$("#gate button").forEach((b) => (b.disabled = true));
}
function gateDecide(what) {
  if (!S.gateOpen || S.gate?.pending) return;
  cancelAnimationFrame(holdRaf);
  if (!S.gate?.api) {
    gateDone(`<b>${esc(what)}</b> by the operator at ${nowHMS()}. ${S.live?.recorded ? "Recorded run: not logged." : "Shown on this console only, not logged."} Assessment only: nothing was switched on.`);
    return;
  }
  gateReview(what);
}
async function gatePost(body) {
  const res = await fetch("/api/gate", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({episode_id: S.episode, answer_hash: S.gate.hash, operator_id: OPERATOR_ID, ...body})});
  let out = {};
  try { out = await res.json(); } catch (_) { /* empty body */ }
  return {ok: res.ok && out.ok !== false, out};
}
async function gateReview(what) {
  const dec = GATE_DECISION[what], r = $("#gRes");
  S.gate.pending = {what, dec};
  $$("#gate .btns button").forEach((b) => (b.disabled = true));
  r.hidden = false; r.textContent = "Sending the decision for review…";
  let rv;
  try { rv = await gatePost({decision: dec, step: "review"}); } catch (_) { rv = {ok: false, out: {}}; }
  if (!S.gate) return;
  if (!rv.ok) { gateFail(what, rv.out); return; }
  S.gate.pending.token = rv.out.confirm_token;
  const wait_s = Number(rv.out.confirm_after_s) || 1;
  r.innerHTML = `<b>${esc(what)}</b>, reviewed at ${nowHMS()}. Confirm to write it to the audit log.
    <div class="acts"><button type="button" class="btn conf" id="gConf" disabled>Confirm <kbd>Enter</kbd></button>
    <button type="button" class="btn" id="gBack">Change decision</button></div>`;
  $("#gConf").onclick = () => gateConfirm();
  $("#gBack").onclick = () => { S.gate.pending = null; r.hidden = true; $$("#gate .btns button").forEach((b) => (b.disabled = false)); holdCancel(); };
  setTimeout(() => { const c = $("#gConf"); if (c) c.disabled = false; }, wait_s * 1000 + 50);
}
async function gateConfirm() {
  const p = S.gate?.pending;
  if (!p || !p.token || $("#gConf")?.disabled) return;
  $("#gConf").disabled = true;
  let cf;
  try { cf = await gatePost({decision: p.dec, step: "confirm", confirm_token: p.token}); } catch (_) { cf = {ok: false, out: {}}; }
  if (!cf.ok) { gateFail(p.what, cf.out); return; }
  S.gate.pending = null;
  gateDone(`<b>${esc(p.what)}</b> by the operator at ${nowHMS()}, written to the audit log (record ${esc(String(cf.out.record_id || "").slice(0, 12))}). Assessment only: nothing was switched on.`);
}
function gateFail(what, out) {
  if (!S.gate) return;                              // the question was closed while the request was in flight
  S.gate.pending = null;
  gateDone(`<b>${esc(what)}</b> by the operator at ${nowHMS()}, but the audit log did not record it: ${esc(GATE_ERR[out?.code] || "the gateway did not answer")}. Assessment only: nothing was switched on.`);
}
function closeGate() { S.gateOpen = false; S.gate = null; $("#gate").hidden = true; $("#gate").innerHTML = ""; }

/* ------------------------------------------------------------------ compare with the untrained model */
function compareBlock() {
  const box = $("#compare"), tid = S.live && S.live.taskId, r = tid && S.replays[tid];
  if (!r || !r.scenescout) { box.hidden = true; return; }
  const base = r.scenescout, ours = S.live.recorded ? S.live.summary : r.sft;
  const out = (s) => (s.end_reason === "submitted" ? (s.score >= 0.95 ? "Correct answer" : s.score > 0 ? "Partly correct" : "Wrong answer") : END_REASON[s.end_reason] || s.end_reason);
  box.innerHTML = `<h2>Same question, untrained model</h2>
    <div class="cols">
      <div class="col"><h3>4B base, untrained</h3><p class="out">${esc(out(base))}</p>
        <dl><dt>Turns</dt><dd>${base.turns} of 8</dd><dt>Time</dt><dd>${base.wall_s} s</dd><dt>Score</dt><dd>${base.score.toFixed(2)}</dd></dl></div>
      <div class="col"><h3>${esc((ours && ours.label) || "SceneScout (recorded)")}</h3><p class="out">${esc(out(ours || {end_reason: "?"}))}</p>
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
    if (rec) { flyToPoints(rec.pts, {minRange: 1400, auto: false, caption: `${shortRef(rec.data.sensor)} bearing ${fmtDeg(rec.data.bearing_deg)}`}); showInfo(rec); }
    return;
  }
  if (b.startsWith("ras:")) {
    const z = S.ep?.weak?.get(b);
    if (z && z.rings.length) flyToPoints(z.rings.flat().map((p) => [p[0], p[1], 0]), {minRange: 500, pitch: -60, auto: false, caption: refLabel(b)});
    return;
  }
  if (S.recs.has(b)) { focusRefs([ref], {color: T.sodium}); showInfo(S.recs.get(b)); }
}
const dl = (rows) => `<dl>${rows.filter(([, v]) => v != null && v !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
function showInfo(rec) {
  const box = $("#info");
  if (!rec) { box.hidden = true; return; }
  const d = rec.data;
  let html = "";
  if (rec.kind === "obj") {
    const FID = {osm: "modelled on a mapped footprint", estimated: "estimated", ml_footprint: "machine-detected footprint, estimated",
                 illustrative: "illustrative detail", modelled: "modelled in the Blender world"};
    html = `<h3>${esc(d.name || d.id)}${d.synthetic ? " <small>(label)</small>" : ""}</h3><div class="sub">${esc(String(d.kind).replace(/_/g, " "))}, ${esc(d.ref)}${d.protected ? ", protected" : ""}</div>` +
      dl([["Top", d.top != null ? `${Number(d.top).toFixed(1)} m above the quay` : "unknown"],
          ["Geometry", d.src === "blender" ? `Blender world, ${esc(FID[d.fidelity] || d.fidelity || "estimated")}` : `Blender world footprint; height ${esc(FID[d.fidelity] || "estimated")}`],
          ["Name", d.synthetic ? "synthetic label" : "mapped name"], ["In agent tools", d.in_tools ? "yes" : "no (shown for picking only)"]]);
  } else if (rec.kind === "zone") {
    html = `<h3>${esc(d.name)}</h3><div class="sub">${esc(d.kind)} zone, ${esc(d.ref)}</div>` + dl([["Source", esc(d.provenance?.source)], ["Status", esc(d.provenance?.status)]]);
  } else if (rec.kind === "sensor") {
    html = `<h3>Sensor ${esc(d.id)}</h3><div class="sub">${d.type === "drone_df" ? "DF drone" : "Fixed AIR7311 DF node"}, ${esc(d.placement || "synthetic")} site</div>` +
      dl([["Mast", `${d.z_m} m`], ["Bands", esc((d.bands || []).map(band).join(", "))], ["Position", `${d.lat.toFixed(5)}, ${d.lon.toFixed(5)}`]]);
  } else if (rec.kind === "trk") {
    const p = rec.pos;
    html = `<h3>${esc(d.ref)}</h3><div class="sub">${esc(d.klass)}, ${esc(band(d.band))}${d.group ? `, swarm ${esc(d.group)}` : ""}</div>` +
      dl([["Track confidence", d.confidence != null ? d.confidence.toFixed(2) : "?"], ["Seen", `t ${d.avail[0]} to ${d.avail[1]} s`],
          ["Now", p ? `${Math.round(p[2])} m up` : "not seen at this time"]]);
  } else if (rec.kind === "ray") {
    html = `<h3>${esc(shortRef(d.sensor))} bearing ${fmtDeg(d.bearing_deg)}</h3><div class="sub">${esc(d.ref)}</div>` +
      dl([["Sensor", esc(shortRef(d.sensor))], ["SNR", d.snr_db != null ? `${d.snr_db} dB` : null], ["Track", esc(d.track)], ["Time", d.t != null ? `t ${d.t} s` : null]]);
  }
  box.innerHTML = `<button class="x" aria-label="Close">×</button>${html}`;
  box.hidden = false;
  box.querySelector(".x").onclick = () => (box.hidden = true);
}
const PICK_ORDER = {trk: 0, grp: 1, sensor: 2, ray: 3, obj: 4, zone: 5};

/* ------------------------------------------------------------------ time, clock, compass */
function timelineTicks() {
  let html = "";
  for (let t = 0; t <= S.dur; t += 30) {
    html += `<i class="${t % 60 === 0 ? "maj" : ""}" style="left:${(t / S.dur) * 100}%"></i>`;
    if (t % 120 === 0) html += `<span style="left:${(t / S.dur) * 100}%">${t}</span>`;
  }
  $("#tlTicks").innerHTML = html;
  $("#tl").setAttribute("aria-valuemax", String(S.dur));
}
function eventTicks(items) {
  $("#tlEvents").innerHTML = items.map((it) => `<b style="left:${(it.t_s / S.dur) * 100}%" title="${esc(`${it.type.replace("_", " ")} ${it.track_ref} ${(it.zone_ref || "").replace("zone:", "")} t ${it.t_s} s`)}" data-t="${it.t_s}"></b>`).join("");
  $$("#tlEvents b").forEach((b) => (b.onclick = (e) => { e.stopPropagation(); setSimTime(Number(b.dataset.t)); }));
}
let lastSec = -1, cullN = 0;
function cullLabels() {
  const cam = S.W.camera.position;
  const d = (o) => cam.distanceTo(o.position);
  for (const rec of S.recs.values()) {
    if (!rec.label || rec.kind === "sensor") continue;
    const max = rec.kind === "zone" ? (rec.data.kind === "restricted" ? 1900 : 16000) : rec.kind === "obj" ? 7000 : rec.kind === "trk" ? (rec.data.group ? 2600 : 9000) : 1e9;
    if (rec.kind === "trk") { if (rec.label.visible && d(rec.label) > max && !S.highlighted.includes(rec.data.ref)) rec.label.visible = false; }
    else rec.label.element.style.visibility = d(rec.label) > max ? "hidden" : "";
  }
}
/* Screen-space label layout, every frame: labels are placed in priority order (verdict evidence, step stamps, other
   analysis marks, sensors, tracks, zones and places); a label that would overlap one already placed moves down in
   22 px rows and gets a 1 px leader to its anchor. Stamps sit 60 px above their mark on a leader. Low-priority
   labels that cannot be placed are hidden rather than stacked. */
let _dcV = null;
function dcPrio(cl) {
  if (cl.contains("callout")) return -1;
  if (cl.contains("rec")) return 7;
  if (cl.contains("stamp")) return 1;
  if (cl.contains("ok") || cl.contains("act")) return cl.contains("zone") || cl.contains("band") ? 2 : 0;
  if (cl.contains("zone") || cl.contains("place")) return 6;
  if (cl.contains("band")) return 5;
  if (cl.contains("trk") || cl.contains("grp")) return 4;
  if (cl.contains("fr")) return 3;
  return 2;
}
function declutter() {
  const W = S.W;
  if (!W || !W.overlay) return;
  const cam = W.camera, w = W.el.clientWidth, h = W.el.clientHeight;
  _dcV = _dcV || cam.position.clone();
  const items = [];
  for (const o of W.overlay.children) {
    const el = o.element;
    if (!el || !o.visible || el.style.display === "none" || el.style.visibility === "hidden") continue;
    _dcV.copy(o.position).project(cam);
    if (_dcV.z < -1 || _dcV.z > 1) continue;
    const key = `${el.className.replace(" lead", "")}|${el.textContent}`;
    if (el._dcKey !== key) { el._dcKey = key; el._dcW = el.offsetWidth; el._dcH = el.offsetHeight; el._dcLead = ""; }
    if (!el._dcW) { el._dcKey = null; continue; }             // not laid out yet (first frame): measure again next frame
    const ax = (_dcV.x + 1) / 2 * w, ay = (1 - _dcV.y) / 2 * h;
    if (ax < -200 || ax > w + 200 || ay < -100 || ay > h + 100) continue;
    items.push({el, ax, ay, pr: dcPrio(el.classList), cw: el._dcW, ch: el._dcH});
  }
  items.sort((a, b) => a.pr - b.pr);
  const placed = [];
  const hit = (x, y, cw, ch) => placed.some((r) => x < r.x + r.w + 3 && x + cw + 3 > r.x && y < r.y + r.h + 2 && y + ch + 2 > r.y);
  for (const it of items) {
    const {el, ax, ay, cw, ch, pr} = it;
    const cl = el.classList, centred = cl.contains("zone") || cl.contains("place") || cl.contains("grp") || cl.contains("band");
    const callL = cl.contains("callout");
    let x = centred ? ax - cw / 2 : ax + 14, y = cl.contains("stamp") ? ay - 60 - ch / 2 : ay - ch / 2;
    if (cl.contains("stamp")) x = ax - cw / 2;
    if (callL) { x = ax + 22; y = ay - 52 - ch / 2; }               // answer callout: up and right of its beacon, on a leader
    /* candidate slots: as wanted, then rows below (stamps: above), rows above, then the other side of the anchor */
    const stampL = cl.contains("stamp") || callL, dir = stampL ? -(callL ? 30 : 22) : 22;
    const alts = [[0, 0]];
    for (let r = 1; r <= 5; r++) alts.push([0, r * dir]);
    for (let r = 1; r <= 3; r++) alts.push([0, -r * dir]);
    if (!centred && !stampL) for (const r of [0, 1, -1, 2, -2]) alts.push([-(cw + 28), r * 22]);
    if (callL) for (const r of [0, 1, 2, 3]) alts.push([-(cw + 44), -r * 30]);
    const outside = (xx, yy) => callL && (xx < 4 || xx + cw > w - 4 || yy < 44 || yy + ch > h - 4);   // callouts stay whole on screen
    let slot = alts.find(([dx, dy]) => !hit(x + dx, y + dy, cw, ch) && !outside(x + dx, y + dy));
    const hide = !slot && pr >= 3;
    if (!slot) slot = [0, 0];
    x += slot[0]; y += slot[1];
    const k = slot[0] || slot[1] ? 1 : 0;
    if (hide) { if (el.style.opacity !== "0") el.style.opacity = "0"; continue; }
    if (el.style.opacity === "0") el.style.opacity = "";
    placed.push({x, y, w: cw, h: ch});
    // CSS2DRenderer centres the element on its anchor; margins move it to the placed rectangle
    const ml = Math.round(x - (ax - cw / 2)), mt = Math.round(y - (ay - ch / 2));
    if (el._dcML !== ml) { el.style.marginLeft = `${ml}px`; el._dcML = ml; }
    if (el._dcMT !== mt) { el.style.marginTop = `${mt}px`; el._dcMT = mt; }
    const lead = cl.contains("stamp") || callL || k > 0;
    if (lead) {
      const lx = Math.max(0, Math.min(cw, ax - x)), ly = Math.max(0, Math.min(ch, ay - y));
      const dx = ax - x - lx, dy = ay - y - ly, L = Math.hypot(dx, dy);
      const sig = `${Math.round(lx)},${Math.round(ly)},${Math.round(L)},${Math.round(Math.atan2(dy, dx) * 57.3)}`;
      if (el._dcLead !== sig) {
        el._dcLead = sig;
        el.classList.toggle("lead", L > 4);
        el.style.setProperty("--lx", `${lx}px`); el.style.setProperty("--ly", `${ly}px`);
        el.style.setProperty("--ll", `${L}px`); el.style.setProperty("--la", `${Math.atan2(dy, dx)}rad`);
      }
    } else if (el._dcLead) { el._dcLead = ""; el.classList.remove("lead"); }
  }
}
function onTick(t, dt = 0.016) {
  S.cam.tick(dt);
  echoTick(dt);
  updateTracks(t);
  if (S.drones) { S.D.highlight(S.highlighted); S.D.update(t); }
  S.B.tick();
  cullLabels();
  declutter();
  $("#tlHead").style.left = `${(t / S.dur) * 100}%`;
  const s = Math.round(t);
  if (s !== lastSec) {
    lastSec = s;
    $("#tlHeadText").textContent = `t ${s} s`;
    $("#sysSim .v").textContent = `t ${s} / ${S.dur} s${S.W.playing ? `, ${S.W.mult}x` : ""}`;
    $("#tl").setAttribute("aria-valuenow", String(s));
  }
  const v = S.W.view();
  $("#northNeedle").setAttribute("transform", `rotate(${-v.heading})`);
  const mpp = S.W.metresPerPixel();
  const nice = [10, 20, 50, 100, 200, 250, 500, 1000, 2000, 5000].find((m) => m / mpp >= 60) || 5000;
  $("#scaleBar").style.width = `${Math.round(nice / mpp)}px`;
  $("#scaleText").textContent = fmtM(nice);
}
function clockTick() { $("#sysClock .v").textContent = new Date().toLocaleTimeString("en-GB", {timeZone: "Asia/Kolkata"}); }
function setPlay(on) { S.W.playing = on; $("#play").textContent = on ? "Pause" : "Play"; $("#play").setAttribute("aria-label", on ? "Pause" : "Play"); }
function scrub(ev) { const r = $("#tl").getBoundingClientRect(); setSimTime(((ev.clientX - r.left) / r.width) * S.dur); }

/* ------------------------------------------------------------------ demo menu, banner, keys */
function demoMenu() {
  const BEAT = ["genuine_drone", "direct_path", "ds_hover", "suggest_one", "jam_links", "actuation"];
  $("#demoList").innerHTML = S.demo.map((d, i) => {
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
  if (recorded || !liveOk()) playRecording(d.task_id, "sft");
  else ask({question: d.question});
}
function toggleDemo() {
  const m = $("#demo");
  m.hidden = !m.hidden;
  $("#demoBtn").setAttribute("aria-expanded", String(!m.hidden));
  if (!m.hidden) $("#demoList button")?.focus();
}
function hideDemo() { $("#demo").hidden = true; $("#demoBtn").setAttribute("aria-expanded", "false"); }
function banner(head, msg, kind = "", ms = 9000) {
  const b = $("#banner");
  b.className = `banner ${kind}`;
  b.innerHTML = `<b>${esc(head)}</b> ${esc(String(msg).trim())}`;
  b.hidden = false;
  clearTimeout(banner.t); banner.t = setTimeout(() => (b.hidden = true), ms);
}
function resetAnalysis() {
  S.epId += 1;
  if (S.es) { S.es.close(); S.es = null; }
  clearOverlays(); clearHighlight(); closeGate();
  S.W.setAnalysis(false); recedeTracks(false);
  $("#verdict").hidden = true; $("#question").hidden = true; $("#stepsBox").hidden = true; $("#compare").hidden = true;
  $("#working").hidden = true; $("#idle").hidden = false; clearInterval(S.tick); $("#qNote").hidden = true;
  S.finished = true; S.ep = null; updateAsk();
  camCaption("Port overview"); S.prevView = null; $("#camBack").hidden = true;
  heldHint(false); renderLegend();
}
function setDrones(on) {
  S.drones = on;
  S.D.show(on);
  $("#dronesBtn").classList.toggle("on", on); $("#dronesBtn").setAttribute("aria-pressed", String(on));
  updateTracks(S.W.simT);
  renderLegend();
  if (on && !S.W.playing && !S.ep) banner("Live drones.", "Observed tracks as flying drones. Press Play or Space to run the timeline; click a drone to select its track.", "ok", 5000);
}
function setFollow(on) {
  S.follow = on;
  $("#followBtn").classList.toggle("on", on); $("#followBtn").setAttribute("aria-pressed", String(on));
  if (on) { S.userCam = false; S.W.userCam = false; }
}
/* select a track (map click or drone click): highlight it, jump the timeline into its span, show its card */
function selectTrack(ref) {
  const rec = S.recs.get(baseRef(ref));
  if (!rec) return;
  focusRefs([ref], {fly: false, color: T.sodium});
  showInfo(rec);
}
function wireUI() {
  const cv = S.W.renderer.domElement;
  let down = null;
  cv.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  cv.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    if (S.cam?.flying) return;
    const dref = S.drones && S.D.pick(e.clientX, e.clientY);
    if (dref) { selectTrack(dref); return; }
    const recs = S.W.pick(e.clientX, e.clientY);
    recs.sort((a, b) => (PICK_ORDER[a.kind] ?? 9) - (PICK_ORDER[b.kind] ?? 9));
    if (recs[0]?.kind === "trk") { selectTrack(recs[0].data.ref); return; }
    showInfo(recs[0]);
  });
  $$(".views [data-view]").forEach((b) => (b.onclick = () => { view(b.dataset.view); if (S.ep) userMoved(); }));
  $("#dronesBtn").onclick = () => setDrones(!S.drones);
  $("#flyBtn").onclick = () => (S.cam.flying ? S.cam.exitFly() : S.cam.enterFly());
  $("#followBtn").onclick = () => setFollow(!S.follow);
  $("#heldHint button").onclick = () => frameAnswer();
  $("#layersBtn").onclick = () => { const l = $("#layers"); l.hidden = !l.hidden; $("#layersBtn").setAttribute("aria-expanded", String(!l.hidden)); };
  $$("[data-layer]").forEach((cb) => (cb.onchange = () => {
    const on = cb.checked, n = cb.dataset.layer;
    if (n === "structures") for (const m of Object.values(S.W.models)) m.visible = on;
    else if (n === "analysis") { isolateStep(null); for (const r of new Set(S.rays.values())) { S.W.show(r.line, on); S.W.show(r.label, on && !r.label.userData.retired); } }
    else if (n === "tracks") LAYER_ON.tracks = on;
    if (n === "analysis" || n === "echo") echoRefresh();
    else for (const o of LAYER[n] || []) S.W.show(o, on);
  }));
  $("#camBack").onclick = () => {
    if (!S.prevView) return;
    S.W.animateTo(S.prevView.target, S.prevView.position, 1.0);
    camCaption(S.prevView.caption); S.prevView = null; $("#camBack").hidden = true;
  };
  const fid = $("#fidelity");
  const tog = () => { const m = $("#fidMore"); m.hidden = !m.hidden; fid.setAttribute("aria-expanded", String(!m.hidden)); };
  fid.onclick = tog; fid.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tog(); } };
  $("#askForm").onsubmit = (e) => { e.preventDefault(); ask(); };
  $("#q").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } };
  $("#q").oninput = () => { const d = S.demo.find((x) => x.question === $("#q").value.trim()); S.taskId = d ? d.task_id : null; };
  $("#demoBtn").onclick = toggleDemo;
  $("#stopBtn").onclick = () => resetAnalysis();
  $("#play").onclick = () => setPlay(!S.W.playing);
  const tl = $("#tl");
  let drag = false;
  tl.onpointerdown = (e) => { drag = true; tl.setPointerCapture(e.pointerId); scrub(e); };
  tl.onpointermove = (e) => { if (drag) scrub(e); };
  tl.onpointerup = () => { drag = false; };
  tl.onkeydown = (e) => { if (e.key === "ArrowRight") setSimTime(now() + 10); if (e.key === "ArrowLeft") setSimTime(now() - 10); };
  wireSearch();
  $$("input[name=rfmap]").forEach((r) => (r.onchange = () => {    // one ground raster at a time: an RF map hides multipath
    if (r.value && $("#mpOn").checked) { $("#mpOn").checked = false; loadMultipath(); }
    loadRfMap(); echoRefresh();
  }));
  $("#rfBand").onchange = loadRfMap; $("#rfHeight").onchange = loadRfMap;
  $("#mpOn").onchange = () => {
    if ($("#mpOn").checked) { const none = document.querySelector("input[name=rfmap][value='']"); if (!none.checked) { none.checked = true; loadRfMap(); } }
    loadMultipath(); echoRefresh();
  };
  $("#mpSensor").onchange = loadMultipath; $("#mpBand").onchange = loadMultipath; $("#mpHeight").onchange = loadMultipath;
  $("#mpFly").onclick = () => flyToSpot(0);
  document.addEventListener("keydown", (e) => {
    const typing = /^(TEXTAREA|INPUT)$/.test(e.target.tagName);
    const mod = e.ctrlKey || e.metaKey || e.altKey;
    /* fly mode: its movement keys (WASD, Q, E, Shift) win over every other binding; Esc or F leaves it */
    if (S.cam?.flying && !typing && !mod) {
      if (e.key === "Escape" || e.key === "f" || e.key === "F") { S.cam.exitFly(); e.preventDefault(); return; }
      if (/^[wasdqeWASDQE]$/.test(e.key) || e.key === "Shift") { e.preventDefault(); return; }
      if (e.key === "t" || e.key === "T" || /^Digit[1-4]$/.test(e.code)) S.cam.exitFly();
    }
    if (S.gateOpen && !typing && !mod) {
      if (S.gate?.pending) { if (e.key === "Enter") { gateConfirm(); e.preventDefault(); } return; }
      if (e.key === "a" || e.key === "A") { if (!e.repeat) holdStart(); e.preventDefault(); return; }
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
    if (typing || mod) return;
    const dig = /^Digit([1-9])$/.exec(e.code);
    if (!$("#demo").hidden && dig) { pickDemo(Number(dig[1]) - 1, e.shiftKey); e.preventDefault(); return; }
    if (e.key === "d" || e.key === "D") { toggleDemo(); e.preventDefault(); }
    else if (e.key === "w" || e.key === "W") { document.documentElement.classList.toggle("wall"); setTimeout(() => S.W.resize(), 50); }
    else if (e.key === "q" || e.key === "Q") { S.hiq = !S.hiq; S.W.setQuality(S.hiq); banner(S.hiq ? "Full resolution." : "Balanced resolution.", S.hiq ? " Rendering at the display's native pixel density." : " Rendering at 1x for speed."); }
    else if (e.key === "t" || e.key === "T") { const top = S.W.topView(); banner(top ? "Plan view." : "Oblique view.", top ? " North up. Press T to return." : ""); e.preventDefault(); }
    else if (e.key === " ") { setPlay(!S.W.playing); e.preventDefault(); }
    else if (e.key === "/") { $("#q").focus(); e.preventDefault(); }
    else if (e.key === "F" && e.shiftKey) { $("#layers").hidden = false; $("#search").focus(); e.preventDefault(); }
    else if (e.key === "f") { S.cam.enterFly(); e.preventDefault(); }
    else if (e.key === "v" || e.key === "V") { frameAnswer(); e.preventDefault(); }
    else if (e.key === "u" || e.key === "U") { setDrones(!S.drones); e.preventDefault(); }
    else if (e.key === "g" || e.key === "G") { setFollow(!S.follow); e.preventDefault(); }
    else if (/^Digit[1-4]$/.test(e.code)) {
      const slot = e.code.slice(5);
      if (e.shiftKey) { const ok = S.cam.save(slot); banner(`View ${slot} saved.`, ok ? ` Press ${slot} to return to it.` : " Kept for this session only (browser storage is unavailable).", "ok", 3500); }
      else if (S.cam.recall(slot)) { userMoved(); camCaption(`Saved view ${slot}`); }
      else banner(`No view ${slot} saved.`, ` Shift+${slot} saves the current view.`, "", 3500);
      e.preventDefault();
    }
  });
  document.addEventListener("keyup", (e) => { if ((e.key === "a" || e.key === "A") && S.gateOpen) holdCancel(); });
}
/* ------------------------------------------------------------------ RF map layers (belief ensemble, D-031) */
const RF_RAMP = {
  ghost_risk: (v) => [255, Math.round(200 - 150 * v), 40, v < 0.05 ? 0 : Math.round(60 + 160 * v)],  // amber -> red
  trust: (v) => [230, 70, 40, v > 0.95 ? 0 : Math.round(210 * (1 - v))],                             // low trust shows red
  df_err: (v) => [46, Math.round(196 - 120 * v), 230, v < 0.05 ? 0 : Math.round(40 + 180 * v)],      // cyan, denser = worse
  blind_spot: (v) => { const n = Math.round(v * 5); return [230, 60, 40, n >= 2 ? 0 : n === 1 ? 120 : 200]; },  // < 2 nodes
};
const RF_CSS = {ghost_risk: "linear-gradient(90deg, rgba(255,200,40,.15), rgba(255,50,40,.85))",
                trust: "linear-gradient(90deg, rgba(230,70,40,.85), rgba(230,70,40,0))",
                df_err: "linear-gradient(90deg, rgba(46,196,230,.15), rgba(46,76,230,.9))",
                blind_spot: "linear-gradient(90deg, rgba(230,60,40,.8), rgba(230,60,40,.47) 40%, rgba(230,60,40,0) 40%)"};
let rfSeq = 0;                                   // only the newest request may draw (controls change faster than fetches)
async function loadRfMap() {
  const seq = ++rfSeq;
  const metric = document.querySelector("input[name=rfmap]:checked")?.value || "";
  const leg = $("#rfLegend");
  if (!metric) { S.W.clearHeat(); leg.hidden = true; return; }
  try {
    const q = new URLSearchParams({band: $("#rfBand").value, height: $("#rfHeight").value, metric});
    const d = await getJSON(`/api/rfmap?${q}`);
    if (seq !== rfSeq) return;
    const raw = Uint8Array.from(atob(d.values), (c) => c.charCodeAt(0));
    const values = halfToFloat(new Uint16Array(raw.buffer));
    const norm = (v) => (v - d.vmin) / (d.vmax - d.vmin);
    S.W.heat({...d, values}, (v) => RF_RAMP[metric](Math.max(0, Math.min(1, norm(v)))));
    $("#rfRamp").style.background = RF_CSS[metric];
    $("#rfLo").textContent = metric === "df_err" ? "0°" : metric === "blind_spot" ? "0 nodes" : String(d.vmin);
    $("#rfHi").textContent = metric === "df_err" ? `${d.vmax}°+` : metric === "blind_spot" ? "5" : String(d.vmax);
    $("#rfText").textContent = `${d.legend}. ${band(d.band)}, drone height ${d.height_m} m. Simulation model.`;
    leg.hidden = false;
  } catch (e) { if (seq === rfSeq) banner("RF map unavailable.", "This map layer is not built for the chosen band and height."); }
}
/* ------------------------------------------------------------------ per-sensor multipath layer (scripts/multipath_layer.py) */
const MP = {seq: 0, cache: new Map(), objs: [], cur: null};
const MP_REFLECT = "#e879f9";
const MP_FLOOR = 0.4;                                               // below this the ground stays clear (light echoes)
const mpColour = (v) => {                                           // v: 0 clean .. 1 heavy echoes
  if (!(v >= MP_FLOOR)) return [0, 0, 0, 0];
  const t = Math.min(1, (v - MP_FLOOR) / (1 - MP_FLOOR));
  return [Math.round(190 + 50 * t), Math.round(70 + 90 * t), 249, Math.round(35 + 175 * t ** 1.4)];
};
function clearMultipath() {
  S.W.clearHeat("mp");
  for (const o of MP.objs) {
    S.W.remove(o);
    o.traverse?.((c) => {                                         // lines: free GPU buffers and drop them from resize updates
      if (!c.material) return;
      const k = S.W.lineMats?.indexOf(c.material) ?? -1;
      if (k >= 0) S.W.lineMats.splice(k, 1);
      c.geometry?.dispose(); c.material.dispose();
    });
  }
  MP.objs = []; MP.cur = null;
}
async function mpFile(band, h) {
  const key = `${band}_h${h}`;
  if (!MP.cache.has(key)) MP.cache.set(key, getJSON(`/ui/data/multipath/${key}.json`).catch((e) => { MP.cache.delete(key); throw e; }));
  return MP.cache.get(key);
}
async function loadMultipath() {
  const seq = ++MP.seq, on = $("#mpOn").checked;
  for (const id of ["#mpSel", "#mpLegend", "#mpSpots", "#mpFly"]) $(id).hidden = !on;
  if (!on) { clearMultipath(); return; }
  const sensor = $("#mpSensor").value, bandId = $("#mpBand").value, h = $("#mpHeight").value;
  try {
    const d = await mpFile(bandId, h);
    if (seq !== MP.seq) return;
    const s = d.sensors.find((x) => x.sensor === sensor);
    clearMultipath();
    if (!s) throw new Error(`no multipath for ${sensor}`);
    const raw = Uint8Array.from(atob(s.score_u8), (c) => c.charCodeAt(0));
    const values = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) values[i] = raw[i] === 255 ? NaN : raw[i] / 250;
    S.W.heat({...s, values}, mpColour, 1.3, "mp");
    const [px, py] = s.pose, ring = [];
    for (let a = 0; a <= 96; a++) ring.push([px + 1500 * Math.sin(a * Math.PI / 48), py + 1500 * Math.cos(a * Math.PI / 48), 1.6]);
    MP.objs.push(S.W.line(ring, {color: MP_REFLECT, width: 1.5, dashed: true, opacity: 0.7, cased: false, dash: 10, gap: 12}));
    s.clusters.forEach((c, i) => {
      for (const r of c.rings) MP.objs.push(S.W.line([...r, r[0]].map((p) => [p[0], p[1], 2]), {color: MP_REFLECT, width: i === 0 ? 2.6 : 1.5, opacity: i === 0 ? 1 : 0.85}));
      MP.objs.push(S.W.label(i < 3 ? `${c.rank} · ${c.label}` : String(c.rank), [c.centroid[0], c.centroid[1], 18], i < 3 ? "mp" : "mp minor"));
    });
    MP.cur = s;
    $("#mpText").textContent = `${sensor}, ${d.band_label}, drone at ${s.height_m} m. Brighter: strong echoes, no clear direct signal, `
      + "spread bearings. Outlined: worst 5 %. Dashed ring: sensor reach. Simulation model.";
    const sp = $("#mpSpots");
    const cap = (w) => w[0].toUpperCase() + w.slice(1);
    sp.innerHTML = s.clusters.length
      ? s.clusters.map((c, i) => `<li tabindex="0" data-i="${i}" title="Fly to hot spot ${c.rank}"><b>${c.rank}</b><span>${esc(c.label)}</span>`
          + `<small>${cap(c.severity)}, about ${c.echoes_mean} strong echoes${c.main_reflector ? `, mostly off ${esc(c.main_reflector)}` : ""}</small></li>`).join("")
      : `<li class="none">No echo hot spots for ${esc(sensor)} at this band and height.</li>`;
    sp.querySelectorAll("li[data-i]").forEach((li) => {
      li.onclick = () => flyToSpot(Number(li.dataset.i));
      li.onkeydown = (e) => { if (e.key === "Enter") flyToSpot(Number(li.dataset.i)); };
    });
    $("#mpFly").hidden = !s.clusters.length;
    $("#mpFly").textContent = `Fly to ${sensor}'s worst area`;
  } catch (e) {
    if (seq === MP.seq) { clearMultipath(); banner("Multipath layer unavailable.", " Build it with scripts/multipath_layer.py."); }
  }
}
function flyToSpot(i) {
  const s = MP.cur, c = s?.clusters[i];
  if (!c) return;
  /* centre the spot, look along the sensor -> spot bearing so the sensor sits in the foreground */
  const dx = c.centroid[0] - s.pose[0], dy = c.centroid[1] - s.pose[1];
  const heading = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
  const ext = Math.max(...c.rings.flat().map((p) => Math.hypot(p[0] - c.centroid[0], p[1] - c.centroid[1])));
  if (S.cam?.flying) S.cam.exitFly();
  S.W.flyTo([c.centroid[0], c.centroid[1], 0], {heading, pitch: -45, dist: Math.min(3000, Math.max(700, 1.5 * Math.hypot(dx, dy) + 2.5 * ext))});
  camCaption(`${s.sensor} multipath hot spot ${c.rank}: ${c.label}`);
  if (S.ep) userMoved();
}
/* ------------------------------------------------------------------ echo map inside DF answers
   A DF-ambiguity answer (direct_path, direct_all, locate, locate_ghost, reject_ghosts, ...), or any answer whose work
   predicted AoA at a sensor, fades in that sensor's per-sensor multipath map (same band, height nearest the track)
   under the answer layer: the bright ground is where this sensor hears strong echoes, so a second bearing there is a
   ghost. It fades out when the answer is cleared. Shown only while the Analysis marks and "Echo map in answers"
   layers are on and no RF map or manual multipath layer holds the ground (one ground raster at a time). */
const ECHO = {key: null, sensor: null, mesh: null, op: 0, want: 0, seq: 0};
const ECHO_MAX = 0.9, ECHO_FADE_S = 0.6, ECHO_HEIGHTS = [5, 40], ECHO_FLOOR = 0.12;
const echoColour = (v) => {                                         // as mpColour, but light echoes show faintly too
  if (!(v >= ECHO_FLOOR)) return [0, 0, 0, 0];
  const t = Math.min(1, (v - ECHO_FLOOR) / (1 - ECHO_FLOOR));
  return [Math.round(190 + 50 * t), Math.round(70 + 90 * t), 249, Math.round(40 + 200 * t ** 1.2)];
};
const hypSensor = (r) => (/^hyp:(S\d+)_/.exec(String(r || "")) || [])[1] || null;
const sensorId = (r) => (r ? String(r).replace(/^sensor:/i, "").toUpperCase() : null);
/** which echo map an answer explains: {sensor, band, h} or null (exported for the capture hook and tests) */
function echoPick(a, E) {
  if (!a || !E) return null;
  const aoa = E.aoa || [];
  if (a.intent !== "df_ambiguity" && !aoa.length) return null;
  const sensor = hypSensor((a.rejected_hyp_refs || [])[0]) || sensorId(aoa[0]?.sensor) || hypSensor((a.chosen_hyp_refs || [])[0])
                 || (a.evidence_refs || []).map(baseRef).filter((r) => r.startsWith("sensor:")).map(sensorId)[0];
  if (!sensor) return null;
  const tr = E.track && S.recs.get(baseRef(E.track));
  const atAoa = aoa.find((x) => sensorId(x.sensor) === sensor) || aoa[0];
  const band = atAoa?.band || E.fitBand || tr?.data?.band || "ism_2400";
  let t = null;
  for (const r of [...(a.chosen_hyp_refs || []), ...(a.evidence_refs || [])]) if (String(r).includes("@")) { t = Number(String(r).split("@")[1]); break; }
  const p = tr && tr.data?.ts ? trackPos(tr.data, t ?? S.W.simT) : null;
  const z = p?.[2] ?? a.position?.z_m ?? E.fits?.[0]?.position?.z_m ?? atAoa?.z ?? 40;
  const h = ECHO_HEIGHTS.reduce((b, x) => (Math.abs(x - z) < Math.abs(b - z) ? x : b));
  return {sensor, band, h};
}
const echoAllowed = () => analysisOn() && $("[data-layer=echo]")?.checked !== false && !$("#mpOn").checked
  && !document.querySelector("input[name=rfmap]:checked")?.value;
async function showEcho(a) {
  const pick = echoPick(a, S.ep);
  if (!pick) return;
  const seq = ++ECHO.seq, key = `${pick.sensor}_${pick.band}_h${pick.h}`;
  try {
    const d = await mpFile(pick.band, pick.h);
    if (seq !== ECHO.seq) return;
    const s = d.sensors.find((x) => x.sensor === pick.sensor);
    if (!s) return;
    if (ECHO.key !== key || !ECHO.mesh) {
      const raw = Uint8Array.from(atob(s.score_u8), (c) => c.charCodeAt(0));
      const values = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i++) values[i] = raw[i] === 255 ? NaN : raw[i] / 250;
      ECHO.mesh = S.W.heat({...s, values}, echoColour, 1.1, "mpa");
      ECHO.op = 0;
    }
    ECHO.mesh.material.opacity = ECHO.op * ECHO_MAX; ECHO.mesh.visible = ECHO.op > 0.001;
    ECHO.key = key; ECHO.sensor = pick.sensor; ECHO.band = pick.band; ECHO.h = pick.h; ECHO.active = true;
    LEGEND.echo[0] = `Echo map for ${pick.sensor}: why one bearing is a ghost`;
    echoRefresh();
  } catch (e) { /* no echo map built for this band and height: the answer stands without it */ }
}
function echoRefresh() {
  const on = !!(ECHO.active && ECHO.mesh && echoAllowed());
  ECHO.want = on ? 1 : 0;
  if (on) legendAdd("echo");
  else if (S.legend.delete("echo")) renderLegend();
}
function hideEcho() {
  ECHO.seq += 1; ECHO.active = false; ECHO.want = 0;
  if (S.legend.delete("echo")) renderLegend();
}
function echoTick(dt) {
  const m = ECHO.mesh;
  if (!m) return;
  if (ECHO.op !== ECHO.want) {
    const k = dt / ECHO_FADE_S;
    ECHO.op = ECHO.want > ECHO.op ? Math.min(ECHO.want, ECHO.op + k) : Math.max(ECHO.want, ECHO.op - k);
    m.material.opacity = ECHO.op * ECHO_MAX; m.visible = ECHO.op > 0.001;
  }
  if (!ECHO.active && ECHO.op <= 0) { S.W.clearHeat("mpa"); ECHO.mesh = null; ECHO.key = null; ECHO.sensor = null; }
}
function halfToFloat(h) {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const x = h[i], s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 0x1f, f = x & 0x3ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

function wireSearch() {
  const inp = $("#search"), list = $("#searchList");
  const index = [];
  for (const [ref, rec] of S.recs) {
    if (!["obj", "zone", "sensor", "trk"].includes(rec.kind)) continue;
    const d = rec.data, name = d.name || d.ref || ref;
    index.push({ref, name, kind: d.kind || rec.kind, keys: [ref, name, d.id, ...(d.aliases || [])].filter(Boolean).map((s) => String(s).toLowerCase())});
  }
  const render = () => {
    const q = inp.value.trim().toLowerCase();
    if (!q) { list.hidden = true; return; }
    const exact = index.filter((x) => x.keys.includes(q));
    const hits = exact.concat(index.filter((x) => !exact.includes(x) && x.keys.some((k) => k.includes(q)))).slice(0, 8);
    list.innerHTML = hits.length ? hits.map((h, i) => `<li data-ref="${esc(h.ref)}" class="${i === 0 ? "act" : ""}"><span>${esc(h.name)}</span><small>${esc(String(h.kind).replace("_", " "))}</small></li>`).join("")
                                 : `<li><small>No match in the port model</small></li>`;
    list.hidden = false;
    list.querySelectorAll("li[data-ref]").forEach((li) => (li.onmousedown = () => { handleRef(li.dataset.ref); list.hidden = true; }));
  };
  inp.oninput = render;
  inp.onkeydown = (e) => { if (e.key === "Enter") { const f = list.querySelector("li[data-ref]"); if (f) { handleRef(f.dataset.ref); list.hidden = true; } } };
  inp.onblur = () => setTimeout(() => (list.hidden = true), 150);
}

/* ------------------------------------------------------------------ video mode and deterministic capture (both opt-in)
   ?video=1 (?video=tl keeps the timeline): public-safe framing for the demo film. Header, side panel, camera toolbar,
   captions and legend are hidden (console.css); the 3D answer layers and the "Simulation" badge stay; map labels that
   would name the port, a frequency or a power level lose those parts (a label left empty is hidden).
   ?capture=1: window.ssCapture lets a script own time and camera (scripts/capture_console_clips.py). freeze() stops the
   render loop and swaps performance.now for a virtual clock; step(dt) advances it, plays the recorded events that fall
   due and renders exactly one frame, so a clip is the same whatever the machine's frame rate. Normal use is unchanged. */
const QS = new URLSearchParams(location.search);
const VIDEO = !!QS.get("video") && QS.get("video") !== "0";
const CAPTURE = QS.get("capture") === "1";
if (VIDEO) document.documentElement.classList.add("video", ...(QS.get("video") === "tl" ? ["video-tl"] : []));
const PUBLIC_FREQ = /\s*\d+(?:[./]\d+)*\s*(?:MHz|GHz)\b/gi;
const PUBLIC_DROP = /\b(?:MHz|GHz|dBm|dB over|AIR7311|KPT|IOCL)\b|Kandla|Deendayal|Indian Oil/i;   // + operators that place the site
function publicText(text) {
  const s = String(text ?? "");
  if (!PUBLIC_DROP.test(s)) return s;
  return s.replace(PUBLIC_FREQ, "").split(/,\s*/).map((p) => p.trim()).filter((p) => p && !PUBLIC_DROP.test(p)).join(", ");
}
function captureApi() {
  const realNow = performance.now.bind(performance);
  const C0 = {vt: 0, frozen: false, rec: null, at: [], next: 0, t: 0, token: -1};
  const enu = (v) => [v.x, -v.z, v.y];
  const camOf = () => { const v = S.W.view(); return {target: enu(v.target), dist: v.dist, heading: v.heading, pitch: v.pitch}; };
  return {
    /** the world and every streamed structure group / vegetation kind are loaded */
    async settle() { await S.W.streamRest(); return {groups: Object.keys(S.W.models).length, veg: S.W.veg.length}; },
    /** stop the wall clock: from here only step() moves time, animations and the camera */
    freeze() {
      if (C0.frozen) return;
      C0.vt = realNow(); performance.now = () => C0.vt;
      S.W.renderer.setAnimationLoop(null);
      S.W.clock.getDelta();
      setPlay(false); setFollow(false);
      C0.frozen = true;
    },
    /** advance virtual time by dt seconds, fire due replay events, render one frame */
    step(dt = 1 / 30) {
      C0.vt += dt * 1000;
      let fired = 0;
      if (C0.rec) {
        C0.t += dt;
        while (C0.next < C0.at.length && C0.at[C0.next] <= C0.t + 1e-9) {
          if (S.epId !== C0.token) { C0.rec = null; break; }
          onEvent(C0.rec.events[C0.next]); C0.next += 1; fired += 1;
        }
        if (C0.rec && C0.next >= C0.at.length) C0.rec = null;
      }
      S.W.frame();
      return {t: C0.t, fired, finished: !!S.finished};
    },
    /** load a recorded run; its events then fire from step(). times: seconds from now per event (defaults to the
        console's own pacing, multiplied by scale). Returns the schedule. */
    async play(taskId, who = "sft", {scale = 1, times = null, delay = 0} = {}) {
      const rec = await getJSON(`/api/replay?task_id=${encodeURIComponent(taskId)}&who=${who}`);
      closeGate(); hideDemo();
      startReplay(rec, taskId, who);
      let acc = delay;
      const pace = replayPace(rec);
      C0.at = times || pace.map((ms) => (acc += (ms / 1000) * scale));
      C0.rec = rec; C0.next = 0; C0.t = 0; C0.token = S.epId;
      return rec.events.map((e, i) => ({i, type: e.type, name: e.name || null, t: C0.at[i]}));
    },
    reset() { C0.rec = null; resetAnalysis(); },
    /** camera: target ENU metres [e, n, u], distance m, compass heading deg, pitch deg (negative looks down) */
    setCamera({target, dist, heading, pitch}) { S.W.anim = null; S.W.flyTo(target, {dist, heading, pitch, dur: 0}); return camOf(); },
    camera: camOf,
    /** the camera pose that frames these ENU points (the console's own framing), without moving the camera */
    fit(points, {pitch = -45, heading = 0, pad = 0.2, minDist = 300} = {}) {
      const keep = {t: S.W.controls.target.clone(), p: S.W.camera.position.clone()};
      frame(S.W, points, {pitch, heading, pad, minDist, dur: 0});
      const out = camOf();
      S.W.controls.target.copy(keep.t); S.W.camera.position.copy(keep.p);
      return out;
    },
    simT(t) { if (t != null) setSimTime(t); updateTracks(S.W.simT); return S.W.simT; },
    trackPos(ref, t) { const r = S.recs.get(baseRef(ref)); return r && r.kind === "trk" ? trackPos(r.data, t ?? S.W.simT) : null; },
    drones(on) { setDrones(!!on); return S.drones; },
    focus(refs, color = "bone") { return focusRefs(refs, {fly: false, color: T[color] || color}); },
    async multipath(sensor = "S4", bandId = "ism_2400", h = "5") {
      $("#mpSensor").value = sensor; $("#mpBand").value = bandId; $("#mpHeight").value = String(h); $("#mpOn").checked = true;
      await loadMultipath();
      const s = MP.cur;
      return s && {sensor: s.sensor, pose: s.pose, clusters: s.clusters.map((c) => ({rank: c.rank, label: c.label, severity: c.severity, centroid: c.centroid,
        ext: Math.max(...c.rings.flat().map((p) => Math.hypot(p[0] - c.centroid[0], p[1] - c.centroid[1])))}))};
    },
    /** the end pose of the layer panel's "Fly to worst area" (flyToSpot) for hot spot i */
    mpSpotView(i = 0) {
      const s = MP.cur, c = s?.clusters[i];
      if (!c) return null;
      const dx = c.centroid[0] - s.pose[0], dy = c.centroid[1] - s.pose[1];
      const ext = Math.max(...c.rings.flat().map((p) => Math.hypot(p[0] - c.centroid[0], p[1] - c.centroid[1])));
      return {target: [c.centroid[0], c.centroid[1], 0], heading: (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360, pitch: -45,
              dist: Math.min(3000, Math.max(700, 1.5 * Math.hypot(dx, dy) + 2.5 * ext))};
    },
    async rfmap(metric = "", bandId = "ism_2400", h = "15") {
      $("#rfBand").value = bandId; $("#rfHeight").value = String(h);
      const r = document.querySelector(`input[name=rfmap][value='${metric}']`); if (r) r.checked = true;
      await loadRfMap();
    },
    sensors() { return Object.fromEntries(S.site.sensors.filter((s) => s.enu).map((s) => [s.id || s.ref || s.name, s.enu])); },
    state() {
      const E = S.ep;
      return {simT: S.W.simT, finished: !!S.finished, step: S.step, ansFrame: E?.ansFrame?.points || null,
              lastFrame: E?.lastFrame?.points || null, drone: E?.drone || null, labels: $$(".wl").filter((l) => l.offsetParent).map((l) => l.textContent)};
    },
  };
}

boot().catch((e) => {
  console.error(e);
  const li = document.createElement("li");
  li.className = "fault";
  li.innerHTML = `<span>Start-up</span><span>The console could not start. Reload the page; if it persists, check the gateway log.</span>`;
  $("#bootList").appendChild(li);
});
