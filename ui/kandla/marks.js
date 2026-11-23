/* Answer marks on the 3D world: the agent-output colour scheme (one set of tokens), hatched area fills, pulsing
   focus beacons, padded camera framing and the analysis legend. Lives beside ss3d.js (world engine) and adds its
   objects to the overlay scene, which is drawn on top of the graded world. Site ENU (x east, y north, z up). */
import * as THREE from "three";

/* Agent-output colours, defined once (docs/design/FINAL.md, "Agent output on a coloured world"). Hue is never the
   only cue: every layer also has its own shape or dash (see LEGEND). */
export const C = {
  answer: "#f2a33a",     // sodium: the accepted answer and its evidence (solid, heavy, cased; beacon; callout)
  act: "#ff5a1f",        // signal: the operator must act (victim diamonds)
  direct: "#f2a33a",     // the chosen direct-path bearing is the answer: solid sodium
  reflect: "#e879f9",    // reflections and ghosts: magenta, dashed; reflector as a hollow ring
  gap: "#2dd4bf",        // weak DF zones and coverage gaps: teal, hatched fill
  jam: "#fb7185",        // jamming footprint level bands: rose, graded fill
  cand: "#a3e635",       // candidate and proposed sites: lime, triangle on a mast
  own: "#7db0cc",        // our sensors, links and DF drone: friendly blue, rectangle
  work: "#ece8e1",       // provisional agent work: bone, thin
  ash: "#9a958c",
};
/* drone layer: class colours of observed tracks (belief classifier argmax) */
export const KLASS = {drone: "#ffd84a", rc_controller: "#c084fc", wifi_device: "#ece8e1", unknown: "#ece8e1", friendly: "#7db0cc"};

const P = (e, n, u = 0) => new THREE.Vector3(e, u, -n);
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let HATCH = null;
function hatchTex() {
  if (HATCH) return HATCH;
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  g.strokeStyle = "#fff"; g.lineWidth = 9;
  for (let k = -64; k <= 128; k += 32) { g.beginPath(); g.moveTo(k, 64); g.lineTo(k + 64, 0); g.stroke(); }
  HATCH = new THREE.CanvasTexture(c);
  HATCH.wrapS = HATCH.wrapT = THREE.RepeatWrapping;
  HATCH.repeat.set(1 / 28, 1 / 28);                 // 28 m per tile: ~20 m stripe pitch on the ground
  HATCH.anisotropy = 4;
  return HATCH;
}

/** Filled area on the ground (overlay scene, drawn over the world): a translucent tint, optionally hatched. */
export function area(W, ring, color, {alpha = 0.28, hatch = false, hatchAlpha = 0.7, z = 1.3, order = 6} = {}) {
  const grp = new THREE.Group();
  const shape = new THREE.Shape(ring.map((p) => new THREE.Vector2(p[0], p[1])));
  const geo = new THREE.ShapeGeometry(shape);
  geo.rotateX(-Math.PI / 2);
  const mk = (opts, o) => {
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({color: new THREE.Color(color), transparent: true, depthTest: false,
      depthWrite: false, side: THREE.DoubleSide, toneMapped: false, ...opts}));
    m.renderOrder = o; return m;
  };
  grp.add(mk({opacity: alpha}, order));
  if (hatch) grp.add(mk({opacity: hatchAlpha, map: hatchTex(), alphaMap: null}, order + 1));
  grp.position.y = z;
  W.overlay.add(grp);
  return grp;
}

/** Pulsing focus beacon at a ground point: a cased core ring that holds a constant screen size and an expanding
    pulse ring (static under prefers-reduced-motion), plus an optional stalk up to an airborne point. */
export class Beacons {
  constructor(W) {
    this.W = W; this.list = [];
    this.ringG = new THREE.RingGeometry(0.74, 1, 48).rotateX(-Math.PI / 2);
    this.dotG = new THREE.CircleGeometry(0.22, 24).rotateX(-Math.PI / 2);
  }
  add(p, color = C.answer) {
    const W = this.W, grp = new THREE.Group();
    const mat = (c, o) => new THREE.MeshBasicMaterial({color: new THREE.Color(c), transparent: true, opacity: o, depthTest: false,
                                                       depthWrite: false, toneMapped: false, side: THREE.DoubleSide});
    const casing = new THREE.Mesh(this.ringG, mat("#000000", 0.75)); casing.renderOrder = 24;
    const core = new THREE.Mesh(this.ringG, mat(color, 1)); core.renderOrder = 25;
    const dot = new THREE.Mesh(this.dotG, mat(color, 1)); dot.renderOrder = 25;
    const pulse = new THREE.Mesh(this.ringG, mat(color, 0.6)); pulse.renderOrder = 23;
    const pulse2 = new THREE.Mesh(this.ringG, mat(color, 0.4)); pulse2.renderOrder = 23;
    grp.add(casing, core, dot, pulse, pulse2);
    grp.position.copy(P(p[0], p[1], 1.6));
    grp.userData = {casing, core, pulse, pulse2, ph: this.list.length * 0.23, beacon: true};
    W.overlay.add(grp);
    this.list.push(grp);
    return grp;
  }
  tick() {
    this.list = this.list.filter((g) => g.parent);
    if (!this.list.length) return;
    const W = this.W, cam = W.camera, h = W.el.clientHeight || 1;
    const tf = 2 * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)) / h;
    const t = performance.now() / 1000, still = reduceMotion();
    for (const g of this.list) {
      if (!g.visible) continue;
      const mpp = cam.position.distanceTo(g.position) * tf;            // metres per screen pixel at the beacon
      const u = g.userData;
      u.core.scale.setScalar(13 * mpp); u.casing.scale.setScalar(15.5 * mpp);
      g.children[2].scale.setScalar(13 * mpp);
      const op = u.dim ? 0.35 : 1;
      if (still) {
        u.pulse.scale.setScalar(30 * mpp); u.pulse.material.opacity = 0.45 * op;
        u.pulse2.scale.setScalar(44 * mpp); u.pulse2.material.opacity = 0.25 * op;
      } else {
        const f = (t * 0.7 + u.ph) % 1, f2 = (f + 0.5) % 1;
        u.pulse.scale.setScalar((14 + 46 * f) * mpp); u.pulse.material.opacity = 0.75 * (1 - f) * (1 - f) * op;
        u.pulse2.scale.setScalar((14 + 46 * f2) * mpp); u.pulse2.material.opacity = 0.75 * (1 - f2) * (1 - f2) * op;
      }
    }
  }
}

/** Fly so every point is on screen with a margin (fraction of the half-view on each side), at the given pitch and
    heading. Distance found by bisection on the real projection, so wide and tall sets both fit. */
export function frame(W, pts, {pitch = -45, heading, pad = 0.2, minDist = 400, maxDist = 30000, dur} = {}) {
  if (!pts.length) return;
  const b = new THREE.Box3();
  for (const p of pts) b.expandByPoint(P(p[0], p[1], p[2] ?? 0));
  const c = b.getCenter(new THREE.Vector3());
  const v = W.view();
  const h = THREE.MathUtils.degToRad(heading ?? v.heading), a = THREE.MathUtils.degToRad(-pitch);
  const dir = P(-Math.sin(h) * Math.cos(a), -Math.cos(h) * Math.cos(a), Math.sin(a));
  const cam = W.camera.clone();
  cam.aspect = W.camera.aspect; cam.updateProjectionMatrix();
  const lim = 1 - pad, q = new THREE.Vector3();
  const fits = (d) => {
    cam.position.copy(c).addScaledVector(dir, d); cam.lookAt(c); cam.updateMatrixWorld();
    for (const p of pts) {
      q.copy(P(p[0], p[1], p[2] ?? 0)).project(cam);
      if (q.z > 1 || Math.abs(q.x) > lim || Math.abs(q.y) > lim) return false;
    }
    return true;
  };
  let lo = 20, hi = maxDist;
  if (!fits(hi)) hi = maxDist;
  for (let k = 0; k < 28; k++) { const m = (lo + hi) / 2; if (fits(m)) hi = m; else lo = m; }
  W.flyTo([c.x, -c.z, c.y], {dist: Math.max(minDist, hi), pitch, heading: heading ?? v.heading, ...(dur != null ? {dur} : {})});
}

/* legend rows: swatch SVG (shape + dash, never colour alone) and wording */
const sv = (inner) => `<svg width="34" height="14" viewBox="0 0 34 14" aria-hidden="true">${inner}</svg>`;
export const LEGEND = {
  answer: ["Answer", sv(`<circle cx="17" cy="7" r="5.5" fill="none" stroke="#000" stroke-width="4"/><circle cx="17" cy="7" r="5.5" fill="none" stroke="${C.answer}" stroke-width="2.2"/><circle cx="17" cy="7" r="1.8" fill="${C.answer}"/>`)],
  answerArea: ["Answer area", sv(`<rect x="2" y="2" width="30" height="10" fill="${C.answer}" fill-opacity=".35" stroke="${C.answer}" stroke-width="2.4"/>`)],
  direct: ["Direct path, chosen", sv(`<line x1="1" y1="7" x2="33" y2="7" stroke="#000" stroke-width="6"/><line x1="1" y1="7" x2="33" y2="7" stroke="${C.direct}" stroke-width="3.5"/>`)],
  reflect: ["Reflection or ghost, rejected", sv(`<line x1="1" y1="7" x2="33" y2="7" stroke="${C.reflect}" stroke-width="2.5" stroke-dasharray="6 4"/>`)],
  echo: ["Echo map: why one bearing is a ghost", sv(`<defs><linearGradient id="lge"><stop offset="0" stop-color="#c046f9" stop-opacity=".15"/><stop offset="1" stop-color="#f0a0f9" stop-opacity=".85"/></linearGradient></defs><rect x="2" y="2" width="30" height="10" fill="url(#lge)"/>`)],
  gap: ["Weak DF zone", sv(`<defs><pattern id="lgh" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="${C.gap}" stroke-width="2.4"/></pattern></defs><rect x="2" y="2" width="30" height="10" fill="url(#lgh)" stroke="${C.gap}" stroke-width="1.5"/>`)],
  jam: ["Jamming footprint, level bands", sv(`<rect x="1" y="1" width="32" height="12" fill="${C.jam}" fill-opacity=".18" stroke="${C.jam}" stroke-width="1"/><rect x="8" y="3.5" width="18" height="7" fill="${C.jam}" fill-opacity=".45" stroke="${C.jam}" stroke-width="1.5"/>`)],
  act: ["Would be hit: operator decides", sv(`<path d="M17 1 L23 7 L17 13 L11 7 Z" fill="${C.act}" stroke="#000" stroke-width="1.2"/>`)],
  cand: ["Candidate site", sv(`<line x1="17" y1="13" x2="17" y2="6" stroke="${C.cand}" stroke-width="2"/><path d="M17 1 L22 8 L12 8 Z" fill="${C.cand}" stroke="#000" stroke-width="1"/>`)],
  own: ["Our sensors and links", sv(`<rect x="11" y="3" width="12" height="8" fill="none" stroke="${C.own}" stroke-width="2"/>`)],
  work: ["Agent work, provisional", sv(`<line x1="1" y1="7" x2="33" y2="7" stroke="${C.work}" stroke-width="1.4" stroke-opacity=".6"/>`)],
  los: ["Line of sight", sv(`<line x1="1" y1="7" x2="33" y2="7" stroke="${C.answer}" stroke-width="2" stroke-dasharray="2 3"/>`)],
  drone: ["Drone (classed UAV)", sv(`<g fill="${KLASS.drone}" stroke="#000" stroke-width=".8"><circle cx="11" cy="4" r="3"/><circle cx="23" cy="4" r="3"/><circle cx="11" cy="10" r="3"/><circle cx="23" cy="10" r="3"/><rect x="14" y="5" width="6" height="4"/></g>`)],
  controller: ["Controller or Wi-Fi device", sv(`<g fill="${KLASS.rc_controller}" stroke="#000" stroke-width=".8"><circle cx="11" cy="4" r="3"/><circle cx="23" cy="4" r="3"/><circle cx="11" cy="10" r="3"/><circle cx="23" cy="10" r="3"/><rect x="14" y="5" width="6" height="4"/></g>`)],
};
