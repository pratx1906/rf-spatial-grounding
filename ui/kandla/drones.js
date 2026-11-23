/* Live drones layer: instanced quadcopters flying the OBSERVED tracks (/api/tracks belief estimates, never truth),
   interpolated over sim time, with spinning rotors, a short fading trail and a class colour. One InstancedMesh for
   the airframes, one for the rotor blades, one LineSegments for every trail: a handful of draw calls for any
   number of tracks. Drawn in the overlay scene (never graded, readable over water and land); each airframe holds
   at least ~34 px on screen so it reads from the port overview, and its true size when the camera is close. */
import * as THREE from "three";
import {KLASS, C} from "/ui/marks.js";

const P = (e, n, u = 0) => new THREE.Vector3(e, u, -n);
const ARM = 0.42, SPAN = 1.2;                     // airframe: rotor centres at +-0.42 m (about a 1.2 m class drone)
const TRAIL_S = 40, TRAIL_N = 16;                 // trail: last 40 s of the track in 16 segments, fading out

function airframe() {
  const parts = [];
  const box = (w, h, d, x = 0, y = 0, z = 0, ry = 0, shade = 1) => {
    const g = new THREE.BoxGeometry(w, h, d).toNonIndexed();
    if (ry) g.rotateY(ry);
    g.translate(x, y, z);
    const n = g.attributes.position.count, col = new Float32Array(n * 3), nr = g.attributes.normal;
    for (let i = 0; i < n; i++) {                   // baked light: top bright, sides mid, belly dark (unlit overlay)
      const k = (nr.getY(i) > 0.5 ? 1.0 : nr.getY(i) < -0.5 ? 0.42 : 0.72) * shade;
      col[3 * i] = col[3 * i + 1] = col[3 * i + 2] = k;
    }
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.deleteAttribute("uv");
    parts.push(g);
  };
  box(0.34, 0.12, 0.34);                                           // hull
  box(0.2, 0.06, 0.12, 0, -0.08, 0.12, 0, 0.6);                    // camera pod
  for (const s of [1, -1]) box(SPAN * 0.95, 0.045, 0.06, 0, 0.02, 0, s * Math.PI / 4, 0.85);   // arms
  for (const [x, z] of [[ARM, ARM], [-ARM, ARM], [ARM, -ARM], [-ARM, -ARM]]) box(0.08, 0.1, 0.08, x * 0.97, 0.06, z * 0.97, 0, 0.9);  // motors
  for (const [x, z] of [[ARM, ARM], [-ARM, ARM], [ARM, -ARM], [-ARM, -ARM]]) box(0.36, 0.012, 0.36, x * 0.97, 0.115, z * 0.97, Math.PI / 4, 0.55);  // prop discs (blur)
  return mergeAll(parts);
}
function mergeAll(gs) {
  let n = 0;
  for (const g of gs) n += g.attributes.position.count;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), nor = new Float32Array(n * 3);
  let o = 0;
  for (const g of gs) {
    pos.set(g.attributes.position.array, o * 3); col.set(g.attributes.color.array, o * 3); nor.set(g.attributes.normal.array, o * 3);
    o += g.attributes.position.count;
  }
  const m = new THREE.BufferGeometry();
  m.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  m.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  m.setAttribute("color", new THREE.BufferAttribute(col, 3));
  m.computeBoundingSphere();
  return m;
}

export class Drones {
  /** tracks: [{ref, klass, group, pts: [[t,x,y,z,...]], ts, avail}] (console3 adds ts/avail); posAt(tr, t) -> [x,y,z]|null */
  constructor(W, tracks, posAt) {
    this.W = W; this.tracks = tracks.filter((t) => t.pts && t.pts.length > 1); this.posAt = posAt;
    this.on = false; this.hi = new Set();
    const n = Math.max(1, this.tracks.length);
    const mat = new THREE.MeshBasicMaterial({vertexColors: true, toneMapped: false});
    this.body = new THREE.InstancedMesh(airframe(), mat, n);
    const blade = new THREE.BoxGeometry(0.4, 0.014, 0.035);
    blade.translate(0, 0.125, 0);
    this.blades = new THREE.InstancedMesh(blade, new THREE.MeshBasicMaterial({color: 0x101010, toneMapped: false}), n * 4);
    for (const m of [this.body, this.blades]) { m.frustumCulled = false; m.renderOrder = 30; m.visible = false; m.count = 0; }
    this.body.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.blades.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.colors = this.tracks.map((tr) => new THREE.Color(KLASS[tr.klass] || KLASS.unknown));
    this.tracks.forEach((_, i) => this.body.setColorAt(i, this.colors[i]));
    // trails: per drone TRAIL_N segments, RGBA vertex colours fading to transparent
    const tp = new Float32Array(n * TRAIL_N * 2 * 3), tc = new Float32Array(n * TRAIL_N * 2 * 4);
    const tg = new THREE.BufferGeometry();
    tg.setAttribute("position", new THREE.BufferAttribute(tp, 3).setUsage(THREE.DynamicDrawUsage));
    tg.setAttribute("color", new THREE.BufferAttribute(tc, 4).setUsage(THREE.DynamicDrawUsage));
    this.trail = new THREE.LineSegments(tg, new THREE.LineBasicMaterial({vertexColors: true, transparent: true, depthTest: false,
                                                                          depthWrite: false, toneMapped: false}));
    this.trail.frustumCulled = false; this.trail.renderOrder = 29; this.trail.visible = false;
    W.overlay.add(this.trail, this.body, this.blades);
    this.idx = [];                                 // instance -> track index (only tracks seen at the current time)
    this._m = new THREE.Matrix4(); this._q = new THREE.Quaternion(); this._q2 = new THREE.Quaternion(); this._s = new THREE.Vector3();
    this._p = new THREE.Vector3(); this._e = new THREE.Euler(0, 0, 0, "YXZ"); this._up = new THREE.Vector3(0, 1, 0);
  }

  show(on) {
    this.on = on;
    this.body.visible = this.blades.visible = this.trail.visible = on;
  }

  highlight(refs) { this.hi = new Set(refs); }

  /** per frame: place every drone seen at sim time t; rotors spin on wall-clock time (they hover when paused) */
  update(t) {
    if (!this.on) return;
    const W = this.W, cam = W.camera, h = W.el.clientHeight || 1;
    const tf = 2 * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)) / h;
    const spin = (performance.now() / 1000) * 70;
    const tp = this.trail.geometry.attributes.position.array, tc = this.trail.geometry.attributes.color.array;
    tp.fill(0); tc.fill(0);
    let n = 0;
    this.idx.length = 0;
    const sodium = new THREE.Color(C.answer);
    this.tracks.forEach((tr, i) => {
      const p = this.posAt(tr, t);
      if (!p) return;
      const prev = this.posAt(tr, Math.max(tr.avail[0], t - 2)) || p;
      const vx = p[0] - prev[0], vy = p[1] - prev[1], sp = Math.hypot(vx, vy);
      const yaw = sp > 0.05 ? Math.atan2(vx, vy) : (i * 1.3) % (2 * Math.PI);    // nose along the ground track
      const tilt = Math.min(0.32, sp * 0.02);                                     // lean into the motion
      this._p.copy(P(p[0], p[1], p[2]));
      const mpp = cam.position.distanceTo(this._p) * tf;
      const hot = this.hi.has(tr.ref);
      const s = Math.max(1, ((hot ? 48 : 34) * mpp) / SPAN);                     // screen floor, true size up close
      this._e.set(-tilt, -yaw, 0);
      this._q.setFromEuler(this._e);
      this._s.setScalar(s);
      this._m.compose(this._p, this._q, this._s);
      this.body.setMatrixAt(n, this._m);
      this.body.setColorAt(n, hot ? sodium : this.colors[i]);
      for (let r = 0; r < 4; r++) {
        const [x, z] = [[ARM, ARM], [-ARM, ARM], [ARM, -ARM], [-ARM, -ARM]][r];
        const off = new THREE.Vector3(x * 0.97, 0, z * 0.97).multiplyScalar(s).applyQuaternion(this._q);
        this._q2.setFromAxisAngle(this._up, spin * (r % 3 === 0 ? 1 : -1) + i);
        const q = this._q.clone().multiply(this._q2);
        this._m.compose(this._p.clone().add(off), q, this._s);
        this.blades.setMatrixAt(n * 4 + r, this._m);
      }
      // trail
      const col = hot ? sodium : this.colors[i];
      let last = p, base = n * TRAIL_N;
      for (let k = 1; k <= TRAIL_N; k++) {
        const tk = t - (k * TRAIL_S) / TRAIL_N;
        if (tk < tr.avail[0]) break;
        const q = this.posAt(tr, tk);
        if (!q) break;
        const j = (base + k - 1) * 6, a0 = 1 - (k - 1) / TRAIL_N, a1 = 1 - k / TRAIL_N;
        tp.set([last[0], last[2], -last[1], q[0], q[2], -q[1]], j);
        tc.set([col.r, col.g, col.b, 0.85 * a0, col.r, col.g, col.b, 0.85 * a1], (base + k - 1) * 8);
        last = q;
      }
      this.idx.push(i);
      n++;
    });
    this.body.count = n; this.blades.count = n * 4;
    this.body.instanceMatrix.needsUpdate = true; this.blades.instanceMatrix.needsUpdate = true;
    if (this.body.instanceColor) this.body.instanceColor.needsUpdate = true;
    this.trail.geometry.attributes.position.needsUpdate = true; this.trail.geometry.attributes.color.needsUpdate = true;
    this.trail.geometry.setDrawRange(0, Math.max(1, this.tracks.length) * TRAIL_N * 2);
  }

  /** track ref under the cursor, or null */
  pick(clientX, clientY) {
    if (!this.on || !this.body.count) return null;
    const r = this.W.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, this.W.camera);
    this.body.computeBoundingSphere();
    const hit = rc.intersectObject(this.body, false)[0];
    if (hit && hit.instanceId != null) return this.tracks[this.idx[hit.instanceId]]?.ref || null;
    // forgiving pick: the nearest drone within 18 px of the cursor
    let best = null, bd = 18;
    const v = new THREE.Vector3();
    for (let k = 0; k < this.body.count; k++) {
      this.body.getMatrixAt(k, this._m); v.setFromMatrixPosition(this._m).project(this.W.camera);
      const dx = (v.x + 1) / 2 * r.width + r.left - clientX, dy = (1 - v.y) / 2 * r.height + r.top - clientY, d = Math.hypot(dx, dy);
      if (d < bd) { bd = d; best = this.tracks[this.idx[k]]?.ref; }
    }
    return best;
  }
}
