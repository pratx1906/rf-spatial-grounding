/* Free camera for the SceneScout console (beside ss3d.js MapControls):
   - orbit about the point under the cursor: right-drag (or Ctrl+left-drag) pivots about the picked structure,
     ground or water point, with damped inertia; no snap (camera and target turn together about the pivot);
   - fly mode (F): first person, WASD move, Q / E down / up, mouse-look under pointer lock, Shift faster, wheel
     sets the speed, Esc leaves; on exit the orbit target is put on what the camera looks at;
   - named views: save the current view to slot 1-4 and recall it (localStorage, guarded).
   Site ENU (x east, y north, z up) is three.js (x, z, -y). */
import * as THREE from "three";

const STORE = "scenescout.views.v1";
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export class FreeCam {
  constructor(W, {onUser = () => {}, onFly = () => {}} = {}) {
    this.W = W; this.onUser = onUser; this.onFly = onFly;
    this.flying = false; this.keys = new Set(); this.speed = 60; this.vel = new THREE.Vector3();
    this.yawV = 0; this.pitchV = 0; this.drag = null; this.pivot = null;
    const ctl = W.controls, el = W.renderer.domElement;
    ctl.enableRotate = false;                                     // rotation is ours: about the cursor's point
    ctl.minDistance = 5; ctl.maxDistance = 40000;
    ctl.maxPolarAngle = THREE.MathUtils.degToRad(100);            // down to the horizon and a little above it
    ctl.zoomToCursor = true;
    ctl.dampingFactor = 0.1;
    el.addEventListener("pointerdown", (e) => this.down(e), {capture: true});   // before MapControls
    window.addEventListener("pointermove", (e) => this.move(e));
    window.addEventListener("pointerup", () => { this.drag = null; });
    el.addEventListener("wheel", (e) => { if (this.flying) { e.preventDefault(); e.stopImmediatePropagation(); this.speed = THREE.MathUtils.clamp(this.speed * (e.deltaY > 0 ? 0.8 : 1.25), 5, 2000); } }, {capture: true, passive: false});
    document.addEventListener("pointerlockchange", () => { if (document.pointerLockElement !== el && this.flying) this.exitFly(); });
    window.addEventListener("keydown", (e) => {
      if (!this.flying || e.ctrlKey || e.metaKey || e.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target?.tagName || "")) return;
      const k = e.key.toLowerCase();
      if ("wasdqe".includes(k) && k.length === 1) { this.keys.add(k); e.preventDefault(); }
      this.shift = e.shiftKey;
    });
    window.addEventListener("keyup", (e) => { this.keys.delete(e.key.toLowerCase()); this.shift = e.shiftKey; });
    window.addEventListener("blur", () => this.keys.clear());
  }

  /* ---------------------------------------------------------------- orbit about the cursor's point */
  hitPoint(x, y) {
    const W = this.W, r = W.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, W.camera);
    const targets = [...Object.values(W.models || {}).filter((m) => m.visible), W.ground, W.water].filter(Boolean);
    const hit = rc.intersectObjects(targets, true)[0];
    if (hit && hit.distance < 60000) return hit.point;
    return null;
  }
  down(e) {
    if (this.flying) { this.W.renderer.domElement.requestPointerLock?.(); return; }
    const rot = e.button === 2 || (e.button === 0 && (e.ctrlKey || e.metaKey));
    if (!rot) return;
    if (e.button === 0) e.stopImmediatePropagation();             // MapControls would pan
    this.pivot = this.hitPoint(e.clientX, e.clientY) || this.W.controls.target.clone();
    this.drag = {x: e.clientX, y: e.clientY};
    this.W.anim = null;
    this.onUser();
  }
  move(e) {
    if (this.flying && document.pointerLockElement === this.W.renderer.domElement) {
      const cam = this.W.camera, eu = new THREE.Euler(0, 0, 0, "YXZ").setFromQuaternion(cam.quaternion);
      eu.y -= e.movementX * 0.0022; eu.x = THREE.MathUtils.clamp(eu.x - e.movementY * 0.0022, -1.5, 1.5);
      cam.quaternion.setFromEuler(eu);
      return;
    }
    if (!this.drag) return;
    const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
    this.drag = {x: e.clientX, y: e.clientY};
    this.yawV = -dx * 0.0042; this.pitchV = dy * 0.0042;
    this.apply(this.yawV, this.pitchV);
  }
  /** turn camera and target together about the pivot: yaw about world up, pitch about the camera's right axis */
  apply(yaw, pitch) {
    const W = this.W, cam = W.camera, t = W.controls.target, pv = this.pivot || t;
    const up = new THREE.Vector3(0, 1, 0);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion).setY(0).normalize();
    // keep the view direction between straight down and 10 deg above the horizon
    const dir = new THREE.Vector3().subVectors(t, cam.position).normalize();
    const el = Math.asin(THREE.MathUtils.clamp(dir.y, -1, 1));
    const np = THREE.MathUtils.clamp(el - pitch, -Math.PI / 2 + 0.01, THREE.MathUtils.degToRad(10)) - el;
    const q = new THREE.Quaternion().setFromAxisAngle(up, yaw).multiply(new THREE.Quaternion().setFromAxisAngle(right, -np));
    for (const v of [cam.position, t]) v.sub(pv).applyQuaternion(q).add(pv);
    if (cam.position.y < 3) { const d = 3 - cam.position.y; cam.position.y += d; t.y += d; }
    cam.lookAt(t);
  }

  /* ---------------------------------------------------------------- fly mode */
  enterFly() {
    if (this.flying) return;
    const W = this.W;
    this.flying = true; W.flying = true; W.anim = null;
    W.controls.enabled = false;
    this.polar = W.controls.maxPolarAngle; W.controls.maxPolarAngle = Math.PI;   // free look up in fly mode
    W.renderer.domElement.requestPointerLock?.();
    this.speed = THREE.MathUtils.clamp(W.camera.position.y * 0.6, 20, 600);
    this.onUser(); this.onFly(true);
  }
  exitFly() {
    if (!this.flying) return;
    const W = this.W, cam = W.camera;
    this.flying = false; W.flying = false; this.keys.clear(); this.vel.set(0, 0, 0);
    if (document.pointerLockElement) document.exitPointerLock?.();
    // the orbit target goes on what the camera looks at (or 400 m ahead), so the hand-over does not move the view
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
    const rc = new THREE.Raycaster(cam.position, fwd, 0, 20000);
    const hit = rc.intersectObjects([...Object.values(W.models || {}), W.ground, W.water].filter(Boolean), true)[0];
    W.controls.target.copy(hit ? hit.point : cam.position.clone().addScaledVector(fwd, 400));
    W.controls.enabled = true; W.controls.maxPolarAngle = this.polar ?? W.controls.maxPolarAngle;
    W.controls.update();
    this.onFly(false);
  }

  /** per frame (from the World tick) */
  tick(dt) {
    const W = this.W, cam = W.camera;
    if (this.flying) {
      const k = this.keys, f = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
      const r = new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion);
      const want = new THREE.Vector3()
        .addScaledVector(f, (k.has("w") ? 1 : 0) - (k.has("s") ? 1 : 0))
        .addScaledVector(r, (k.has("d") ? 1 : 0) - (k.has("a") ? 1 : 0))
        .addScaledVector(new THREE.Vector3(0, 1, 0), (k.has("e") ? 1 : 0) - (k.has("q") ? 1 : 0));
      if (want.lengthSq() > 0) want.normalize().multiplyScalar(this.speed * (this.shift ? 4 : 1));
      this.vel.lerp(want, Math.min(1, dt * 6));                     // eased start and stop
      cam.position.addScaledVector(this.vel, dt);
      if (cam.position.y < 2) cam.position.y = 2;
      W.controls.target.copy(cam.position).addScaledVector(f, 50);  // keeps ss3d's near plane and shadows sensible
      return;
    }
    if (!this.drag && (Math.abs(this.yawV) > 1e-5 || Math.abs(this.pitchV) > 1e-5)) {   // damped inertia after a drag
      const decay = reduceMotion() ? 0 : Math.exp(-dt * 9);
      this.yawV *= decay; this.pitchV *= decay;
      this.apply(this.yawV, this.pitchV);
    }
  }

  /* ---------------------------------------------------------------- named views */
  load() { try { return JSON.parse(localStorage.getItem(STORE) || "{}") || {}; } catch (_) { return {}; } }
  save(slot) {
    const W = this.W, v = {t: W.controls.target.toArray(), p: W.camera.position.toArray()};
    const all = this.load(); all[slot] = v;
    try { localStorage.setItem(STORE, JSON.stringify(all)); return true; } catch (_) { this.mem = {...(this.mem || {}), [slot]: v}; return false; }
  }
  recall(slot) {
    const v = this.load()[slot] || (this.mem || {})[slot];
    if (!v) return false;
    if (this.flying) this.exitFly();
    this.W.animateTo(new THREE.Vector3(...v.t), new THREE.Vector3(...v.p), 1.0);
    return true;
  }
}
