/* SceneScout 3D engine (three.js): the Blender Kandla world rendered live in the console.
   Same sources and frame as the Blender build (docs/WORLD_BUILD.md): OSM/Overture structures exported from Blender,
   5 m terrain with the creek bed, Sentinel-2 ground (structures removed), the Blender HDRI sky, sun at 245/28 deg,
   AgX tone mapping, haze. The camera look comes from the .blend itself (envbuild/blender/web_look.py -> manifest.look):
   Blender's own view transform as a 3D LUT (AgX + "Medium High Contrast", exposure -0.45), the world HDRI with its
   mapping rotation and strength, the KW_sun direction / colour / strength, the compositor mist (quadratic, also over
   the sky). Without manifest.look the older approximations below stay in force. Evidence overlays live in a separate scene drawn on top (never graded, never occluded).
   Site ENU (metres, x east, y north, z up) maps to three.js as (x, z, -y). */
import * as THREE from "three";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
import {DRACOLoader} from "three/addons/loaders/DRACOLoader.js";
import {RGBELoader} from "three/addons/loaders/RGBELoader.js";
import {MapControls} from "three/addons/controls/MapControls.js";
import {EffectComposer} from "three/addons/postprocessing/EffectComposer.js";
import {RenderPass} from "three/addons/postprocessing/RenderPass.js";
import {GTAOPass} from "three/addons/postprocessing/GTAOPass.js";
import {ShaderPass} from "three/addons/postprocessing/ShaderPass.js";
import {OutputPass} from "three/addons/postprocessing/OutputPass.js";
import {Pass, FullScreenQuad} from "three/addons/postprocessing/Pass.js";
import {mergeGeometries} from "three/addons/utils/BufferGeometryUtils.js";
import {CSS2DRenderer, CSS2DObject} from "three/addons/renderers/CSS2DRenderer.js";
import {Line2} from "three/addons/lines/Line2.js";
import {LineMaterial} from "three/addons/lines/LineMaterial.js";
import {LineGeometry} from "three/addons/lines/LineGeometry.js";

const DRACO_PATH = "/ui/vendor/three/examples/jsm/libs/draco/gltf/";   // vendored three.js 0.169.0 (no CDN)
export const P = (e, n, u = 0) => new THREE.Vector3(e, u, -n);
const toENU = (v) => [v.x, -v.z, v.y];
const SUN_AZ = 245, SUN_EL = 28;                  // the Blender build's sun (illustrative late afternoon)
const HAZE = new THREE.Color(0.70, 0.71, 0.72);   // the Blender compositor's mist colour
// groups the opening view needs; the rest stream in after the first interactive frame
const FIRST = ["jetties", "tanks", "tank_farms", "ships", "pipelines"];
// thin lattice and wire groups: members are narrower than an atlas texel, so the bake bleeds dark; draw them flat
const FLAT = {power: [0.40, 0.41, 0.42], fences: [0.52, 0.53, 0.54]};
const FLY_S = 1.0;                                // camera flights take about a second
const KEY_PAN = 0.9, KEY_ZOOM = 1.6, KEY_ORBIT = 70;   // keyboard: view-heights/s, e-folds/s, deg/s

// Blender material name -> web material (texture set, metre scale, linear tint, imagery tint via vertex colour)
const MAT = {
  concrete: {aid: "concrete_floor_worn_001", s: 4, tint: [0.40, 0.385, 0.35], vc: true},
  // vc: the building's Sentinel-2 colour (roof_col) replaces the tint where the imagery gave one, as in bl_pbr.py
  concrete_dark: {aid: "concrete_wall_006", s: 3, tint: [0.26, 0.25, 0.235]},
  bund_floor: {aid: "gravel_concrete_03", s: 4, tint: [0.22, 0.21, 0.19]},
  roof_flat: {aid: "gravel_concrete_03", s: 3, tint: [0.36, 0.35, 0.33], vc: true},
  roof_metal: {aid: "corrugated_iron_02", s: 3, tint: [0.52, 0.53, 0.54], vc: true, metal: 0.25, rough: 0.55},
  wall_cladding: {aid: "corrugated_iron_02", s: 3, tint: [0.55, 0.56, 0.57], vc: true, metal: 0.2, rough: 0.6},
  cladding: {aid: "corrugated_iron_02", s: 3, tint: [0.75, 0.76, 0.77], metal: 0.3, rough: 0.5},
  wall: {aid: "white_rough_plaster", s: 2.5, tint: [0.66, 0.64, 0.60], vc: true},
  tank_paint: {aid: "metal_plate_02", s: 3, tint: [0.70, 0.70, 0.68], metal: 0.15, rough: 0.45},
  white_paint: {aid: "metal_plate_02", s: 2.5, tint: [0.66, 0.66, 0.64], rough: 0.4},
  white: {aid: "metal_plate_02", s: 2.5, tint: [0.66, 0.66, 0.64], rough: 0.4},
  steel: {aid: "metal_plate_02", s: 2.5, tint: [0.28, 0.29, 0.30], metal: 0.5, rough: 0.45},
  red: {aid: "metal_plate_02", s: 2.5, tint: [0.50, 0.05, 0.035], rough: 0.45},
  yellow: {aid: "metal_plate_02", s: 2.5, tint: [0.62, 0.40, 0.04], rough: 0.45},
  hull: {aid: "metal_plate_02", s: 3, tint: [0.16, 0.17, 0.18], metal: 0.3, rough: 0.5},
  hull_red: {aid: "rusty_metal_02", s: 2.5, tint: [0.26, 0.045, 0.03], rough: 0.6},
  wharf_yard: {aid: "concrete_floor_worn_001", s: 5, tint: [0.42, 0.40, 0.36]},
  asphalt: {aid: "worn_asphalt", s: 4, tint: [0.11, 0.11, 0.11], rough: 0.9},
  asphalt_plain: {aid: "worn_asphalt", s: 4, tint: [0.12, 0.12, 0.12], rough: 0.9},
  ballast: {aid: "tarred_gravel", s: 2, tint: [0.30, 0.28, 0.26], rough: 0.95},
  galvanised: {aid: "metal_plate_02", s: 2, tint: [0.62, 0.63, 0.64], metal: 0.8, rough: 0.35},
  deck_plate: {aid: "metal_plate_02", s: 2, tint: [0.52, 0.47, 0.38], rough: 0.6},
  roof_cone: {aid: "metal_plate_02", s: 3, tint: [0.70, 0.70, 0.68], rough: 0.5},
  floating_roof: {aid: "metal_plate_02", s: 3, tint: [0.45, 0.45, 0.44], metal: 0.3, rough: 0.5},
};

/** Soft bloom after the .blend's compositor Glare (manifest.look.bloom: threshold, strength, size), in scene-linear
    light before the view transform: a soft-knee bright pass, a half-resolution mip chain (13-tap down, tent up; the
    chain depth follows size), added back at strength. Only the sun disc, sun glitter and hot glints pass threshold 1.5. */
class SoftBloomPass extends Pass {
  constructor(w, h, {threshold = 1.5, strength = 0.5, size = 0.4} = {}) {
    super();
    this.levels = Math.max(3, Math.min(8, Math.round(4 + 8 * size)));
    this.down = []; this.up = [];
    const vs = "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }";
    const mk = (fs, uniforms) => new THREE.ShaderMaterial({uniforms, vertexShader: vs, fragmentShader: fs, depthTest: false, depthWrite: false});
    this.mPre = mk(`uniform sampler2D tSrc; uniform vec2 uTexel; uniform float uTh; varying vec2 vUv;
      vec3 bp(vec3 c){ float b = max(c.r, max(c.g, c.b)), kn = uTh * 0.5, sq = clamp(b - uTh + kn, 0.0, 2.0 * kn);
        sq = sq * sq / (4.0 * kn + 1e-5); return c * max(sq, b - uTh) / max(b, 1e-5); }
      void main(){ vec2 o = uTexel * 0.5; vec3 c = bp(texture2D(tSrc, vUv + vec2(-o.x, -o.y)).rgb) + bp(texture2D(tSrc, vUv + vec2(o.x, -o.y)).rgb)
        + bp(texture2D(tSrc, vUv + vec2(-o.x, o.y)).rgb) + bp(texture2D(tSrc, vUv + o).rgb);
        gl_FragColor = vec4(min(c * 0.25, vec3(200.0)), 1.0); }`, {tSrc: {value: null}, uTexel: {value: new THREE.Vector2()}, uTh: {value: threshold}});
    this.mDown = mk(`uniform sampler2D tSrc; uniform vec2 uTexel; varying vec2 vUv;
      vec3 s(vec2 d){ return texture2D(tSrc, vUv + d * uTexel).rgb; }
      void main(){ vec3 c = s(vec2(0.0)) * 0.125 + (s(vec2(-1.0, -1.0)) + s(vec2(1.0, -1.0)) + s(vec2(-1.0, 1.0)) + s(vec2(1.0, 1.0))) * 0.125
        + (s(vec2(-2.0, -2.0)) + s(vec2(2.0, -2.0)) + s(vec2(-2.0, 2.0)) + s(vec2(2.0, 2.0))) * 0.03125
        + (s(vec2(-2.0, 0.0)) + s(vec2(2.0, 0.0)) + s(vec2(0.0, -2.0)) + s(vec2(0.0, 2.0))) * 0.0625;
        gl_FragColor = vec4(c, 1.0); }`, {tSrc: {value: null}, uTexel: {value: new THREE.Vector2()}});
    this.mUp = mk(`uniform sampler2D tSrc, tAdd; uniform vec2 uTexel; varying vec2 vUv;
      vec3 s(vec2 d){ return texture2D(tSrc, vUv + d * uTexel).rgb; }
      void main(){ vec3 c = (s(vec2(-1.0, -1.0)) + s(vec2(1.0, -1.0)) + s(vec2(-1.0, 1.0)) + s(vec2(1.0, 1.0))
        + 2.0 * (s(vec2(-1.0, 0.0)) + s(vec2(1.0, 0.0)) + s(vec2(0.0, -1.0)) + s(vec2(0.0, 1.0))) + 4.0 * s(vec2(0.0))) / 16.0;
        gl_FragColor = vec4(c + texture2D(tAdd, vUv).rgb, 1.0); }`, {tSrc: {value: null}, tAdd: {value: null}, uTexel: {value: new THREE.Vector2()}});
    this.mOut = mk(`uniform sampler2D tSrc, tBloom; uniform float uK; varying vec2 vUv;
      void main(){ vec4 c = texture2D(tSrc, vUv); gl_FragColor = vec4(c.rgb + texture2D(tBloom, vUv).rgb * uK, c.a); }`,
      {tSrc: {value: null}, tBloom: {value: null}, uK: {value: strength / this.levels}});
    this.q = new FullScreenQuad(null);
    this.setSize(w, h);
  }
  setSize(w, h) {
    for (const t of [...this.down, ...this.up]) t.dispose();
    this.down = []; this.up = [];
    const o = {type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter};
    for (let i = 0; i < this.levels; i++) {
      const lw = Math.max(1, w >> (i + 1)), lh = Math.max(1, h >> (i + 1));
      this.down.push(new THREE.WebGLRenderTarget(lw, lh, o)); this.up.push(new THREE.WebGLRenderTarget(lw, lh, o));
    }
  }
  draw(r, m, target) { this.q.material = m; r.setRenderTarget(target); this.q.render(r); }
  render(r, writeBuffer, readBuffer) {
    const D = this.down, U = this.up, n = this.levels;
    this.mPre.uniforms.tSrc.value = readBuffer.texture;
    this.mPre.uniforms.uTexel.value.set(1 / readBuffer.width, 1 / readBuffer.height);
    this.draw(r, this.mPre, D[0]);
    for (let i = 1; i < n; i++) {
      this.mDown.uniforms.tSrc.value = D[i - 1].texture; this.mDown.uniforms.uTexel.value.set(1 / D[i - 1].width, 1 / D[i - 1].height);
      this.draw(r, this.mDown, D[i]);
    }
    let src = D[n - 1];
    for (let i = n - 2; i >= 0; i--) {                  // tent upsample of the coarser sum + this level
      this.mUp.uniforms.tSrc.value = src.texture; this.mUp.uniforms.tAdd.value = D[i].texture;
      this.mUp.uniforms.uTexel.value.set(1 / src.width, 1 / src.height);
      this.draw(r, this.mUp, U[i]); src = U[i];
    }
    this.mOut.uniforms.tSrc.value = readBuffer.texture; this.mOut.uniforms.tBloom.value = src.texture;
    this.draw(r, this.mOut, this.renderToScreen ? null : writeBuffer);
  }
}

export class World {
  constructor(container, labelHost) {
    this.el = container;
    this.listeners = {};
    this.pickables = [];
    this.anim = null;
    this.userCam = false;
    this.grade = 0; this.gradeTarget = 0;
    this.simT = 0; this.playing = false; this.mult = 10; this.dur = 600;
    this.tracks = [];
    this.veg = [];
    this.sprites = [];
    this.tex = new THREE.TextureLoader();
    this.texCache = {};
    this.labelHost = labelHost;
  }

  on(ev, fn) { (this.listeners[ev] ||= []).push(fn); }
  emit(ev, ...a) { for (const f of this.listeners[ev] || []) f(...a); }

  async init(site, onProgress = () => {}, opts = {}) {
    const el = this.el;
    const T = window.__ss3dT = {t0: performance.now()};
    const mark = (k) => { T[k] = Math.round(performance.now() - T.t0); };
    const r = this.renderer = new THREE.WebGLRenderer({antialias: false, powerPreference: "high-performance"});
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));   // 60 fps holds at native density on the RTX 5090
    r.setSize(el.clientWidth, el.clientHeight);
    r.toneMapping = THREE.AgXToneMapping;
    r.toneMappingExposure = 0.95;              // brighter than the Blender AgX -0.45 build: richer colour on screens
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    el.appendChild(r.domElement);
    this.labels = new CSS2DRenderer();
    this.labels.setSize(el.clientWidth, el.clientHeight);
    Object.assign(this.labels.domElement.style, {position: "absolute", inset: "0", pointerEvents: "none", zIndex: "1"});
    el.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.overlay = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(HAZE, 2.8e-5);
    const cam = this.camera = new THREE.PerspectiveCamera(40, el.clientWidth / el.clientHeight, 2, 90000);
    cam.position.copy(P(-2200, -5200, 2400));
    const ctl = this.controls = new MapControls(cam, r.domElement);
    ctl.enableDamping = true; ctl.dampingFactor = 0.08;
    ctl.rotateSpeed = 0.5;                       // audit P2-4: 360 px used to swing 124 deg of heading
    ctl.maxPolarAngle = THREE.MathUtils.degToRad(84);
    ctl.minDistance = 25; ctl.maxDistance = 30000;
    ctl.zoomToCursor = true;
    ctl.target.copy(P(200, -600, 0));
    ctl.addEventListener("start", () => { this.userCam = true; this.anim = null; this.emit("usercam"); });
    this.site = site;
    this.wireKeys();
    r.domElement.addEventListener("dblclick", (e) => this.flyToPick(e.clientX, e.clientY));

    // sky + image-based lighting from the Blender HDRI
    onProgress("sky", 0);
    const manP = fetch("/world/manifest.json").then((x) => x.json());
    const draco = new DRACOLoader().setDecoderPath(DRACO_PATH).setWorkerLimit(6);
    draco.preload();
    const gl = this.gltf = new GLTFLoader().setDRACOLoader(draco);
    const [hdr, man] = await Promise.all([new RGBELoader().loadAsync("/world/sky_1k.hdr"), manP]);
    this.manifest = man;
    const L = this.look = man.look || null;
    hdr.mapping = THREE.EquirectangularReflectionMapping;
    this.sky = hdr;
    const pm = new THREE.PMREMGenerator(r);
    this.scene.environment = pm.fromEquirectangular(hdr).texture;
    pm.dispose();
    if (L) {
      // Blender: one Background node (HDRI x strength) lights the scene and is what the camera sees; the compositor
      // then mixes the mist colour over the sky (mist pass = 1 there). No hemisphere fill: the .blend has none.
      this.scene.environmentIntensity = L.world.strength;
      this.buildSky(L);
      const m = L.mist;
      // Blender mist (compositor): mix(image, colour, factor * clamp((d - start) / depth)^2); three.js linear Fog carries
      // start / end, the chunk below makes the ramp quadratic and scales it (ShaderChunk is global: only Fog, not FogExp2)
      THREE.ShaderChunk.fog_fragment = THREE.ShaderChunk.fog_fragment.replace("float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );",
        `float fogM = clamp( ( vFogDepth - fogNear ) / ( fogFar - fogNear ), 0.0, 1.0 ); float fogFactor = ${m.factor.toFixed(4)} * fogM * fogM;`);
      this.scene.fog = new THREE.Fog(new THREE.Color(...m.colour), m.start_m, m.start_m + m.depth_m);
      this.lut = await this.loadLut(L.lut).catch((e) => { console.warn("look LUT", e); return null; });
    } else {
      this.scene.background = hdr;
      this.scene.backgroundIntensity = 1.0;
      this.scene.environmentIntensity = 0.32;
    }

    const sAz = L ? L.sun.azimuth_deg : SUN_AZ, sEl = L ? L.sun.elevation_deg : SUN_EL;
    const az = THREE.MathUtils.degToRad(sAz), elv = THREE.MathUtils.degToRad(sEl);
    this.sunDir = P(Math.sin(az) * Math.cos(elv), Math.cos(az) * Math.cos(elv), Math.sin(elv)).normalize();
    // three.js directional intensity = irradiance, like a Blender sun's strength (W/m^2): same number, same result
    const sun = this.sun = L ? new THREE.DirectionalLight(new THREE.Color(...L.sun.colour), L.sun.strength) : new THREE.DirectionalLight(0xffeed6, 4.2);
    sun.castShadow = true;
    sun.shadow.mapSize.set(4096, 4096);
    sun.shadow.bias = -0.0003; sun.shadow.normalBias = 0.6;
    const sc = sun.shadow.camera;
    sc.left = -1400; sc.right = 1400; sc.top = 1400; sc.bottom = -1400; sc.near = 10; sc.far = 9000;
    this.scene.add(sun, sun.target);
    if (!L) this.scene.add(new THREE.HemisphereLight(0xd8e2ea, 0x6b5c45, 0.12));

    mark("sky");
    onProgress("ground", 0);
    await this.buildGround(man.ground);
    mark("ground");
    onProgress("water", 0);
    this.buildWater();
    const groups = Object.keys(man.tiles).filter((g) => !["sensors"].includes(g));
    this.models = {};
    let n = 0;
    const loadGroup = async (g) => {
      try {
        const t = man.tiles[g];
        const [gltf, maps] = await Promise.all([gl.loadAsync(`/world/${t.file}`), FLAT[g] ? null : this.bakedMaps(t.baked)]);
        const grp = this.merge(gltf.scene, FLAT[g] ? {flat: FLAT[g]} : maps);
        grp.name = g;
        this.scene.add(grp);
        this.models[g] = grp;
        if (g === "ships" || g === "jetties") this.addFoam(grp);
      } catch (e) { console.warn("world group", g, e); }
      onProgress("structures", ++n / groups.length);
      this.emit("group", g, n, groups.length);
    };
    const first = groups.filter((g) => FIRST.includes(g)), rest = groups.filter((g) => !FIRST.includes(g));
    this.buildComposer();
    window.addEventListener("resize", () => this.resize());
    this.clock = new THREE.Clock();
    await Promise.all(first.map(loadGroup));
    mark("first");
    await this.warm();
    r.setAnimationLoop(() => this.frame());
    mark("warm");
    /* after the first interactive frame: the remaining structure groups, then vegetation (never blocks the operator).
       Loaded in parallel right after init, they competed with the console's first frame (S6). opts.deferRest:
       nothing streams until streamRest() (called once the curtain lifts); groups are then built one at a time
       with a frame between them. */
    let go = null;
    const gate = opts.deferRest ? new Promise((res) => { go = res; }) : Promise.resolve();
    const frame = () => new Promise((res) => requestAnimationFrame(() => res()));
    this.ready = gate.then(async () => {
      for (const g of rest) { await loadGroup(g); await frame(); }
      if (man.vegetation) await this.loadVegetation(gl, man.vegetation);
      mark("all"); this.warm(); this.emit("loaded", {groups: Object.keys(this.models).length, veg: this.veg.length});
    });
    this.streamRest = () => { if (go) go(); return this.ready; };
    this.addBeacons();
    return {groups: groups.length, veg: Object.keys(man.vegetation || {}).length};
  }

  /** Blender's view transform (AgX + look) baked by web_look.py: a 65^3 strip PNG -> RGBA8 3D texture (x = r, y = g, z = b). */
  async loadLut(lut) {
    const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = `/world/${lut.file}`; });
    const n = lut.n, cv = document.createElement("canvas");
    cv.width = img.width; cv.height = img.height;
    const cx = cv.getContext("2d", {willReadFrequently: true, colorSpace: "srgb"});
    cx.drawImage(img, 0, 0);
    const px = cx.getImageData(0, 0, cv.width, cv.height).data, d = new Uint8Array(n * n * n * 4);
    for (let b = 0; b < n; b++) for (let g = 0; g < n; g++) for (let r = 0; r < n; r++) {
      const s = ((n - 1 - g) * cv.width + r + n * b) * 4, o = ((b * n + g) * n + r) * 4;   // PNG row 0 = top = g n-1
      d[o] = px[s]; d[o + 1] = px[s + 1]; d[o + 2] = px[s + 2]; d[o + 3] = 255;
    }
    const t = new THREE.Data3DTexture(d, n, n, n);
    t.format = THREE.RGBAFormat; t.type = THREE.UnsignedByteType;
    t.minFilter = t.magFilter = THREE.LinearFilter;
    t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
    t.unpackAlignment = 1; t.needsUpdate = true;
    return {tex: t, ...lut};
  }

  /** The camera's sky as Blender renders it: the HDRI x world strength, then the compositor's mist mix (mist = 1 on the
      background). The PMREM environment and the water reflections keep the un-misted sky, as in Blender. */
  buildSky(L) {
    const m = new THREE.ShaderMaterial({
      uniforms: {tSky: {value: this.sky}, uK: {value: L.world.strength}, uHaze: {value: new THREE.Color(...L.mist.colour)},
                 uMix: {value: L.mist.factor}},
      vertexShader: "varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",
      fragmentShader: `uniform sampler2D tSky; uniform float uK, uMix; uniform vec3 uHaze; varying vec3 vW;
        void main(){ vec3 d = normalize(vW - cameraPosition);
          vec2 uv = vec2(atan(d.z, d.x) * 0.15915494 + 0.5, asin(clamp(d.y, -1.0, 1.0)) * 0.31830989 + 0.5);
          gl_FragColor = vec4(mix(textureLod(tSky, uv, 0.0).rgb * uK, uHaze, uMix), 1.0); }`,
      side: THREE.BackSide, depthWrite: false, depthTest: false, fog: false,
    });
    m.name = "sky";
    const s = this.skyMesh = new THREE.Mesh(new THREE.SphereGeometry(40000, 48, 24), m);
    s.frustumCulled = false; s.renderOrder = -10;
    this.scene.add(s);
  }

  /** Compile every shader once before the first frame shows (no hitch on the first orbit). */
  async warm() {
    try { await this.renderer.compileAsync(this.scene, this.camera); } catch (e) { /* older drivers: compile lazily */ }
  }

  /** Baked atlases from envbuild/blender/bake_web.py (albedo sRGB, AO linear); glTF UV convention (no flip). */
  async bakedMaps(b) {
    if (!b) return null;
    const load = (f, srgb) => this.tex.loadAsync(`/world/${f}`).then((t) => {
      t.flipY = false; t.anisotropy = 8; t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      t.needsUpdate = true; return t;
    });
    const [map, ao, gi] = await Promise.all([load(b.albedo, true), load(b.ao, false), b.gi ? load(b.gi, false) : null]);
    return {map, ao, gi, giScale: b.gi_scale || 1};
  }

  /** One mesh per material per group: the Blender export keeps every tank, shed and pipe as its own object
      (thousands of draw calls); merged, a group costs a handful. A baked group has one material, so one call. */
  merge(root, maps = null) {
    root.updateMatrixWorld(true);
    const buckets = new Map();
    root.traverse((o) => {
      if (!o.isMesh) return;
      const vc = !maps && !!(MAT[o.material.name]?.vc && o.geometry.attributes.color);
      const key = maps ? "baked" : `${o.material.name}|${vc}`;
      if (!buckets.has(key)) buckets.set(key, {src: o.material, vc, geoms: []});
      let g = o.geometry.clone();
      const keep = maps ? ["position", "normal", "uv"] : vc ? ["position", "normal", "color"] : ["position", "normal"];
      for (const a of Object.keys(g.attributes)) if (!keep.includes(a)) g.deleteAttribute(a);
      if (!g.attributes.normal) g.computeVertexNormals();
      if (vc && g.attributes.color.itemSize === 4) {       // RGBA -> RGB so every geometry in the bucket matches
        const c = g.attributes.color, rgb = new Float32Array(c.count * 3);
        for (let i = 0; i < c.count; i++) { rgb[3 * i] = c.getX(i); rgb[3 * i + 1] = c.getY(i); rgb[3 * i + 2] = c.getZ(i); }
        g.setAttribute("color", new THREE.BufferAttribute(rgb, 3));
      }
      if (vc && g.attributes.color.normalized) {
        const c = g.attributes.color, f = new Float32Array(c.count * 3);
        for (let i = 0; i < c.count; i++) { f[3 * i] = c.getX(i); f[3 * i + 1] = c.getY(i); f[3 * i + 2] = c.getZ(i); }
        g.setAttribute("color", new THREE.BufferAttribute(f, 3));
      }
      if (!g.index) { const idx = new Uint32Array(g.attributes.position.count); for (let i = 0; i < idx.length; i++) idx[i] = i; g.setIndex(new THREE.BufferAttribute(idx, 1)); }
      g.applyMatrix4(o.matrixWorld);
      buckets.get(key).geoms.push(g);
    });
    const out = new THREE.Group();
    for (const b of buckets.values()) {
      const merged = mergeGeometries(b.geoms, false);
      if (!merged) continue;
      const m = new THREE.Mesh(merged, maps ? this.bakedMaterial(maps) : this.material(b.src, merged));
      m.castShadow = true; m.receiveShadow = true;
      out.add(m);
    }
    return out;
  }

  texture(path, srgb) {
    if (this.texCache[path]) return this.texCache[path];
    const t = this.tex.load(path);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 8;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    this.texCache[path] = t;
    return t;
  }

  /** Blender material -> PBR material with world-space triplanar texture detail (the Blender build used box /
      object-space projection, so the meshes need no UVs) and the per-building imagery tint as vertex colour. */
  material(src, geom) {
    const def = MAT[src.name];
    const hasVC = !!geom.attributes.color;
    if (!def) {
      const m = new THREE.MeshStandardMaterial({color: src.color, roughness: src.roughness ?? 0.7, metalness: src.metalness ?? 0});
      m.name = src.name;
      return m;
    }
    const m = new THREE.MeshStandardMaterial({color: new THREE.Color(...def.tint), roughness: def.rough ?? 0.8,
                                              metalness: def.metal ?? 0, vertexColors: !!def.vc && hasVC});
    m.name = src.name;
    const diff = this.texture(`/world/tex/${def.aid}_diff.jpg`, true), rough = this.texture(`/world/tex/${def.aid}_rough.jpg`, false);
    m.onBeforeCompile = (sh) => {
      sh.uniforms.tDiff = {value: diff}; sh.uniforms.tRough = {value: rough}; sh.uniforms.tScale = {value: 1 / def.s};
      sh.vertexShader = "varying vec3 vWP; varying vec3 vWN;\n" + sh.vertexShader.replace("#include <worldpos_vertex>",
        "#include <worldpos_vertex>\n vWP = (modelMatrix * vec4(transformed, 1.0)).xyz; vWN = normalize(mat3(modelMatrix) * objectNormal);");
      sh.fragmentShader = "uniform sampler2D tDiff; uniform sampler2D tRough; uniform float tScale; varying vec3 vWP; varying vec3 vWN;\n" +
        "vec4 tri(sampler2D t){ vec3 w = pow(abs(vWN), vec3(4.0)); w /= (w.x + w.y + w.z);\n" +
        "  return texture2D(t, vWP.zy * tScale) * w.x + texture2D(t, vWP.xz * tScale) * w.y + texture2D(t, vWP.xy * tScale) * w.z; }\n" +
        sh.fragmentShader
          .replace("#include <color_fragment>", "#ifdef USE_COLOR\n if (dot(vColor.rgb, vec3(1.0)) > 0.03) diffuseColor.rgb = vColor.rgb;\n#endif")
          .replace("#include <map_fragment>", "#include <map_fragment>\n vec3 dt = tri(tDiff).rgb; diffuseColor.rgb *= clamp(dt / max(dot(dt, vec3(0.333)), 0.05) * 0.75 + 0.25, 0.0, 1.8);")
          .replace("#include <roughnessmap_fragment>", "#include <roughnessmap_fragment>\n roughnessFactor = clamp(mix(roughnessFactor, tri(tRough).g, 0.55), 0.04, 1.0);");
    };
    m.customProgramCacheKey = () => `tri_${def.aid}_${def.vc ? 1 : 0}`;
    return m;
  }

  /** Baked Blender look: Cycles albedo of the procedural materials, lit live by the same sun and sky; baked AO on the
      ambient term. A faint world-space grain keeps close-ups from looking smeared (texels are 0.1-0.6 m). */
  bakedMaterial({map, ao, flat, gi, giScale}) {
    // flat lattice / wire groups: a rough mid-grey (metal 0.6 / rough 0.45 mirrored the bright sky: near wires read as
    // white streaks where the Blender render shows thin dark lines)
    if (flat) return Object.assign(new THREE.MeshStandardMaterial({color: new THREE.Color(...flat), roughness: 0.6, metalness: 0.3}), {name: "flat"});
    const m = new THREE.MeshStandardMaterial({map, aoMap: ao, aoMapIntensity: 1.0, roughness: 0.78, metalness: 0.0});
    m.name = "baked";
    const grain = this.texture("/world/tex/concrete_floor_worn_001_diff.jpg", true);
    m.onBeforeCompile = (sh) => {
      sh.uniforms.tGrain = {value: grain};
      if (gi) {
        /* baked indirect light (bake_web.py --gi: Cycles bounce off the ground and neighbouring structures, lighting
           only): outgoing radiance albedo x light, added after the AO (the bake has its own occlusion) */
        sh.uniforms.tGI = {value: gi}; sh.uniforms.uGIK = {value: giScale};
        sh.fragmentShader = "uniform sampler2D tGI; uniform float uGIK;\n" + sh.fragmentShader.replace("#include <aomap_fragment>",
          "#include <aomap_fragment>\n reflectedLight.indirectDiffuse += material.diffuseColor * texture2D(tGI, vAoMapUv).rgb * uGIK;");
      }
      sh.vertexShader = "varying vec3 vWP; varying vec3 vWN;\n" + sh.vertexShader.replace("#include <worldpos_vertex>",
        "#include <worldpos_vertex>\n vWP = (modelMatrix * vec4(transformed, 1.0)).xyz; vWN = normalize(mat3(modelMatrix) * objectNormal);");
      sh.fragmentShader = "uniform sampler2D tGrain; varying vec3 vWP; varying vec3 vWN;\n" + sh.fragmentShader.replace("#include <map_fragment>",
        "#include <map_fragment>\n vec3 w = pow(abs(vWN), vec3(4.0)); w /= (w.x + w.y + w.z);\n" +
        " vec3 g = texture2D(tGrain, vWP.zy / 3.0).rgb * w.x + texture2D(tGrain, vWP.xz / 3.0).rgb * w.y + texture2D(tGrain, vWP.xy / 3.0).rgb * w.z;\n" +
        " float near = 1.0 - smoothstep(60.0, 400.0, length(vWP - cameraPosition));\n" +
        " diffuseColor.rgb *= mix(1.0, clamp(dot(g, vec3(0.333)) / 0.45, 0.6, 1.4), 0.35 * near);");
    };
    m.customProgramCacheKey = () => (gi ? "baked_grain_gi" : "baked_grain");
    return m;
  }

  async buildGround(baked) {
    const [meta, buf, bmap, bao, bcls] = await Promise.all([
      fetch("/world/terrain.json").then((x) => x.json()), fetch("/world/terrain.bin").then((x) => x.arrayBuffer()),
      baked ? this.tex.loadAsync(`/world/${baked.albedo}`) : null, baked?.ao ? this.tex.loadAsync(`/world/${baked.ao}`) : null,
      baked?.cls ? this.tex.loadAsync(`/world/${baked.cls}`) : null]);
    const hz = new Float32Array(buf);
    const {nx, ny, x0, x1, y0, y1} = meta, H = meta.ground_half_m;
    this.tile = {x0, x1, y0, y1};
    const ht = this.heightTex = new THREE.DataTexture(hz, nx, ny, THREE.RedFormat, THREE.FloatType);
    ht.magFilter = ht.minFilter = THREE.LinearFilter; ht.needsUpdate = true;
    const pos = new Float32Array(nx * ny * 3), uv = new Float32Array(nx * ny * 2);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i, x = x0 + (x1 - x0) * i / (nx - 1), y = y0 + (y1 - y0) * j / (ny - 1);
      pos[3 * k] = x; pos[3 * k + 1] = hz[k]; pos[3 * k + 2] = -y;
      uv[2 * k] = (x + H) / (2 * H); uv[2 * k + 1] = (y + H) / (2 * H);
    }
    const idx = [];
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
      idx.push(a, b, c, b, d, c);          // counter-clockwise seen from above (normals up)
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    const map = this.texture("/world/ground.jpg", true);
    map.wrapS = map.wrapT = THREE.ClampToEdgeWrapping;
    map.anisotropy = 16;
    const detail = this.texture("/world/detail.jpg", true);
    for (const t of [bmap, bao]) if (t) { t.anisotropy = 16; t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; }
    if (bmap) bmap.colorSpace = THREE.SRGBColorSpace;
    if (bcls) { bcls.wrapS = bcls.wrapT = THREE.ClampToEdgeWrapping; bcls.colorSpace = THREE.NoColorSpace; }
    const paved = this.texture("/world/tex/concrete_floor_worn_001_diff.jpg", true), grav = this.texture("/world/tex/gravel_concrete_03_diff.jpg", true);
    const mat = new THREE.MeshStandardMaterial({map, roughness: 0.96, metalness: 0});
    // Blender ground (bake_web.py --ground): the Sentinel-2 colour with the build's mottling, worn patches and the
    // structures' contact occlusion, 1.2 m texels over the detailed tile; the 10 m Sentinel-2 square stays the far field.
    // Near the camera the 1.2 m texels would read as blur, so photo detail is blended in by ground class
    // (web_ground.py ground_cls: paved yards / bank and mudflat / open land), luminance-normalised so the baked colour
    // stays the colour; two scales per class and a large-scale variation keep it from tiling.
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.tDet = {value: detail}; sh.uniforms.tBake = {value: bmap}; sh.uniforms.tBakeAO = {value: bao};
      sh.uniforms.tCls = {value: bcls}; sh.uniforms.tPaved = {value: paved}; sh.uniforms.tGrav = {value: grav};
      sh.uniforms.uTile = {value: new THREE.Vector4(x0, x1, y0, y1)};
      sh.vertexShader = "varying vec3 vGP;\n" + sh.vertexShader.replace("#include <worldpos_vertex>",
        "#include <worldpos_vertex>\n vGP = (modelMatrix * vec4(transformed, 1.0)).xyz;");
      sh.fragmentShader = "uniform sampler2D tDet, tBake, tBakeAO, tCls, tPaved, tGrav; uniform vec4 uTile; varying vec3 vGP;\n" +
        "vec3 nrm(vec3 c){ return c / max(dot(c, vec3(0.333)), 0.03); }\n" +
        sh.fragmentShader.replace("#include <map_fragment>",
        "#include <map_fragment>\n" +
        " vec2 tuv = vec2((vGP.x - uTile.x) / (uTile.y - uTile.x), (-vGP.z - uTile.z) / (uTile.w - uTile.z));\n" +
        (bmap ? " diffuseColor.rgb = texture2D(tBake, tuv).rgb;\n" : "") +
        (bao ? " diffuseColor.rgb *= mix(1.0, texture2D(tBakeAO, tuv).r, 0.85);\n" : "") +
        " float cd = length(vGP - cameraPosition);\n" +
        " float dn = 1.0 - smoothstep(250.0, 1600.0, cd);\n" +
        (bcls ? " vec3 cls = texture2D(tCls, tuv).rgb;\n" : " vec3 cls = vec3(0.0);\n") +
        " vec2 q = vGP.xz;\n" +
        " vec3 land = nrm(texture2D(tDet, q / 6.0).rgb) * 0.6 + nrm(texture2D(tDet, q / 27.0 + 0.37).rgb) * 0.4;\n" +
        " vec3 yard = nrm(texture2D(tPaved, q / 5.0).rgb) * 0.55 + nrm(texture2D(tGrav, q / 11.0 + 0.21).rgb) * 0.45;\n" +
        " vec3 mud = vec3(dot(nrm(texture2D(tDet, q / 9.0 + 0.6).rgb), vec3(0.333)));\n" +
        " vec3 dd = mix(mix(land, yard, cls.r), mud, cls.g * (1.0 - cls.r));\n" +
        // greener pixels (scrub, mangrove) get clumpy darker patches instead of soil grain
        " float veg = clamp((diffuseColor.g - max(diffuseColor.r, diffuseColor.b)) * 14.0, 0.0, 1.0);\n" +
        " float clump = texture2D(tGrav, q / 3.1).g;\n" +
        " dd = mix(dd, vec3(0.72 + 0.5 * clump), veg * 0.7);\n" +
        // look mode: photo detail as luminance only, so the Blender bake's hue stays the hue (the photo sets' brown
        // tint over-saturated the near ground against the Blender render)
        (this.look ? " dd = vec3(dot(dd, vec3(0.333)));\n" : "") +
        " float big = texture2D(tDet, q / 173.0 + 0.11).r;\n" +
        " diffuseColor.rgb *= mix(1.0, 0.86 + 0.28 * big, 0.6 * dn);\n" +
        " diffuseColor.rgb *= mix(vec3(1.0), clamp(dd * 0.8 + 0.2, 0.0, 1.8), " + (bmap ? "0.85 * dn + 0.1" : "1.0") + ");");
    };
    mat.customProgramCacheKey = () => `ground_${bmap ? 1 : 0}_${bao ? 1 : 0}_${bcls ? 1 : 0}_${this.look ? 1 : 0}`;
    const tile = new THREE.Mesh(g, mat);
    tile.receiveShadow = true;
    this.scene.add(tile);
    // far field: the same Sentinel-2 square around the tile, then haze
    const shape = new THREE.Shape([new THREE.Vector2(-H, -H), new THREE.Vector2(H, -H), new THREE.Vector2(H, H), new THREE.Vector2(-H, H)]);
    shape.holes.push(new THREE.Path([new THREE.Vector2(x0 + 5, y0 + 5), new THREE.Vector2(x0 + 5, y1 - 5), new THREE.Vector2(x1 - 5, y1 - 5), new THREE.Vector2(x1 - 5, y0 + 5)]));
    const fg = new THREE.ShapeGeometry(shape);
    const p = fg.attributes.position, fuv = new Float32Array(p.count * 2);
    for (let k = 0; k < p.count; k++) { fuv[2 * k] = (p.getX(k) + H) / (2 * H); fuv[2 * k + 1] = (p.getY(k) + H) / (2 * H); }
    fg.setAttribute("uv", new THREE.BufferAttribute(fuv, 2));
    fg.rotateX(-Math.PI / 2);
    const far = new THREE.Mesh(fg, new THREE.MeshStandardMaterial({map, roughness: 1}));
    far.position.y = -0.4;
    this.scene.add(far);
    const ring = new THREE.Mesh(new THREE.RingGeometry(H * 0.98, 60000, 64), new THREE.MeshBasicMaterial({color: 0x8d8574}));
    ring.rotateX(-Math.PI / 2); ring.position.y = -0.6;
    this.scene.add(ring);
    this.ground = tile;
    this.waterZ = meta.water_z_m;
  }

  /** Turbid creek, after bl_pbr.water_live: Fresnel sky reflection (Schlick, F0 0.02, so the sky takes over at grazing
      angles), a sun glint lobe, sediment colour that lifts in the shallows (depth from the 5 m terrain), and foam on
      the shore band and around hulls and jetty piles (mask drawn from the Blender geometry at the waterline).
      Ripples are four scrolling normal samples whose strength fades with distance (no shimmer far out).
      Look mode (manifest.look + manifest.water): the body is lit by the sky's real irradiance (HDRI sun included, as
      Blender's world light), and the wake foam of bl_pbr.water_live is reproduced: the Blender flow field's wake
      channel x a 3 m noise stretched along the current, advected with the ILLUSTRATIVE ebb flow (two-phase flow map,
      6 s period), Blender's foam colour. Beyond the noise's pixel footprint the foam fades to its mean, not to zero. */
  buildWater() {
    const nrm = this.texture("/world/waternormals.jpg", false);
    const sky = this.sky;
    sky.generateMipmaps = true; sky.minFilter = THREE.LinearMipmapLinearFilter; sky.needsUpdate = true;
    const t = this.tile;
    const fc = this.foamCanvas = document.createElement("canvas");
    fc.width = fc.height = 2048;
    const foam = this.foamTex = new THREE.CanvasTexture(fc);
    foam.colorSpace = THREE.NoColorSpace;
    const sunCol = this.sun.color.clone().multiplyScalar(this.sun.intensity);
    const L = this.look, Wf = this.manifest?.water;
    const look = !!(L && Wf);
    let flow = null;
    if (look) {
      flow = this.tex.load(`/world/${Wf.flow}`);
      flow.colorSpace = THREE.NoColorSpace; flow.wrapS = flow.wrapT = THREE.ClampToEdgeWrapping;
      flow.minFilter = flow.magFilter = THREE.LinearFilter; flow.generateMipmaps = false;
    }
    let shoreT = null;                                 // signed shore distance (web_ground.py water_shore.png)
    if (look && Wf.shore) {
      shoreT = this.tex.load(`/world/${Wf.shore}`);
      shoreT.colorSpace = THREE.NoColorSpace; shoreT.wrapS = shoreT.wrapT = THREE.ClampToEdgeWrapping;
      shoreT.minFilter = shoreT.magFilter = THREE.LinearFilter; shoreT.generateMipmaps = false;
    }
    const eUp = L?.world.irradiance_up || [0, 0, 0];
    const u = this.waterU = {
      uTime: {value: 0}, tNrm: {value: nrm}, tSky: {value: sky}, tH: {value: this.heightTex}, tFoam: {value: foam},
      uTile: {value: new THREE.Vector4(t.x0, t.x1, t.y0, t.y1)}, uWaterZ: {value: this.waterZ ?? -3.5},
      uSun: {value: this.sunDir.clone()}, uSunCol: {value: sunCol},
      // turbid creek: the Sentinel-2 creek pixels' hue (linear R:G:B ~ 1 : 0.87 : 0.52, silt-laden brown-olive)
      // look mode: a little greyer, after bl_pbr.water_live (70% neutral silt / deep tones, 30% desaturated imagery hue),
      // fitted to the Blender render's creek patches (render_parity.md)
      uDeep: {value: look ? new THREE.Color(0.090, 0.088, 0.072) : new THREE.Color(0.096, 0.089, 0.064)},
      uSilt: {value: look ? new THREE.Color(0.148, 0.138, 0.110) : new THREE.Color(0.158, 0.142, 0.100)},
      uFogCol: {value: this.scene.fog.color.clone()}, uFogD: {value: this.scene.fog.density ?? 0},
      // look mode: Blender mist (start, depth, factor), sky reflection at the world strength, no extra dulling
      uMist: {value: L ? new THREE.Vector3(L.mist.start_m, L.mist.depth_m, L.mist.factor) : new THREE.Vector3(0, 1, 0)},
      uSkyK: {value: L ? L.world.strength : 1.0}, uReflK: {value: L ? 1.0 : 0.85}, uAmbK: {value: L ? 1.0 : 0.9},
      // ripples: (wavelength scale, slope near, slope far, fade distance m)
      uWave: {value: new THREE.Vector4(1, 0.6, 0.08, 5000)},
      uAmbE: {value: new THREE.Vector3(...eUp).multiplyScalar(1 / Math.PI)},
      tFlow: {value: flow},
      uFlow: {value: look ? new THREE.Vector4(Wf.x0, Wf.y0, Wf.nx * Wf.step_m, Wf.ny * Wf.step_m) : new THREE.Vector4(0, 0, 1, 1)},
      uFlowH: {value: look ? new THREE.Vector3(0.5 / Wf.nx, 0.5 / Wf.ny, Wf.vscale) : new THREE.Vector3(0, 0, 1)},
      uFoamCol: {value: new THREE.Color(0.50, 0.48, 0.43)},
      tShore: {value: shoreT}, uShore: {value: new THREE.Vector2(Wf?.shore_lo ?? 20, Wf?.shore_span ?? 25)},
    };
    const m = new THREE.ShaderMaterial({
      uniforms: u, defines: {LOOK: look ? 1 : 0, SHORE: shoreT ? 1 : 0},
      vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
      fragmentShader: `
        uniform float uTime, uWaterZ, uFogD; uniform vec4 uWave; uniform sampler2D tNrm, tSky, tH, tFoam; uniform vec4 uTile;
        uniform vec3 uSun, uSunCol, uDeep, uSilt, uFogCol, uMist; uniform float uSkyK, uReflK, uAmbK; varying vec3 vW;
        uniform vec3 uAmbE, uFlowH, uFoamCol; uniform vec4 uFlow; uniform sampler2D tFlow, tShore; uniform vec2 uShore;
        // 2D simplex noise (Ashima Arts / Stefan Gustavson, MIT)
        vec3 m289(vec3 x){ return x - floor(x * (1.0 / 289.0)) * 289.0; } vec2 m289(vec2 x){ return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec3 perm(vec3 x){ return m289(((x * 34.0) + 1.0) * x); }
        float snoise(vec2 v){ const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
          vec2 i = floor(v + dot(v, C.yy)); vec2 x0 = v - i + dot(i, C.xx); vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
          vec4 x12 = x0.xyxy + C.xxzz; x12.xy -= i1; i = m289(i);
          vec3 p = perm(perm(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
          vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0); m = m * m; m = m * m;
          vec3 x = 2.0 * fract(p * C.www) - 1.0; vec3 h = abs(x) - 0.5; vec3 ox = floor(x + 0.5); vec3 a0 = x - ox;
          m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
          vec3 g; g.x = a0.x * x0.x + h.x * x0.y; g.yz = a0.yz * x12.xz + h.yz * x12.yw; return 130.0 * dot(m, g); }
        float fbm(vec2 q){ float s = 0.0, a = 1.0, w = 0.0; for (int k = 0; k < 4; k++){ s += a * snoise(q); w += a; a *= 0.65; q = q * 2.03 + 17.1; } return s / w; }
        vec3 skyAt(vec3 d, float lod){ d.y = max(d.y, 0.02); d = normalize(d);
          vec2 uv = vec2(atan(d.z, d.x) * 0.15915494 + 0.5, asin(clamp(d.y, -1.0, 1.0)) * 0.31830989 + 0.5);
          return textureLod(tSky, uv, lod).rgb * uSkyK; }
        void main(){
          vec2 p = vW.xz / uWave.x; float dist = length(cameraPosition - vW);
          vec4 n = texture2D(tNrm, p / 61.0 + uTime * vec2(0.0042, 0.0031)) + texture2D(tNrm, p / 157.0 - uTime * vec2(0.0023, -0.0035))
                 + texture2D(tNrm, p / 23.0 + uTime * vec2(-0.0081, 0.0067)) + texture2D(tNrm, p / 9.0 - uTime * vec2(0.012, 0.0095));
          float amp = mix(uWave.y, uWave.z, smoothstep(150.0, uWave.w, dist));
          vec3 N = normalize(vec3((n.x * 0.5 - 1.0) * amp, 1.0, (n.y * 0.5 - 1.0) * amp));
          vec3 V = normalize(cameraPosition - vW);
          float cosv = max(dot(N, V), 0.0);
          float F = 0.02 + 0.98 * pow(1.0 - cosv, 5.0);
          vec3 R = reflect(-V, N);
          vec3 refl = skyAt(R, mix(2.5, 5.0, smoothstep(300.0, 6000.0, dist)));
          // tile coordinates (ENU) -> terrain height -> water depth
          vec2 tuv = vec2((vW.x - uTile.x) / (uTile.y - uTile.x), (-vW.z - uTile.z) / (uTile.w - uTile.z));
          bool inTile = all(greaterThan(tuv, vec2(0.0))) && all(lessThan(tuv, vec2(1.0)));
          float depth = inTile ? uWaterZ - texture2D(tH, tuv).r : 8.0;
          float shallow = 1.0 - smoothstep(0.0, 3.5, depth);
          float sunUp = max(uSun.y, 0.0);
#if LOOK
          vec3 amb = uAmbE;                                // sky irradiance / pi (web_look.py), HDRI sun included
#else
          vec3 amb = skyAt(vec3(0.0, 1.0, 0.0), 6.0);
#endif
          vec3 body = mix(uDeep, uSilt, 0.25 + 0.6 * shallow) * (uSunCol * sunUp * 0.318 + amb * uAmbK);
          vec3 col = mix(body, refl * uReflK, F);          // muddy surface film: sky reflection a little duller
          float sd = max(dot(R, uSun), 0.0);
#if LOOK
          // sun glitter as Blender's Principled water (bl_pbr.water_live): GGX, roughness 0.05 near -> 0.24 far (so far
          // water does not sparkle), Schlick F0 0.02 at the half vector; clamped so single texels do not flare the bloom
          { vec3 Hh = normalize(uSun + V); float ndh = max(dot(N, Hh), 0.0), ndl = max(dot(N, uSun), 0.0);
            float rg = mix(0.05, 0.24, smoothstep(150.0, 3500.0, dist)), a2 = rg * rg * rg * rg;
            float dd = ndh * ndh * (a2 - 1.0) + 1.0, D = a2 / (3.14159265 * dd * dd);
            float k = rg * rg * 0.5, vis = 0.25 / ((ndl * (1.0 - k) + k) * (cosv * (1.0 - k) + k));
            float Fh = 0.02 + 0.98 * pow(1.0 - max(dot(V, Hh), 0.0), 5.0);
            col += min(uSunCol * (D * vis * Fh * ndl), vec3(60.0)); }
#else
          col += uSunCol * (pow(sd, 1400.0) * 1.2 + pow(sd, 160.0) * 0.035) * (0.4 + F);
#endif
          // foam: shore band and hull / pile wakes, broken up by the ripple texture
          float fm = inTile ? texture2D(tFoam, tuv).r : 0.0;
#if SHORE
          float shore = 0.0;                               // the shore-distance band below replaces the 5 m depth band
#else
          float shore = inTile ? 0.55 * (1.0 - smoothstep(0.05, 0.45, depth)) * step(-0.3, depth) : 0.0;
#endif
          float brk = smoothstep(0.42, 0.78, texture2D(tNrm, p / 5.3 + uTime * vec2(0.013, -0.009)).g * 0.6 + n.x * 0.1);
          // broken streaks, not a haze: the splatted pile/hull mask is soft, so only its ripple-broken crests show
          float foam = clamp(max(fm * 0.55, shore) * (0.1 + 1.0 * brk), 0.0, 1.0) * (1.0 - smoothstep(1500.0, 5000.0, dist));
#if LOOK
          // Blender wake foam (bl_pbr.water_live): wake (flow field) x noise(3 m, stretched 1 : 0.45 along y), advected
          vec2 q = vec2(vW.x, -vW.z);
          vec3 fl = texture2D(tFlow, (q - uFlow.xy) / uFlow.zw + uFlowH.xy).rgb;
          vec2 vel = (fl.rg - 0.5) * uFlowH.z;
          float base = uTime / 6.0 + (snoise(q / 250.0) * 0.5 + 0.5) * 3.0;
          float ph0 = fract(base), ph1 = fract(base + 0.5), bl = abs(1.0 - 2.0 * ph0);
          vec2 s0 = (q - vel * ph0 * 6.0) * vec2(1.0, 0.45) / 3.0, s1 = (q - vel * ph1 * 6.0 + vec2(37.1, 11.3)) * vec2(1.0, 0.45) / 3.0;
          float fw = length(fwidth(q)) / 3.0;                // noise cells per pixel
          float fac = 0.5 + 0.17 * mix(fbm(s1), fbm(s0), bl);
          float fn = mix(clamp((fac - 0.50) / 0.16, 0.0, 1.0), 0.1, smoothstep(0.5, 2.0, fw));
          float wf = clamp((fl.b - 0.08) / 0.62, 0.0, 1.0) * fn;
#if SHORE
          // Blender's breaking band along the banks: maprange(sd, -14, -3) x maprange(sd, 0.5, -1.5) x noise(0.47 .. 0.60)
          float sdm = texture2D(tShore, (q - uFlow.xy) / uFlow.zw + uFlowH.xy).r * uShore.y - uShore.x;
          float band = clamp((sdm + 14.0) / 11.0, 0.0, 1.0) * clamp((0.5 - sdm) / 2.0, 0.0, 1.0);
          float fs = mix(clamp((fac - 0.47) / 0.13, 0.0, 1.0), 0.25, smoothstep(0.5, 2.0, fw));
          wf += band * fs;
#endif
          foam = clamp(foam + wf, 0.0, 1.0);
          vec3 foamCol = uFoamCol * (uSunCol * sunUp * 0.318 + amb);
#else
          vec3 foamCol = vec3(0.62, 0.61, 0.57) * (uSunCol * sunUp * 0.318 + amb);
#endif
          col = mix(col, foamCol, foam * 0.85);
          float mz = clamp((dist - uMist.x) / uMist.y, 0.0, 1.0);
          float f = uMist.z > 0.0 ? uMist.z * mz * mz : 1.0 - exp(-uFogD * uFogD * dist * dist);
          gl_FragColor = vec4(mix(col, uFogCol, f), 1.0);
        }`,
    });
    m.name = "water";
    const w = this.water = new THREE.Mesh(new THREE.PlaneGeometry(60000, 60000), m);
    w.rotation.x = -Math.PI / 2;
    w.position.y = this.waterZ ?? -3.5;
    this.scene.add(w);
  }

  /** Waterline foam from a loaded group: vertices within 1.2 m of the water level (hull plating, piles, fenders)
      are splatted into the foam mask over the detailed tile, then softened. */
  addFoam(grp) {
    const t = this.tile, c = this.foamCanvas;
    if (!t || !c) return;
    const ctx = c.getContext("2d"), W = c.width, H = c.height, wz = this.waterZ ?? -3.5;
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    grp.traverse((o) => {
      if (!o.isMesh) return;
      const p = o.geometry.attributes.position;
      for (let i = 0; i < p.count; i += 1) {
        const y = p.getY(i);
        if (Math.abs(y - wz) > 1.2) continue;
        const px = ((p.getX(i) - t.x0) / (t.x1 - t.x0)) * W, py = (1 - (-p.getZ(i) - t.y0) / (t.y1 - t.y0)) * H;
        ctx.fillRect(px - 1, py - 1, 2.2, 2.2);
      }
    });
    if (!this._foamBlur) {                           // blur once both groups are in (cheap on a 2k canvas)
      this._foamBlur = setTimeout(() => {
        const tmp = document.createElement("canvas"); tmp.width = W; tmp.height = H;
        const tc = tmp.getContext("2d"); tc.filter = "blur(2.5px)"; tc.drawImage(c, 0, 0);
        ctx.clearRect(0, 0, W, H); ctx.drawImage(tmp, 0, 0); ctx.drawImage(tmp, 0, 0);
        this.foamTex.needsUpdate = true; this._foamBlur = null;
      }, 50);
    }
    this.foamTex.needsUpdate = true;
  }

  async loadVegetation(gl, veg) {
    await Promise.all(Object.entries(veg).map(async ([name, v]) => {
      try {
        // instances: int16 [x/0.25, y/0.25, z/0.01, rot/1e-4, scale/1e-3] per row (scripts/web_ground.py), else JSON rows
        const rowsP = v.bin ? fetch(`/world/${v.bin}`).then((x) => x.arrayBuffer()).then((b) => {
          const q = new Int16Array(b), out = new Array(q.length / 5);
          for (let i = 0; i < out.length; i++) {
            const k = 5 * i;
            out[i] = [q[k] * 0.25, q[k + 1] * 0.25, q[k + 2] * 0.01, q[k + 3] * 1e-4, q[k + 4] * 1e-3];
          }
          return out;
        }) : fetch(`/world/${v.instances}`).then((x) => x.json());
        const [gltf, rows] = await Promise.all([gl.loadAsync(`/world/${v.file}`), rowsP]);
        let mesh = null;
        gltf.scene.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
        if (!mesh) return;
        const mat = mesh.material;
        mat.side = THREE.DoubleSide;
        const cap = 6000;
        const im = new THREE.InstancedMesh(mesh.geometry, mat, cap);
        im.castShadow = false; im.receiveShadow = true;
        im.count = 0;
        im.frustumCulled = false;
        this.scene.add(im);
        this.veg.push({name, im, rows, cap});
      } catch (e) { console.warn("vegetation", name, e); }
    }));
    this.vegAt = null;
  }

  updateVegetation() {
    const t = this.controls.target, camH = Math.max(80, this.camera.position.y);
    const R = Math.min(1800, Math.max(500, camH * 1.8)), R2 = R * R;
    const key = `${Math.round(t.x / 60)}_${Math.round(t.z / 60)}_${Math.round(R / 100)}`;
    if (key === this.vegAt) return;
    this.vegAt = key;
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    for (const v of this.veg) {
      let n = 0;
      for (const row of v.rows) {
        const dx = row[0] - t.x, dz = -row[1] - t.z;
        if (dx * dx + dz * dz > R2) continue;
        p.set(row[0], row[2], -row[1]); q.setFromAxisAngle(up, row[3]); s.setScalar(row[4]);
        m.compose(p, q, s);
        v.im.setMatrixAt(n++, m);
        if (n >= v.cap) break;
      }
      v.im.count = n;
      v.im.instanceMatrix.needsUpdate = true;
    }
  }

  buildComposer() {
    const r = this.renderer, w = this.el.clientWidth, h = this.el.clientHeight;
    const rt = new THREE.WebGLRenderTarget(w, h, {samples: 4, type: THREE.HalfFloatType});
    const c = this.composer = new EffectComposer(r, rt);
    c.addPass(new RenderPass(this.scene, this.camera));
    try {
      const ao = this.ao = new GTAOPass(this.scene, this.camera, w, h);
      ao.updateGtaoMaterial({radius: 6, distanceExponent: 1.5, thickness: 4, scale: 1.2, samples: 8});
      ao.blendIntensity = 0.85;
      c.addPass(ao);
    } catch (e) { console.warn("GTAO unavailable", e); }
    this.gradePass = new ShaderPass({
      uniforms: {tDiffuse: {value: null}, amount: {value: 0}},
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
      fragmentShader: "uniform sampler2D tDiffuse; uniform float amount; varying vec2 vUv;\n" +
        "void main(){ vec4 c = texture2D(tDiffuse, vUv); float l = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));\n" +
        // analysis grade: the world stays in colour (at most 15% toward luminance, 15% dimmer) so answer marks pop
        " vec3 g = mix(c.rgb, vec3(l), amount * 0.15) * (1.0 - 0.15 * amount); gl_FragColor = vec4(g, c.a); }",
    });
    c.addPass(this.gradePass);
    if (this.look?.bloom) c.addPass(this.bloomPass = new SoftBloomPass(w, h, this.look.bloom));
    if (this.lut) {
      // Blender's display transform instead of three.js AgX: exposure (stops), log2 domain around 0.18, 3D LUT -> sRGB
      const L = this.lut;
      c.addPass(this.lookPass = new ShaderPass({
        uniforms: {tDiffuse: {value: null}, tLut: {value: L.tex}, uExp: {value: Math.pow(2, this.look.exposure_stops)},
                   uLo: {value: L.log2_lo}, uHi: {value: L.log2_hi}, uN: {value: L.n}, uGrey: {value: L.mid_grey}},
        vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
        fragmentShader: "precision highp sampler3D;\nuniform sampler2D tDiffuse; uniform sampler3D tLut; uniform float uExp, uLo, uHi, uN, uGrey; varying vec2 vUv;\n" +
          "void main(){ vec4 c = texture2D(tDiffuse, vUv);\n" +
          " vec3 t = clamp((log2(max(c.rgb * uExp, vec3(1e-10)) / uGrey) - uLo) / (uHi - uLo), 0.0, 1.0);\n" +
          " gl_FragColor = vec4(texture(tLut, t * ((uN - 1.0) / uN) + 0.5 / uN).rgb, 1.0); }",
      }));
    } else c.addPass(new OutputPass());
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h); this.composer.setSize(w, h); this.labels.setSize(w, h);
    for (const m of this.lineMats || []) m.resolution.set(w, h);
  }

  setQuality(hi) {
    this.renderer.setPixelRatio(hi ? Math.min(window.devicePixelRatio || 1, 2) : 1);
    this.resize();
  }

  /* ------------------------------------------------------------------ per frame */
  frame() {
    const dt = Math.min(0.1, this.clock.getDelta());
    this._fpsN = (this._fpsN || 0) + 1; this._fpsT = (this._fpsT || 0) + dt;
    if (this._fpsT >= 1) { window.__fps = Math.round(this._fpsN / this._fpsT); this._fpsN = 0; this._fpsT = 0; }
    if (this.anim) this.stepAnim();
    this.keyMove(dt);
    this.controls.update();
    if (this.playing) {
      this.simT = Math.min(this.dur, this.simT + dt * this.mult);
      if (this.simT >= this.dur) { this.playing = false; this.emit("stop"); }
    }
    this.emit("tick", this.simT, dt);
    if (Math.abs(this.grade - this.gradeTarget) > 0.001) {
      this.grade += (this.gradeTarget - this.grade) * Math.min(1, dt * 5);
      this.gradePass.uniforms.amount.value = this.grade;
    }
    // shadow box follows the view (snapped to texels so shadows do not shimmer)
    const t = this.controls.target, d = this.camera.position.distanceTo(t);
    const half = Math.min(4000, Math.max(500, d * 0.9));
    const sc = this.sun.shadow.camera;
    if (Math.abs(sc.right - half) > 50) { sc.left = -half; sc.right = half; sc.top = half; sc.bottom = -half; sc.updateProjectionMatrix(); }
    const texel = (2 * half) / 4096;
    const tx = Math.round(t.x / texel) * texel, tz = Math.round(t.z / texel) * texel;
    this.sun.target.position.set(tx, 0, tz);
    this.sun.position.set(tx, 0, tz).addScaledVector(this.sunDir, 5000);
    this.camera.near = Math.max(1.5, d / 600);
    this.camera.updateProjectionMatrix();
    this.updateVegetation();
    // haze: the Blender mist is ground-level aerial perspective; looking down from kilometres up it thins
    if (this.scene.fog.isFogExp2) {
      const fd = 2.8e-5 * THREE.MathUtils.clamp(2500 / Math.max(1, this.camera.position.y), 0.3, 1);
      this.scene.fog.density = fd;
      if (this.waterU) this.waterU.uFogD.value = fd;
    }
    if (this.waterU) this.waterU.uTime.value += dt;
    if (this.skyMesh) this.skyMesh.position.copy(this.camera.position);
    if (this.beacons) this.pulseBeacons();
    this.spriteScale();
    const r = this.renderer;
    this.composer.render();
    r.autoClear = false;
    r.clearDepth();
    r.render(this.overlay, this.camera);
    r.autoClear = true;
    this.labels.render(this.overlay, this.camera);
  }

  /** Live layer: a slow pulse ring on the ground at each fixed DF sensor and a small lamp at the mast top (synthetic
      sensor sites, as everywhere in the console). Drawn in the world scene, so structures occlude them. */
  addBeacons() {
    this.beacons = [];
    const ringG = new THREE.RingGeometry(0.82, 1, 64).rotateX(-Math.PI / 2), lampG = new THREE.SphereGeometry(1.6, 16, 12);
    for (const s of this.site?.sensors || []) {
      if (s.type === "drone_df" || s.lon == null) continue;
      const e = this.enu(s.lon, s.lat);
      if (!e) continue;
      const ring = new THREE.Mesh(ringG, new THREE.MeshBasicMaterial({color: 0x7fd6c8, transparent: true, depthWrite: false, toneMapped: false}));
      ring.position.copy(P(e[0], e[1], 0.6));
      const lamp = new THREE.Mesh(lampG, new THREE.MeshBasicMaterial({color: 0x9ff0e2, transparent: true, toneMapped: false}));
      lamp.position.copy(P(e[0], e[1], (s.z_m || 10) + 1.5));
      this.scene.add(ring, lamp);
      this.beacons.push({ring, lamp, ph: this.beacons.length * 0.37});
    }
  }

  pulseBeacons() {
    const t = performance.now() / 1000, d = this.camera.position.distanceTo(this.controls.target);
    const k = THREE.MathUtils.clamp(d / 1500, 0.4, 4);          // stays readable from the overview and from close up
    for (const b of this.beacons) {
      const f = ((t * 0.45 + b.ph) % 1);
      b.ring.scale.setScalar((12 + 70 * f) * k);
      b.ring.material.opacity = 0.55 * (1 - f) * (1 - f);
      b.lamp.scale.setScalar(Math.max(1, k * 0.8));
      b.lamp.material.opacity = 0.55 + 0.45 * Math.max(0, Math.sin(t * 2.6 + b.ph * 6));
    }
  }

  spriteScale() {
    const h = this.el.clientHeight, p11 = this.camera.projectionMatrix.elements[5];
    for (const s of this.sprites) {
      if (!s.parent) continue;
      const k = (2 * s.userData.px) / (p11 * h);
      s.scale.set(k, k, 1);
    }
  }

  setAnalysis(on) { this.gradeTarget = on ? 1 : 0; }

  /** RF map heat layer (D-031): a ground raster in the MAIN scene, so tanks and buildings occlude it.
      g: {nx, ny, x0, y0, dx, dy, values: Float32Array (row = y index, south to north)}; colour(v) -> [r,g,b,a]. */
  heat(g, colour, z = 1.2, slot = "rf") {
    this.clearHeat(slot);
    const cv = document.createElement("canvas");
    cv.width = g.nx; cv.height = g.ny;
    const ctx = cv.getContext("2d"), img = ctx.createImageData(g.nx, g.ny);
    for (let iy = 0; iy < g.ny; iy++) for (let ix = 0; ix < g.nx; ix++) {
      const v = g.values[iy * g.nx + ix], o = ((g.ny - 1 - iy) * g.nx + ix) * 4;
      const c = Number.isFinite(v) ? colour(v) : [0, 0, 0, 0];
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = c[3];
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.LinearFilter;
    const w = g.nx * g.dx, h = g.ny * g.dy;
    const geo = new THREE.PlaneGeometry(w, h);
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshBasicMaterial({map: tex, transparent: true, depthWrite: false, polygonOffset: true,
                                             polygonOffsetFactor: -2, polygonOffsetUnits: -2, toneMapped: false});
    const m = new THREE.Mesh(geo, mat);
    (this.heatMeshes ||= {})[slot] = m;
    m.position.set(g.x0 - g.dx / 2 + w / 2, z, -(g.y0 - g.dy / 2 + h / 2));
    m.renderOrder = 2;
    this.scene.add(m);
    return m;
  }

  /** Remove one heat slot ("rf": RF map layers, "mp": per-sensor multipath, "mpa": the echo map inside a DF answer). */
  clearHeat(slot = "rf") {
    const m = this.heatMeshes?.[slot];
    if (!m) return;
    this.scene.remove(m);
    m.geometry.dispose(); m.material.map.dispose(); m.material.dispose();
    delete this.heatMeshes[slot];
  }

  /* ------------------------------------------------------------------ camera */
  view() {
    const c = this.camera.position, t = this.controls.target;
    const d = c.clone().sub(t);
    const dist = d.length();
    const heading = (THREE.MathUtils.radToDeg(Math.atan2(-d.x, d.z)) + 360) % 360;   // compass bearing the camera looks along
    const pitch = -THREE.MathUtils.radToDeg(Math.asin(d.y / dist));
    return {target: t.clone(), position: c.clone(), dist, heading, pitch};
  }

  flyTo(targetENU, {dist, heading, pitch = -35, dur = FLY_S} = {}) {
    const v = this.view();
    const h = THREE.MathUtils.degToRad(heading ?? v.heading), a = THREE.MathUtils.degToRad(-pitch);
    const tgt = P(...targetENU);
    const off = P(-Math.sin(h) * Math.cos(a), -Math.cos(h) * Math.cos(a), Math.sin(a)).multiplyScalar(dist ?? v.dist);
    this.animateTo(tgt, tgt.clone().add(off), dur);
  }

  /** Site lon/lat -> ENU metres (same ellipsoid tangent plane as console3.js lonLatToEnu). */
  enu(lon, lat) {
    const a = 6378137, e2 = 0.00669437999014, o = this.site?.meta?.origin;
    if (!o) return null;
    const p = (o.lat * Math.PI) / 180, s = Math.sin(p);
    const N = a / Math.sqrt(1 - e2 * s * s), M = (a * (1 - e2)) / Math.pow(1 - e2 * s * s, 1.5);
    return [((lon - o.lon) * Math.PI / 180) * N * Math.cos(p), ((lat - o.lat) * Math.PI / 180) * M];
  }

  /** Points the Top-down preset must show: every DF sensor and every jetty (footprint corners). */
  topPoints() {
    const pts = [];
    for (const s of this.site?.sensors || []) {
      const e = s.enu || (s.lon != null ? this.enu(s.lon, s.lat) : null);
      if (e) pts.push([e[0], e[1], 0]);
    }
    for (const o of this.site?.objects || []) {
      if (o.kind !== "jetty" && !/jetty/i.test(o.name || "")) continue;
      if (o.polygon?.length) for (const [lon, lat] of o.polygon) { const e = this.enu(lon, lat); if (e) pts.push([e[0], e[1], 0]); }
      else if (o.enu) pts.push([o.enu[0], o.enu[1], 0]);
    }
    return pts;
  }

  /** North-up plan view that frames all sensors and jetties, with a margin for the side panels. */
  topFrame(dur = FLY_S) {
    const pts = this.topPoints();
    if (!pts.length) return false;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const tf = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)), asp = this.camera.aspect;
    const hx = (x1 - x0) / 2 + 150, hy = (y1 - y0) / 2 + 150;
    const dist = Math.max(hy / tf, hx / (tf * asp)) * 1.12;         // margin: overlays sit on the canvas edges
    this.flyTo([(x0 + x1) / 2, (y0 + y1) / 2, 0], {dist, heading: 0, pitch: -89.5, dur});
    return true;
  }

  /** Plan view (T key): north up, framing all sensors and jetties.
   *  Called again from the plan view it returns to the previous oblique view. Returns true when entering it. */
  topView(dur = FLY_S) {
    const v = this.view();
    if (this.preTop && v.pitch < -80) {
      const p = this.preTop; this.preTop = null;
      this.animateTo(p.target, p.position, dur);
      return false;
    }
    this.preTop = {target: v.target, position: v.position};
    if (!this.topFrame(dur)) this.flyTo(toENU(v.target), {dist: Math.max(v.dist, 900), heading: 0, pitch: -89.5, dur});
    return true;
  }

  animateTo(target, position, dur) {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || dur <= 0) { this.anim = null; this.controls.target.copy(target); this.camera.position.copy(position); return; }
    dur = Math.min(dur, FLY_S * 1.1);                // every flight about a second (audit P2-4: presets took 1.85 s)
    this.anim = {t0: performance.now(), dur: dur * 1000, ft: this.controls.target.clone(), fp: this.camera.position.clone(), tt: target, tp: position};
  }

  /** Double-click on the world: fly to the structure (or ground, or water) under the cursor, keeping the heading. */
  flyToPick(clientX, clientY) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, this.camera);
    const targets = [...Object.values(this.models || {}).filter((m) => m.visible), this.ground, this.water].filter(Boolean);
    const hit = rc.intersectObjects(targets, true)[0];
    if (!hit) return null;
    const v = this.view();
    const dist = THREE.MathUtils.clamp(v.dist * 0.45, 180, 1500);
    if (!this.userCam) { this.userCam = true; this.emit("usercam"); }
    this.flyTo(toENU(hit.point), {dist, heading: v.heading, pitch: Math.min(v.pitch, -25)});
    return toENU(hit.point);
  }

  /* ------------------------------------------------------------------ keyboard camera */
  /** Arrows pan, Shift+arrows orbit, + / - (and PageUp / PageDown) zoom; held keys move smoothly. Ignored while
   *  typing, with Ctrl/Alt/Meta, on the timeline, or while the operator gate is open. WASD is off by default because
   *  console3.js binds W (wall mode), A (gate hold) and D (demo menu); set `world.wasd = true` to turn it on. */
  wireKeys() {
    this.keys = new Set();
    this.wasd = false;
    const mapKey = (e) => {
      const k = e.key;
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "+", "=", "-", "_", "PageUp", "PageDown"].includes(k)) return k;
      if (this.wasd && /^[wasdWASD]$/.test(k)) return {w: "ArrowUp", s: "ArrowDown", a: "ArrowLeft", d: "ArrowRight"}[k.toLowerCase()];
      return null;
    };
    const blocked = (e) => {
      const t = e.target, tag = t?.tagName || "";
      if (e.ctrlKey || e.metaKey || e.altKey) return true;
      if (/^(INPUT|TEXTAREA|SELECT)$/.test(tag) || t?.isContentEditable) return true;
      if (t?.getAttribute?.("role") === "slider" || t?.id === "tl") return true;
      return !!window.__scenescout?.gateOpen;
    };
    window.addEventListener("keydown", (e) => {
      const k = mapKey(e);
      if (!k || blocked(e)) return;
      this.keys.add(k); this.keyShift = e.shiftKey;
      this.anim = null;
      if (!this.userCam) { this.userCam = true; this.emit("usercam"); }
      e.preventDefault();
    });
    window.addEventListener("keyup", (e) => { const k = mapKey(e); if (k) this.keys.delete(k); this.keyShift = e.shiftKey; });
    window.addEventListener("blur", () => this.keys.clear());
  }

  keyMove(dt) {
    if (!this.keys?.size || this.flying) return;            // fly mode (camera.js) owns the keys
    const K = this.keys, cam = this.camera, t = this.controls.target;
    const off = cam.position.clone().sub(t), dist = off.length();
    const ax = (K.has("ArrowRight") ? 1 : 0) - (K.has("ArrowLeft") ? 1 : 0);
    const ay = (K.has("ArrowUp") ? 1 : 0) - (K.has("ArrowDown") ? 1 : 0);
    const az = (K.has("-") || K.has("_") || K.has("PageDown") ? 1 : 0) - (K.has("+") || K.has("=") || K.has("PageUp") ? 1 : 0);
    const orbit = this.keyShift && (ax || ay);
    if (orbit) {                                                // orbit about the target
      const sph = new THREE.Spherical().setFromVector3(off);
      sph.theta -= THREE.MathUtils.degToRad(KEY_ORBIT) * ax * dt;
      sph.phi = THREE.MathUtils.clamp(sph.phi - THREE.MathUtils.degToRad(KEY_ORBIT) * ay * dt * 0.6, 0.01, this.controls.maxPolarAngle);
      off.setFromSpherical(sph);
    } else if (ax || ay) {                                      // pan on the ground plane, screen-aligned
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
      if (Math.abs(fwd.y) > 0.98) fwd.set(0, 1, 0).applyQuaternion(cam.quaternion);   // plan view: screen-up is forward
      fwd.y = 0; fwd.normalize();
      const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
      const step = Math.max(dist, 60) * KEY_PAN * dt;
      const mv = right.multiplyScalar(ax * step).addScaledVector(fwd, ay * step);
      t.add(mv); cam.position.add(mv);
    }
    if (az) off.setLength(THREE.MathUtils.clamp(dist * Math.exp(az * KEY_ZOOM * dt), this.controls.minDistance, this.controls.maxDistance));
    if (az || orbit) cam.position.copy(t).add(off);
  }

  stepAnim() {
    const a = this.anim, k = Math.min(1, (performance.now() - a.t0) / a.dur);
    const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    this.controls.target.lerpVectors(a.ft, a.tt, e);
    // arc the flight a little upward so long moves do not skim the ground
    const lift = Math.sin(Math.PI * e) * a.fp.distanceTo(a.tp) * 0.12;
    this.camera.position.lerpVectors(a.fp, a.tp, e).add(new THREE.Vector3(0, lift, 0));
    if (k >= 1) this.anim = null;
  }

  fit(pointsENU, {pitch = -38, heading, minDist = 500, mul = 2.4, dur = FLY_S} = {}) {
    if (!pointsENU.length) return;
    const b = new THREE.Box3();
    for (const p of pointsENU) b.expandByPoint(P(...p));
    const c = b.getCenter(new THREE.Vector3()), r = b.getBoundingSphere(new THREE.Sphere()).radius;
    this.flyTo(toENU(c), {dist: Math.max(r * mul, minDist), pitch, heading, dur});
  }

  /* ------------------------------------------------------------------ overlays */
  line(ptsENU, {color = "#ece8e1", width = 2, dashed = false, opacity = 1, cased = true, dash = 14, gap = 10} = {}) {
    const flat = ptsENU.flatMap((p) => [p[0], p[2] ?? 0, -p[1]]);
    const grp = new THREE.Group();
    const mk = (col, w, dsh) => {
      const g = new LineGeometry(); g.setPositions(flat);
      const m = new LineMaterial({color: new THREE.Color(col), linewidth: w, dashed: dsh, dashSize: dash, gapSize: gap, transparent: true,
                                  opacity, depthTest: false, depthWrite: false, worldUnits: false, toneMapped: false});
      m.resolution.set(this.el.clientWidth, this.el.clientHeight);
      (this.lineMats ||= []).push(m);
      const l = new Line2(g, m);
      if (dsh) l.computeLineDistances();
      l.renderOrder = 10;
      return l;
    };
    if (cased && !dashed) grp.add(mk("#000000", width + 2.5, false));
    const top = mk(color, width, dashed);
    top.renderOrder = 11;
    grp.add(top);
    grp.userData.top = top;
    this.overlay.add(grp);
    return grp;
  }

  setLine(grp, {color, width, opacity}) {
    const m = grp.userData.top.material;
    if (color) m.color.set(color);
    if (width) { m.linewidth = width; if (grp.children.length > 1) grp.children[0].material.linewidth = width + 2.5; }
    if (opacity != null) {
      m.opacity = opacity;
      if (grp.children.length > 1) grp.children[0].material.opacity = Math.min(1, opacity * 0.5);   // casing fades with the line
    }
  }

  fill(ringENU, color, alpha, z = 1.2) {
    const shape = new THREE.Shape(ringENU.map((p) => new THREE.Vector2(p[0], p[1])));
    const g = new THREE.ShapeGeometry(shape);
    g.rotateX(-Math.PI / 2);
    const m = new THREE.MeshBasicMaterial({color: new THREE.Color(color), transparent: true, opacity: alpha, depthTest: false,
                                           depthWrite: false, side: THREE.DoubleSide, toneMapped: false});
    const mesh = new THREE.Mesh(g, m);
    mesh.position.y = z;
    mesh.renderOrder = 5;
    this.overlay.add(mesh);
    return mesh;
  }

  sprite(canvas, pENU, px = 20) {
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    const s = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, depthTest: false, depthWrite: false, sizeAttenuation: false, toneMapped: false}));
    s.position.copy(P(...pENU));
    s.userData.px = px;
    s.renderOrder = 20;
    this.overlay.add(s);
    this.sprites.push(s);
    return s;
  }

  setSprite(s, canvas, px) {
    s.material.map.dispose();
    s.material.map = new THREE.CanvasTexture(canvas);
    s.material.map.colorSpace = THREE.SRGBColorSpace;
    s.material.needsUpdate = true;
    if (px) s.userData.px = px;
  }

  label(text, pENU, cls = "", parent = null) {
    const div = document.createElement("div");
    div.className = `wl ${cls}`;
    div.textContent = text;
    const o = new CSS2DObject(div);
    o.position.copy(parent ? new THREE.Vector3() : P(...pENU));
    (parent || this.overlay).add(o);
    return o;
  }

  remove(o) {
    if (!o) return;
    o.traverse?.((c) => { if (c.element && c.element.parentNode) c.element.parentNode.removeChild(c.element); });
    if (o.element && o.element.parentNode) o.element.parentNode.removeChild(o.element);
    o.parent?.remove(o);
    const i = this.sprites.indexOf(o);
    if (i >= 0) this.sprites.splice(i, 1);
  }

  show(o, on) { if (o) { o.visible = on; o.traverse?.((c) => { if (c.element) c.element.style.display = on ? "" : "none"; }); } }

  /* ------------------------------------------------------------------ picking */
  addPick(obj, rec) { obj.userData.rec = rec; this.pickables.push(obj); }

  pick(clientX, clientY) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, this.camera);
    const hits = rc.intersectObjects(this.pickables.filter((o) => o.visible !== false && (o.parent ? o.parent.visible !== false : true)), false);
    return hits.map((h) => h.object.userData.rec).filter(Boolean);
  }

  /** ground point under the cursor (for the scale bar) */
  metresPerPixel() {
    const d = this.camera.position.distanceTo(this.controls.target);
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    return (2 * d * Math.tan(fov / 2)) / this.el.clientHeight;
  }
}
