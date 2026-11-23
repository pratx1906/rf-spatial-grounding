/* SceneScout world layer: the exported Blender structures (Draco GLB, site ENU frame) on Cesium terrain + imagery,
   a creek water surface, sun lighting, and the "analysis grade" that quiets the world while evidence is on screen.
   Belief-side only: geometry and provenance, never simulator truth. */
"use strict";

window.SSWorld = (() => {
  const C = window.Cesium;
  // Groups built from OSM footprints (plus illustrative detail). The Overture ML footprints are excluded: the ML
  // model labels many unmapped tanks as box buildings, which would contradict the imagery (docs/WORLD_BUILD.md).
  const TILE_GROUPS = ["tanks", "tank_farms", "jetties", "buildings", "cranes", "ships", "pipelines", "power", "yard",
                       "stockpiles", "rail", "roads", "fences"];
  const W = {viewer: null, M: null, h0: 0, models: {}, water: null, grade: 0, gradeTarget: 0, gradeRaf: 0, imagery: null,
             tint: null, ready: false};

  function enuMatrix(origin, h0) {
    return C.Transforms.eastNorthUpToFixedFrame(C.Cartesian3.fromDegrees(origin.lon, origin.lat, h0));
  }

  async function loadTiles(onProgress) {
    let man;
    try { man = await (await fetch("/world/manifest.json")).json(); } catch (_) { return 0; }
    const groups = TILE_GROUPS.filter((g) => man.tiles[g]);
    let n = 0;
    await Promise.all(groups.map(async (g) => {
      try {
        const m = await C.Model.fromGltfAsync({
          url: `/world/${man.tiles[g].file}`, modelMatrix: W.M, forwardAxis: C.Axis.X, upAxis: C.Axis.Y,
          shadows: C.ShadowMode.ENABLED, colorBlendMode: C.ColorBlendMode.MIX, colorBlendAmount: 0,
          color: C.Color.WHITE, backFaceCulling: true});
        W.viewer.scene.primitives.add(m);
        W.models[g] = m;
      } catch (e) { console.warn("world tile failed", g, e); }
      n += 1;
      if (onProgress) onProgress(n, groups.length, g);
    }));
    return Object.keys(W.models).length;
  }

  function addWater(ring) {
    if (!ring || ring.length < 3) return;
    const flat = ring.flatMap((p) => [p[0], p[1]]);
    const mat = new C.Material({fabric: {type: "Water", uniforms: {
      baseWaterColor: new C.Color(0.12, 0.13, 0.12, 0.30),      // translucent: the imagery's real creek shows through
      blendColor: new C.Color(0.16, 0.15, 0.12, 0.30),
      normalMap: C.buildModuleUrl("Assets/Textures/waterNormals.jpg"),
      frequency: 5000.0, animationSpeed: 0.003, amplitude: 4.0, specularIntensity: 0.22}}});
    W.waterMat = mat;
    W.water = W.viewer.scene.primitives.add(new C.GroundPrimitive({
      geometryInstances: new C.GeometryInstance({geometry: new C.PolygonGeometry({
        polygonHierarchy: new C.PolygonHierarchy(C.Cartesian3.fromDegreesArray(flat))})}),
      appearance: new C.EllipsoidSurfaceAppearance({material: mat, aboveGround: false}),
      classificationType: C.ClassificationType.TERRAIN}));
  }

  function applyGrade(a) {
    // a in [0,1]: 0 = natural daylight; 1 = analysis mode (world drops to a warm graphite monochrome).
    const L = W.imagery;
    if (L) {
      L.saturation = 0.62 - 0.56 * a;     // idle imagery is already quieter, so salt pans do not read as signal orange
      L.brightness = 0.92 - 0.5 * a;
      L.contrast = 1 + 0.05 * a;
      L.gamma = 1 + 0.15 * a;
    }
    for (const m of Object.values(W.models)) {
      m.color = W.tint;
      m.colorBlendAmount = 0.82 * a;
    }
    if (W.waterMat) {
      const k = 1 - 0.55 * a;
      W.waterMat.uniforms.baseWaterColor = new C.Color(0.12 * k, 0.13 * k, 0.12 * k, 0.30);
      W.waterMat.uniforms.blendColor = new C.Color(0.16 * k, 0.15 * k, 0.12 * k, 0.30);
    }
    W.viewer.scene.requestRender();
  }

  function setAnalysis(on, ms = 600) {
    W.gradeTarget = on ? 1 : 0;
    cancelAnimationFrame(W.gradeRaf);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const from = W.grade, t0 = performance.now(), dur = reduce ? 0 : ms;
    const step = (now) => {
      const k = dur ? Math.min(1, (now - t0) / dur) : 1;
      W.grade = from + (W.gradeTarget - from) * (1 - Math.pow(1 - k, 3));
      applyGrade(W.grade);
      if (k < 1) W.gradeRaf = requestAnimationFrame(step);
    };
    W.gradeRaf = requestAnimationFrame(step);
  }

  async function init(viewer, site, {tint = "#3a352c", onProgress} = {}) {
    W.viewer = viewer;
    W.tint = C.Color.fromCssColorString(tint);
    const scene = viewer.scene;
    scene.light = new C.SunLight();
    scene.globe.enableLighting = false;          // imagery evenly lit; models get a fixed afternoon sun (console.js)
    scene.globe.dynamicAtmosphereLighting = false;
    scene.globe.depthTestAgainstTerrain = true;
    scene.globe.showGroundAtmosphere = true;
    scene.fog.enabled = true;
    scene.fog.density = 2.2e-4;
    scene.highDynamicRange = false;
    scene.postProcessStages.fxaa.enabled = true;
    viewer.shadows = true;
    scene.shadowMap.softShadows = true;
    scene.shadowMap.size = 4096;
    scene.shadowMap.maximumDistance = 6000;
    W.imagery = viewer.imageryLayers.get(0);
    const o = site.meta.origin;
    const carto = C.Cartographic.fromDegrees(o.lon, o.lat);
    try { await C.sampleTerrainMostDetailed(viewer.terrainProvider, [carto]); } catch (_) { /* ellipsoid fallback */ }
    W.h0 = Number.isFinite(carto.height) ? carto.height : 0;
    W.M = enuMatrix(o, W.h0);
    const creek = (site.zones || []).find((z) => z.ref === "zone:kandla_creek");
    // The OSM creek polygon does not follow the imagery's shoreline closely enough: a water surface on it shows hard
    // edges. Off by default (the imagery's creek is real); ?water=1 turns it on for close-up shots.
    if (creek && creek.polygon && new URLSearchParams(location.search).has("water")) addWater(creek.polygon);
    const n = await loadTiles(onProgress);
    W.ready = true;
    applyGrade(0);
    return {tiles: n, h0: W.h0};
  }

  function enuToCartesian(x, y, z = 0) {
    return C.Matrix4.multiplyByPoint(W.M, new C.Cartesian3(x, y, z), new C.Cartesian3());
  }

  return {init, setAnalysis, enuToCartesian, state: W, TILE_GROUPS};
})();
