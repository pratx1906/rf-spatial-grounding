# SceneScout operator console (CesiumJS)

Served by `serve/gateway.py` at `/` (assets under `/ui/`). The old 2D stub chart is still at `/stub`.

```
wsl -d Ubuntu-24.04 --cd "C:\Users\Pratyush Dixit\Desktop\Urbexsim" -- .venv-wsl/bin/python -m serve.gateway --port 8080
# other site: --site data/stub_site  (or SCENESCOUT_SITE=...)
```

Open http://localhost:8080 in Chrome or Edge. The Cesium ion token is read from `token.txt` at the repo root by
`GET /api/cesium_token`. Without it the page falls back to OpenStreetMap tiles on a flat ellipsoid.

## Visual checklist

1. **Load:** within about 10 s the overview shows Bing aerial imagery of Kandla Creek, grey extruded tanks, amber
   protected tank farms and jetties, magenta translucent restricted-zone volumes, a dashed white port perimeter, and a
   yellow hero-zone outline.
2. **Fidelity badge** (top left) reads "Geometry: OpenStreetMap, heights estimated · Sensors and drones synthetic ·
   RF: geometric_stub". Clicking it expands the full fidelity, terrain, imagery and origin. "© OpenStreetMap
   contributors" sits bottom right above the timeline.
3. **Heights:** tanks and buildings stand on the ground. They should not float and should not be sunk about 50 m. Sensor
   masts are vertical lines from the ground to the labelled marker (S1–S5; D1 is the drone-borne DF).
4. **Click** a tank, zone, sensor or track. The info card (bottom right) shows name, kind, height, and provenance
   (source, status, name source, height source). Clicking inside a restricted volume still picks the tank inside it.
5. **Provenance colours** toggle: OSM objects with estimated heights turn blue, derived tank farms violet, and
   synthetic ships red. The legend updates.
6. **Time:** press play on the animation widget. Drones move with about 120 s trails (60 s for swarm members), ground
   ellipses and altitude tethers. Swarm G1 is violet and G2 green, each with a "G1 · swarm of 8" label. Single drones
   are cyan. trk:6071 (confidence 0.51) is grey and dashed. The HUD shows `t = … / 600 s`.
7. **Presets:** Port overview, Hero zone (top-down, north up), Oblique, Jetties, Tank farms, and Follow swarm G1
   (camera tracks the group centroid and starts playback). Any fly-to or other preset releases the follow.
8. **Agent:** with the model server running, ask a question. The trace shows one card per tool (name, short args,
   latency, summary, ref chips; errors in red with a hint). Each result that returns obj:/zone:/trk:/sensor: refs
   highlights those entities in yellow and flies to them.
9. **DF:** a `df_hypotheses` result draws 3 km rays from each sensor (pale), plus a faint dashed ray towards the
   track estimate. When the answer intent is `df_ambiguity`, chosen rays turn solid yellow, rejected rays dashed
   magenta, and the DF fix is marked.
10. **Answer card:** shows Yes/No (or the count, or the abstain reason in plain words), a confidence bar,
    assumptions, clickable evidence chips, and an amber "Needs operator confirmation" badge when
    `requires_human_confirm` is set.
11. **Scripted mode:** this mode is enabled only when `data/demo_tasks_kandla.json` exists. Until then the chips are
    model-mode examples. With `--site data/stub_site`, scripted mode replays the expert on `data/demo_tasks.json`.
12. **Responsiveness:** the page uses request-render mode, so with the clock paused the GPU should be idle. "Low
    graphics" lowers the resolution scale and hides trails.

## Known data artefacts (not UI bugs)
- One OSM "jetty" polygon is a long wedge towards KICT. This makes the Jetties preset frame wide.
- `obj:tank_t001` is a large (about 400 m) OSM storage-tank polygon.
