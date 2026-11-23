"""M1 gate: tool p95 under 100 ms on the stub (SPEC 13), measured through the full executor path."""

import random
import time

import numpy as np

GATE_MS = 100.0


def test_tool_p95_under_gate(env, site):
    from tasks.base import Task

    rng = random.Random(0)
    tids = list(site.tracks)
    zones = [f"zone:{z}" for z in site.zones]
    objs = [f"obj:{o}" for o in site.objects]

    def args(name):
        tid = rng.choice(tids)
        pts = site.points[tid]["t"]
        t = float(pts[rng.randrange(len(pts))])
        return {
            "site_resolve": {"query": rng.choice(["tank farm a", "jetty 3", "cargo area", "fuel depot", "W07"])},
            "site_describe": {"ref": rng.choice(objs + zones)},
            "region_contents": {"region": {"type": "circle", "center": {"x_m": rng.uniform(-1000, 1000),
                                                                        "y_m": rng.uniform(-500, 800)},
                                           "radius_m": rng.uniform(100, 800)}},
            "spatial_relation": {"a_ref": rng.choice(objs), "b_ref": rng.choice(objs),
                                 "relation": rng.choice(["distance", "bearing", "within_m", "north_of"])},
            "inside_zone": {"subject": f"trk:{tid}@{t:g}", "zone_ref": rng.choice(zones)},
            "track_search": {"filters": {"time_range_s": [t, t + 60],
                                         "region": {"type": "ref", "ref": rng.choice(zones)}}},
            "track_detail": {"track_ref": f"trk:{tid}", "at_time_s": t},
            "df_hypotheses": {"track_ref": f"trk:{tid}", "at_time_s": round(t / 10) * 10},
        }[name]

    timings = {}
    for name in ("site_resolve", "site_describe", "region_contents", "spatial_relation", "inside_zone",
                 "track_search", "track_detail", "df_hypotheses"):
        env.reset(Task("perf", "operator_query", "perf", "perf", {"track_refs": []}))
        ms = []
        for _ in range(150):
            a = args(name)
            t0 = time.perf_counter()
            env.executor.run([{"name": name, "arguments": a}], 1)
            ms.append((time.perf_counter() - t0) * 1000)
            env.executor.cache.clear()          # measure real work, not cache hits
            env.executor.obs_tokens = 0
        timings[name] = float(np.percentile(ms, 95))
    print({k: round(v, 2) for k, v in timings.items()})
    assert max(timings.values()) < GATE_MS, timings
