import duckdb, yaml, numpy as np, sys
d = sys.argv[1]
fe = yaml.safe_load(open(f"{d}/friendly_emitters.yaml"))["emitters"]
sens = {s["id"]: s["pose"] for s in yaml.safe_load(open(f"{d}/sensors.yaml"))["sensors"]}
tb = {k: v["heading_bias_deg"] for k, v in yaml.safe_load(open(f"{d}/truth/world_variant.yaml"))["nodes"].items()}
c = duckdb.connect(f"{d}/observed.duckdb", read_only=True)
B = c.execute("select sensor_id, f_mhz, bearing_deg, snr_db from bursts").fetchall()
for s, p in sorted(sens.items()):
    res = []
    for e in fe:
        dist = np.hypot(e["x_m"] - p["x_m"], e["y_m"] - p["y_m"])
        if dist < 50: continue
        tbrg = np.degrees(np.arctan2(e["x_m"] - p["x_m"], e["y_m"] - p["y_m"])) % 360
        for sid, f, b, snr in B:
            if sid == s and b is not None and abs(f - e["freq_mhz"]) < 0.3:
                res.append(((b - tbrg + 180) % 360) - 180)
    r = np.array(res)
    print(s, "n", len(r), "median", np.round(np.median(r), 2) if len(r) else None, "true", round(tb.get(s, 0), 2),
          "frac within 5 of true", np.round(np.mean(np.abs(r - tb.get(s, 0)) < 5), 2) if len(r) else None)
