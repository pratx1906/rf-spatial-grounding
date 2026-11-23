import duckdb, yaml, numpy as np, sys
d, S = sys.argv[1], sys.argv[2]
fe = yaml.safe_load(open(f"{d}/friendly_emitters.yaml"))["emitters"]
p = {s["id"]: s["pose"] for s in yaml.safe_load(open(f"{d}/sensors.yaml"))["sensors"]}[S]
c = duckdb.connect(f"{d}/observed.duckdb", read_only=True)
B = c.execute("select f_mhz, bearing_deg, snr_db, track_id from bursts where sensor_id=?", [S]).fetchall()
for e in fe:
    dist = np.hypot(e["x_m"] - p["x_m"], e["y_m"] - p["y_m"])
    tbrg = np.degrees(np.arctan2(e["x_m"] - p["x_m"], e["y_m"] - p["y_m"])) % 360
    r = np.array([((b - tbrg + 180) % 360) - 180 for f, b, snr, t in B if b is not None and abs(f - e["freq_mhz"]) < 0.3])
    if len(r): print(e["id"], e["band"], int(dist), "n", len(r), "med", round(float(np.median(r)),1), "MAD", round(float(np.median(abs(r-np.median(r)))),1), np.round(np.percentile(r,[10,50,90]),1))
