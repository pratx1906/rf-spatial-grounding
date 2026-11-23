import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import duckdb
d = "data/scenarios_v3/kandla_s15_X"
o = duckdb.connect(f"{d}/observed.duckdb", read_only=True)
print(o.execute("describe bursts").fetchall())
print(o.execute("select * from bursts limit 5").fetchall())
print(o.execute("select count(*), count(bearing_deg), count(track_id), min(snr_db), median(snr_db), max(snr_db), median(bw_khz), median(dur_us) from bursts").fetchall())
tr = duckdb.connect(f"{d}/truth/truth.duckdb", read_only=True)
print(tr.execute("show tables").fetchall())
print(tr.execute("describe track_labels").fetchall())
print(tr.execute("describe emitter_points").fetchall())
