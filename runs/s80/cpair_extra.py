"""Not pre-registered: per-family F scores per arm (seed mean) and B-C on the other families."""
import json, sys
sys.path.insert(0, "runs/s80")
import numpy as np
import cpair_analysis as c
rng = np.random.default_rng(1)
fams = ["near_object", "locate_ghost", "weak_low_alt", "loiter_start"]
tab = {a: {f: round(float(np.mean([v for v, _ in c.per_task(a, "F", f).values()])), 3) for f in fams} for a in "ABC"}
print(json.dumps(tab))
oa, ob, oc = (c.per_task(a, "F") for a in "ABC")
oc = {t: v for t, v in oc.items() if v[1] in c.OTHER}
oa2 = {t: v for t, v in oa.items() if v[1] in c.OTHER}
print("B-C other macro", c.boot(c.sub(oc, ob), macro_fams=c.OTHER, rng=rng))
print("C-A other macro", c.boot(c.sub(oa2, oc), macro_fams=c.OTHER, rng=rng))
for f in c.OTHER:
    print(f, "B-C", c.boot(c.sub(c.per_task("C", "F", f), c.per_task("B", "F", f)), rng=rng)["ci95"])
