"""World-side spectrum environment for the v2 observation model (RF phase 2, Task 5).

    python -m envbuild.spectrum_env      (WSL, CPU) -> data/site_kandla/world_cache_v2/spectrum_env.yaml

Emitters the DF network hears besides drones. Two groups:
  known     the site's friendly emitters (data/site_kandla/friendly_emitters.yaml; the agent sees these through
            collateral_check), copied with their stated parameters;
  unknown   SYNTHETIC background the operator's picture lacks: Wi-Fi APs on buildings (2.4 GHz ch 1/6/11, 5.8 GHz
            ch 149/157/165), one crane remote per jetty (865-867 MHz licence-exempt band in India), a second LTE
            B40 carrier at 2390 MHz (inside the AIR7311 2.4 GHz tuning window, off the ISM channels) and an S-band
            VTS radar (3.05 GHz, pulsed: high peak, 0.1 % duty).
Positions are drawn from a fixed seed on real site objects; every unknown emitter is labelled `synthetic`. Values
(EIRP, duty, bandwidth) are typical figures, `estimated`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely import wkt

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "data" / "site_kandla"
OUT = SITE / "world_cache_v2" / "spectrum_env.yaml"
SEED = 7101

BW_MHZ = {"cctv_link": 20.0, "tank_telemetry": 0.125, "port_wifi": 20.0, "lte_site": 20.0, "vhf_base": 0.025}
DUTY = {"cctv_link": 1.0, "tank_telemetry": 0.02, "port_wifi": 0.3, "lte_site": 1.0, "vhf_base": 0.2}


def build() -> dict:
    rng = random.Random(SEED)
    rows = pq.read_table(SITE / "objects.parquet").to_pylist()
    buildings = sorted([r for r in rows if r["kind"] == "building" and r["height_m"]], key=lambda r: r["id"])
    jetties = sorted([r for r in rows if r["kind"] == "jetty"], key=lambda r: r["id"])
    known = yaml.safe_load((SITE / "friendly_emitters.yaml").read_text())["emitters"]
    out = []
    for e in known:
        out.append({"id": e["id"], "kind": e["kind"], "known_to_operator": True, "f_mhz": float(e["freq_mhz"]),
                    "bw_mhz": BW_MHZ.get(e["kind"], 1.0), "eirp_dbm": float(e["eirp_dbm"]),
                    "duty": DUTY.get(e["kind"], 0.5), "x_m": e["x_m"], "y_m": e["y_m"], "z_m": e["z_m"],
                    "placement": e.get("placement", "synthetic"), "source": "friendly_emitters.yaml"})

    def roof(r):
        c = wkt.loads(r["wkt"]).representative_point()
        return round(c.x, 1), round(c.y, 1), round(float(r["height_m"]) + 2.0, 1)

    for i, b in enumerate(rng.sample(buildings, 10)):
        x, y, z = roof(b)
        out.append({"id": f"bg_wifi24_{i}", "kind": "wifi_ap", "known_to_operator": False,
                    "f_mhz": rng.choice([2412.0, 2437.0, 2462.0]), "bw_mhz": 20.0, "eirp_dbm": 20.0,
                    "duty": round(rng.uniform(0.1, 0.4), 2), "x_m": x, "y_m": y, "z_m": z, "mount": b["id"]})
    for i, b in enumerate(rng.sample(buildings, 4)):
        x, y, z = roof(b)
        out.append({"id": f"bg_wifi58_{i}", "kind": "wifi_ap", "known_to_operator": False,
                    "f_mhz": rng.choice([5745.0, 5785.0, 5825.0]), "bw_mhz": rng.choice([20.0, 40.0]), "eirp_dbm": 23.0,
                    "duty": round(rng.uniform(0.1, 0.3), 2), "x_m": x, "y_m": y, "z_m": z, "mount": b["id"]})
    for j in jetties:
        c = wkt.loads(j["wkt"]).representative_point()
        out.append({"id": f"bg_crane_{j['id']}", "kind": "crane_remote", "known_to_operator": False,
                    "f_mhz": round(rng.uniform(865.2, 866.8), 3), "bw_mhz": 0.025, "eirp_dbm": 10.0, "duty": 0.2,
                    "x_m": round(c.x, 1), "y_m": round(c.y, 1), "z_m": 12.0, "mount": j["id"]})
    tall = max(buildings, key=lambda r: (r["height_m"], r["id"]))
    x, y, z = roof(tall)
    out.append({"id": "bg_lte_b40_2", "kind": "lte_site", "known_to_operator": False, "f_mhz": 2390.0, "bw_mhz": 20.0,
                "eirp_dbm": 58.0, "duty": 1.0, "x_m": x, "y_m": y, "z_m": z + 18.0, "mount": tall["id"]})
    j1 = wkt.loads(jetties[0]["wkt"]).representative_point()
    out.append({"id": "bg_vts_radar", "kind": "sband_radar", "known_to_operator": False, "f_mhz": 3050.0,
                "bw_mhz": 10.0, "eirp_dbm": 85.0, "duty": 0.001, "x_m": round(j1.x, 1), "y_m": round(j1.y, 1),
                "z_m": 35.0, "mount": jetties[0]["id"], "note": "peak EIRP; pulsed"})
    for e in out:
        if not e.get("known_to_operator"):
            e["placement"] = "synthetic"
            e["values"] = "estimated"
    return {"seed": SEED, "fidelity": "synthetic background + the site's friendly emitters", "emitters": out}


def main() -> None:
    env = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(env, sort_keys=False))
    kinds = {}
    for e in env["emitters"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(json.dumps({"emitters": len(env["emitters"]), "kinds": kinds, "out": str(OUT)}))


if __name__ == "__main__":
    main()
