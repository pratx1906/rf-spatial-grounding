"""Build configs/atmos_table.json with the open-source ITU-Rpy (MIT), isolated from the locked agent venv:

    uv run --no-project --with itur python scripts/atmos_table.py

Specific attenuation (dB/km) per band x rain rate x relative humidity at 30 C, sea level, Kandla latitude:
gaseous (ITU-R P.676, water-vapour density from RH via P.453) + rain (ITU-R P.838, horizontal polarisation,
horizontal path). Consumed by sim/obs_model/atmosphere.py (world side, D-031).
"""

import json
from pathlib import Path

import itur
import itur.models.itu453 as p453
import itur.models.itu676 as p676
import itur.models.itu838 as p838
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "configs" / "atmos_table.json"
BANDS = {"sub_ghz_868_915": 0.9, "ism_2400": 2.44, "ism_5800": 5.8}
RAIN = [0.0, 5.0, 25.0, 50.0]
RH = [50.0, 85.0, 95.0]
T_C, P_HPA = 30.0, 1010.0


def rho_g_m3(rh_pct: float, t_c: float) -> float:
    """Water-vapour density from relative humidity (P.453: e = H*e_s/100, rho = 216.7 e / T)."""
    es = float(np.asarray(p453.saturation_vapour_pressure(t_c, P_HPA).value))
    e = rh_pct * es / 100.0
    return 216.7 * e / (t_c + 273.15)


def main() -> None:
    table = {"t_c": T_C, "p_hpa": P_HPA, "rain_mm_h": RAIN, "rh_pct": RH, "bands": {}, "itur_version": itur.__version__,
             "models": "gas ITU-R P.676 (gamma0 + gammaw), rain ITU-R P.838 (pol. tilt 0, elevation 0)"}
    for band, f in BANDS.items():
        rows = []
        for r in RAIN:
            row = []
            for h in RH:
                rho = rho_g_m3(h, T_C)
                T_K = T_C + 273.15                    # P.676 functions take absolute temperature (K)
                gas = float(np.asarray(p676.gamma0_exact(f, P_HPA, rho, T_K).value)
                            + np.asarray(p676.gammaw_exact(f, P_HPA, rho, T_K).value))
                rain = float(np.asarray(p838.rain_specific_attenuation(r, f, 0.0, 0.0).value)) if r > 0 else 0.0
                row.append(round(gas + rain, 6))
            rows.append(row)
        table["bands"][band] = rows
    OUT.write_text(json.dumps(table, indent=1))
    print(json.dumps(table["bands"]))


if __name__ == "__main__":
    main()
