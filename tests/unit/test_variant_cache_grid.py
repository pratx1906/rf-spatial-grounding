"""Living-world variant caches (D-031): same grid as nominal, and the W0 control reproduces the nominal cache.

Skipped when the variant caches have not been traced (Windows GPU step, scripts/precompute_variants.py).
"""

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "data" / "site_kandla"
NOM = SITE / "rf_cache"
BANDS = ("sub_ghz_868_915", "ism_2400", "ism_5800")
SENSORS = ("S1", "S2", "S3", "S4", "S5")


def variant_roots():
    roots = sorted((SITE / "world_cache" / "rf").glob("*")) + sorted((NOM / "ensemble").glob("*"))
    return [r for r in roots if r.is_dir()]


def pairs(root: Path):
    for s in SENSORS:
        for b in BANDS:
            f = root / f"air7311_{s}" / "spec_diff" / f"{b}.npz"
            if f.exists():
                yield s, b, f, NOM / f"air7311_{s}" / "spec_diff" / f"{b}.npz"


@pytest.mark.skipif(not variant_roots(), reason="no variant caches traced")
def test_variant_caches_share_the_nominal_grid():
    n = 0
    for root in variant_roots():
        for _, _, f, nom in pairs(root):
            a, b = np.load(f), np.load(nom)
            for k in ("xs", "ys", "heights"):
                assert np.array_equal(a[k], b[k]), (f, k)
            n += 1
    assert n > 0


def strongest(z):
    p = z["h_re"] ** 2 + z["h_im"] ** 2
    i = p.argmax(-1)[..., None]
    pt = np.take_along_axis(z["ptype"], i, -1)[..., 0]
    az = np.take_along_axis(z["az_deg"], i, -1)[..., 0]
    pw = 10 * np.log10(np.maximum(np.take_along_axis(p, i, -1)[..., 0], 1e-30))
    return pt, az, pw


W0 = SITE / "world_cache" / "rf" / "W0"


# Sionna's ray launching has a run-to-run noise floor: re-tracing the SAME nominal scene (S4, 2.4 GHz) changes the
# strongest path in 0.73% of cells (az > 0.1 deg in 0.66%), almost all by a path found in one run and missed in the
# other (runs/rf_variants/nominal_rerun, D-031). The control must stay within that floor, not below it.
NOISE_FLOOR = 0.01


@pytest.mark.skipif(not W0.exists(), reason="W0 control not traced")
def test_w0_control_reproduces_the_nominal_cache():
    checked = 0
    for s, b, f, nom in pairs(W0):
        pt1, az1, pw1 = strongest(np.load(f))
        pt0, az0, pw0 = strongest(np.load(nom))
        valid = (pt0 != 255) | (pt1 != 255)
        same_type = (pt0 == pt1)[valid].mean()
        both = valid & (pt0 != 255) & (pt1 != 255)
        daz = np.abs((az1 - az0 + 180) % 360 - 180)[both]
        dpw = np.abs(pw1 - pw0)[both]
        assert same_type >= 0.995, (s, b, same_type)
        assert (daz > 0.1).mean() < NOISE_FLOOR, (s, b, float((daz > 0.1).mean()))
        assert (dpw > 0.1).mean() < NOISE_FLOOR, (s, b, float((dpw > 0.1).mean()))
        assert np.median(dpw) < 1e-3                       # where both runs find the same paths, values are exact
        checked += 1
    assert checked == len(SENSORS) * len(BANDS)
