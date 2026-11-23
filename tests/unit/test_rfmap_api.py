"""Gateway RF map layers for the console (D-031): ghost risk, model trust, DF accuracy on the union grid."""

import base64
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
MAPS = REPO / "data" / "site_kandla" / "rf_cache" / "maps" / "ism_2400.npz"
pytestmark = pytest.mark.skipif(not MAPS.exists(), reason="RF maps not built")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from serve.gateway import app

    return TestClient(app)


@pytest.mark.parametrize("metric", ["ghost_risk", "trust", "df_err", "blind_spot"])
def test_rfmap_returns_the_union_grid_within_range(client, metric):
    r = client.get("/api/rfmap", params={"band": "ism_2400", "height": 40, "metric": metric})
    assert r.status_code == 200
    assert len(r.content) < 2_000_000
    d = r.json()
    v = np.frombuffer(base64.b64decode(d["values"]), "<f2").astype("f4").reshape(d["ny"], d["nx"])
    m = np.load(MAPS)
    assert (d["nx"], d["ny"]) == (len(m["xs"]), len(m["ys"]))
    f = v[np.isfinite(v)]
    assert f.size and f.min() >= d["vmin"] - 1e-3 and f.max() <= d["vmax"] + 1e-3
    assert d["legend"]


def test_rfmap_rejects_unknown_metric_and_band(client):
    assert client.get("/api/rfmap", params={"band": "ism_2400", "height": 40, "metric": "truth"}).status_code == 404
    assert client.get("/api/rfmap", params={"band": "x", "height": 40, "metric": "trust"}).status_code == 404
