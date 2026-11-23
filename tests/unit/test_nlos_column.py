"""S7.1: track_points.nlos_suspect is additive (nullable, default false); readers of older sets keep working."""

import shutil

import duckdb

from sim.belief.site_view import SiteView


def test_stub_points_have_the_flag_false_by_default(stub_dir, site):
    con = duckdb.connect(str(stub_dir / "observed.duckdb"), read_only=True)
    cols = [r[1] for r in con.execute("PRAGMA table_info('track_points')").fetchall()]
    assert cols[-1] == "nlos_suspect"
    assert con.execute("select count(*) from track_points where nlos_suspect").fetchone()[0] == 0
    con.close()
    tid = next(iter(site.points))
    assert len(site.points[tid]["nlos"]) == len(site.points[tid]["t"]) and not site.points[tid]["nlos"].any()


def test_site_view_reads_sets_without_the_column(stub_dir, tmp_path):
    d = tmp_path / "old_set"
    shutil.copytree(stub_dir, d)
    con = duckdb.connect(str(d / "observed.duckdb"))
    con.execute("DROP INDEX IF EXISTS tp_idx")
    con.execute("ALTER TABLE track_points DROP COLUMN nlos_suspect")
    con.close()
    sv = SiteView(d)
    tid = next(iter(sv.points))
    assert not sv.points[tid]["nlos"].any()


def test_products_v2_points_are_causal_rows_with_the_flag(tmp_path):
    import math
    from types import SimpleNamespace

    from sim.obs_model.observe_v2 import Detection, products_v2

    sens = [SimpleNamespace(id=i, type="fixed_df", x=x, y=y, z=10.0)
            for i, x, y in (("S1", 0.0, 0.0), ("S2", 2000.0, 0.0), ("S3", 1000.0, 2000.0))]
    dets = []
    for t in range(0, 200, 10):
        px, py = 400.0 + 8.0 * t, 900.0
        for s in sens:
            brg = math.degrees(math.atan2(px - s.x, py - s.y)) % 360
            dets.append(Detection(float(t), s.id, "ism_2400", 2440.0, 10000.0, 1e-3, 1e-3, "ocusync_like", 0.9,
                                  [(brg, 3.0, 25.0)], math.nan, 25.0, -60.0))
    obs = products_v2(1, [], sens, dets, tmp_path, 5.0)
    assert len(obs["tracks"]) == 1
    rows = obs["points"]
    assert rows and all(len(r) == 11 and isinstance(r[10], bool) for r in rows)
    cut = products_v2(1, [], sens, [d for d in dets if d.t <= 100], tmp_path, 5.0)["points"]
    full = {r[1]: r[2:5] for r in rows}
    assert all(full[r[1]] == r[2:5] for r in cut)            # causal: later detections never move earlier points
