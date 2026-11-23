import duckdb

from sim.stub_site import generate

TABLES = {"observed.duckdb": ["tracks", "track_points", "bearings", "events"],
          "truth/truth.duckdb": ["emitters", "emitter_points", "track_labels", "hyp_labels"]}


def _dump(d):
    out = {}
    for db, tables in TABLES.items():
        con = duckdb.connect(str(d / db), read_only=True)
        for t in tables:
            out[t] = con.execute(f"SELECT * FROM {t} ORDER BY ALL").fetchall()
        con.close()
    return out


def test_same_seed_same_world_bit_for_bit(tmp_path, stub_dir):
    m = generate(tmp_path / "again", seed=0)
    assert _dump(tmp_path / "again") == _dump(stub_dir)
    assert (tmp_path / "again" / "objects.parquet").read_bytes() == (stub_dir / "objects.parquet").read_bytes()
    assert m["version_hash"]


def test_different_seed_different_world(tmp_path, stub_dir):
    generate(tmp_path / "other", seed=1)
    assert _dump(tmp_path / "other")["track_points"] != _dump(stub_dir)["track_points"]


def test_tool_outputs_are_deterministic(stub_dir, tasks):
    from harness.env import AkshaEnv
    from tasks.expert import solve

    def run():
        env = AkshaEnv(stub_dir)
        out = []
        for t in tasks[:15]:
            env.reset(t)
            solve(env, t)
            out.append([(e.get("tool"), e.get("result_summary")) for e in env.tracer.events if e.get("tool")])
            out.append(env.reward_parts["total"])
        return out

    assert run() == run()
