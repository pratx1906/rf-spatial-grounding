"""demo-v2 S5.1 / S5.3 / S4 requests: gateway endpoints (operator gate, POST /api/ask, one run per client, localhost
default, health with the LLM probe, Blender objects for the console)."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytest.importorskip("httpx")


@pytest.fixture()
def gw(stub_dir, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from harness.audit import AuditLog, OperatorGate
    from serve import gateway

    monkeypatch.setattr(gateway, "SITE_DIR", Path(stub_dir))
    monkeypatch.setattr(gateway, "AUDIT", AuditLog(tmp_path))
    monkeypatch.setattr(gateway, "GATE", OperatorGate(gateway.AUDIT, min_confirm_delay_s=0.0))
    monkeypatch.setattr(gateway, "_llm_probe", lambda: {"ok": False, "models": [], "error": "test"})
    return gateway, TestClient(gateway.app)


def _fake_run(events):
    def run(env, question, *a, **kw):
        env.reset(__import__("serve.agent", fromlist=["free_task"]).free_task(question))
        yield {"type": "start", "episode_id": env.episode_id}
        yield from events(env)
    return run


def _sse(text):
    return [json.loads(x[6:]) for x in text.split("\n\n") if x.startswith("data: ")]


def test_default_host_is_localhost():
    from serve.gateway import parse_args

    assert parse_args([]).host == "127.0.0.1"
    assert parse_args(["--host", "0.0.0.0"]).host == "0.0.0.0"


def test_ask_accepts_post_and_get(gw, monkeypatch):
    gateway, c = gw
    monkeypatch.setattr(gateway, "run_model", _fake_run(lambda env: iter([{"type": "done"}])))
    r = c.post("/api/ask", json={"q": "Is any drone near the jetty?"})
    assert r.status_code == 200 and _sse(r.text)[-1]["type"] == "done"
    r = c.get("/api/ask", params={"q": "Is any drone near the jetty?"})
    assert r.status_code == 200
    assert c.post("/api/ask", json={"q": ""}).status_code == 400


def test_one_concurrent_run_per_client(gw, monkeypatch):
    import threading
    import time

    gateway, c = gw
    go, running = threading.Event(), threading.Event()

    def slow(env):
        running.set()
        go.wait(10)
        yield {"type": "done"}

    monkeypatch.setattr(gateway, "run_model", _fake_run(slow))
    first = {}
    th = threading.Thread(target=lambda: first.setdefault("r", c.post("/api/ask", json={"q": "first question"})))
    th.start()
    assert running.wait(10)
    time.sleep(0.05)
    second = c.post("/api/ask", json={"q": "second question"})
    go.set()
    th.join(10)
    assert first["r"].status_code == 200
    assert second.status_code == 429 and second.json()["code"] == "busy"
    monkeypatch.setattr(gateway, "run_model", _fake_run(lambda env: iter([{"type": "done"}])))
    assert c.post("/api/ask", json={"q": "third question"}).status_code == 200      # released after the stream


def test_gate_round_trip_through_the_gateway(gw, monkeypatch):
    from harness.audit import answer_hash

    gateway, c = gw
    ans = {"intent": "collateral", "fp_ref": "fp:1", "violations": [], "requires_human_confirm": True,
           "evidence_refs": ["fp:1"], "confidence": 0.7}

    def events(env):
        h = answer_hash(env.episode_id, env.site.version, ans)
        yield {"type": "answer", "answer": ans, "answer_hash": h, "status": "verified", "gate_required": True,
               "flagged": False, "flags": [], "end_reason": "submitted"}
        yield {"type": "done"}

    monkeypatch.setattr(gateway, "run_model", _fake_run(events))
    ev = _sse(c.post("/api/ask", json={"q": "jam check"}).text)
    a = next(e for e in ev if e["type"] == "answer")
    ep = next(e for e in ev if e["type"] == "start")["episode_id"]
    body = {"episode_id": ep, "answer_hash": a["answer_hash"], "operator_id": "op_1", "decision": "defer",
            "step": "review"}
    r1 = c.post("/api/gate", json=body)
    assert r1.status_code == 200
    r2 = c.post("/api/gate", json={**body, "step": "confirm", "confirm_token": r1.json()["confirm_token"]})
    assert r2.status_code == 200 and r2.json()["recorded"] and r2.json()["decision"] == "defer"
    assert c.get(f"/api/gate/{ep}").json()["decision"] == "defer"
    assert c.post("/api/gate", json={**body, "answer_hash": "f" * 64}).status_code == 409


def test_health_reports_llm_and_model(gw):
    _, c = gw
    h = c.get("/api/health").json()
    assert h["ok"] and h["llm"]["ok"] is False and "model" in h


def test_served_model_prefers_sft(monkeypatch):
    from serve import gateway

    monkeypatch.delenv("SCENESCOUT_SERVED_MODEL", raising=False)
    monkeypatch.setattr(gateway, "_llm_probe", lambda: {"ok": True, "models": ["scenescout", "sft"]})
    assert gateway.served_model() == "sft"
    monkeypatch.setattr(gateway, "_llm_probe", lambda: {"ok": True, "models": ["scenescout"]})
    assert gateway.served_model() == "scenescout"
    monkeypatch.setenv("SCENESCOUT_SERVED_MODEL", "rl")
    assert gateway.served_model() == "rl"


@pytest.mark.skipif(not (REPO / "data/site_kandla/rf_scene_blender/objects.json").exists()
                    or not (REPO / "data/site_kandla/meta.yaml").exists(), reason="Kandla Blender export missing")
def test_blender_objects_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    from harness.env import site_view
    from serve import gateway

    monkeypatch.setattr(gateway, "SITE_DIR", REPO / "data" / "site_kandla")
    d = TestClient(gateway.app).get("/api/blender_objects").json()
    assert d["source"] == "blender" and d["geometry"]
    objs = d["objects"]
    tools = {o["ref"] for o in objs if o["in_tools"]}
    site = site_view(REPO / "data" / "site_kandla")
    assert tools and tools <= {o.ref for o in site.objects.values()}
    assert "obj:jetty_j1" in tools and "obj:tank_t001" in tools
    assert not any(o["kind"] in ("road",) for o in objs)
    bl = [o for o in objs if not o["in_tools"]]
    assert any(o["collection"] == "KW_ships" for o in bl) and any(o["collection"] == "KW_buildings_ml" for o in bl)
    assert not any(o["collection"] in ("KW_power", "KW_rail", "KW_fences", "KW_pipelines") for o in objs)
    for o in objs[:50] + bl[:50]:
        assert len(o["ring"]) >= 3 and len(o["center"]) == 2 and o["top_m"] is not None
        assert {"ref", "id", "name", "kind", "collection", "fidelity", "name_source", "base_m"} <= set(o)
    tank = next(o for o in objs if o["ref"] == "obj:tank_t001")
    assert tank["collection"] == "KW_tanks" and 5 < tank["top_m"] < 40
