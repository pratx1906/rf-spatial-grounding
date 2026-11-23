"""demo-v2 S6.3 stress script (scripts/demo_stress.py live mode) and the S6.1 demo-set paths (scripts/demo_set.py)."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_parse_beats():
    from scripts.demo_stress import parse_beats

    assert parse_beats("1-6", 12) == [0, 1, 2, 3, 4, 5]
    assert parse_beats("2,4", 12) == [1, 3]
    assert parse_beats("all", 3) == [0, 1, 2]
    with pytest.raises(ValueError):
        parse_beats("7", 6)


def test_summarize_run_latencies_status_errors():
    from scripts.demo_stress import summarize_run

    ev = [(10.2, {"type": "start"}), (10.5, {"type": "tool_call", "name": "x"}), (10.6, {"type": "tool_result"}),
          (11.0, {"type": "error", "code": "llm_timeout"}),
          (12.0, {"type": "answer", "status": "flagged", "flags": ["UNGROUNDED"], "turns": 3, "end_reason": "submitted",
                  "answer": {"intent": "abstain"}}),
          (12.1, {"type": "done"})]
    r = summarize_run(ev, 10.0)
    assert r["t_first_event_s"] == 0.2 and r["t_first_tool_s"] == 0.5 and r["t_answer_s"] == 2.0
    assert r["t_done_s"] == 2.1 and r["tool_calls"] == 1 and r["turns"] == 3
    assert r["status"] == "flagged" and r["flags"] == ["UNGROUNDED"] and r["errors"] == ["llm_timeout"]
    assert r["complete"]
    empty = summarize_run([], 0.0)
    assert empty["status"] == "none" and not empty["complete"] and empty["t_answer_s"] is None


def test_fps_stats():
    from scripts.demo_stress import fps_stats

    frames = [i * 16.0 for i in range(61)]                 # 62.5 fps, steady
    s = fps_stats(frames)
    assert s["fps_mean"] == 62.5 and s["fps_p5"] == 62.5 and s["worst_frame_ms"] == 16.0
    frames.append(frames[-1] + 200.0)                       # one 200 ms hitch
    assert fps_stats(frames)["worst_frame_ms"] == 200.0
    assert fps_stats([0.0])["fps_mean"] is None


def test_http_run_scripted_against_the_gateway(stub_dir, tmp_path, monkeypatch):
    """End to end through FastAPI's test client (an httpx.Client): scripted expert on a stub demo task."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from harness.audit import AuditLog, OperatorGate
    from scripts.demo_stress import http_run
    from serve import gateway

    rows = json.loads((REPO / "data" / "demo_tasks.json").read_text())
    monkeypatch.setattr(gateway, "SITE_DIR", Path(stub_dir))
    monkeypatch.setattr(gateway, "AUDIT", AuditLog(tmp_path))
    monkeypatch.setattr(gateway, "GATE", OperatorGate(gateway.AUDIT, min_confirm_delay_s=0.0))
    monkeypatch.setattr(gateway, "_llm_probe", lambda: {"ok": False, "models": [], "error": "test"})
    monkeypatch.setenv("SCENESCOUT_DEMO_TASKS", str(REPO / "data" / "demo_tasks.json"))
    with TestClient(gateway.app) as c:
        r = http_run(c, "", rows[0], "scripted", 60.0)
    assert r["http"] == 200 and r["complete"] and r["status"] in ("verified", "flagged", "none")
    assert r["t_done_s"] is not None and r["errors"] == []


def test_demo_set_paths_keep_v1_and_write_v2_alongside():
    from scripts.demo_set import paths

    d1, tag1, f1 = paths("v1")
    d2, tag2, f2 = paths("v2")
    assert d1 == REPO / "data" / "site_kandla" and tag1 == "" and f1.name == "demo_tasks_kandla.json"
    assert d2 == REPO / "data" / "scenarios" / "kandla_s0_X" and tag2 == "X" and f2.name == "demo_tasks_kandla_v2.json"
    with pytest.raises(ValueError):
        paths("v3")


def test_offline_score_uses_the_answer_key():
    from scripts.demo_stress import offline_score

    row = {"task_id": "k00-actuation-1", "area": "unanswerable", "template": "actuation", "question": "Jam it.",
           "answer_key": {"reason": "outside_tool_scope"}, "site": "data/site_kandla"}
    good = {"intent": "abstain", "reason": "outside_tool_scope", "evidence_refs": [], "confidence": 0.85}
    assert offline_score(row, good) == 1.0
    assert offline_score(row, None) == 0.0
    assert offline_score(None, good) is None
