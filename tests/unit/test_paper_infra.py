"""Paper-program infrastructure (plan 2026-10-28 P0): pipeline programs, dashboard data, manifests, GPU queue runner."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import paper_manifest as pm
from scripts import pipeline_dashboard as dash
from scripts import pipeline_status as ps

REPO = Path(__file__).resolve().parents[2]


def test_pipeline_status_programs(monkeypatch):
    try:
        assert ps.set_program("paper") == REPO / "runs" / "paper" and ps.LOGS == REPO / "runs" / "paper" / "logs"
        monkeypatch.setenv(ps.ENV_VAR, "paper")
        assert ps.program_root() == REPO / "runs" / "paper"
        monkeypatch.delenv(ps.ENV_VAR)
        assert ps.program_root() == REPO / "runs" / "rfconv"
        with pytest.raises(ValueError):
            ps.program_root("nope")
    finally:
        ps.set_program("rfconv")
    assert ps.STATUS == REPO / "runs" / "rfconv" / "status"


def test_run_logged_program(tmp_path):
    env = {**os.environ, "SCENESCOUT_PIPELINE_LOGS": str(tmp_path)}
    r = subprocess.run(["bash", str(REPO / "scripts" / "run_logged.sh"), "--program", "paper", "PX", "sh", "-c",
                        "echo hi; exit 5"], env=env, capture_output=True, text=True)
    assert r.returncode == 5
    txt = (tmp_path / "PX.log").read_text()
    assert "=== START PX" in txt and "hi" in txt and "rc=5" in txt
    bad = subprocess.run(["bash", str(REPO / "scripts" / "run_logged.sh"), "--program", "x", "PX", "true"],
                         capture_output=True, text=True)
    assert bad.returncode == 2


def test_dashboard_paper_data(tmp_path, monkeypatch):
    try:
        cfg = dash.configure("paper")
        assert cfg["port"] == 8096 and dash.PLAN["P4"] == (18.0, ["P3"]) and "P12" in dash.ORDER
        monkeypatch.setattr(dash, "RF", tmp_path)
        q = tmp_path / "queue"
        for d in ("pending", "running", "done"):
            (q / d).mkdir(parents=True)
        (q / "pending" / "020_P2_ladder.sh").write_text("true\n")
        (q / "pending" / "010_P1_models.sh").write_text("true\n")
        (q / "running" / "005_P1_x.sh").write_text("true\n")
        (q / "running" / "005_P1_x.json").write_text(json.dumps({"started": 1.0, "peak_vram_mb": 900}))
        (q / "done" / "001_P0_t.json").write_text(json.dumps({"name": "001_P0_t.sh", "task": "P0", "exit": 0,
                                                               "ended": 5, "episodes_per_s": 1.5}))
        qd = dash.queue_data()
        assert [j["name"] for j in qd["pending"]] == ["010_P1_models.sh", "020_P2_ladder.sh"]
        assert qd["running"][0]["task"] == "P1" and qd["done"][0]["episodes_per_s"] == 1.5
        assert qd["runner"]["alive"] is False
        lg = tmp_path / "results" / "P4_B" / "seed0"
        lg.mkdir(parents=True)
        (lg / "log.jsonl").write_text("\n".join([
            json.dumps({"iter": 1, "reward_mean": 0.4, "val_before_update": {"score": 0.8}}),
            "not json",
            json.dumps({"iter": 2, "reward_mean": 0.5, "abstain_rate": 0.1, "kl": 0.02}),
            json.dumps({"step": 3, "reward": {"mean": 0.6}})]))
        g = dash.grpo_data()["runs"][0]
        assert g["run"] == "P4_B/seed0" and g["n"] == 3 and g["x_max"] == 3
        assert g["reward"] == [[1.0, 0.4], [2.0, 0.5], [3.0, 0.6]] and g["val"] == [[1.0, 0.8]]
        assert g["abstain"] == [[2.0, 0.1]] and g["kl"] == [[2.0, 0.02]]
        sd = tmp_path / "results" / "P1" / "sft11_v6"
        sd.mkdir(parents=True)
        (sd / "summary.json").write_text(json.dumps({"model": "sft11", "set": "v6", "n": 700, "score": 0.82,
                                                     "ci95": [0.8, 0.84]}))
        (tmp_path / "results" / "P1" / "broken").mkdir()
        (tmp_path / "results" / "P1" / "broken" / "summary.json").write_text("{")
        rows = dash.results_data()["rows"]
        assert len(rows) == 1 and rows[0]["task"] == "P1" and rows[0]["ci95"] == [0.8, 0.84]
        assert dash.job_task("150_P4_pilot.sh") == "P4" and dash.job_task("readme.sh") == "misc"
    finally:
        dash.configure("rfconv")
    assert dash.PLAN["T6"] == (4.0, ["T5", "T2b"]) and dash.STATUS == REPO / "runs" / "rfconv" / "status"


def _eval(set_path="eval/frozen/v6/tasks.jsonl"):
    rows = [{"task_id": f"k{s}X-fam-{i}", "area": "operator_query" if i % 2 else "df_ambiguity",
             "score": float((i + s) % 2), "abstained": i == 0} for s in (15, 16) for i in range(10)]
    return {"policy": "model", "model_id": "sft_v11", "set": set_path, "seed": 0, "samples": 1, "ablation": "none",
            "summary": {"operator_query": {"n": 10, "score": 0.5, "abstain_rate": 0.0},
                        "_all": {"n": 20, "score": 0.5, "macro_score": 0.5, "wallclock_s": 10.0}}, "rows": rows}


def test_manifest_and_summary(tmp_path, monkeypatch):
    e = tmp_path / "res" / "P1" / "sft11_v6" / "eval.json"
    e.parent.mkdir(parents=True)
    e.write_text(json.dumps(_eval()))
    ad = tmp_path / "adapter"
    ad.mkdir()
    (ad / "adapter_config.json").write_text("{}")
    ledger = tmp_path / "LEDGER.md"
    monkeypatch.setenv("PAPER_JOB", "010_P1_models.sh")
    s = pm.write(e, "P1", cmd="python -m eval.run ...", model="sft11", adapter=str(ad), ledger=True, ledger_path=ledger)
    assert s["set"] == "v6" and s["n"] == 20 and s["episodes_per_s"] == 2.0 and s["n_clusters"] == 2
    assert s["ci95"][0] <= 0.5 <= s["ci95"][1] and s["abstain_rate"] == 0.1
    man = json.loads((e.parent / "manifest.json").read_text())
    assert man["set_sha256"] == pm.sha256_file(REPO / "eval/frozen/v6/tasks.jsonl") and man["adapter_sha256"]
    assert man["job"] == "010_P1_models.sh" and "dirty_files" in man and man["cmd"].startswith("python")
    line = ledger.read_text().strip()
    assert " P1 sft11 on v6: score 0.500 CI [" in line and line.endswith("summary.json)")
    st = pm.job_stats("010_P1_models.sh", 20000, results=tmp_path / "res")
    assert st == {"runs": 1, "episodes": 20, "eval_s": 10.0, "episodes_per_s": 2.0}
    assert json.loads((e.parent / "manifest.json").read_text())["peak_vram_mb"] == 20000
    assert pm.set_name("runs/s79/v14/heldout_tasks.jsonl") == "heldout_tasks"
    out = tmp_path / "cli"                                          # CLI: --cmd must not clash with the sub-command
    assert pm.main(["write", "--eval", str(e), "--task", "P1", "--out-dir", str(out), "--cmd", "x y"]) == 0
    assert json.loads((out / "manifest.json").read_text())["cmd"] == "x y" and (out / "summary.json").exists()


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="runner is a WSL/Linux script")
def test_queue_runner_end_to_end(tmp_path):
    q = tmp_path / "queue"
    (q / "pending").mkdir(parents=True)
    (q / "pending" / "020_P9_fail.sh").write_text("echo fail; exit 4\n")
    (q / "pending" / "010_P1_ok.sh").write_text(
        "# gpu_state: bench\n"
        "echo task=$PAPER_TASK job=$PAPER_JOB lock=$AKSHA_GPU_LOCK prog=$SCENESCOUT_PIPELINE\n"
        "test -f \"$OUTER\" && echo outer-held\n")
    env = {**os.environ, "PAPER_QUEUE_DIR": str(q), "PAPER_RUNNER_ONCE": "1", "PAPER_POLL_S": "1",
           "PAPER_SAMPLE_S": "1", "PAPER_OUTER_LOCK": str(tmp_path / "outer.lock"), "OUTER": str(tmp_path / "outer.lock"),
           "PAPER_NVSMI": "/bin/false", "SCENESCOUT_PIPELINE_LOGS": str(tmp_path / "logs"), "PAPER_PY": sys.executable}
    r = subprocess.run(["bash", str(REPO / "runs" / "paper" / "queue" / "runner.sh")], env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    ok = json.loads((q / "done" / "010_P1_ok.json").read_text())
    bad = json.loads((q / "done" / "020_P9_fail.json").read_text())
    assert ok["exit"] == 0 and ok["task"] == "P1" and ok["gpu_state"] == "bench" and bad["exit"] == 4
    assert ok["ended"] <= bad["started"]                                   # lexical order, one at a time
    log = (tmp_path / "logs" / "P1.log").read_text()
    assert "task=P1 job=010_P1_ok.sh" in log and str(q / "running" / ".inner_lock") in log and "prog=paper" in log
    assert "outer-held" in log and not (tmp_path / "outer.lock").exists()   # outer lock held during, released after
    assert not list((q / "pending").iterdir()) and (q / "done" / "010_P1_ok.sh").exists()
    assert json.loads((q / "runner.json").read_text())["state"] == "stopped"
