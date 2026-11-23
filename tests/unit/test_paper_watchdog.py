"""runs/paper/watchdog.py: failed RL jobs are requeued (at most 3 times, named to run next); others are left alone."""

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("watchdog", Path(__file__).parents[2] / "runs/paper/watchdog.py")
wd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wd)

RL = "#!/usr/bin/env bash\n# gpu_state: rl\nSEED=0 bash runs/paper/rl/arm_B.sh\n"
EVAL = "#!/usr/bin/env bash\nbash runs/paper/ladder/x.sh\n"


def finish(q: Path, name: str, body: str, exit_code: int) -> None:
    (q / "done" / name).write_text(body)
    (q / "done" / name.replace(".sh", ".json")).write_text(json.dumps({"exit": exit_code, "task": "P4"}))


def test_retry_name_sorts_next():
    assert wd.retry_name("204_P4_B_s0_resume2.sh", 1) == "2041_P4_B_s0_resume2.sh"
    assert "2041_x" < "205_x"                              # runs before the next queued job


def test_failed_rl_job_requeued_three_times(tmp_path, monkeypatch):
    for d in ("done", "pending", "staging", "running"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(wd, "Q", tmp_path)
    monkeypatch.setattr(wd, "LOG", tmp_path / "wd.log")
    monkeypatch.setattr(wd, "status_note", lambda *a: None)
    st = {"retries": {}, "root_of": {}, "handled": []}
    finish(tmp_path, "204_P4_B.sh", RL, 1)
    finish(tmp_path, "205_P2_eval.sh", EVAL, 1)            # not an RL job: left alone
    finish(tmp_path, "206_P4_C.sh", RL, 0)                 # succeeded: left alone
    wd.rl_retries(st)
    assert sorted(p.name for p in (tmp_path / "pending").iterdir()) == ["2041_P4_B.sh"]
    assert "watchdog retry 1/3" in (tmp_path / "pending" / "2041_P4_B.sh").read_text()
    wd.rl_retries(st)                                      # same failure is handled once
    assert len(list((tmp_path / "pending").iterdir())) == 1
    for k in (1, 2, 3):                                    # the retries fail too
        n = f"204{k}_P4_B.sh"
        (tmp_path / "pending" / n).unlink(missing_ok=True)
        finish(tmp_path, n, RL, 1)
        wd.rl_retries(st)
    assert sorted(p.name for p in (tmp_path / "pending").iterdir()) == []
    assert st["retries"]["204_P4_B.sh"] == 3
    assert "3 retries used" in (tmp_path / "wd.log").read_text()
