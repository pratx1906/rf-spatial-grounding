"""demo-v2 S5.6: the LLM warm-up sends exactly the agent's first-turn prefix (so vLLM's prefix cache is hot for the
first operator question), is best effort, and the gateway waits for the served model before warming."""

from pathlib import Path

import copy

from test_harness_runloop import ABSTAIN
from test_harness_runloop import FakeLLM as _FakeLLM


class FakeLLM(_FakeLLM):
    """Snapshots each request (run_task appends to the same messages list after the call)."""

    def _create(self, **kw):
        out = super()._create(**kw)
        self.kw[-1] = copy.deepcopy(kw)
        return out


def test_warm_up_sends_the_first_turn_request_prefix(env):
    from serve.agent import free_task, run_task, warm_up

    q = "Is any drone near Jetty 1?"
    run_llm, warm_llm = FakeLLM([ABSTAIN]), FakeLLM([ABSTAIN])
    list(run_task(env, free_task(q), client=run_llm, model="sft"))
    res = warm_up(env, client=warm_llm, model="sft", question=q)
    first, warm = run_llm.kw[0], warm_llm.kw[0]
    assert res["ok"] and res["prompt_tokens"] == 1000
    assert warm["max_tokens"] == 1 and warm["model"] == "sft"
    for k in ("messages", "tools", "tool_choice", "temperature", "parallel_tool_calls"):
        assert warm[k] == first[k], k
    assert env.done and env.end_reason == "warmup"


def test_warm_up_with_another_question_shares_the_static_prefix(env):
    from serve.agent import free_task, run_task, warm_up

    run_llm, warm_llm = FakeLLM([ABSTAIN]), FakeLLM([ABSTAIN])
    list(run_task(env, free_task("Which sensor hears the drone best?"), client=run_llm))
    warm_up(env, client=warm_llm)
    a, b = run_llm.kw[0]["messages"], warm_llm.kw[0]["messages"]
    assert a[0]["role"] == "system" and a[0] == b[0]          # byte-identical system prompt
    assert run_llm.kw[0]["tools"] == warm_llm.kw[0]["tools"]  # byte-identical tool schemas


def test_warm_up_failure_is_soft(env):
    from serve.agent import warm_up

    res = warm_up(env, client=FakeLLM([], exc=ConnectionError("refused")))
    assert res["ok"] is False and "ConnectionError" in res["error"]
    assert env.done


def test_gateway_warms_once_the_served_model_is_listed(stub_dir, monkeypatch):
    from serve import gateway

    probes = iter([{"ok": False, "models": []}, {"ok": True, "models": ["scenescout"]},
                   {"ok": True, "models": ["scenescout", "sft"]}])
    state = {"p": {"ok": False, "models": []}}

    def probe():
        return state["p"]

    def tick(_s):
        state["p"] = next(probes)

    calls = []
    monkeypatch.delenv("SCENESCOUT_SERVED_MODEL", raising=False)
    monkeypatch.setattr(gateway, "SITE_DIR", Path(stub_dir))
    monkeypatch.setattr(gateway, "_llm_probe", probe)
    monkeypatch.setattr(gateway.time, "sleep", tick)
    monkeypatch.setattr(gateway, "warm_up", lambda env, model: calls.append(model) or {"ok": True, "ms": 5})
    res = gateway.warm_llm(wait_s=60, poll_s=0)
    assert res["state"] == "done" and calls == ["scenescout"]


def test_gateway_warm_up_times_out_quietly(monkeypatch):
    from serve import gateway

    monkeypatch.setattr(gateway, "_llm_probe", lambda: {"ok": False, "models": []})
    monkeypatch.setattr(gateway.time, "sleep", lambda s: None)
    assert gateway.warm_llm(wait_s=0.01, poll_s=0)["state"] == "timeout"


def test_llm_launch_flags():
    from serve.llm import build_cmd, parse_args

    base = build_cmd(parse_args([]), 0.7)
    assert "--enable-prefix-caching" in base and "--speculative-config" not in base and "--max-num-seqs" not in base
    a = parse_args(["--gpu-util", "0.45", "--max-model-len", "10240", "--max-num-seqs", "4", "--spec-ngram", "4",
                    "--lora", "sft=checkpoints/sft_v11"])
    cmd = build_cmd(a, a.gpu_util)
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.45"
    assert cmd[cmd.index("--max-model-len") + 1] == "10240" and cmd[cmd.index("--max-num-seqs") + 1] == "4"
    import json

    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert spec["method"] == "ngram" and spec["num_speculative_tokens"] == 4
    assert cmd[-1] == "sft=checkpoints/sft_v11"
