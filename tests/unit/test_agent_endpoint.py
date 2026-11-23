"""API policy plumbing (plan 2026-10-28 T2): provider clients, request options, tool_choice fallback, usage and
key hygiene. Mocked clients only; no network."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from serve import agent
from serve.agent import LLMOptions, build_request, is_reasoning_model, make_client, redact
from tasks.base import Task

REPO = Path(__file__).resolve().parents[2]
FAKE_KEY = "sk-test-SECRET-0123456789abcdef"


def test_local_client_is_the_unchanged_default(monkeypatch):
    c = make_client()
    assert str(c.base_url).rstrip("/") == agent.LLM_URL and c.api_key == "local"
    assert make_client("local").api_key == "local"


@pytest.mark.parametrize("provider,var,url", [
    ("openrouter", "OPENROUTER_API_KEY", "openrouter.ai"), ("openai", "OPENAI_API_KEY", "api.openai.com"),
    ("gemini", "GEMINI_API_KEY", "generativelanguage.googleapis.com")])      # anthropic: native SDK (test_anthropic_backend)
def test_provider_clients_take_keys_from_env(monkeypatch, provider, var, url):
    monkeypatch.setenv(var, FAKE_KEY)
    c = make_client(provider)
    assert url in str(c.base_url) and c.api_key == FAKE_KEY


def test_missing_key_names_the_variable_only(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_client("openai")
    with pytest.raises(ValueError):
        make_client("nope")


def test_reasoning_models_are_matched_by_prefix():
    assert is_reasoning_model("o3-mini") and is_reasoning_model("openai/gpt-5-mini") and is_reasoning_model("gpt-5")
    assert not is_reasoning_model("gpt-4.1") and not is_reasoning_model("scenescout")
    assert is_reasoning_model("my-model", prefixes=("my-",))


def test_build_request_local_default_unchanged():
    kw = build_request(LLMOptions(), "scenescout", [], [], 0.0, 512, "required", 30.0)
    assert kw == {"model": "scenescout", "messages": [], "tools": [], "tool_choice": "required", "temperature": 0.0,
                  "max_tokens": 512, "parallel_tool_calls": True, "timeout": 30.0}


def test_build_request_api_options():
    o = LLMOptions(provider="openai", max_completion_tokens=4096, reasoning_effort="low", seed=5)
    kw = build_request(o, "o4-mini", [], [], 0.0, 512, "required", 30.0)
    assert "temperature" not in kw and kw["max_completion_tokens"] == 4096 and "max_tokens" not in kw
    assert kw["reasoning_effort"] == "low" and kw["seed"] == 5
    kw = build_request(LLMOptions(provider="openai"), "gpt-4.1", [], [], 0.0, 512, "required", 30.0)
    assert kw["temperature"] == 0.0 and kw["max_completion_tokens"] == 512        # env cap when no API cap
    kw = build_request(LLMOptions(provider="openrouter", reasoning_effort="high"), "x/y", [], [], 0.0, 512, "auto", 1)
    assert kw["extra_body"] == {"reasoning": {"effort": "high"}} and kw["tool_choice"] == "auto"
    kw = build_request(LLMOptions(provider="anthropic", max_completion_tokens=2048), "claude-x", [], [], 0.0, 512,
                       "required", 1)
    assert kw["max_tokens"] == 2048


def test_build_request_gpt5_class_reasoning_model():
    """GPT-5-class (reasoning) models: no temperature, max_completion_tokens (never max_tokens), reasoning_effort,
    tool_choice required and parallel tool calls kept."""
    o = LLMOptions(provider="openai", max_completion_tokens=16000, reasoning_effort="medium")
    for m in ("gpt-5", "gpt-5-mini", "gpt-5.1", "openai/gpt-5"):
        kw = build_request(o, m, [], [], 0.0, 512, "required", 30.0)
        assert "temperature" not in kw and "max_tokens" not in kw and kw["max_completion_tokens"] == 16000
        assert kw["reasoning_effort"] == "medium" and kw["tool_choice"] == "required"
        assert kw["parallel_tool_calls"] is True


class _Err400(Exception):
    status_code = 400


class _FakeAPI:
    """Rejects tool_choice=required with a 400 (like some providers); replays tool calls otherwise."""

    def __init__(self, turns, reject_required=True, raise_with_key=False):
        self.turns = list(turns)
        self.kw = []
        self.reject_required = reject_required
        self.raise_with_key = raise_with_key
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.kw.append(kw)
        if self.raise_with_key:
            raise RuntimeError(f"401 invalid api key {FAKE_KEY}")
        if self.reject_required and kw["tool_choice"] == "required":
            raise _Err400("tool_choice 'required' is not supported")
        calls = self.turns.pop(0)
        tcs = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
               for i, (n, a) in enumerate(calls)]
        usage = SimpleNamespace(completion_tokens=30, prompt_tokens=1000,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
                                prompt_tokens_details=SimpleNamespace(cached_tokens=600))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=tcs))],
                               usage=usage)


def _site():
    site = REPO / "data" / "scenarios_v3" / "kandla_s15_X"
    if not (site / "meta.yaml").exists():
        pytest.skip("v3 scenario not built")
    return site


SUB = [("submit", {"answer": {"intent": "abstain", "reason": "outside_tool_scope", "evidence_refs": [],
                              "confidence": 0.9}})]
TASK = Task("t1", "unanswerable", "actuation", "Jam the drone now.", {"reason": "outside_tool_scope"})


def test_400_on_required_falls_back_to_auto_once():
    from harness.env import AkshaEnv
    from serve.agent import run_task

    env = AkshaEnv(_site())
    llm = _FakeAPI([SUB])
    ev = list(run_task(env, TASK, client=llm, options=LLMOptions(provider="openai")))
    assert [k["tool_choice"] for k in llm.kw] == ["required", "auto"]
    a = [e for e in ev if e["type"] == "assistant"][0]
    assert a["tool_choice"] == "auto" and a["usage"]["prompt_tokens"] == 1000
    assert a["usage"]["reasoning_tokens"] == 12 and a["usage"]["cached_tokens"] == 600
    assert env.answer_score == 1.0


def test_eval_row_records_provider_fallback_usage_and_no_key(monkeypatch):
    from eval.run import run_one

    _site()
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    t = Task.from_row({**TASK.to_row(), "site": "data/scenarios_v3/kandla_s15_X"})
    llm = _FakeAPI([SUB])
    r = run_one(t, "model", "api:o4-mini", 0.0, 0, "o4-mini", opts={"client": llm, "provider": "openai",
                                                                   "options": LLMOptions(provider="openai")})
    assert r["provider"] == "openai" and r["tool_choice_used"] == "auto"
    assert r["usage"] == {"prompt_tokens": 1000, "completion_tokens": 30, "reasoning_tokens": 12,
                          "cached_tokens": 600, "calls": 1}
    assert FAKE_KEY not in json.dumps(r)
    # a provider error that echoes the key is redacted in the row
    bad = _FakeAPI([SUB], raise_with_key=True)
    r = run_one(t, "model", "api:o4-mini", 0.0, 0, "o4-mini", opts={"client": bad, "provider": "openai",
                                                                   "options": LLMOptions(provider="openai")})
    assert r["error"] and FAKE_KEY not in json.dumps(r) and "[REDACTED]" in r["error"]


def test_local_rows_keep_required(monkeypatch):
    from eval.run import run_one

    _site()
    t = Task.from_row({**TASK.to_row(), "site": "data/scenarios_v3/kandla_s15_X"})
    llm = _FakeAPI([SUB], reject_required=False)
    r = run_one(t, "model", "m", 0.0, 0, "scenescout", opts={"client": llm})
    assert r["provider"] == "local" and r["tool_choice_used"] == "required" and r["score"] == 1.0
    assert "max_tokens" in llm.kw[0] and "seed" in llm.kw[0]


def test_redact_scrubs_known_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    assert redact(f"bad key {FAKE_KEY}!") == "bad key [REDACTED]!"
    assert redact(None) is None


def test_api_cost_estimate(tmp_path):
    rows = [{"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 100_000, "cached_tokens": 500_000,
                       "reasoning_tokens": 0, "calls": 8}} for _ in range(4)]
    p = tmp_path / "smoke.json"
    p.write_text(json.dumps({"rows": rows}))
    out = subprocess.run([sys.executable, str(REPO / "scripts/api_cost.py"), str(p), "--in-per-m", "2",
                          "--out-per-m", "8", "--cached-in-per-m", "0.5", "--tasks", "1000", "--json"],
                         capture_output=True, text=True, check=True, cwd=REPO)
    r = json.loads(out.stdout)
    # per task: 0.5M uncached * 2 + 0.5M cached * 0.5 + 0.1M * 8 = 1 + 0.25 + 0.8 = 2.05 $
    assert r["usd_per_task"] == pytest.approx(2.05) and r["usd_total"] == pytest.approx(2050.0)
    assert r["n_rows"] == 4


class _RejectsParam(_FakeAPI):
    """400 on a named request parameter (some reasoning models reject parallel_tool_calls)."""

    def __init__(self, turns, param):
        super().__init__(turns, reject_required=False)
        self.param = param

    def _create(self, **kw):
        if self.param in kw:
            self.kw.append(kw)
            raise _Err400(f"Unsupported parameter: '{self.param}' is not supported with this model.")
        return super()._create(**kw)


def test_400_on_unsupported_param_drops_it_for_the_episode():
    from harness.env import AkshaEnv
    from serve.agent import run_task

    env = AkshaEnv(_site())
    llm = _RejectsParam([SUB], "parallel_tool_calls")
    ev = list(run_task(env, TASK, client=llm, model="gpt-5", options=LLMOptions(provider="openai")))
    assert "parallel_tool_calls" in llm.kw[0] and "parallel_tool_calls" not in llm.kw[1]
    assert llm.kw[1]["tool_choice"] == "required"                       # not confused with the tool_choice fallback
    a = [e for e in ev if e["type"] == "assistant"][0]
    assert a["dropped_params"] == ["parallel_tool_calls"] and env.answer_score == 1.0


def test_local_assistant_event_has_no_api_extras():
    from harness.env import AkshaEnv
    from serve.agent import run_task

    env = AkshaEnv(_site())
    ev = list(run_task(env, TASK, client=_FakeAPI([SUB], reject_required=False)))
    a = [e for e in ev if e["type"] == "assistant"][0]
    assert set(a) == {"type", "turn", "text", "gen_ms", "tokens", "n_calls", "usage", "tool_choice"}


def test_api_cost_cache_writes_and_opus_preset(tmp_path):
    rows = [{"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 100_000, "cached_tokens": 500_000,
                       "cache_creation_tokens": 200_000, "reasoning_tokens": 0, "calls": 8}} for _ in range(2)]
    p = tmp_path / "smoke.json"
    p.write_text(json.dumps({"rows": rows}))
    out = subprocess.run([sys.executable, str(REPO / "scripts/api_cost.py"), str(p), "--preset", "claude-opus-5-5",
                          "--json"], capture_output=True, text=True, check=True, cwd=REPO)
    r = json.loads(out.stdout)
    # 0.3M uncached * 4 + 0.5M read * 0.20 + 0.2M write * 5 + 0.1M out * 20 = 1.2 + 0.1 + 1.0 + 2.0 = 4.3 $
    assert r["usd_per_task"] == pytest.approx(4.3) and r["prices_per_m"]["cache_write"] == 5.0
    assert r["tokens_per_task"]["cache_creation_tokens"] == 200_000
    # explicit prices override the preset; without a write price, writes bill at the input price
    out = subprocess.run([sys.executable, str(REPO / "scripts/api_cost.py"), str(p), "--in-per-m", "2",
                          "--out-per-m", "8", "--cached-in-per-m", "0.5", "--json"],
                         capture_output=True, text=True, check=True, cwd=REPO)
    assert json.loads(out.stdout)["usd_per_task"] == pytest.approx(0.6 + 0.25 + 0.4 + 0.8)
