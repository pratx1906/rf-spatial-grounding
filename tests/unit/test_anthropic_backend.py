"""Native Claude (Anthropic Messages API) backend for the agent loop (plan 2026-10-28 T2, frontier baselines).

Mocked SDK clients only; no network. Checks the Claude Opus 5.5 request rules (no sampling params, no thinking
budget, no forced tool_choice, effort via output_config, server-side refusal fallback), prompt caching markers,
parallel tool results in one user message, preserved thinking (append-only history), the one-shot nudge on a
text-only turn, stop reasons, refusal, usage mapping and key hygiene.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from serve.agent import LLMOptions
from tasks.base import Task

REPO = Path(__file__).resolve().parents[2]
FAKE_KEY = "sk-ant-test-SECRET-0123456789abcdef"
SUB = {"answer": {"intent": "abstain", "reason": "outside_tool_scope", "evidence_refs": [], "confidence": 0.9}}
TASK = Task("t1", "unanswerable", "actuation", "Jam the drone now.", {"reason": "outside_tool_scope"})
SITE = "data/scenarios_v3/kandla_s15_X"


def _site():
    if not (REPO / SITE / "meta.yaml").exists():
        pytest.skip("v3 scenario not built")
    return REPO / SITE


class Blk(SimpleNamespace):
    def model_dump(self, mode="python", exclude_none=False):
        return dict(vars(self))


def tool_use(i, name, inp):
    return Blk(type="tool_use", id=f"toolu_{i}", name=name, input=inp)


def thinking(sig):
    return Blk(type="thinking", thinking="", signature=sig)


def text(t):
    return Blk(type="text", text=t)


def usage(inp=100, out=50, cw=0, cr=0, iterations=None):
    return SimpleNamespace(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=cw,
                           cache_read_input_tokens=cr, iterations=iterations)


def resp(content, stop="tool_use", u=None, details=None, model="claude-opus-5-5"):
    return SimpleNamespace(content=content, stop_reason=stop, stop_details=details, usage=u or usage(), model=model)


class FakeAnthropic:
    """Replays canned responses; records every request (messages snapshotted: the backend appends after a call)."""

    def __init__(self, responses, raise_with_key=False):
        self.responses = list(responses)
        self.kw, self.snap, self.route = [], [], []
        self.raise_with_key = raise_with_key
        self.messages = SimpleNamespace(create=lambda **kw: self._create("messages", **kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self._create("beta", **kw)))

    def _create(self, route, **kw):
        if self.raise_with_key:
            raise RuntimeError(f"401 invalid x-api-key {FAKE_KEY}")
        self.kw.append(kw)
        self.route.append(route)
        self.snap.append([dict(m) for m in kw["messages"]])
        return self.responses.pop(0)


def _run(responses, options=None, **kw):
    from harness.env import AkshaEnv
    from serve.agent import run_task

    env = AkshaEnv(_site())
    llm = FakeAnthropic(responses)
    ev = list(run_task(env, TASK, client=llm, model="claude-opus-5-5",
                       options=options or LLMOptions(provider="anthropic"), **kw))
    return env, llm, ev


def test_request_follows_opus55_rules_and_caches_the_static_prefix():
    env, llm, _ = _run([resp([thinking("s1"), tool_use(1, "submit", SUB)])])
    kw = llm.kw[0]
    for banned in ("temperature", "top_p", "top_k", "seed", "parallel_tool_calls", "max_completion_tokens"):
        assert banned not in kw
    assert kw.get("thinking", {"type": "adaptive"}) == {"type": "adaptive"}         # never budget / disabled
    assert "budget_tokens" not in json.dumps(kw.get("thinking") or {})
    assert kw["tool_choice"] == {"type": "auto"}
    assert kw["output_config"] == {"effort": "medium"} and kw["max_tokens"] == 16000
    # server-side refusal fallback on by default (beta route, "default" scalar form + its header)
    assert llm.route == ["beta"] and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    # caching: last tool + system carry the marker; nothing else in tools does; growing tail auto-cached
    assert kw["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in t for t in kw["tools"][:-1])
    assert kw["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert kw["cache_control"] == {"type": "ephemeral"}
    # the env's OpenAI-format schemas map 1:1, order kept (deterministic prefix)
    obs_tools = env.tools
    assert [t["name"] for t in kw["tools"]] == [t["function"]["name"] for t in obs_tools]
    assert kw["tools"][0]["input_schema"] == obs_tools[0]["function"]["parameters"]
    assert kw["tools"][0]["description"] == obs_tools[0]["function"]["description"]
    # system = env system prompt + the tool-protocol instruction; first message is the user question
    from serve.anthropic_backend import SYSTEM_SUFFIX

    assert kw["system"][0]["text"] == env.system + SYSTEM_SUFFIX and "submit" in SYSTEM_SUFFIX
    assert llm.snap[0] == [{"role": "user", "content": TASK.question}]
    assert env.answer_score == 1.0 and env.end_reason == "submitted"


def test_effort_cap_and_fallback_flag():
    o = LLMOptions(provider="anthropic", reasoning_effort="high", max_completion_tokens=8000, anthropic_fallback=False)
    _, llm, ev = _run([resp([tool_use(1, "submit", SUB)])], options=o)
    kw = llm.kw[0]
    assert llm.route == ["messages"] and "betas" not in kw and "fallbacks" not in kw
    assert kw["output_config"] == {"effort": "high"} and kw["max_tokens"] == 8000
    a = [e for e in ev if e["type"] == "assistant"][0]
    assert a["reasoning_effort"] == "high" and a["tool_choice"] == "auto"


def test_parallel_results_in_one_user_message_and_thinking_preserved():
    first = [thinking("sig-A"), text("Looking."), tool_use(1, "track_search", {}), tool_use(2, "no_such_tool", {})]
    env, llm, _ = _run([resp(first), resp([thinking("sig-B"), tool_use(3, "submit", SUB)])])
    m = llm.snap[1]
    assert [x["role"] for x in m] == ["user", "assistant", "user"]
    assert m[1]["content"] is first                                   # verbatim: same blocks, same order
    res = m[2]["content"]
    assert [b["type"] for b in res] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in res] == ["toolu_1", "toolu_2"]
    assert not res[0].get("is_error") and res[1]["is_error"] is True
    assert json.loads(res[0]["content"].split("\n", 1)[0])["ok"] is True
    # append-only: the second request starts with exactly the first request's messages
    assert m[:1] == llm.snap[0]
    # system/tools byte-identical across turns (cache prefix + thinking binding)
    assert json.dumps(llm.kw[0]["system"]) == json.dumps(llm.kw[1]["system"])
    assert json.dumps(llm.kw[0]["tools"]) == json.dumps(llm.kw[1]["tools"])
    assert env.answer_score == 1.0


def test_text_only_turn_gets_one_nudge():
    from serve.anthropic_backend import NUDGE

    env, llm, ev = _run([resp([text("I think it is jamming.")], stop="end_turn"),
                         resp([tool_use(1, "submit", SUB)])])
    assert len(llm.kw) == 2 and env.turn == 1                        # nudge is inside one env turn
    last = llm.snap[1][-1]
    assert last["role"] == "user" and last["content"] == [{"type": "text", "text": NUDGE}]
    a = [e for e in ev if e["type"] == "assistant"][0]
    assert a["nudges"] == 1 and a["stop_reasons"] == ["end_turn", "tool_use"]
    assert env.answer_score == 1.0


def test_text_only_after_nudge_goes_to_env_format_error():
    env, llm, ev = _run([resp([text("a")], stop="end_turn"), resp([text("b")], stop="end_turn"),
                         resp([tool_use(1, "submit", SUB)])])
    assert len(llm.kw) == 3 and env.turn == 2 and env.format_failures == 1
    # env's format-error message goes back as a user text block after the second text-only reply
    assert llm.snap[2][-1]["role"] == "user" and "Format error" in llm.snap[2][-1]["content"][0]["text"]
    assert [e["nudges"] for e in ev if e["type"] == "assistant"] == [1, 0]


def test_pause_turn_is_resumed_without_a_user_message():
    env, llm, ev = _run([resp([text("working")], stop="pause_turn"), resp([tool_use(1, "submit", SUB)])])
    assert len(llm.kw) == 2 and llm.snap[1][-1]["role"] == "assistant"
    assert [e for e in ev if e["type"] == "assistant"][0]["nudges"] == 0


def test_refusal_ends_the_episode_and_is_recorded():
    det = SimpleNamespace(category="cyber", explanation="declined")
    env, llm, ev = _run([resp([], stop="refusal", details=det)])
    assert env.done and env.end_reason == "refusal" and env.answer is None
    err = [e for e in ev if e["type"] == "error"]
    assert err and err[0]["code"] == "refusal"
    a = [e for e in ev if e["type"] == "assistant"][0]
    assert a["refusal"] == "cyber" and a["stop_reasons"] == ["refusal"]


def test_fallback_ran_and_usage_mapping():
    it = [SimpleNamespace(type="message", input_tokens=10, output_tokens=0, cache_creation_input_tokens=0,
                          cache_read_input_tokens=0),
          SimpleNamespace(type="fallback_message", input_tokens=100, output_tokens=40,
                          cache_creation_input_tokens=0, cache_read_input_tokens=0)]
    _, _, ev = _run([resp([tool_use(1, "site_describe", {})], u=usage(100, 50, cw=7000)),
                     resp([tool_use(2, "submit", SUB)], u=usage(30, 20, cr=7100, iterations=it),
                          model="claude-opus-4-8")])
    a1, a2 = [e for e in ev if e["type"] == "assistant"]
    assert a1["usage"] == {"prompt_tokens": 7100, "completion_tokens": 50, "reasoning_tokens": 0,
                           "cached_tokens": 0, "cache_creation_tokens": 7000, "input_tokens": 100}
    assert a1["tokens"] == 50 and a1["fallback"] is False
    # with iterations the per-attempt sum is the billing truth
    assert a2["usage"]["prompt_tokens"] == 110 and a2["usage"]["completion_tokens"] == 40
    assert a2["fallback"] is True and a2["served_by"] == "claude-opus-4-8"


def test_eval_row_records_anthropic_fields_and_never_the_key(monkeypatch):
    from eval.run import run_one

    _site()
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    t = Task.from_row({**TASK.to_row(), "site": SITE})
    llm = FakeAnthropic([resp([text("hm")], stop="end_turn", u=usage(100, 5, cw=7000)),
                         resp([tool_use(1, "submit", SUB)], u=usage(20, 30, cr=7000))])
    o = {"client": llm, "provider": "anthropic", "options": LLMOptions(provider="anthropic")}
    r = run_one(t, "model", "api:claude-opus-5-5", 0.0, 0, "claude-opus-5-5", opts=o)
    assert r["provider"] == "anthropic" and r["tool_choice_used"] == "auto" and r["score"] == 1.0
    assert r["usage"]["cache_creation_tokens"] == 7000 and r["usage"]["cached_tokens"] == 7000
    assert r["usage"]["prompt_tokens"] == 14120 and r["usage"]["completion_tokens"] == 35
    assert r["api"] == {"nudges": 1, "stop_reasons": {"end_turn": 1, "tool_use": 1}, "fallback_ran": False,
                        "refusal": None, "reasoning_effort": "medium", "served_by": ["claude-opus-5-5"]}
    assert FAKE_KEY not in json.dumps(r)
    bad = FakeAnthropic([], raise_with_key=True)
    r = run_one(t, "model", "api:claude-opus-5-5", 0.0, 0, "claude-opus-5-5",
                opts={**o, "client": bad})
    assert r["error"] and FAKE_KEY not in json.dumps(r) and "[REDACTED]" in r["error"]


def test_transcript_is_json_serialisable():
    _, _, ev = _run([resp([thinking("s"), tool_use(1, "site_describe", {})]), resp([tool_use(2, "submit", SUB)])])
    tr = [e for e in ev if e["type"] == "transcript"][0]
    s = json.dumps(tr)
    assert '"signature": "s"' in s and '"tool_result"' in s


def test_large_max_tokens_streams():
    from serve.anthropic_backend import STREAM_ABOVE

    final = resp([tool_use(1, "submit", SUB)])

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return final

    llm = FakeAnthropic([])
    calls = []
    llm.beta.messages.stream = lambda **kw: calls.append(kw) or Stream()
    from harness.env import AkshaEnv
    from serve.agent import run_task

    env = AkshaEnv(_site())
    list(run_task(env, TASK, client=llm, model="claude-opus-5-5",
                  options=LLMOptions(provider="anthropic", max_completion_tokens=STREAM_ABOVE + 1)))
    assert calls and calls[0]["max_tokens"] == STREAM_ABOVE + 1 and env.answer_score == 1.0


def test_make_client_anthropic(monkeypatch, tmp_path):
    import anthropic

    from serve.agent import make_client

    for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY") as ei:
        make_client("anthropic")
    assert "ant auth login" in str(ei.value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    c = make_client("anthropic")
    assert isinstance(c, anthropic.Anthropic)


@pytest.mark.parametrize("model,effort,fallback", [
    ("claude-opus-5-5", "medium", True), ("claude-sonnet-5-5", "high", True),
    ("claude-haiku-5-5", "medium", False), ("claude-opus-5", "high", True), ("claude-opus-4-8", None, False)])
def test_request_shape_per_model(model, effort, fallback):
    from harness.env import AkshaEnv
    from serve.agent import run_task

    env = AkshaEnv(_site())
    llm = FakeAnthropic([resp([thinking("s"), tool_use(1, "submit", SUB)], model=model)])
    ev = list(run_task(env, TASK, client=llm, model=model, options=LLMOptions(provider="anthropic")))
    kw = llm.kw[0]
    assert kw["model"] == model and kw["tool_choice"] == {"type": "auto"}
    for banned in ("temperature", "top_p", "top_k", "thinking", "seed"):
        assert banned not in kw
    assert kw.get("output_config") == ({"effort": effort} if effort else None)
    assert (llm.route == ["beta"] and kw["fallbacks"] == "default"
            and kw["betas"] == ["server-side-fallback-2026-07-01"]) if fallback else \
        (llm.route == ["messages"] and "fallbacks" not in kw and "betas" not in kw)
    assert kw["tools"][-1]["cache_control"] == {"type": "ephemeral"} and env.answer_score == 1.0
    assert [e for e in ev if e["type"] == "assistant"][0]["reasoning_effort"] == effort


def test_sonnet55_effort_flag_is_sent_explicitly():
    from harness.env import AkshaEnv
    from serve.agent import run_task

    env = AkshaEnv(_site())
    llm = FakeAnthropic([resp([tool_use(1, "submit", SUB)])])
    list(run_task(env, TASK, client=llm, model="claude-sonnet-5-5",
                  options=LLMOptions(provider="anthropic", reasoning_effort="medium")))
    assert llm.kw[0]["output_config"] == {"effort": "medium"}
