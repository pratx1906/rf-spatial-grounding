"""Agent loop against the OpenAI-compatible LLM server, streaming events for the UI (SPEC 7.1, 7.7).

Events (dicts with "type"): start, assistant, tool_call, tool_result, ledger, answer, error, done.
The same AkshaEnv runs training, evaluation and the UI; here rewards are simply not shown.

Run-loop limits (demo-v2 S5.3): every LLM call has a client timeout (never the OpenAI default of 600 s); with
`max_wallclock_s` (the gateway passes the env config's budget; eval and training pass nothing, so their numbers are
unchanged) the episode ends with reason `wallclock` once the budget is spent. An LLM error ends it with `llm_error`.
Either way the trace gets an end event and the UI an `answer` event and `done`.
`error` events carry a `code`: llm_unreachable | llm_timeout | wallclock | format_retry (gateway: internal).
The `answer` event says how to present the answer: `status` verified | flagged | none (flagged = the verifier
rejected it: never show it as the headline), `flagged`, `flags` (verifier codes), `answer_hash` (binds the operator
gate decision to this exact answer), `gate_required` (verified and requires_human_confirm), `confidence`,
`verifier_rules` (per-rule results). With an `audit` log every question writes one `question` record (S5.5).

Providers (paper plan 2026-10-28 T2): `make_client(provider)` gives the local vLLM server (default, unchanged), an
OpenAI-compatible API endpoint (openrouter, openai, gemini) with its key from an env var only (never logged, never
written to rows; `redact` scrubs error text), or the native Anthropic SDK client (anthropic: default credential
resolution, ANTHROPIC_API_KEY or an `ant auth login` profile). `run_task` keeps one env / ledger / verifier loop for all
of them and talks to the model through a per-episode backend with one turn contract (`complete`, `add_results`,
`transcript`): `_ChatBackend` (OpenAI Chat Completions: local + OpenAI-compatible; the local request is unchanged) or
`serve.anthropic_backend.AnthropicBackend` (Claude Messages API; its request rules are documented there).
`LLMOptions` carries the per-request API settings: a completion token cap separate from the env budgets, reasoning
effort, seed, the Anthropic refusal-fallback switch; temperature is omitted for reasoning models. On the chat path an
API that rejects `tool_choice="required"` with a 400 is retried with "auto", and a 400 naming an optional parameter
(parallel_tool_calls, seed, reasoning_effort) drops it, both kept for the rest of the episode; assistant events report
`tool_choice` and `usage` (prompt, completion, reasoning, cached tokens; Anthropic adds cache-creation and uncached
input tokens plus `stop_reasons`, `nudges`, `fallback`, `refusal`, `reasoning_effort`, `served_by`).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, replace

from openai import OpenAI

from harness.audit import AuditLog, answer_hash
from harness.env import AkshaEnv
from tasks.base import Task

LLM_URL = "http://127.0.0.1:8001/v1"
MODEL = os.environ.get("SCENESCOUT_SERVED_MODEL", "scenescout")   # e.g. "sft" / "rl" (vLLM --lora name)
LLM_TIMEOUT_S = float(os.environ.get("SCENESCOUT_LLM_TIMEOUT_S", "120"))   # per call, when no wall-clock budget
_ADAPTERS: dict[str, dict] = {}

# provider -> (base_url, key env vars in order of preference, completion-cap field)
PROVIDERS = {
    "local": (LLM_URL, (), "max_tokens"),
    "openrouter": ("https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",), "max_completion_tokens"),
    "openai": ("https://api.openai.com/v1", ("OPENAI_API_KEY",), "max_completion_tokens"),
    "anthropic": ("https://api.anthropic.com", ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"), "max_tokens"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
               "max_tokens"),
}
# models that reject a temperature (prefix of the name, with or without a "vendor/" prefix); eval/run.py can override
REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def make_client(provider: str = "local", timeout_s: float | None = None):
    """Client for a provider: OpenAI-compatible, or the native Anthropic SDK client for "anthropic". Keys come from
    the environment only (Anthropic: the SDK's default resolution, which also reads an `ant auth login` profile); a
    missing key names the variable, never a value."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; one of {sorted(PROVIDERS)}")
    url, vars_, _ = PROVIDERS[provider]
    if provider == "local":
        return OpenAI(base_url=url, api_key="local")
    if provider == "anthropic":
        import anthropic

        c = anthropic.Anthropic(max_retries=3, **({"timeout": timeout_s} if timeout_s else {}))
        if c.api_key is None and getattr(c, "auth_token", None) is None and getattr(c, "credentials", None) is None:
            raise RuntimeError("provider anthropic: no credentials; set ANTHROPIC_API_KEY in the environment or run "
                               "`ant auth login`")
        return c
    key = next((os.environ[v] for v in vars_ if os.environ.get(v)), None)
    if not key:
        raise RuntimeError(f"provider {provider}: set {' or '.join(vars_)} in the environment")
    return OpenAI(base_url=url, api_key=key, max_retries=3, **({"timeout": timeout_s} if timeout_s else {}))


def redact(text: str | None) -> str | None:
    """Replace any provider key present in the environment by [REDACTED] (error messages may echo a key)."""
    if not text:
        return text
    for _, vars_, _ in PROVIDERS.values():
        for v in vars_:
            k = os.environ.get(v)
            if k and len(k) >= 8:
                text = text.replace(k, "[REDACTED]")
    return text


def is_reasoning_model(model: str, prefixes: tuple[str, ...] | None = None) -> bool:
    prefixes = REASONING_PREFIXES if prefixes is None else tuple(prefixes)
    name = model.lower()
    return any(n.startswith(p) for n in (name, name.rsplit("/", 1)[-1]) for p in prefixes)


@dataclass(frozen=True)
class LLMOptions:
    """Per-request settings. The defaults reproduce the local request exactly (no seed, max_tokens = env budget)."""
    provider: str = "local"
    max_completion_tokens: int | None = None     # API completion cap (incl. reasoning); None = env budget per turn
    reasoning_effort: str | None = None          # e.g. low | medium | high (API reasoning models)
    seed: int | None = None
    reasoning_prefixes: tuple[str, ...] | None = None   # None = REASONING_PREFIXES
    anthropic_fallback: bool = True              # anthropic: server-side refusal fallback where the model has one


def build_request(opts: LLMOptions, model: str, messages: list, tools: list, temperature: float,
                  env_max_tokens: int, tool_choice: str, timeout: float) -> dict:
    kw: dict = {"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice}
    if not (opts.provider != "local" and is_reasoning_model(model, opts.reasoning_prefixes)):
        kw["temperature"] = temperature
    kw[PROVIDERS[opts.provider][2]] = opts.max_completion_tokens or env_max_tokens
    kw["parallel_tool_calls"] = True
    kw["timeout"] = timeout
    if opts.seed is not None:
        kw["seed"] = opts.seed
    if opts.reasoning_effort:
        if opts.provider == "openrouter":
            kw["extra_body"] = {"reasoning": {"effort": opts.reasoning_effort}}
        else:
            kw["reasoning_effort"] = opts.reasoning_effort
    return kw


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
    det, pdet = getattr(u, "completion_tokens_details", None), getattr(u, "prompt_tokens_details", None)
    return {"prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(det, "reasoning_tokens", 0) or 0) if det is not None else 0,
            "cached_tokens": int(getattr(pdet, "cached_tokens", 0) or 0) if pdet is not None else 0}


# optional chat parameters an API may reject with a 400 naming them; dropped for the rest of the episode
DROPPABLE_PARAMS = ("parallel_tool_calls", "seed", "reasoning_effort")


class _ChatBackend:
    """OpenAI Chat Completions turn contract (local vLLM and the OpenAI-compatible APIs). For provider "local" the
    request and the message history are exactly what run_task sent before the backend split."""

    def __init__(self, client, opts: LLMOptions, model: str, obs: dict, temperature: float, env_max_tokens: int):
        self.client, self.opts, self.model = client, opts, model
        self.temperature, self.env_max_tokens = temperature, env_max_tokens
        self.messages = list(obs["messages"])
        self.tools = obs["tools"]
        self.tool_choice = "required"
        self.dropped: list[str] = []

    def _create(self, timeout: float):
        kw = build_request(self.opts, self.model, self.messages, self.tools, self.temperature, self.env_max_tokens,
                           self.tool_choice, timeout)
        for k in self.dropped:
            kw.pop(k, None)
        return self.client.chat.completions.create(**kw)

    def complete(self, timeout: float) -> dict:
        # Constrained decoding (SPEC 10.2): every turn must be a well-formed tool call; answers go via submit.
        while True:
            try:
                resp = self._create(timeout)
                break
            except Exception as e:  # noqa: BLE001
                if self.opts.provider == "local" or getattr(e, "status_code", None) != 400:
                    raise
                text = str(e)
                bad = next((p for p in DROPPABLE_PARAMS if p in text and p not in self.dropped), None)
                if bad is not None:                 # e.g. a reasoning model without parallel_tool_calls
                    self.dropped.append(bad)
                elif self.tool_choice == "required":
                    # some APIs reject tool_choice="required" (400): retry with "auto", kept for the episode
                    self.tool_choice = "auto"
                else:
                    raise
        msg = resp.choices[0].message
        calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                 for tc in (msg.tool_calls or [])]
        assistant = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
        self.messages.append(assistant)
        return {"text": msg.content or "", "calls": calls, "usage": _usage(resp), "tool_choice": self.tool_choice,
                "end": None, "extra": {"dropped_params": list(self.dropped)} if self.dropped else {}}

    def add_results(self, calls: list[dict], tool_msgs: list[dict], results: list[dict],
                    user_msgs: list[dict]) -> None:
        for c, m in zip(calls, tool_msgs, strict=False):
            self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": m["content"]})
        self.messages.extend(user_msgs)             # format-error nudge

    def transcript(self) -> list[dict]:
        return self.messages


def make_backend(client, opts: LLMOptions, model: str, obs: dict, temperature: float, env_max_tokens: int):
    if opts.provider == "anthropic":
        from serve.anthropic_backend import AnthropicBackend

        return AnthropicBackend(client, opts, model, obs)
    return _ChatBackend(client, opts, model, obs, temperature, env_max_tokens)


def free_task(question: str, clarification: str | None = None) -> Task:
    return Task(f"ui-{uuid.uuid4().hex[:8]}", "free", "free", question.strip(), {}, split="ui",
                clarification=clarification)


def run_model(env: AkshaEnv, question: str, client: OpenAI | None = None, temperature: float = 0.0,
              clarification: str | None = None, **kw) -> Iterator[dict]:
    """kw: model, max_wallclock_s, audit, request (see run_task)."""
    yield from run_task(env, free_task(question, clarification), client=client, temperature=temperature, **kw)


def run_task(env: AkshaEnv, task: Task, client=None, temperature: float = 0.0, model: str = MODEL,
             max_wallclock_s: float | None = None, audit: AuditLog | None = None,
             request: dict | None = None, seed: int | None = None,
             options: LLMOptions | None = None) -> Iterator[dict]:
    """The model loop on a full Task (answer key, clarification, drone payload kept): UI, eval and RL rollouts.
    max_wallclock_s: None or <= 0 = no wall-clock budget (eval, training); the gateway passes env.cfg's.
    seed: None = unchanged (env.reset seed 0, no seed in the request); else env.reset(task, seed) + request seed.
    options: provider request settings (LLMOptions); None = the local default request."""
    opts = options or LLMOptions()
    if seed is not None:
        opts = replace(opts, seed=seed)
    client = client or make_client(opts.provider)
    question = task.question
    t_start = time.perf_counter()
    budget = max_wallclock_s if max_wallclock_s and max_wallclock_s > 0 else None
    obs = env.reset(task, seed=seed if seed is not None else 0)
    backend = make_backend(client, opts, model, obs, temperature, env.cfg.max_tokens_per_turn)
    turns: list[dict] = []
    yield {"type": "start", "episode_id": env.episode_id, "question": question, "mode": "model",
           "data_version": env.site.version, "template_hash": env.template_hash}
    while not env.done:
        timeout = LLM_TIMEOUT_S
        if budget is not None:
            left = budget - (time.perf_counter() - t_start)
            if left <= 0:
                env.finish_external("wallclock")
                yield {"type": "error", "code": "wallclock",
                       "message": f"Stopped: the {budget:g} s time budget for one question was used up."}
                break
            timeout = min(timeout, left)
        t0 = time.perf_counter()
        try:
            out = backend.complete(timeout)
        except Exception as e:  # noqa: BLE001 - surfaced to the operator, never crashes the UI
            code = "llm_timeout" if "timeout" in type(e).__name__.lower() else "llm_unreachable"
            env.finish_external("llm_error")
            yield {"type": "error", "code": code, "message": redact(f"LLM server: {type(e).__name__}: {e}")}
            break
        gen_ms = round((time.perf_counter() - t0) * 1000)
        use, calls = out["usage"], out["calls"]
        usage, prompt_tokens = use["completion_tokens"], use["prompt_tokens"]
        turns.append({"turn": env.turn + 1, "prompt_tokens": prompt_tokens, "completion_tokens": usage,
                      "gen_ms": gen_ms, "tools": [c["name"] for c in calls]})
        yield {"type": "assistant", "turn": env.turn + 1, "text": out["text"], "gen_ms": gen_ms,
               "tokens": usage, "n_calls": len(calls), "usage": use, "tool_choice": out["tool_choice"],
               **out["extra"]}
        if out["end"] is not None:                  # e.g. a Claude refusal: the episode ends without an answer
            reason, message = out["end"]
            env.finish_external(reason)
            yield {"type": "error", "code": reason, "message": message}
            break
        for c in calls:
            yield {"type": "tool_call", "turn": env.turn + 1, "name": c["name"], "arguments": _parse(c["arguments"])}
        action = {"tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls],
                  "text": out["text"], "gen_tokens": usage}
        out_msgs, _, done, info = env.step(action)
        tool_msgs = [m for m in out_msgs if m["role"] == "tool"]
        user_msgs = [m for m in out_msgs if m["role"] == "user"]
        backend.add_results(calls, tool_msgs, info["results"], user_msgs)
        for c, m, res in zip(calls, tool_msgs, info["results"], strict=False):
            yield _tool_event(env, c["name"], res)
        for m in user_msgs:                         # format-error nudge
            yield {"type": "error", "code": "format_retry", "message": m["content"]}
        delta = tool_msgs[-1]["content"].split("\nLEDGER+ ", 1)[1] if tool_msgs and "\nLEDGER+ " in \
            tool_msgs[-1]["content"] else ""
        if delta:
            yield {"type": "ledger", "turn": env.turn, "delta": delta}
    yield {"type": "transcript", "messages": backend.transcript()}   # exactly what the model saw (RL / debugging)
    events = list(_finish(env))
    if audit is not None:
        _audit_question(audit, env, events[0], model, client, turns, request or {"question": question},
                        time.perf_counter() - t_start)
    yield from events


def warm_up(env: AkshaEnv, client=None, model: str = MODEL, question: str = "Status check.",
            timeout_s: float = 120.0) -> dict:
    """One throw-away LLM request with exactly the first-turn request of `run_task` (same system prompt, tool
    schemas, tool_choice, adapter), max_tokens=1. It fills vLLM's prefix cache with the static prefix (~7k tokens)
    and loads the adapter, so the operator's first question does not pay the cold prefill (demo-v2 S5.6). Serving
    only: no env step, no trace, no audit record. Returns {ok, ms, prompt_tokens} or {ok: False, error}."""
    client = client or OpenAI(base_url=LLM_URL, api_key="local")
    obs = env.reset(free_task(question))
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model, messages=list(obs["messages"]), tools=obs["tools"], tool_choice="required",
            temperature=0.0, max_tokens=1, parallel_tool_calls=True, timeout=timeout_s)
    except Exception as e:  # noqa: BLE001 - best effort; the first question then simply runs cold
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        env.finish_external("warmup")
    return {"ok": True, "ms": round((time.perf_counter() - t0) * 1000),
            "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0) if resp.usage else 0}


def run_expert(env: AkshaEnv, task: Task) -> Iterator[dict]:
    """Scripted fallback: the same UI flow driven by the scripted expert on a generated task."""
    if task.params:                       # Kandla families carry structured slots
        from tasks.expert_kandla import solve
    else:
        from tasks.expert import solve

    events: list[dict] = []
    obs = env.reset(task)
    orig = env.step

    def step(action):
        for c in action.get("tool_calls") or []:
            events.append({"type": "tool_call", "turn": env.turn + 1, "name": c["name"],
                           "arguments": _parse(c.get("arguments"))})
        out = orig(action)
        for c, res in zip(action.get("tool_calls") or [], out[3]["results"], strict=False):
            events.append(_tool_event(env, c["name"], res))
        return out

    env.step = step
    try:
        solve(env, task)
    finally:
        env.step = orig
    yield {"type": "start", "episode_id": env.episode_id, "question": task.question, "mode": "scripted",
           "data_version": env.site.version, "template_hash": env.template_hash}
    _ = obs
    yield from events
    yield from _finish(env)


def _tool_event(env: AkshaEnv, name: str, res: dict) -> dict:
    from serve.ui_geom import ui_extras

    ev = {"type": "tool_result", "turn": env.turn, "name": name, "result": res}
    try:
        ui = ui_extras(env, name, res)
    except Exception:  # noqa: BLE001 - geometry for display only; never breaks the episode
        ui = None
    if ui:
        ev["ui"] = ui
    return ev


def _finish(env: AkshaEnv) -> Iterator[dict]:
    ans = env.answer.model_dump(mode="json") if env.answer is not None else None
    vr = env.verifier_result or {}
    ok = ans is not None and bool(vr.get("ok")) and env.end_reason == "submitted"
    status = "none" if ans is None else "verified" if ok else "flagged"
    flags = [r["code"] for r in (env.verifier_rules or []) if not r["ok"]] if status == "flagged" else []
    if status == "flagged" and not flags:
        flags = [vr.get("code") or env.end_reason]
    yield {"type": "answer", "answer": ans, "verifier": env.verifier_result, "end_reason": env.end_reason,
           "turns": env.turn, "ledger": env.ledger.render(env.turn), "status": status,
           "flagged": status == "flagged", "flags": flags,
           "answer_hash": answer_hash(env.episode_id, env.site.version, ans) if ans is not None else None,
           "gate_required": bool(ok and ans.get("requires_human_confirm")),
           "confidence": ans.get("confidence") if ans is not None else None, "verifier_rules": env.verifier_rules}
    yield {"type": "done"}


def _adapter(client, model: str) -> dict:
    """Which weights answered: the served model's vLLM entry (LoRA root path) and the sha256 of the adapter file when
    it is readable here. Cached per process; best effort (never fails a question)."""
    if model in _ADAPTERS:
        return _ADAPTERS[model]
    info: dict = {"root": None, "parent": None, "sha256": None}
    try:
        m = next((m for m in client.models.list().data if m.id == model), None)
        if m is not None:
            info["root"], info["parent"] = getattr(m, "root", None), getattr(m, "parent", None)
    except Exception:  # noqa: BLE001
        pass
    if info["root"]:
        import hashlib
        from pathlib import Path

        try:
            f = Path(info["root"]) / "adapter_model.safetensors"
            if f.exists():
                h = hashlib.sha256()
                with f.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                info["sha256"] = h.hexdigest()
        except OSError:
            pass
    _ADAPTERS[model] = info
    return info


def _audit_question(audit: AuditLog, env: AkshaEnv, ans_ev: dict, model: str, client, turns: list[dict],
                    request: dict, wall_s: float) -> None:
    """S5.5: one audit record per question: request, model + adapter, tokens per turn, tool calls, per-rule verifier
    results, the answer and its hash. The operator decision is a later `gate` record (AuditLog.trace joins them)."""
    try:
        audit.append({
            "kind": "question", "episode_id": env.episode_id, "request": request,
            "model": {"served": model, "model_id": env.model_id, "adapter": _adapter(client, model),
                      "template_hash": env.template_hash},
            "data_version": env.site.version, "turns": turns,
            "tokens": {"prompt": sum(t["prompt_tokens"] for t in turns),
                       "completion": sum(t["completion_tokens"] for t in turns)},
            "tool_calls": env.executor.calls, "verifier": env.verifier_result, "verifier_rules": env.verifier_rules,
            "end_reason": env.end_reason, "status": ans_ev["status"], "flags": ans_ev["flags"],
            "answer_hash": ans_ev["answer_hash"], "answer": ans_ev["answer"], "confidence": ans_ev["confidence"],
            "gate_required": ans_ev["gate_required"], "wallclock_s": round(wall_s, 3)})
    except OSError as e:                        # an unwritable audit dir must not take the answer away, but say so
        import sys

        print(f"[serve.agent] AUDIT LOG WRITE FAILED for {env.episode_id}: {e}", file=sys.stderr, flush=True)


def _parse(args):
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"_raw": args}
    return args or {}
