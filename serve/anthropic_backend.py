"""Native Claude backend (Anthropic Messages API, official `anthropic` SDK) for the shared agent loop.

Frontier baseline (paper plan 2026-10-28 T2). `serve.agent.run_task` drives the same AkshaEnv / ledger / verifier loop
for every provider; this module only implements the per-turn contract for `--provider anthropic`:
`complete(timeout) -> turn dict` (text, tool calls, usage, extras) and `add_results(...)` (tool results back).

Claude request rules, Opus 5.5 and Sonnet 5.5 (claude-api skill: python README, tool-use, prompt-caching, model-migration):
  - thinking: the `thinking` field is never sent (adaptive is the default; a budget or "disabled" is a 400 on Opus
    5.5 and Sonnet 5.5); depth is `output_config.effort`, set explicitly from --reasoning-effort or the model's API
    default (MODEL_RULES: Opus 5.5 "medium", Sonnet 5.5 "high"); no temperature / top_p / top_k ever.
  - forced tool_choice (any / tool) is a 400: tool_choice is {"type": "auto"}, the system prompt gets SYSTEM_SUFFIX
    (every reply is tool calls; the answer only via `submit`), and a reply without a tool_use block is re-prompted
    once per turn with NUDGE (counted in `nudges`); a second text-only reply goes to the env as a format error.
  - parallel tool use: every tool_result of a turn goes back in ONE user message, `is_error` on failed calls; the
    env's own user messages (format errors) follow as text blocks in that message.
  - preserved thinking: the full `response.content` (thinking blocks included) is appended unchanged; the history is
    append-only and system / tools are built once per episode (byte-identical prefix: cache hits, thinking binding).
  - stop reasons are recorded per turn; `refusal` ends the episode (end_reason "refusal", stop_details.category
    recorded); `pause_turn` is resumed by re-sending; `max_tokens` with no tool call is treated like text-only.
  - server-side refusal fallback (`fallbacks: "default"`, beta server-side-fallback-2026-07-01) is on by default
    (LLMOptions.anthropic_fallback) for models that support it (MODEL_RULES; not Haiku 5.5); whether a fallback model served the turn is read from `usage.iterations`
    (`fallback_message`) and recorded with the serving model.
  - prompt caching: cache_control on the last tool definition and on the system block (tools -> system is the
    stable prefix) plus top-level automatic caching for the growing conversation tail (3 of 4 breakpoints).
Usage per turn: input (uncached), output, cache_creation, cache_read tokens, mapped onto the row keys the cost
script reads: prompt_tokens = input + cache_creation + cache_read, cached_tokens = cache_read,
cache_creation_tokens, input_tokens, completion_tokens = output (thinking included; not reported separately).
"""

from __future__ import annotations

import json

DEFAULT_MAX_TOKENS = 16000          # thinking + reply per call; the env's 512-token turn budget is far too small
STREAM_ABOVE = 16000                # larger max_tokens: stream and take .get_final_message() (SDK long-request rule)
FALLBACK_BETA = "server-side-fallback-2026-07-01"   # pairs with the `fallbacks: "default"` scalar form only
MAX_PAUSE_RESUMES = 3
EPHEMERAL = {"type": "ephemeral"}

# Request rules by model-id prefix (first match wins; more specific ids first). effort: the API default, sent
# explicitly (None = send output_config only when --reasoning-effort is given). fallback: the model supports the
# server-side `fallbacks: "default"` form on the Claude API. Every claude-* id gets the same loop: no thinking field
# (adaptive where thinking exists), no sampling parameters, tool_choice auto + SYSTEM_SUFFIX + one nudge.
MODEL_RULES = (
    ("claude-opus-5-5", {"effort": "medium", "fallback": True}),      # thinking always on; forced tool_choice 400
    ("claude-sonnet-5-5", {"effort": "high", "fallback": True}),      # "disabled" thinking 400; recalibrated effort
    ("claude-haiku-5-5", {"effort": "medium", "fallback": False}),    # no server-side fallback for this model
    ("claude-opus-5", {"effort": "high", "fallback": True}),
    ("claude-", {"effort": None, "fallback": False}),                 # older / unknown: API defaults, no fallback
)


def model_rules(model: str) -> dict:
    name = model.lower().rsplit("/", 1)[-1]
    return next((dict(r) for p, r in MODEL_RULES if name.startswith(p)), {"effort": None, "fallback": False})

SYSTEM_SUFFIX = ("\n\nTool protocol: every reply must consist of tool calls only; do not answer in plain text. "
                 "Gather evidence with the tools, then give the final answer only by calling the `submit` tool.")
NUDGE = "Your last reply had no tool call. Reply with tool calls only: gather evidence with tools, then call submit."


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """OpenAI-format function schemas (env obs["tools"]) -> Anthropic tool definitions, order kept; the last one
    carries the cache breakpoint for the tools prefix."""
    out = []
    for t in tools:
        f = t.get("function", t)
        out.append({"name": f["name"], "description": f.get("description", ""),
                    "input_schema": f.get("parameters") or {"type": "object", "properties": {}}})
    if out:
        out[-1]["cache_control"] = dict(EPHEMERAL)
    return out


def _n(obj, k: str) -> int:
    return int(getattr(obj, k, 0) or 0)


def map_usage(resp) -> dict:
    """Token usage of one response. With server-side fallback, `usage.iterations` (one entry per attempt) is the
    billing truth; top-level usage covers only the attempt that produced the message."""
    u = getattr(resp, "usage", None)
    its = list(getattr(u, "iterations", None) or []) if u is not None else []
    parts = its or ([u] if u is not None else [])
    inp = sum(_n(p, "input_tokens") for p in parts)
    cw = sum(_n(p, "cache_creation_input_tokens") for p in parts)
    cr = sum(_n(p, "cache_read_input_tokens") for p in parts)
    out = sum(_n(p, "output_tokens") for p in parts)
    return {"prompt_tokens": inp + cw + cr, "completion_tokens": out, "reasoning_tokens": 0, "cached_tokens": cr,
            "cache_creation_tokens": cw, "input_tokens": inp}


def fallback_ran(resp) -> bool:
    u = getattr(resp, "usage", None)
    return any(getattr(e, "type", None) == "fallback_message" for e in (getattr(u, "iterations", None) or []))


def _jsonable(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json", exclude_none=True)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


class AnthropicBackend:
    """One episode's conversation with Claude. Built once per episode from the env's first observation."""

    def __init__(self, client, opts, model: str, obs: dict):
        self.client, self.model = client, model
        rules = model_rules(model)
        self.effort = opts.reasoning_effort or rules["effort"]
        self.max_tokens = opts.max_completion_tokens or DEFAULT_MAX_TOKENS
        self.fallback = bool(getattr(opts, "anthropic_fallback", True)) and rules["fallback"]
        sys_text = "\n\n".join(m["content"] for m in obs["messages"] if m["role"] == "system") + SYSTEM_SUFFIX
        self.system = [{"type": "text", "text": sys_text, "cache_control": dict(EPHEMERAL)}]
        self.tools = to_anthropic_tools(obs["tools"])
        self.messages: list[dict] = [{"role": m["role"], "content": m["content"]}
                                     for m in obs["messages"] if m["role"] != "system"]
        self.tool_choice = "auto"

    # ---------------------------------------------------------------------------------------------------------
    def _request(self, timeout: float) -> dict:
        kw = {"model": self.model, "max_tokens": self.max_tokens, "system": self.system, "tools": self.tools,
              "tool_choice": {"type": "auto"}, "messages": list(self.messages),
              "cache_control": dict(EPHEMERAL), "timeout": timeout}
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        if self.fallback:
            kw["betas"] = [FALLBACK_BETA]
            kw["fallbacks"] = "default"
        return kw

    def _call(self, timeout: float):
        kw = self._request(timeout)
        api = self.client.beta.messages if self.fallback else self.client.messages
        if self.max_tokens > STREAM_ABOVE:
            with api.stream(**kw) as s:
                return s.get_final_message()
        return api.create(**kw)

    def complete(self, timeout: float) -> dict:
        """One env turn: the request plus at most one nudge (text-only reply) and up to MAX_PAUSE_RESUMES pause_turn
        resumes. Usage is summed over every request of the turn."""
        use: dict = {}
        stops: list[str] = []
        nudges = pauses = 0
        fb, refusal, served, end = False, None, None, None
        texts: list[str] = []
        calls: list[dict] = []
        while True:
            resp = self._call(timeout)
            for k, v in map_usage(resp).items():
                use[k] = use.get(k, 0) + v
            stop = getattr(resp, "stop_reason", None)
            stops.append(stop)
            fb = fb or fallback_ran(resp)
            served = getattr(resp, "model", None) or served
            if stop == "refusal":                 # branch on stop_reason; stop_details is informational
                det = getattr(resp, "stop_details", None)
                refusal = getattr(det, "category", None) if det is not None else None
                end = ("refusal", f"LLM refusal (category {refusal}): the model declined this request.")
                break
            blocks = list(resp.content or [])
            if blocks:
                self.messages.append({"role": "assistant", "content": resp.content})     # verbatim, append-only
            texts += [b.text for b in blocks if getattr(b, "type", None) == "text" and getattr(b, "text", "")]
            calls = [{"id": b.id, "name": b.name, "arguments": json.dumps(b.input)}
                     for b in blocks if getattr(b, "type", None) == "tool_use"]
            if calls:
                break
            if stop == "pause_turn" and pauses < MAX_PAUSE_RESUMES:
                pauses += 1
                continue
            if nudges >= 1:
                break
            nudges += 1
            self.messages.append({"role": "user", "content": [{"type": "text", "text": NUDGE}]})
        return {"text": "\n".join(texts), "calls": calls, "usage": use, "tool_choice": self.tool_choice, "end": end,
                "extra": {"stop_reasons": stops, "nudges": nudges, "fallback": fb, "refusal": refusal,
                          "reasoning_effort": self.effort, "served_by": served}}

    def add_results(self, calls: list[dict], tool_msgs: list[dict], results: list[dict],
                    user_msgs: list[dict]) -> None:
        """All tool results of the turn (one per tool_use id, in call order) in one user message; calls the env did
        not execute (episode ended mid-turn) get an error result so the history stays valid."""
        blocks: list[dict] = []
        for i, c in enumerate(calls):
            if i < len(tool_msgs):
                b = {"type": "tool_result", "tool_use_id": c["id"], "content": tool_msgs[i]["content"]}
                if i < len(results) and isinstance(results[i], dict) and results[i].get("ok") is False:
                    b["is_error"] = True
            else:
                b = {"type": "tool_result", "tool_use_id": c["id"], "is_error": True,
                     "content": "Not executed: the episode ended earlier in this turn."}
            blocks.append(b)
        blocks += [{"type": "text", "text": m["content"]} for m in user_msgs]
        if blocks:
            self.messages.append({"role": "user", "content": blocks})

    def transcript(self) -> list[dict]:
        """JSON-serialisable copy of exactly what the model saw (system first, as in the OpenAI transcript)."""
        return [{"role": "system", "content": self.system[0]["text"]}] + _jsonable(self.messages)
