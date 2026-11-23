"""Tool executor (SPEC 7.2): validate -> budget -> dedupe/cache -> run with timeout -> cap output -> ledger -> trace."""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout

from pydantic import ValidationError

from harness.config import EnvConfig
from harness.guard import guard_roots
from harness.ledger import Ledger
from tools.errors import ToolError
from tools.registry import REGISTRY, ToolContext, load_all

_HEALTH_MEMO: dict = {}

UNIT_HINTS = {"_km": "_m (metres)", "_ft": "_m (metres)", "_mi": "_m (metres)", "_ghz": "_mhz (MHz)",
              "_khz": "_mhz (MHz)", "_hz": "_mhz (MHz)", "_rad": "_deg (degrees)", "_min": "_s (seconds)",
              "_ms": "_s (seconds)", "_h": "_s (seconds)", "_dbw": "_dbm (dBm)"}


def canonical(args: dict) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def validation_hint(e: ValidationError) -> list[dict]:
    """Field-level hints (SPEC 5 rule 5: errors are instructions)."""
    out = []
    for err in e.errors()[:6]:
        loc = ".".join(str(x) for x in err["loc"])
        hint = err["msg"]
        if err["type"] == "missing":
            hint = f"required field {str(err['loc'][-1])!r} is missing"
        elif err["type"] == "extra_forbidden":
            field = str(err["loc"][-1])
            unit = next((v for k, v in UNIT_HINTS.items() if field.endswith(k)), None)
            hint = (f"unknown field {field!r}; use the {unit} form of this field" if unit
                    else f"unknown field {field!r}; see the tool schema")
        out.append({"field": loc, "hint": hint})
    return out


class Executor:
    def __init__(self, site, config: EnvConfig, ledger: Ledger, guard: tuple[str, ...] = (), trace=None):
        load_all()
        self.ctx = ToolContext(site, options={"fit_methods_disabled": tuple(config.fit_methods_disabled)})
        self.cfg = config
        # ablation switches: disabled tools, and opt-in tools this config does not expose, answer NOT_PERMITTED
        self.blocked = set(config.tools_disabled) | {n for n, s in REGISTRY.items()
                                                     if s.opt_in and n not in config.tools_extra}
        self.ledger = ledger
        self.guard = guard
        self.trace = trace
        self.cache: dict[tuple, dict] = {}
        self.obs_tokens = 0
        self.calls = 0
        self.call_names: list[str] = []  # every call in order (paper G1b: which tools were used)
        self.valid_calls = 0
        self.repeat_calls = 0
        self.state_version = 0
        self.timeouts: list[dict] = []     # every TIMEOUT this episode (AUDIT_CODE C8): {tool, turn, timeout_ms}
        self.pool = ThreadPoolExecutor(max_workers=config.parallel_tool_calls, thread_name_prefix="tool")

    @property
    def state_hash(self) -> str:
        return hashlib.sha256(f"{self.state_version}".encode()).hexdigest()[:4]

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    # -------------------------------------------------------------------------------------------------------
    def _error(self, tool: str, err: ToolError | dict) -> dict:
        return {"ok": False, "tool": tool, "error": err.to_dict() if isinstance(err, ToolError) else err}

    def _stateful(self, name: str) -> bool:
        spec = REGISTRY.get(name)
        return bool(spec and spec.stateful)

    def _validate(self, name: str, raw):
        spec = REGISTRY.get(name)
        if spec is None or spec.fn is None:
            known = sorted(n for n, s in REGISTRY.items() if s.fn is not None and n not in self.blocked)
            raise ToolError("INVALID_ARGS", f"unknown tool {name!r}", f"Available tools: {', '.join(known)}.")
        if name in self.blocked:
            raise ToolError("NOT_PERMITTED", f"tool {name!r} is not available in this configuration",
                            "Use the tools listed in your schema; submit or abstain with the evidence you have.")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as e:
                raise ToolError("INVALID_ARGS", f"arguments are not valid JSON: {e.msg}",
                                "Send arguments as a JSON object.") from None
        try:
            return spec, spec.args.model_validate(raw or {})
        except ValidationError as e:
            raise ToolError("INVALID_ARGS", f"{len(e.errors())} invalid argument(s)",
                            "Fix the listed fields and call again.", fields=validation_hint(e)) from None

    def _run(self, spec, args):
        memo = spec.name == "sensor_consistency" and self.cfg.require_health_check
        if memo:   # T-FIX T2: one health check per scenario and window, reused across episodes (belief side only)
            mk = (str(getattr(self.site, "dir", id(self.site))), canonical(args.model_dump(mode="json")))
            if mk in _HEALTH_MEMO:
                return _HEALTH_MEMO[mk]
        with guard_roots(self.guard):
            out = spec.fn(self.ctx, args)
        if memo:
            _HEALTH_MEMO[mk] = out
        return out

    def run(self, calls: list[dict], turn: int) -> list[dict]:
        """Execute up to `parallel_tool_calls` independent calls concurrently. Returns envelopes in order."""
        if len(calls) > 1 and any(self._stateful(c.get("name", "")) for c in calls):
            # stateful tools change the world: never run them concurrently; one call at a time, in order
            out = []
            for idx, c in enumerate(calls):
                if idx >= self.cfg.parallel_tool_calls:
                    out.append(self._error(c.get("name", ""), ToolError(
                        "BUDGET_EXCEEDED", f"max {self.cfg.parallel_tool_calls} parallel calls per turn",
                        "Make the remaining calls in the next turn.")))
                    continue
                out += self.run([c], turn)
            return out
        prepared, results = [], [None] * len(calls)
        for i, c in enumerate(calls):
            name = c.get("name", "")
            self.calls += 1
            self.call_names.append(name)
            if i >= self.cfg.parallel_tool_calls:
                results[i] = self._error(name, ToolError(
                    "BUDGET_EXCEEDED", f"max {self.cfg.parallel_tool_calls} parallel calls per turn",
                    "Make the remaining calls in the next turn."))
                continue
            if self.obs_tokens >= self.cfg.max_obs_tokens:
                results[i] = self._error(name, ToolError(
                    "BUDGET_EXCEEDED", f"observation budget of {self.cfg.max_obs_tokens} tokens is used up",
                    "Submit now with the evidence you have, or abstain (insufficient_evidence)."))
                continue
            try:
                spec, args = self._validate(name, c.get("arguments"))
            except ToolError as e:
                results[i] = self._error(name, e)
                continue
            key = (name, canonical(args.model_dump(mode="json")), self.state_hash)
            if key in self.cache:
                self.repeat_calls += 1
                env = dict(self.cache[key])
                env["warnings"] = list(env.get("warnings", [])) + ["repeat_call"]
                results[i] = env
                continue
            prepared.append((i, spec, args, key))

        futures = [(i, spec, args, key, time.perf_counter(), self.pool.submit(self._run, spec, args))
                   for i, spec, args, key in prepared]
        for i, spec, args, key, t0, fut in futures:
            timeout = self.cfg.timeout_ms(spec.name) / 1000
            try:
                out = fut.result(timeout=timeout)
                env = {"ok": True, "tool": spec.name, "data": out.data, "refs": out.refs}
                if out.definition:
                    env["definition"] = out.definition
                env["warnings"] = list(out.warnings)
                self.valid_calls += 1
            except FutTimeout:
                self.timeouts.append({"tool": spec.name, "turn": turn, "timeout_ms": round(timeout * 1000)})
                env = self._error(spec.name, ToolError("TIMEOUT", f"{spec.name} exceeded {timeout * 1000:.0f} ms",
                                                       "Narrow the query (smaller region or time range)."))
            except ToolError as e:
                env = self._error(spec.name, e)
                self.valid_calls += 1      # a well-formed call that the world answered with an error
            except Exception as e:  # noqa: BLE001  a tool bug must not kill the episode (or a whole eval run)
                env = self._error(spec.name, ToolError("INTERNAL_ERROR", f"{spec.name} failed: {type(e).__name__}",
                                                       "Try another tool or different arguments; report if it repeats."))
            env["cost_ms"] = round((time.perf_counter() - t0) * 1000)
            env["state_hash"] = self.state_hash
            env = self._cap(env)
            if env["ok"]:
                self.ledger.register(env["refs"], spec.name, turn)
            elif env["error"].get("candidates"):   # refs offered in an error were returned by a tool too
                self.ledger.register([c["ref"] for c in env["error"]["candidates"] if "ref" in c], spec.name, turn)
            self.ledger.record(spec.name, args.model_dump(mode="json"), env)
            if env["ok"] or env["error"]["code"] not in ("TIMEOUT",):
                self.cache[key] = env
            if spec.stateful and env["ok"]:
                self.state_version += 1           # the world changed: later identical calls are not repeats
            results[i] = env
            if self.trace:
                self.trace(turn=turn, tool=spec.name, args=args.model_dump(mode="json"), ok=env["ok"],
                           cost_ms=env["cost_ms"], result_summary=_summary(env))
        for env in results:
            self.obs_tokens += self.cfg.tokens(json.dumps(env, separators=(",", ":")))
        return results

    def _cap(self, env: dict) -> dict:
        """Per-tool output cap (SPEC 7.2 step 5): trim the longest list and say so; never drop silently."""
        cap = self.cfg.per_tool_output_tokens
        text = json.dumps(env, separators=(",", ":"))
        if self.cfg.tokens(text) <= cap or not env.get("ok"):
            return env
        data = env["data"]
        lists = [k for k, v in data.items() if isinstance(v, list) and v]
        while lists and self.cfg.tokens(json.dumps(env, separators=(",", ":"))) > cap:
            k = max(lists, key=lambda k: len(json.dumps(data[k])))
            if len(data[k]) <= 1:
                lists.remove(k)
                continue
            data[k] = data[k][: max(1, len(data[k]) // 2)]
            env["warnings"] = sorted(set(env["warnings"]) | {f"truncated:{k}"})
        env["refs"] = [r for r in env["refs"] if r in json.dumps(data)]
        return env


def _summary(env: dict) -> str:
    if not env["ok"]:
        return env["error"]["code"]
    d = env["data"]
    if "total" in d:
        return f"total={d['total']}"
    return re.sub(r"\s+", " ", json.dumps(d)[:80])
