"""AkshaEnv: one environment API for training, data generation, evaluation and the UI (SPEC 3).

    env = AkshaEnv("data/stub_site")
    obs = env.reset(task)                                   # {"messages": [system, user], "tools": [...]}
    msgs, reward_parts, done, info = env.step({"tool_calls": [{"name": ..., "arguments": {...}}]})

Context is append-only (D-003): each step returns new messages only. The ledger delta rides on the last tool
message of the turn. Reward is computed from the task's precomputed answer key; truth is never read (D-019).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from pydantic import ValidationError

from harness.answers import drop_off_intent
from harness.config import EnvConfig
from harness.executor import Executor, validation_hint
from harness.guard import forbidden_roots, truth_guard
from harness.ledger import Ledger
from harness.prompt import static_prefix
from harness.trace import Tracer
from harness.verifier import verify
from sim.belief.site_view import SiteView
from tasks.base import Task
from tasks.scoring import score_answer
from tools.control import SubmitArgs
from tools.errors import ToolError
from tools.registry import REGISTRY
from train.rewards.episode import EpisodeStats, RewardConfig, episode_reward

_SITES: dict[str, SiteView] = {}
CONTROL = {"declare_intent", "ask_user", "submit"}


def site_view(site_dir: str | Path) -> SiteView:
    """One SiteView per site per process: many envs (e.g. 32 TRL rollouts) share it."""
    key = str(Path(site_dir).resolve())
    if key not in _SITES:
        _SITES[key] = SiteView(site_dir)
        _warm_trust(_SITES[key])
    return _SITES[key]


def _warm_trust(site: SiteView) -> None:
    """Load the belief trust layer once per process (about 1.4 s per band on Kandla). Cold, the first model_trust call
    overran its 250 ms tool timeout, so the agent saw TIMEOUT instead of the trust values (demo-v2 S5.4)."""
    from sim.belief.trust import trust_at

    maps = Path(site.rf_dir) / "maps"
    if not maps.exists():
        return
    for f in sorted(maps.glob("*.npz")):
        for sid, s in site.sensors.items():
            if s.get("type") == "fixed_df":
                p = s["pose"]
                trust_at(site.rf_dir, sid, f.stem, p["x_m"], p["y_m"], 40.0)


class AkshaEnv:
    def __init__(self, site_dir: str | Path, config: EnvConfig | None = None, trace_dir: str | Path | None = None,
                 reward_cfg: RewardConfig | None = None, model_id: str = "unknown"):
        self.site_dir = Path(site_dir)
        self.cfg = config or EnvConfig.load()
        self.reward_cfg = reward_cfg or RewardConfig()
        with truth_guard(self.site_dir):
            self.site = site_view(self.site_dir)
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.model_id = model_id
        self.system, self.tools, self._schemas_json = static_prefix(self.cfg)
        self.done = True

    # -------------------------------------------------------------------------------------------------------
    def reset(self, task: Task | dict, seed: int = 0) -> dict:
        self.task = task if isinstance(task, Task) else Task.from_row(task)
        self.seed = seed
        self.episode_id = uuid.uuid4().hex[:12]
        self.ledger = Ledger()
        self.tracer = Tracer(self.trace_dir, self.episode_id, self.task.task_id, seed, self.model_id,
                             self.template_hash, self.site.version)
        self.executor = Executor(self.site, self.cfg, self.ledger, forbidden_roots(self.site_dir), self.tracer.event)
        if self.task.drone_payload:
            from tools.drone import DroneSim

            self.executor.ctx.episode["drone"] = DroneSim(self.task.drone_payload)
        self.turn = 0
        self.gen_tokens = 0
        self.asked = 0
        self.format_failures = 0
        self.verifier_failures = 0
        self._health_hinted = False
        self.off_intent_dropped = 0      # lenient_submit diagnostic: submit fields dropped this episode
        self.answer = None
        self.verifier_result = None
        self.verifier_rules: list[dict] | None = None
        self.end_reason = "none"
        self.done = False
        self.reward_parts: dict | None = None
        self._last_results: list[dict] = []
        self.t_start = time.time()
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": self.task.question}]
        self.tracer.event(turn=0, role="user", content=self.task.question)
        return {"messages": msgs, "tools": self.tools}

    @property
    def template_hash(self) -> str:
        import hashlib

        return hashlib.sha256((self.system + self._schemas_json).encode()).hexdigest()[:16]

    # -------------------------------------------------------------------------------------------------------
    def step(self, action: dict) -> tuple[list[dict], dict | None, bool, dict]:
        """One assistant turn: `{"tool_calls": [...], "gen_tokens": n}` or `{"text": "..."}` (format error)."""
        assert not self.done, "episode is over; call reset()"
        with truth_guard(self.site_dir):
            self.turn += 1
            self.gen_tokens += int(action.get("gen_tokens", 0))
            calls = action.get("tool_calls") or []
            if not calls:
                msgs = self._format_error(action.get("text", ""))
            else:
                msgs = self.call_tools(calls)
            if not self.done and self.turn >= self.cfg.max_turns:
                self._finish("max_turns")
            info = {"turn": self.turn, "results": self._last_results, "verifier": self.verifier_result,
                    "ledger": self.ledger.render(self.turn)}
            return msgs, self.reward_parts if self.done else None, self.done, info

    def call_tools(self, calls: list[dict]) -> list[dict]:
        """Execute one turn's tool calls (also used directly by the TRL adapter). Returns tool messages. Guarded on
        its own (AUDIT_CODE minor 6): direct callers get the same truth guard as `step` (nesting is harmless)."""
        with truth_guard(self.site_dir):
            return self._call_tools(calls)

    def _call_tools(self, calls: list[dict]) -> list[dict]:
        self._last_results = []
        msgs = []
        regular = [c for c in calls if c.get("name") not in CONTROL]
        envs = self.executor.run(regular, self.turn) if regular else []
        it = iter(envs)
        for c in calls:
            env = next(it) if c.get("name") not in CONTROL else self._control(c, solo=len(calls) == 1)
            self._last_results.append(env)
            msgs.append({"role": "tool", "name": c.get("name", ""), "content": json.dumps(env, separators=(",", ":"))})
            if self.done:
                break
        delta = self.ledger.take_delta()
        if delta and msgs:
            msgs[-1]["content"] += "\n" + delta
        return msgs

    # -------------------------------------------------------------------------------------------------------
    def _control(self, call: dict, solo: bool) -> dict:
        name, raw = call["name"], call.get("arguments") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as e:
                return _err(name, ToolError("INVALID_ARGS", f"arguments are not valid JSON: {e.msg}",
                                            "Send arguments as a JSON object."))
        self.executor.calls += 1
        self.executor.call_names.append(name)
        if name == "declare_intent":
            intent = raw.get("intent")
            self.executor.valid_calls += 1
            self.tracer.event(turn=self.turn, tool=name, args=raw)
            return {"ok": True, "tool": name, "data": {"intent": intent,
                                                       "guidance": "abstain if out_of_scope" if intent == "out_of_scope"
                                                       else "resolve entities, then gather evidence"}}
        if name == "ask_user":
            if self.asked >= self.cfg.max_ask_user:
                return _err(name, ToolError("BUDGET_EXCEEDED", "ask_user allowed once per episode",
                                            "Choose one interpretation and state it in assumptions."))
            self.asked += 1
            self.executor.valid_calls += 1
            reply = self.task.clarification or "No further information is available; use your best judgement."
            self.tracer.event(turn=self.turn, tool=name, args=raw, result_summary=reply)
            return {"ok": True, "tool": name, "data": {"reply": reply}}
        # submit
        if not solo:
            return _err(name, ToolError("INVALID_ARGS", "submit must be the only call in its turn",
                                        "Finish evidence gathering, then call submit alone."))
        if self.cfg.submit_drop_off_intent and isinstance(raw, dict) and isinstance(raw.get("answer"), dict):
            kept, dropped = drop_off_intent(raw["answer"])
            if dropped:
                raw = {**raw, "answer": kept}
                self.off_intent_dropped += len(dropped)
        try:
            answer = SubmitArgs.model_validate(raw).answer
        except ValidationError as e:
            self.format_failures += 1
            if self.format_failures > self.cfg.max_format_retries:
                self._finish("format_failed")
            return _err(name, ToolError("INVALID_ARGS", "answer does not match the schema",
                                        "Fix the listed fields and submit again.", fields=validation_hint(e)))
        if (self.cfg.require_health_check and answer.intent in ("operator_query", "df_ambiguity")
                and "sensor_consistency" not in self.executor.call_names and not getattr(self, "_health_hinted", False)):
            self._health_hinted = True        # T-FIX T2: one redirecting hint per episode, then any submit is judged
            return _err(name, ToolError("HEALTH_CHECK_REQUIRED", "no sensor-health check in this episode",
                                        "Call sensor_consistency (no args, or at_time_s) first; if a node is suspect, "
                                        "prefer evidence that does not rest on it, then submit again."))
        self.executor.valid_calls += 1
        res = verify(answer, self.ledger, self._recompute, self.task.singular_reference,
                     episode=self.executor.ctx.episode, rf_rows=self._rf_rows)
        self.verifier_result = res.to_dict()
        self.verifier_rules = res.rules
        self.tracer.event(turn=self.turn, tool=name, args=raw, verifier=self.verifier_result, verifier_rules=res.rules)
        if not res.ok and not self.cfg.verifier_off:      # verifier_off: logged above, never rejects (ablation)
            self.verifier_failures += 1
            if self.verifier_failures > self.cfg.max_verifier_retries:
                self._finish("verifier_failed", answer)
            return {"ok": False, "tool": name, "error": {"code": res.code, "message": res.message, **res.details}}
        self._finish("submitted", answer)
        return {"ok": True, "tool": name, "data": {"accepted": True}}

    def _recompute(self, tool: str, args: dict) -> dict:
        """Deterministic re-run for the verifier; bypasses the ledger, cache and budgets on purpose."""
        spec = REGISTRY[tool]
        try:
            out = spec.fn(self.executor.ctx, spec.args.model_validate(args))
            return {"ok": True, "tool": tool, "data": out.data}
        except (ToolError, ValidationError) as e:
            return {"ok": False, "tool": tool, "error": str(e)}

    def _rf_rows(self, track: str, t: float) -> list[dict]:
        """Per-node trust / ghost risk at the track estimate (belief ensemble maps) for the RF-ambiguity rule. The
        harness computes it itself, so the rule does not depend on the model calling model_trust."""
        from tools.df import _band

        tid = Ledger.base(track).split(":", 1)[-1]
        if tid not in self.site.points:
            return []
        band = str(_band(self.executor.ctx, tid, t))
        env = self._recompute("model_trust", {"track_ref": f"trk:{tid}@{t:g}", "band": band, "per_sensor": True})
        return (env.get("data") or {}).get("per_sensor") or [] if env.get("ok") else []

    def finish_external(self, reason: str) -> None:
        """End the episode from outside the step loop (serve: wall-clock budget, LLM error) so the trace always
        gets an end event and the UI an answer event. No-op when the episode is already over."""
        if self.done:
            return
        with truth_guard(self.site_dir):
            self._finish(reason)

    def _format_error(self, text: str) -> list[dict]:
        self.format_failures += 1
        self._last_results = []
        if self.format_failures > self.cfg.max_format_retries:
            self._finish("format_failed")
            return []
        return [{"role": "user", "content": "Format error: reply with tool calls only. Gather evidence with tools, "
                                            "then call submit."}]

    def _finish(self, reason: str, answer=None) -> None:
        self.done = True
        self.answer = answer
        self.end_reason = reason
        ras = self.executor.ctx.episode.get("ras", {})
        resolved = {k: v["geom"] for k, v in ras.items() if "geom" in v}
        score = score_answer(self.task, answer, resolved) if answer is not None else 0.0
        ex = self.executor
        stats = EpisodeStats(
            turns=self.turn, max_turns=self.cfg.max_turns, generated_tokens=self.gen_tokens,
            max_tokens_per_turn=self.cfg.max_tokens_per_turn,
            valid_call_fraction=(ex.valid_calls / ex.calls) if ex.calls else 0.0, repeat_calls=ex.repeat_calls,
            answered=answer is not None and reason == "submitted", grounded=bool(self.verifier_result
                                                                                  and self.verifier_result["ok"]),
            verifier_failed_twice=reason == "verifier_failed", format_failed_twice=reason == "format_failed",
            unanswerable=self.task.area == "unanswerable",
            abstained=answer is not None and answer.intent == "abstain",
            confidence=answer.confidence if answer is not None else None, answer_score=score)
        self.reward_parts = episode_reward(stats, self.reward_cfg)
        self.answer_score = score
        self.tracer.event(turn=self.turn, role="end", reason=reason, reward_parts=self.reward_parts,
                          answer=answer.model_dump(mode="json") if answer is not None else None,
                          wallclock_s=round(time.time() - self.t_start, 3))
        ex.close()


def _err(tool: str, e: ToolError) -> dict:
    return {"ok": False, "tool": tool, "error": e.to_dict()}
