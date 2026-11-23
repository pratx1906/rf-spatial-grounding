"""LLM parser + scripted expert baseline (plan 2026-10-28 T4, E3; audit M5): "why an LLM agent and not a script?".

An LLM reads the question text only (never `task.params`, the answer key or any tool output) and returns the task
family and its structured slots as JSON; `tasks.expert_kandla.solve` then runs on those parsed slots with the real
tools. The few-shot prompt has one example per family of the train split, held-out and compositional families
excluded, so held-out families can only be solved if the parser maps them onto a train family's slots.
A reply that is not JSON, names no family, or makes the expert fail ends in an abstain (`parsed.ok` False).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from eval.trivial import abstain
from harness.env import AkshaEnv
from tasks.base import Task

REPO = Path(__file__).resolve().parents[1]
TRAIN_TASKS = REPO / "data" / "tasks" / "x3" / "train.jsonl"
SYSTEM = ("You convert a counter-UAS operator's question into a task family and its structured slots. Reply with "
          "one JSON object only: {\"family\": <one of the listed families>, \"params\": {<slots>}}. Use exactly the "
          "slot names and value formats of the examples (refs like trk:0123, sensor ids like S3, times in seconds "
          "as numbers, band ids like ism_2400). If nothing fits, reply {\"family\": null, \"params\": {}}.")
_EXAMPLES: dict[str, tuple[list, dict]] = {}


def _excluded() -> set[str]:
    try:
        from scripts.make_tasks import COMPOSITIONAL, HELD_OUT
    except Exception:  # noqa: BLE001
        return set()
    return set(HELD_OUT) | set(COMPOSITIONAL)


def few_shot_examples(path: str | Path = TRAIN_TASKS, exclude: set[str] | None = None) -> list[tuple[str, str, dict]]:
    """(question, family, params): the first train-split task of each family in file order, sorted by family."""
    exclude = _excluded() if exclude is None else exclude
    first: dict[str, tuple[str, str, dict]] = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        fam = r["template"]
        if r.get("split", "train") != "train" or fam in exclude or fam in first:
            continue
        first[fam] = (r["question"], fam, r.get("params") or {})
    return [first[f] for f in sorted(first)]


def family_areas(path: str | Path = TRAIN_TASKS) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["template"], r["area"])
    return out


def load_examples(path: str | Path = TRAIN_TASKS) -> tuple[list, dict]:
    key = str(path)
    if key not in _EXAMPLES:
        _EXAMPLES[key] = (few_shot_examples(path), family_areas(path))
    return _EXAMPLES[key]


def messages_for(question: str, examples: list[tuple[str, str, dict]]) -> list[dict]:
    fams = sorted({f for _, f, _ in examples})
    shots = "\n".join(f"Q: {q}\nA: {json.dumps({'family': f, 'params': p})}" for q, f, p in examples)
    user = (f"Families: {', '.join(fams)}\n\nExamples:\n{shots}\n\nQ: {question}\nA:" if examples
            else f"Q: {question}\nA:")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or not isinstance(d.get("family"), str) or not isinstance(d.get("params"), dict):
        return None
    return d


def parse_task(client, model: str, task: Task, examples: list, areas: dict | None = None,
               request_kw: dict | None = None) -> tuple[Task | None, dict]:
    """Ask the parser for (family, params) from task.question only. Returns a shadow Task for the expert (same
    task_id and question; parsed template, params and area; no answer key) and an info dict for the eval row."""
    kw = request_kw or {"temperature": 0.0, "max_tokens": 512}
    resp = client.chat.completions.create(model=model, messages=messages_for(task.question, examples), **kw)
    text = resp.choices[0].message.content or ""
    u = getattr(resp, "usage", None)
    info = {"ok": False, "family": None, "params": None, "raw": text[:500],
            "usage": {"prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                      "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0)}}
    d = _parse_json(text)
    if d is None:
        return None, info
    info.update(ok=True, family=d["family"], params=d["params"])
    area = (areas or {}).get(d["family"], "free")
    return Task(task.task_id, area, d["family"], task.question, {}, split=task.split,
                singular_reference=task.singular_reference, scenario_seed=task.scenario_seed, site=task.site,
                params=d["params"]), info


def solve(env: AkshaEnv, task: Task, client, model: str, examples: list | None = None, areas: dict | None = None,
          request_kw: dict | None = None) -> dict:
    """env must already be reset on the real task (scoring uses its answer key). Returns the parse info."""
    from tasks.expert_kandla import solve as expert_solve

    if examples is None:
        examples, areas = load_examples()
    shadow, info = parse_task(client, model, task, examples, areas, request_kw)
    if shadow is not None:
        try:
            expert_solve(env, shadow)
        except Exception as e:  # noqa: BLE001 - parsed slots the expert cannot use: the pipeline fails
            info.update(ok=False, solve_error=f"{type(e).__name__}: {e}"[:300])
    if not env.done:
        abstain(env)
    return info
