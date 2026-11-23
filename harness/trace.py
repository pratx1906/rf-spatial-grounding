"""Episode traces as JSONL (SPEC 7.6). Kept in memory always; written to disk when a trace_dir is given."""

from __future__ import annotations

import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, trace_dir: Path | None, episode_id: str, task_id: str, seed: int, model_id: str,
                 template_hash: str, data_version: str):
        self.events: list[dict] = []
        self.base = {"episode_id": episode_id, "task_id": task_id, "seed": seed, "model_id": model_id,
                     "template_hash": template_hash, "data_version": data_version}
        self.path = Path(trace_dir) / f"{episode_id}.jsonl" if trace_dir else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, **kw) -> None:
        ev = {**self.base, "t": round(time.time(), 3), **kw}
        self.events.append(ev)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(ev, default=str) + "\n")
