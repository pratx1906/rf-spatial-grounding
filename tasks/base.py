"""Task record. The answer key is computed offline from truth and travels with the task (DECISIONS D-019);
it is never rendered into any message the agent sees."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Task:
    task_id: str
    area: str                      # operator_query | df_ambiguity | ... | unanswerable
    template: str
    question: str
    answer_key: dict
    split: str = "train"
    singular_reference: bool = False
    clarification: str | None = None   # simulated operator's reply to ask_user
    categories: list[str] = field(default_factory=list)   # e.g. ["entity_ambiguous", "decoy"]
    scenario_seed: int = 0
    site: str = ""                     # site/scenario directory the task belongs to (empty: env default)
    drone_payload: dict | None = None  # drone-search tasks: offline-prepared waypoint measurements (D-024)
    params: dict = field(default_factory=dict)   # structured question slots (for the scripted expert; never rendered)

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> "Task":
        return cls(**{k: row[k] for k in cls.__dataclass_fields__ if k in row})
