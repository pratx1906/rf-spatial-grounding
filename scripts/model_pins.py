"""Pinned Hugging Face model revisions (configs/models.yaml; paper Revision 2, AUDIT_CODE minor 9).

    from scripts.model_pins import revision
    AutoModelForCausalLM.from_pretrained(BASE, revision=revision(BASE), ...)
    python scripts/model_pins.py            # check the local cache snapshots against the pins
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PINS = REPO / "configs" / "models.yaml"
HUB = REPO / ".cache" / "huggingface" / "hub"


def pins() -> dict[str, str]:
    return dict(yaml.safe_load(PINS.read_text())["models"])


def revision(model_id: str) -> str | None:
    """The pinned commit of a HF model id (None if not pinned: then the caller loads `main` and should say so)."""
    return pins().get(model_id)


def local_snapshots(model_id: str, hub: Path = HUB) -> list[str]:
    d = hub / ("models--" + model_id.replace("/", "--")) / "snapshots"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def check(hub: Path = HUB) -> dict[str, dict]:
    """Per pinned model: the pin, the local snapshots and whether the pinned one is present."""
    return {m: {"pin": r, "local": local_snapshots(m, hub), "present": r in local_snapshots(m, hub)}
            for m, r in pins().items()}


if __name__ == "__main__":
    res = check()
    print(json.dumps(res, indent=1))
    sys.exit(0 if all(v["present"] or not v["local"] for v in res.values()) else 1)
