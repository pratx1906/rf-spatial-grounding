"""Tool registry and model-facing schemas (SPEC 6.2: names final, one-line descriptions, compact schemas)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel


@dataclass
class ToolOutput:
    data: dict
    refs: list[str] = field(default_factory=list)
    definition: dict | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ToolContext:
    site: object                      # sim.belief.site_view.SiteView
    episode: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)   # config switches the tools honour, e.g. fit_methods_disabled


@dataclass(frozen=True)
class ToolSpec:
    name: str
    family: str
    description: str
    args: type[BaseModel]
    fn: Callable[[ToolContext, BaseModel], ToolOutput] | None   # None: handled by the env (control tools)
    stateful: bool = False
    schema_override: dict | None = None     # compact model-facing schema; validation still uses `args`
    opt_in: bool = False                    # registered, but exposed only when EnvConfig.tools_extra names it


REGISTRY: dict[str, ToolSpec] = {}


def register(name: str, family: str, description: str, args: type[BaseModel], stateful: bool = False,
             opt_in: bool = False):
    assert "\n" not in description and len(description) <= 100, f"{name}: description must be one short line"

    def deco(fn):
        REGISTRY[name] = ToolSpec(name, family, description, args, fn, stateful, opt_in=opt_in)
        return fn
    return deco


def register_control(name: str, description: str, args: type[BaseModel], schema_override: dict | None = None) -> None:
    REGISTRY[name] = ToolSpec(name, "H", description, args, None, stateful=True, schema_override=schema_override)


# Constraints the model does not need to see: the executor validates them and answers with field-level hints.
_DROP_KEYS = {"title", "additionalProperties", "discriminator", "minLength", "maxLength", "minItems", "maxItems",
              "exclusiveMinimum", "exclusiveMaximum", "minimum", "maximum"}


# Region (circle | polygon | ref) inlines to ~300 tokens per tool; the model sees one compact object instead.
_PT = {"type": "object", "properties": {"x_m": {"type": "number"}, "y_m": {"type": "number"}}}
REGION_COMPACT = {"type": "object", "description": "{type:ref,ref} | {type:circle,center,radius_m} | {type:polygon,points}",
                  "properties": {"type": {"enum": ["ref", "circle", "polygon"]}, "ref": {"type": "string"},
                                 "center": _PT, "radius_m": {"type": "number"},
                                 "points": {"type": "array", "items": _PT}}}


def _is_region(node: dict, defs: dict) -> bool:
    opts = node.get("oneOf") or node.get("anyOf") or []
    names = {o.get("$ref", "").split("/")[-1] for o in opts if isinstance(o, dict)}
    return {"Circle", "PolygonRegion", "RefRegion"} <= names


def _inline(schema: dict) -> dict:
    """Model-facing schema: $defs inlined, Optional[X] -> X, null defaults and validation-only keys dropped.

    Smaller schemas keep the static prefix under its token budget (SPEC 7.5). Full validation still runs
    server-side on the Pydantic models, so nothing is lost.
    """
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if _is_region(node, defs):
                return copy.deepcopy(REGION_COMPACT)
            if "anyOf" in node and any(isinstance(o, dict) and _is_region(o, defs) for o in node["anyOf"]):
                return copy.deepcopy(REGION_COMPACT)                     # Optional[Region]
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            if "anyOf" in node:
                opts = [o for o in node["anyOf"] if o != {"type": "null"}]
                if len(opts) == 1:
                    merged = {k: v for k, v in node.items() if k != "anyOf"}
                    merged.update(opts[0])
                    return walk(merged)
            out = {}
            for k, v in node.items():
                if k in _DROP_KEYS or (k == "default" and v is None):
                    continue
                out[k] = walk(v)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)


def tool_schemas(names: list[str] | None = None) -> list[dict]:
    """OpenAI-style function schemas, in a stable order (part of the byte-identical static prefix)."""
    out = []
    for name in names or default_names():
        spec = REGISTRY[name]
        params = spec.schema_override or _inline(spec.args.model_json_schema())
        out.append({"type": "function", "function": {
            "name": spec.name, "description": spec.description, "parameters": params}})
    return out


def default_names() -> list[str]:
    """The default tool set: everything registered except opt-in tools (the byte-identical prefix)."""
    return sorted(n for n, s in REGISTRY.items() if not s.opt_in)


def schemas_json(names: list[str] | None = None) -> str:
    return json.dumps(tool_schemas(names), separators=(",", ":"), sort_keys=True)


def load_all() -> dict[str, ToolSpec]:
    """Import every tool family so it registers itself."""
    from tools import consistency, control, coverage, df, drone, effects, replay, signals, site, tracks  # noqa: F401

    return REGISTRY
