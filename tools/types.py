"""Shared argument types and helpers (SPEC 6.1). Field names carry units; extra fields are rejected."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Band = Literal["sub_ghz_868_915", "ism_2400", "ism_5800", "gnss_l1", "gnss_l2", "vhf_marine", "cellular_lte", "other"]
ObjKind = Literal["jetty", "tank_farm", "berth", "warehouse", "crane", "tower", "substation", "building", "gate",
                  "helipad"]
ZoneKind = Literal["perimeter", "restricted", "corridor", "aviation", "named_region"]
Klass = Literal["drone", "rc_controller", "wifi_device", "unknown"]
RefKind = Literal["obj", "zone", "sensor", "trk", "hyp", "fp", "evt", "wp", "ras", "nodecand", "cand", "fe"]

REF_RE = re.compile(r"^(obj|zone|sensor|trk|hyp|fp|evt|wp|ras|nodecand|cand|fe):([A-Za-z0-9_\-]+)(@(\d+(\.\d+)?))?$")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point(Strict):
    x_m: float
    y_m: float
    z_m: float | None = None


class Circle(Strict):
    type: Literal["circle"]
    center: Point
    radius_m: Annotated[float, Field(gt=0, le=20000)]


class PolygonRegion(Strict):
    type: Literal["polygon"]
    points: Annotated[list[Point], Field(min_length=3, max_length=64)]


class RefRegion(Strict):
    type: Literal["ref"]
    ref: str


Region = Annotated[Circle | PolygonRegion | RefRegion, Field(discriminator="type")]


class Ellipse(Strict):
    kind: Literal["ellipse"]
    a_m: float
    b_m: float
    theta_deg: float


def parse_ref(ref: str) -> tuple[str, str, float | None]:
    """'trk:0042@120' -> ('trk', '0042', 120.0). Raises ValueError on malformed refs."""
    m = REF_RE.match(ref.strip())
    if not m:
        raise ValueError(f"malformed ref {ref!r}; expected <kind>:<id>[@<sim_t>]")
    return m.group(1), m.group(2), float(m.group(4)) if m.group(4) else None


def r1(v: float | None) -> float | None:
    """Token-lean rounding (positions 0.1 m, angles 0.1 deg; SPEC 5 rule 8)."""
    return None if v is None else round(float(v), 1)


def pt(x: float, y: float, z: float | None = None) -> dict:
    d = {"x_m": r1(x), "y_m": r1(y)}
    if z is not None:
        d["z_m"] = r1(z)
    return d
