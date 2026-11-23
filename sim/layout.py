"""Procedural stub-site layout: objects, zones, sensors (SPEC 4 data contract). Deterministic per seed.

Fictitious port; the real Kandla twin replaces it via config only (SPEC 4: no code changes).
Frame: local ENU metres. Land is y >= -300; the creek (water) is y < -300.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

EXTENT = (-1500.0, -1200.0, 1500.0, 1000.0)   # minx, miny, maxx, maxy
SHORE_Y = -300.0


@dataclass
class Obj:
    id: str
    name: str
    kind: str
    geom: Polygon
    height_m: float
    material: str
    protected: bool = False
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class Zone:
    id: str
    name: str
    kind: str          # perimeter | restricted | corridor | aviation | named_region
    geom: Polygon
    aliases: list[str] = field(default_factory=list)


@dataclass
class Sensor:
    id: str
    type: str          # fixed_df | drone_df | radar | eo
    x: float
    y: float
    z: float
    bands: list[str]
    array: dict = field(default_factory=lambda: {"kind": "uca", "n": 8, "radius_m": 0.12})
    mount: str = ""


def _rect(cx: float, cy: float, w: float, h: float) -> Polygon:
    return box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def build_layout(seed: int) -> tuple[list[Obj], list[Zone], list[Sensor]]:
    rng = random.Random(seed)

    def j(v: float, s: float = 15.0) -> float:
        return v + rng.uniform(-s, s)

    objs: list[Obj] = []
    # Oil terminal (west): tank farms and oil jetties are protected assets.
    objs += [
        Obj("tank_farm_a", "Tank Farm A", "tank_farm", _rect(j(-1050), j(200), 300, 200), 18, "steel", True,
            ["TFA", "tank farm alpha", "south tank farm"], ["oil_terminal"]),
        Obj("tank_farm_b", "Tank Farm B", "tank_farm", _rect(j(-680), j(250), 240, 200), 18, "steel", True,
            ["TFB", "tank farm bravo", "east tank farm"], ["oil_terminal"]),
        Obj("tank_farm_north", "North Tank Farm", "tank_farm", _rect(j(-900), j(650), 280, 180), 20, "steel", True,
            ["NTF", "north fuel depot", "fuel depot"], ["oil_terminal"]),
    ]
    for i, x in enumerate((-1200, -1000, -800, -600), start=1):
        objs.append(Obj(f"oil_jetty_{i}", f"Oil Jetty {i}", "jetty", _rect(j(x, 5), SHORE_Y - 175, 30, 350), 6,
                        "concrete", True, [f"OJ-{i}", f"jetty {i}", f"oil jetty no {i}"], ["oil_terminal", "water"]))
    objs.append(Obj("cargo_jetty", "Cargo Jetty", "jetty", _rect(j(1300, 5), SHORE_Y - 150, 60, 300), 6, "concrete",
                    False, ["CJ", "container jetty"], ["water"]))
    for i in range(6):
        objs.append(Obj(f"berth_{i + 1}", f"Berth {i + 1}", "berth", _rect(-200 + 250 * i, SHORE_Y - 20, 120, 40), 4,
                        "concrete", False, [f"B{i + 1}", f"berth no {i + 1}"], ["waterfront"]))
    n = 0
    for row, y in enumerate((-150, 30, 210, 390)):
        for col, x in enumerate((-300, 100, 500, 900, 1300)):
            n += 1
            objs.append(Obj(f"warehouse_w{n:02d}", f"Warehouse W{n:02d}", "warehouse",
                            _rect(j(x, 20), j(y, 10), rng.uniform(110, 150), rng.uniform(55, 75)),
                            round(rng.uniform(10, 16), 1), "metal", False, [f"W{n:02d}", f"warehouse {n}"], ["cargo"]))
    for i, x in enumerate((0, 450, 950, 1200), start=1):
        objs.append(Obj(f"crane_c{i}", f"Crane C{i}", "crane", _rect(j(x, 10), SHORE_Y + 40, 14, 14), 45, "steel",
                        False, [f"C{i}", f"quay crane {i}"], ["waterfront"]))
    objs += [
        Obj("control_tower", "Port Control Tower", "tower", _rect(100, 700, 20, 20), 45, "concrete", True,
            ["PCT", "control tower", "VTS tower"], ["admin"]),
        Obj("substation_main", "Main Substation", "substation", _rect(600, 800, 60, 40), 8, "metal", True,
            ["substation", "power station"], ["utility"]),
        Obj("fire_station", "Fire Station", "building", _rect(-300, 800, 40, 30), 8, "concrete", False,
            ["fire brigade"], ["emergency"]),
        Obj("admin_building", "Admin Building", "building", _rect(0, 850, 80, 40), 15, "concrete", False,
            ["administration", "port office"], ["admin"]),
        Obj("main_gate", "Main Gate", "gate", _rect(300, 960, 20, 10), 5, "concrete", False,
            ["gate 1", "entry gate"], ["access"]),
        Obj("helipad", "Helipad", "helipad", _rect(900, 700, 30, 30), 0.5, "concrete", False,
            ["heliport", "H pad"], ["aviation"]),
    ]

    terminal = unary_union([o.geom for o in objs if "oil_terminal" in o.tags]).envelope
    cargo = unary_union([o.geom for o in objs if o.kind == "warehouse"]).envelope
    sub = next(o for o in objs if o.id == "substation_main").geom
    heli = next(o for o in objs if o.id == "helipad").geom
    zones = [
        Zone("perimeter", "Port Perimeter", "perimeter", box(EXTENT[0], SHORE_Y, EXTENT[2], EXTENT[3]),
             ["port boundary", "fence line"]),
        Zone("oil_terminal_restricted", "Oil Terminal Restricted Zone", "restricted", terminal.buffer(100).envelope,
             ["oil terminal", "terminal restricted area"]),
        Zone("oil_terminal", "Oil Terminal", "named_region", terminal, ["oil terminal area"]),
        Zone("substation_restricted", "Substation Restricted Zone", "restricted", sub.buffer(150).envelope,
             ["substation zone"]),
        Zone("helipad_zone", "Helipad Zone", "aviation", heli.buffer(300).envelope, ["heliport zone"]),
        Zone("cargo_area", "Cargo Area", "named_region", cargo.buffer(30).envelope, ["warehouse district"]),
        Zone("north_approach", "North Approach Corridor", "corridor", box(-200, 1000, 200, 2200), ["north corridor"]),
        Zone("creek_approach", "Creek Approach Corridor", "corridor", box(500, -1200, 1500, SHORE_Y),
             ["creek corridor"]),
        Zone("creek", "Creek Channel", "named_region", box(EXTENT[0], EXTENT[1], EXTENT[2], SHORE_Y),
             ["the creek", "channel", "waterway"]),
    ]

    df_bands = ["sub_ghz_868_915", "ism_2400", "ism_5800"]
    sensors = [
        Sensor("S1", "fixed_df", 100, 700, 48, df_bands, mount="obj:control_tower"),
        Sensor("S2", "fixed_df", -1400, 0, 20, df_bands, mount="mast_west"),
        Sensor("S3", "fixed_df", 1400, 0, 20, df_bands, mount="mast_east"),
        Sensor("S4", "fixed_df", -250, -270, 15, df_bands, mount="quay_mast"),
        Sensor("S5", "fixed_df", 900, 950, 25, df_bands, mount="mast_north"),
        Sensor("R1", "radar", 0, 500, 30, [], array={"kind": "radar"}, mount="radar_mast"),
        Sensor("D1", "drone_df", 0, 0, 60, ["ism_2400", "ism_5800"], mount="drone_home"),
    ]
    return objs, zones, sensors
