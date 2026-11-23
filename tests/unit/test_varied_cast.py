"""S7.3: varied-cast scenario generator option (audit G Q48). The default cast must stay bit-identical."""

import hashlib
import json
from dataclasses import dataclass

import pytest
from shapely.geometry import box

from sim.world import kandla_scenario as KS


@dataclass
class _Obj:
    kind: str
    geom: object
    id: str = ""


def _site():
    objs = [_Obj("tank_farm", box(x, y, x + 200, y + 150), f"tf{i}")
            for i, (x, y) in enumerate([(-600, -900), (-200, -1300), (400, -800), (-1200, -400), (900, 200)])]
    objs += [_Obj("jetty", box(x, 900 + 40 * i, x + 30, 1300 + 40 * i), f"j{i}")
             for i, x in enumerate([500, 700, 900, 1100, 1300])]
    zones = {"jetty_restricted": _Obj("zone", box(400, 800, 1400, 1500))}
    return zones, objs


def _fingerprint(ems) -> str:
    rows = [[e.id, e.kind, e.band_schedule, e.freq_mhz, round(e.eirp_dbm, 6), e.protocol, e.t_start, e.t_end,
             [[round(v, 4) for v in p] for p in e.points[:: 25]], e.swarm_id] for e in ems]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()[:16]


# fingerprints of the default cast taken BEFORE the varied option was added (seed -> sha)
DEFAULT_FP = {0: "b9b8cad141bfd218", 13: "27ad16c731761544", 40: "8b38abf775e92dd2"}


@pytest.mark.parametrize("seed", sorted(DEFAULT_FP))
def test_default_cast_is_unchanged(seed):
    zones, objs = _site()
    assert _fingerprint(KS.make_emitters(seed, zones, objs)) == DEFAULT_FP[seed]
    assert _fingerprint(KS.make_emitters(seed, zones, objs, cast="default")) == DEFAULT_FP[seed]


@pytest.mark.parametrize("seed", [100, 101, 102, 103, 104, 105, 106, 107])
def test_varied_cast_ranges(seed):
    zones, objs = _site()
    ems = KS.make_emitters(seed, zones, objs, cast="varied")
    drones = [e for e in ems if e.kind == "drone"]
    assert 1 <= len(drones) <= 20
    assert len({e.id for e in ems}) == len(ems)
    for e in ems:
        assert e.points, e.id
        if e.kind in ("drone", "friendly_drone"):
            assert all(5.0 - 1e-6 <= p[3] <= 150.0 + 1e-6 for p in e.points), e.id
        assert 8.0 <= e.eirp_dbm <= 30.0
        assert e.band_schedule[0][1] in KS.PROTO


def test_varied_cast_spans_the_distribution_and_is_deterministic():
    zones, objs = _site()
    casts = [KS.make_emitters(s, zones, objs, cast="varied") for s in range(200, 240)]
    n = [sum(e.kind == "drone" for e in c) for c in casts]
    alts = [p[3] for c in casts for e in c if e.kind == "drone" for p in e.points]
    assert min(n) <= 4 and max(n) >= 15                     # drone count varies 1-20
    assert min(alts) < 15 and max(alts) > 120                # low (< 40 m) and high altitudes occur
    eirp = [e.eirp_dbm for c in casts for e in c if e.kind == "drone"]
    assert max(eirp) - min(eirp) > 12
    dens = [max(sum(e.band_schedule[0][1] == b for e in c if e.kind == "drone") for b in KS.PROTO)
            / max(1, sum(e.kind == "drone" for e in c)) for c in casts]
    assert min(dens) < 0.6 and max(dens) == 1.0              # same-band density varies
    assert _fingerprint(KS.make_emitters(205, zones, objs, cast="varied")) == _fingerprint(casts[5])


def test_unknown_cast_is_rejected():
    zones, objs = _site()
    with pytest.raises(ValueError):
        KS.make_emitters(1, zones, objs, cast="nope")
