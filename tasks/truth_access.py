"""The ONLY module that reads truth/ (SPEC 12.1). Used offline by task generators; never at episode time."""

from __future__ import annotations

from pathlib import Path

import duckdb


class TruthStore:
    def __init__(self, site_dir: str | Path):
        self.con = duckdb.connect(str(Path(site_dir) / "truth" / "truth.duckdb"), read_only=True)

    def emitters(self) -> dict[str, str]:
        return dict(self.con.execute("SELECT emitter_id, kind FROM emitters").fetchall())

    def track_of(self) -> dict[str, tuple[str, str]]:
        """track_id -> (emitter_id, emitter_kind)"""
        return {t: (e, k) for t, e, k in self.con.execute("SELECT * FROM track_labels").fetchall()}

    def position(self, emitter_id: str, t: float) -> tuple[float, float, float] | None:
        r = self.con.execute("SELECT x_m, y_m, z_m FROM emitter_points WHERE emitter_id = ? AND t_s = ?",
                             [emitter_id, t]).fetchone()
        return r

    def trajectories(self) -> dict[str, dict]:
        """emitter_id -> {kind, swarm_id, t, xyz} for offline rendering of the world (never agent-visible)."""
        meta = {e: (k, s) for e, k, s in self.con.execute("SELECT emitter_id, kind, swarm_id FROM emitters").fetchall()}
        out: dict[str, dict] = {}
        for e, t, x, y, z in self.con.execute(
                "SELECT emitter_id, t_s, x_m, y_m, z_m FROM emitter_points ORDER BY emitter_id, t_s").fetchall():
            d = out.setdefault(e, {"kind": meta.get(e, ("?", None))[0], "swarm_id": meta.get(e, ("?", None))[1],
                                   "t": [], "xyz": []})
            d["t"].append(t)
            d["xyz"].append([round(x, 2), round(y, 2), round(z, 2)])
        return out

    def emitter_meta(self) -> dict[str, dict]:
        """emitter_id -> {kind, swarm_id, bands: [(from_t, band)], eirp_dbm}"""
        import json

        return {e: {"kind": k, "swarm_id": s, "bands": json.loads(b), "eirp_dbm": p}
                for e, k, b, p, s in self.con.execute(
                    "SELECT emitter_id, kind, bands, eirp_dbm, swarm_id FROM emitters").fetchall()}

    def hyp_labels_full(self) -> dict[str, tuple[str, str | None]]:
        return {h: (p, v) for h, p, v in self.con.execute("SELECT hyp_id, path_type, via_obj FROM hyp_labels").fetchall()}

    def hyp_labels(self) -> dict[str, str]:
        return dict(self.con.execute("SELECT hyp_id, path_type FROM hyp_labels").fetchall())

    def close(self) -> None:
        self.con.close()
