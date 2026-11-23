"""Evidence ledger: harness-maintained working memory (SPEC 7.3) and the verifier's grounding source.

Append-only in context (D-003): after each tool result the harness appends only the ledger *delta* (new facts).
`render()` gives the full ledger for UI and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RefInfo:
    ref: str
    tool: str
    turn: int
    summary: str = ""


@dataclass
class Ledger:
    refs: dict[str, RefInfo] = field(default_factory=dict)
    resolved: list[str] = field(default_factory=list)
    tracks: dict[str, str] = field(default_factory=dict)
    hyps: dict[str, list[str]] = field(default_factory=dict)          # track@t -> hyp refs
    checks: list[str] = field(default_factory=list)
    check_calls: list[tuple[str, dict, object]] = field(default_factory=list)   # (tool, args, value) for recompute
    open_questions: list[str] = field(default_factory=list)
    fits: list[dict] = field(default_factory=list)          # fit_candidates results (verifier only; not in context)
    last_search_total: int | None = None
    _pending: list[str] = field(default_factory=list)

    @staticmethod
    def base(ref: str) -> str:
        return ref.split("@", 1)[0]

    def has(self, ref: str) -> bool:
        """A ref counts as grounded with or without its @t suffix."""
        return ref in self.refs or self.base(ref) in self.refs

    def register(self, refs: list[str], tool: str, turn: int) -> None:
        for r in refs:
            for key in {r, self.base(r)}:
                self.refs.setdefault(key, RefInfo(key, tool, turn))

    def record(self, tool: str, args: dict, envelope: dict) -> None:
        """Extract compact facts from a tool result into the ledger and the pending delta."""
        new: list[str] = []
        if not envelope.get("ok"):
            err = envelope["error"]
            if err["code"] in ("AMBIGUOUS_ENTITY", "AMBIGUOUS_REFERENCE"):
                q = f"ambiguous: {args.get('query', '')!r} -> " + "|".join(c["ref"] for c in err.get("candidates", []))
                self.open_questions.append(q)
                new.append(f"open: {q}")
            return self._push(new)
        d = envelope["data"]
        if tool == "site_resolve":
            c = d["candidates"][0]
            line = f"resolved: {args.get('query')!r} -> {c['ref']} ({c['score']})"
            self.resolved.append(line)
            new.append(line)
        elif tool in ("track_search", "track_detail"):
            items = d["items"] if tool == "track_search" else [d]
            if tool == "track_search":
                self.last_search_total = d["total"]
                new.append(f"search: total={d['total']} page={len(items)}")
            for it in items[:6]:
                p, u = it["pos"], it["uncertainty"]
                s = (f"{it['ref']} ({it['band'] if 'band' in it else it['signal']['band']}, t={it['t_s']}, "
                     f"pos {p['x_m']},{p['y_m']}, u={u['a_m']}m, conf {it['confidence']})")
                self.tracks[it["ref"]] = s
                new.append(f"track: {s}")
        elif tool == "df_hypotheses":
            key = f"{d['track']}@{d['t_s']}"
            for s in d["sensors"]:
                hs = [f"{h['ref']} {h['bearing_deg']}deg" for h in s["hyps"]]
                self.hyps.setdefault(key, []).extend(h["ref"] for h in s["hyps"])
                new.append(f"hyps {s['sensor']}: " + ", ".join(hs))
        elif tool == "fit_candidates":
            self.fits.append(d)                     # kept for the verifier (explained_as, positions); no delta line
        elif tool in ("inside_zone", "spatial_relation"):
            subject = args.get("subject") if tool == "inside_zone" else args.get("a_ref")
            other = args.get("zone_ref") if tool == "inside_zone" else args.get("b_ref")
            val = d.get("inside", d.get("value", d.get("value_m", d.get("value_deg"))))
            line = f"check: {tool}({subject}, {other}{', ' + args['relation'] if 'relation' in args else ''}) -> {val}"
            self.checks.append(line)
            self.check_calls.append((tool, dict(args), val))
            new.append(line)
        self._push(new)

    def _push(self, lines: list[str]) -> None:
        self._pending.extend(lines)

    def take_delta(self) -> str:
        """New facts since the last call, as the text appended after a tool result."""
        lines, self._pending = self._pending, []
        return ("LEDGER+ " + " | ".join(lines)) if lines else ""

    def render(self, turn: int) -> str:
        parts = [f"LEDGER (turn {turn})"]
        parts += self.resolved
        parts += [f"track: {v}" for v in self.tracks.values()]
        parts += [f"hyps {k}: {', '.join(v)}" for k, v in self.hyps.items()]
        parts += self.checks
        parts.append(f"open_questions: {self.open_questions}")
        return "\n".join(parts)
