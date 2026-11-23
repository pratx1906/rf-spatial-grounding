"""Append-only audit log and the server-side operator gate (demo-v2 S5.1, S5.5; guardrails audit T1, T10).

AuditLog: one JSONL file per UTC day under runs/audit_log/, opened only in append mode. Every record carries
`record_id`, `ts`, `prev_hash` (the previous record's `hash`, "genesis" for the first line of a file) and `hash`
(sha256 of the canonical JSON of the record without `hash`), so an edited or deleted line breaks the chain
(`AuditLog.verify`). Record kinds: `question` (serve/agent.py, one per operator question) and `gate` (one per gate
step). `trace(episode_id)` joins a question with its operator decision.

OperatorGate: the operator's decision on an answer that requires human confirmation (collateral / effector
assessments) is recorded on the server, bound to the exact answer by `answer_hash`, in two steps:
  1. step "review": {episode_id, answer_hash, operator_id, decision} -> a one-time confirm_token
  2. step "confirm": the same fields + confirm_token, at least `min_confirm_delay_s` later -> the recorded decision
A decision is recorded once; answers that failed verification or need no confirmation get no gate.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
import uuid
from pathlib import Path

GENESIS = "genesis"
DECISIONS = ("acknowledge", "dispute", "defer")
OPERATOR_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def answer_hash(episode_id: str, data_version: str, answer: dict | None) -> str:
    """sha256 over the canonical answer JSON, the episode and the data version: binds a decision to one answer."""
    return hashlib.sha256(_canon({"episode_id": episode_id, "data_version": data_version,
                                  "answer": answer}).encode()).hexdigest()


class AuditLog:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._last: dict[str, str] = {}

    def _file(self, ts: float) -> Path:
        return self.root / f"{time.strftime('%Y-%m-%d', time.gmtime(ts))}.jsonl"

    def _tail_hash(self, f: Path) -> str:
        if str(f) not in self._last:
            last = GENESIS
            if f.exists():
                lines = [x for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
                if lines:
                    last = json.loads(lines[-1])["hash"]
            self._last[str(f)] = last
        return self._last[str(f)]

    def append(self, record: dict) -> dict:
        with self._lock:
            ts = time.time()
            f = self._file(ts)
            f.parent.mkdir(parents=True, exist_ok=True)
            rec = {**record, "record_id": uuid.uuid4().hex[:16], "ts": round(ts, 3), "prev_hash": self._tail_hash(f)}
            rec.pop("hash", None)
            rec["hash"] = hashlib.sha256(_canon(rec).encode()).hexdigest()
            with f.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
                fh.flush()
            self._last[str(f)] = rec["hash"]
            return rec

    @staticmethod
    def verify(f: str | Path) -> tuple[bool, int | None]:
        """(True, None) when every line hashes correctly and links to the previous; else (False, bad line index)."""
        prev = GENESIS
        for i, line in enumerate(x for x in Path(f).read_text(encoding="utf-8").splitlines() if x.strip()):
            rec = json.loads(line)
            h = rec.pop("hash", None)
            if rec.get("prev_hash") != prev or hashlib.sha256(_canon(rec).encode()).hexdigest() != h:
                return False, i
            prev = h
        return True, None

    def records(self, episode_id: str | None = None):
        for f in sorted(self.root.glob("*.jsonl")) if self.root.exists() else []:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if episode_id is None or rec.get("episode_id") == episode_id:
                        yield rec

    def trace(self, episode_id: str) -> dict:
        """The question record of an episode with its recorded operator decision (None if none yet)."""
        recs = list(self.records(episode_id))
        q = next((r for r in recs if r.get("kind") == "question"), None)
        dec = [r for r in recs if r.get("kind") == "gate" and r.get("step") == "confirm"]
        if q is not None:
            q = {**q, "operator_decision": None}
        out = {"question": q, "gate": [r for r in recs if r.get("kind") == "gate"],
               "operator_decision": dec[-1] if dec else None}
        if q is not None and dec:
            out["question"]["operator_decision"] = {k: dec[-1][k] for k in ("decision", "operator_id", "record_id", "ts")}
        return out


class OperatorGate:
    def __init__(self, audit: AuditLog, min_confirm_delay_s: float = 1.0, token_ttl_s: float = 120.0,
                 window_s: float = 1800.0):
        self.audit = audit
        self.min_delay, self.ttl, self.window = min_confirm_delay_s, token_ttl_s, window_s
        self._lock = threading.Lock()
        self._answers: dict[str, dict] = {}

    def register(self, episode_id: str, answer_hash_: str, gate_required: bool, **meta) -> None:
        """Called by the gateway for every answer it delivers."""
        with self._lock:
            now = time.time()
            for k in [k for k, v in self._answers.items() if now - v["t"] > 4 * self.window]:
                del self._answers[k]
            self._answers[episode_id] = {"hash": answer_hash_, "gate_required": bool(gate_required), "t": now,
                                         "pending": None, "decided": None, **meta}

    def status(self, episode_id: str) -> dict | None:
        a = self._answers.get(episode_id)
        if a is None:
            return None
        d = a["decided"] or {}
        return {"episode_id": episode_id, "answer_hash": a["hash"], "gate_required": a["gate_required"],
                "decided": a["decided"] is not None, "decision": d.get("decision"), "operator_id": d.get("operator_id"),
                "record_id": d.get("record_id")}

    @staticmethod
    def _err(code: int, key: str, msg: str) -> tuple[int, dict]:
        return code, {"ok": False, "code": key, "message": msg}

    def submit(self, req: dict, client: str = "") -> tuple[int, dict]:
        """Returns (HTTP status, body)."""
        step, decision, op = req.get("step"), req.get("decision"), req.get("operator_id")
        if step not in ("review", "confirm"):
            return self._err(422, "bad_step", "step must be 'review' or 'confirm'")
        if decision not in DECISIONS:
            return self._err(422, "bad_decision", f"decision must be one of {', '.join(DECISIONS)}")
        if not isinstance(op, str) or not OPERATOR_RE.match(op):
            return self._err(422, "bad_operator", "operator_id: 1-40 letters, digits, '_', '.', '-'")
        ep, h = req.get("episode_id"), req.get("answer_hash")
        client_h = hashlib.sha256(str(client).encode()).hexdigest()[:12]
        with self._lock:
            a = self._answers.get(ep) if isinstance(ep, str) else None
            if a is None:
                return self._err(404, "unknown_episode", "no answer was delivered for this episode_id")
            if h != a["hash"]:
                return self._err(409, "hash_mismatch", "answer_hash does not match the delivered answer")
            if not a["gate_required"]:
                return self._err(409, "no_gate", "this answer needs no operator decision (not a verified "
                                                  "answer that requires human confirmation)")
            if a["decided"] is not None:
                return self._err(409, "already_decided", f"decision already recorded ({a['decided']['decision']})")
            now = time.time()
            if now - a["t"] > self.window:
                return self._err(410, "expired", "the decision window for this answer has closed")
            base = {"kind": "gate", "episode_id": ep, "answer_hash": h, "operator_id": op, "decision": decision,
                    "step": step, "client": client_h}
            if step == "review":
                token = secrets.token_urlsafe(16)
                rec = self.audit.append(base)
                a["pending"] = {"token": token, "operator_id": op, "decision": decision, "t": now}
                return 200, {"ok": True, "step": "review", "episode_id": ep, "answer_hash": h, "decision": decision,
                             "confirm_token": token, "confirm_after_s": self.min_delay, "expires_in_s": self.ttl,
                             "record_id": rec["record_id"]}
            p = a["pending"]
            if p is None:
                return self._err(409, "review_first", "send step 'review' first")
            if not secrets.compare_digest(str(req.get("confirm_token") or ""), p["token"]):
                return self._err(403, "bad_token", "confirm_token does not match the review step")
            if op != p["operator_id"] or decision != p["decision"]:
                return self._err(409, "mismatch", "confirm must repeat the reviewed operator_id and decision")
            if now - p["t"] > self.ttl:
                a["pending"] = None
                return self._err(410, "token_expired", "the review expired; send step 'review' again")
            if now - p["t"] < self.min_delay:
                return self._err(409, "too_soon", f"confirm at least {self.min_delay:g} s after the review")
            rec = self.audit.append({**base, "review_record_ts": round(p["t"], 3)})
            a["decided"] = {"decision": decision, "operator_id": op, "record_id": rec["record_id"], "ts": rec["ts"]}
            a["pending"] = None
            return 200, {"ok": True, "step": "confirm", "recorded": True, "record_id": rec["record_id"],
                         "episode_id": ep, "answer_hash": h, "decision": decision, "operator_id": op,
                         "recorded_at": rec["ts"]}
