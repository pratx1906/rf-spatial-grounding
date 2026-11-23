"""Stable error codes; every error carries a corrective hint (SPEC 5 rule 5, 6.1)."""

from __future__ import annotations

CODES = ("INVALID_ARGS", "UNKNOWN_REF", "AMBIGUOUS_ENTITY", "AMBIGUOUS_REFERENCE", "NOT_FOUND", "OUT_OF_COVERAGE",
         "BUDGET_EXCEEDED", "TIMEOUT", "NOT_PERMITTED", "INTERNAL_ERROR",
         "HEALTH_CHECK_REQUIRED")   # T-FIX T2 harness gate (harness/env.py)


class ToolError(Exception):
    def __init__(self, code: str, message: str, hint: str, candidates: list[dict] | None = None, **extra):
        assert code in CODES, code
        super().__init__(message)
        self.code, self.message, self.hint, self.candidates, self.extra = code, message, hint, candidates, extra

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.candidates:
            d["candidates"] = self.candidates
        d.update(self.extra)
        d["hint"] = self.hint
        return d
