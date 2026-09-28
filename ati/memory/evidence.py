"""Evidence registry.

Evidence is only what the system itself produced and recorded: experiment results, walk-forward
runs, adversarial reports, holdout evaluations, decisions, trade reviews, datasets. Each is
registered with a content hash and the instant it became available. Decision and memory records
may only cite registered evidence, so evidence cannot be fabricated by writing an id into a record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ati.core.time import ensure_utc
from ati.ledger.journal import Journal, decode

EVIDENCE_KINDS = frozenset({"dataset", "experiment", "walk_forward", "adversarial", "holdout", "promotion",
                            "decision", "trade_review", "observation"})


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    ref_id: str
    available_at: datetime

    def __post_init__(self) -> None:
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind {self.kind!r}")
        if not self.ref_id:
            raise ValueError("evidence ref_id required")
        object.__setattr__(self, "available_at", ensure_utc(self.available_at, "evidence.available_at"))


class EvidenceRegistry:
    def __init__(self, journal: Journal):
        self.journal = journal
        self._refs: dict[str, EvidenceRef] = {}
        for entry in journal.entries("evidence"):
            p = decode(entry.payload)
            self._refs[p["ref_id"]] = EvidenceRef(p["kind"], p["ref_id"], p["available_at"])

    def register(self, kind: str, ref_id: str, available_at: datetime, summary: str = "") -> EvidenceRef:
        ref = EvidenceRef(kind, ref_id, available_at)
        existing = self._refs.get(ref_id)
        if existing is not None:
            if existing != ref:
                raise ValueError(f"evidence {ref_id} already registered differently")
            return existing
        self.journal.append("evidence", {"kind": kind, "ref_id": ref_id, "available_at": ref.available_at,
                                         "summary": summary[:500]})
        self._refs[ref_id] = ref
        return ref

    def resolve(self, ref_id: str) -> EvidenceRef:
        try:
            return self._refs[ref_id]
        except KeyError:
            raise KeyError(f"unregistered evidence {ref_id!r}") from None

    def __contains__(self, ref_id: object) -> bool:
        return ref_id in self._refs
