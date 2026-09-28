"""Evidence registry.

Evidence is only what the system itself produced and recorded: experiment results, walk-forward
runs, adversarial reports, holdout evaluations, decisions, trade reviews, datasets. Each is
registered with a content hash and the instant it became available. Decision and memory records
may only cite registered evidence, so evidence cannot be fabricated by writing an id into a record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ati.core.errors import ProvenanceError
from ati.core.time import ensure_utc
from ati.ledger.journal import Journal, decode

EVIDENCE_KINDS = frozenset({"dataset", "experiment", "walk_forward", "adversarial", "holdout", "promotion",
                            "decision", "trade_review", "observation"})


@dataclass(frozen=True)
class EvidenceRef:
    """A reference to system-produced evidence. ``data_status``, ``dataset_id`` and ``strategy`` are
    canonical metadata set by the registry at registration; a caller-constructed ref that differs from
    the registered one is rejected wherever refs are accepted."""

    kind: str
    ref_id: str
    available_at: datetime
    data_status: str = "UNKNOWN"      # category of the system (journal) that produced the evidence
    dataset_id: str | None = None     # dataset the evidence was computed on, when applicable
    strategy: str | None = None       # strategy identity (definition hash or strategy id), when applicable

    def __post_init__(self) -> None:
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind {self.kind!r}")
        if not self.ref_id:
            raise ValueError("evidence ref_id required")
        object.__setattr__(self, "available_at", ensure_utc(self.available_at, "evidence.available_at"))


class EvidenceRegistry:
    """Canonical evidence records. The data category of every record is the category the registry's
    journal is bound to (a system never mixes categories), never a caller-supplied value."""

    def __init__(self, journal: Journal):
        self.journal = journal
        self.data_status: str = str(journal.attrs.get("data_status", "UNKNOWN"))
        self._refs: dict[str, EvidenceRef] = {}
        for entry in journal.entries("evidence"):
            p = decode(entry.payload)
            recorded = p.get("data_status", self.data_status)
            if recorded != self.data_status:
                raise ProvenanceError(f"evidence {p['ref_id']} recorded as {recorded} in a {self.data_status} journal")
            self._refs[p["ref_id"]] = EvidenceRef(p["kind"], p["ref_id"], p["available_at"], self.data_status,
                                                  p.get("dataset_id"), p.get("strategy"))

    def register(self, kind: str, ref_id: str, available_at: datetime, summary: str = "", *,
                 dataset_id: str | None = None, strategy: str | None = None) -> EvidenceRef:
        ref = EvidenceRef(kind, ref_id, available_at, self.data_status, dataset_id, strategy)
        existing = self._refs.get(ref_id)
        if existing is not None:
            if existing != ref:
                raise ValueError(f"evidence {ref_id} already registered differently")
            return existing
        self.journal.append("evidence", {"kind": kind, "ref_id": ref_id, "available_at": ref.available_at,
                                         "data_status": self.data_status, "dataset_id": dataset_id,
                                         "strategy": strategy, "summary": summary[:500]})
        self._refs[ref_id] = ref
        return ref

    def resolve(self, ref_id: str) -> EvidenceRef:
        try:
            return self._refs[ref_id]
        except KeyError:
            raise KeyError(f"unregistered evidence {ref_id!r}") from None

    def __contains__(self, ref_id: object) -> bool:
        return ref_id in self._refs
