"""Structured institutional memory.

Append-only. An entry is never edited or deleted; a changed belief is a *new* entry that
``supersedes`` the old one, so the history of what the system believed, and when, is preserved.

Kinds and their evidence requirements (enforced):

| Kind                 | Meaning                                        | Required evidence                       | Confidence cap |
|----------------------|------------------------------------------------|-----------------------------------------|----------------|
| HYPOTHESIS           | unvalidated idea                               | none                                    | 0.5            |
| FINDING              | evidence-backed observation                    | >=1 experiment/walk_forward/trade_review/observation | 0.7 |
| VALIDATED_FINDING    | finding that survived OOS + holdout testing    | walk_forward AND holdout AND adversarial | 0.95          |
| REJECTED_HYPOTHESIS  | idea that failed testing (kept forever)        | >=1 experiment/walk_forward/holdout/adversarial | 1.0     |
| MISTAKE              | recurring decision failure                     | >=2 decision or trade_review            | 0.9            |
| LESSON               | behavioral lesson that survived testing        | >=1 experiment/walk_forward/holdout      | 0.8            |

Queries are point-in-time: ``query(as_of=T)`` only returns entries available at or before T.

Integrity rules (Research & Memory Integrity 1.1), checked on ``add`` and re-checked on every reload:

- Category gate: the store's category is the ``data_status`` its journal is bound to, and must equal
  the evidence registry's. Doctrinal kinds (FINDING, VALIDATED_FINDING, MISTAKE, LESSON) are accepted
  only in a market-evidence store (REAL, HISTORICAL, DELAYED). MOCK, SYNTHETIC, UNKNOWN or any other
  category may hold HYPOTHESIS and REJECTED_HYPOTHESIS only. Nothing is silently downgraded.
- Evidence is judged by its canonical registry record, never by caller-supplied metadata; every
  cited ref must equal its registration, and one ref may be cited only once.
- Independence: a VALIDATED_FINDING needs walk_forward, holdout and adversarial evidence that all carry
  dataset identities; the holdout dataset must differ from every development dataset cited, and the
  holdout and adversarial evidence must concern the same strategy identity. Missing identity fails.
- Dataset claims: an entry may only list dataset ids that its canonical evidence actually carries.
- Supersession: a VALIDATED_FINDING can only be superseded by an entry citing new testing evidence
  (experiment, walk_forward or holdout not cited by the finding it replaces).
- Entries in the journal that violate these rules (e.g. written by older code) are quarantined on
  reload: they stay in the immutable journal but are never returned by queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ati.core.canonical import sha256_hex, sha256_text
from ati.core.errors import LifecycleError, LookaheadError, MemoryIntegrityError, ProvenanceError
from ati.core.time import ensure_utc
from ati.ledger.journal import Journal, decode
from ati.memory.evidence import EvidenceRef, EvidenceRegistry


class MemoryKind(str, Enum):
    HYPOTHESIS = "HYPOTHESIS"
    FINDING = "FINDING"
    VALIDATED_FINDING = "VALIDATED_FINDING"
    REJECTED_HYPOTHESIS = "REJECTED_HYPOTHESIS"
    MISTAKE = "MISTAKE"
    LESSON = "LESSON"


CONFIDENCE_CAP = {
    MemoryKind.HYPOTHESIS: 0.5, MemoryKind.FINDING: 0.7, MemoryKind.VALIDATED_FINDING: 0.95,
    MemoryKind.REJECTED_HYPOTHESIS: 1.0, MemoryKind.MISTAKE: 0.9, MemoryKind.LESSON: 0.8,
}

_TESTING = {"experiment", "walk_forward", "holdout", "adversarial"}
MARKET_CATEGORIES = frozenset({"REAL", "HISTORICAL", "DELAYED"})
DOCTRINAL = frozenset({"FINDING", "VALIDATED_FINDING", "MISTAKE", "LESSON"})


def _evidence_ok(kind: MemoryKind, refs: tuple[EvidenceRef, ...]) -> str | None:
    kinds = [r.kind for r in refs]
    if kind is MemoryKind.FINDING and not set(kinds) & {"experiment", "walk_forward", "trade_review", "observation"}:
        return "FINDING requires experiment, walk_forward, trade_review or observation evidence"
    if kind is MemoryKind.VALIDATED_FINDING and not {"walk_forward", "holdout", "adversarial"} <= set(kinds):
        return "VALIDATED_FINDING requires walk_forward, holdout and adversarial evidence"
    if kind is MemoryKind.REJECTED_HYPOTHESIS and not set(kinds) & _TESTING:
        return "REJECTED_HYPOTHESIS requires testing evidence"
    if kind is MemoryKind.MISTAKE and sum(k in ("decision", "trade_review") for k in kinds) < 2:
        return "MISTAKE requires at least two decision/trade_review references (recurring)"
    if kind is MemoryKind.LESSON and not set(kinds) & {"experiment", "walk_forward", "holdout"}:
        return "LESSON requires experiment, walk_forward or holdout evidence"
    return None


def _check_independence(refs: list[EvidenceRef]) -> None:
    """Holdout confirmation must come from a dataset distinct from all development evidence."""
    holdouts = [r for r in refs if r.kind == "holdout"]
    development = [r for r in refs if r.kind in ("walk_forward", "adversarial", "experiment")]
    for r in holdouts + development:
        if not r.dataset_id:
            raise MemoryIntegrityError(f"{r.kind} evidence {r.ref_id} has no dataset identity; independence unverifiable")
    dev_datasets = {r.dataset_id for r in development}
    for h in holdouts:
        if h.dataset_id in dev_datasets:
            raise MemoryIntegrityError("holdout evidence uses a development dataset; not independent confirmation")
    adversarial = [r for r in refs if r.kind == "adversarial"]
    if not holdouts or not adversarial:
        raise MemoryIntegrityError("validated finding requires holdout and adversarial evidence")
    for h in holdouts:
        if not h.strategy or any(a.strategy != h.strategy for a in adversarial):
            raise MemoryIntegrityError("holdout and adversarial evidence do not concern the same strategy")


def _check_validated_supersession(old: "MemoryEntry", new_refs: list[EvidenceRef]) -> None:
    """A validated conclusion is replaced only by *new* testing evidence, never by opinion."""
    if old.kind is not MemoryKind.VALIDATED_FINDING:
        return
    old_ids = {r.ref_id for r in old.evidence}
    if not any(r.kind in ("experiment", "walk_forward", "holdout") and r.ref_id not in old_ids for r in new_refs):
        raise LifecycleError("validated knowledge can only be superseded by new testing evidence")


def fingerprint(statement: str, strategy_kind: str = "") -> str:
    tokens = sorted(set(re.findall(r"[a-z0-9]+", statement.lower())))
    return sha256_text(strategy_kind + "|" + " ".join(tokens))[:16]


@dataclass(frozen=True)
class MemoryEntry:
    kind: MemoryKind
    statement: str
    created_at: datetime
    provenance: str
    evidence: tuple[EvidenceRef, ...]
    confidence: float
    dataset_ids: tuple[str, ...] = ()
    strategy_keys: tuple[str, ...] = ()
    strategy_kind: str = ""
    supersedes: str | None = None
    lifecycle: str = "ACTIVE"

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", ensure_utc(self.created_at, "created_at"))
        if not self.statement.strip() or len(self.statement) > 2000:
            raise ValueError("statement must be 1..2000 characters")
        if not self.provenance:
            raise ValueError("provenance required")
        if not (0.0 <= self.confidence <= CONFIDENCE_CAP[self.kind]):
            raise ValueError(f"confidence {self.confidence} exceeds cap {CONFIDENCE_CAP[self.kind]} for {self.kind.value}")
        for ref in self.evidence:
            if ref.available_at > self.created_at:
                raise LookaheadError(f"memory cites evidence {ref.ref_id} not available until {ref.available_at}")

    @property
    def entry_id(self) -> str:
        return "mem_" + sha256_hex(self)[:20]

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.statement, self.strategy_kind)


class MemoryStore:
    def __init__(self, journal: Journal, evidence: EvidenceRegistry):
        self.journal = journal
        self.evidence = evidence
        self.data_status: str = str(journal.attrs.get("data_status", "UNKNOWN"))
        if self.data_status != evidence.data_status:
            raise ProvenanceError(f"memory journal is {self.data_status} but evidence registry is {evidence.data_status}")
        self._entries: dict[str, MemoryEntry] = {}
        self._superseded: set[str] = set()
        self.quarantined: dict[str, str] = {}
        for entry in journal.entries("memory"):
            p = decode(entry.payload)["entry"]
            m = MemoryEntry(MemoryKind(p["kind"]), p["statement"], p["created_at"], p["provenance"],
                            tuple(EvidenceRef(e["kind"], e["ref_id"], e["available_at"], e.get("data_status", "UNKNOWN"),
                                              e.get("dataset_id"), e.get("strategy")) for e in p["evidence"]),
                            p["confidence"], tuple(p["dataset_ids"]), tuple(p["strategy_keys"]), p["strategy_kind"],
                            p["supersedes"], p["lifecycle"])
            try:
                self._validate(m)  # restart never weakens validation
            except (MemoryIntegrityError, LifecycleError, KeyError, ValueError) as exc:
                self.quarantined[m.entry_id] = str(exc)
                continue
            self._entries[m.entry_id] = m
            if m.supersedes:
                self._superseded.add(m.supersedes)

    @property
    def accepts_doctrine(self) -> bool:
        return self.data_status in MARKET_CATEGORIES

    def _validate(self, entry: MemoryEntry) -> None:
        ids = [r.ref_id for r in entry.evidence]
        if len(ids) != len(set(ids)):
            raise MemoryIntegrityError("duplicate evidence reference: each piece of evidence may be cited once")
        canonical = []
        for ref in entry.evidence:
            registered = self.evidence.resolve(ref.ref_id)  # KeyError → fabricated evidence
            if registered != ref:
                raise MemoryIntegrityError(f"evidence {ref.ref_id} does not match its canonical registration")
            canonical.append(registered)
        if entry.kind.value in DOCTRINAL:
            if not self.accepts_doctrine:
                raise MemoryIntegrityError(f"{entry.kind.value} is institutional doctrine; a {self.data_status} store "
                                           "may only hold HYPOTHESIS or REJECTED_HYPOTHESIS")
            if any(r.data_status not in MARKET_CATEGORIES for r in canonical):
                raise MemoryIntegrityError("doctrine may only cite market-evidence records")
            evidenced = {r.dataset_id for r in canonical if r.dataset_id}
            if not set(entry.dataset_ids) <= evidenced:
                raise MemoryIntegrityError("entry lists dataset ids its evidence does not carry")
        problem = _evidence_ok(entry.kind, tuple(canonical))
        if problem:
            raise MemoryIntegrityError(problem)
        if entry.kind is MemoryKind.VALIDATED_FINDING:
            _check_independence(canonical)
        if entry.supersedes is not None:
            if entry.supersedes not in self._entries:
                raise LifecycleError(f"cannot supersede unknown entry {entry.supersedes}")
            if entry.supersedes in self._superseded:
                raise LifecycleError(f"{entry.supersedes} is already superseded")
            old = self._entries[entry.supersedes]
            if old.kind is MemoryKind.REJECTED_HYPOTHESIS and entry.kind is not MemoryKind.REJECTED_HYPOTHESIS:
                if not {r.kind for r in entry.evidence} & {"holdout", "walk_forward"}:
                    raise LifecycleError("a rejected hypothesis can only be revived with new out-of-sample evidence")
            _check_validated_supersession(old, canonical)

    def add(self, entry: MemoryEntry) -> str:
        self._validate(entry)
        if entry.entry_id in self._entries:
            return entry.entry_id
        self.journal.append("memory", {"entry_id": entry.entry_id, "entry": entry})
        self._entries[entry.entry_id] = entry
        if entry.supersedes:
            self._superseded.add(entry.supersedes)
        return entry.entry_id

    def get(self, entry_id: str) -> MemoryEntry:
        return self._entries[entry_id]

    def query(self, as_of: datetime, kind: MemoryKind | None = None, include_superseded: bool = False) -> list[MemoryEntry]:
        as_of = ensure_utc(as_of)
        superseded_by_then = {e.supersedes for e in self._entries.values() if e.supersedes and e.created_at <= as_of}
        return sorted(
            (e for eid, e in self._entries.items()
             if e.created_at <= as_of and (kind is None or e.kind is kind)
             and (include_superseded or eid not in superseded_by_then)),
            key=lambda e: e.created_at,
        )

    def prior_rejections(self, statement: str, strategy_kind: str = "") -> list[MemoryEntry]:
        """Has this idea already failed? Used before new research to avoid rediscovering false edges."""
        fp = fingerprint(statement, strategy_kind)
        return [e for e in self._entries.values() if e.kind is MemoryKind.REJECTED_HYPOTHESIS and e.fingerprint == fp]

    def __len__(self) -> int:
        return len(self._entries)
