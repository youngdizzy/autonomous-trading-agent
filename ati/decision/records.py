"""Immutable decision records.

Every trade candidate produces exactly one ``DecisionRecord``, written to the journal *before* any
order is submitted. It answers: why, with what information, supported by what evidence, what would
invalidate it, what the risk engine permitted, and what was decided.

Only concise stated reasons are persisted. There is deliberately no field for private reasoning or
chain-of-thought, and free-text fields are length-capped.

``decision_id`` is derived from (strategy hash, symbol, information cutoff), so the same decision
opportunity maps to the same id — and therefore the same client order id — even across restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from ati.core.canonical import sha256_hex
from ati.core.errors import LookaheadError
from ati.core.time import ensure_utc
from ati.ledger.journal import Journal
from ati.memory.evidence import EvidenceRef, EvidenceRegistry
from ati.risk.engine import Check

MAX_TEXT = 600


class FinalDecision(str, Enum):
    EXECUTE = "EXECUTE"
    NO_TRADE = "NO_TRADE"
    REJECTED_BY_RISK = "REJECTED_BY_RISK"
    BLOCKED_BY_REVIEW = "BLOCKED_BY_REVIEW"
    INVALID_REASONING_OUTPUT = "INVALID_REASONING_OUTPUT"


def make_decision_id(strategy_hash: str, symbol: str, cutoff: datetime, purpose: str = "entry") -> str:
    return "dec_" + sha256_hex({"s": strategy_hash, "sym": symbol, "t": ensure_utc(cutoff), "p": purpose})[:24]


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    timestamp: datetime
    symbol: str
    strategy_id: str
    strategy_version: int
    strategy_hash: str
    mode: str
    data_status: str
    market_context: tuple[tuple[str, str], ...]
    available_information_cutoff: datetime
    signal: str
    thesis: str
    invalidation_condition: str
    expected_risk: Decimal
    expected_reward: Decimal | None
    estimated_cost: Decimal
    proposed_size: Decimal | None
    approved_size: Decimal
    risk_constraints: tuple[Check, ...]
    risk_limits_hash: str
    evidence_references: tuple[EvidenceRef, ...]
    research_references: tuple[str, ...]
    confidence: float | None
    reasoning_tier: str
    adversarial_objections: tuple[str, ...]
    adversarial_verdict: str
    final_decision: FinalDecision
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp, "timestamp"))
        object.__setattr__(self, "available_information_cutoff", ensure_utc(self.available_information_cutoff, "cutoff"))
        if self.available_information_cutoff > self.timestamp:
            raise LookaheadError("information cutoff is after the decision timestamp")
        for ref in self.evidence_references:
            if ref.available_at > self.available_information_cutoff:
                raise LookaheadError(f"decision cites {ref.kind}:{ref.ref_id} available after the cutoff")
        for name in ("thesis", "invalidation_condition", "reason", "signal"):
            if len(getattr(self, name)) > MAX_TEXT:
                raise ValueError(f"{name} exceeds {MAX_TEXT} characters (concise reasons only)")
        for o in self.adversarial_objections:
            if len(o) > MAX_TEXT:
                raise ValueError("objection too long")
        if self.final_decision is FinalDecision.EXECUTE:
            if not self.thesis.strip() or not self.invalidation_condition.strip():
                raise ValueError("an executed decision requires a thesis and an invalidation condition")
            if self.approved_size <= 0:
                raise ValueError("an executed decision requires a positive approved size")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be within [0, 1]")

    @property
    def record_hash(self) -> str:
        return sha256_hex(self)


class DecisionLog:
    def __init__(self, journal: Journal, evidence: EvidenceRegistry):
        self.journal = journal
        self.evidence = evidence
        self._ids: set[str] = {e.payload["decision_id"] for e in journal.entries("decision")}
        self.last: DecisionRecord | None = None

    def record(self, rec: DecisionRecord) -> str:
        if rec.decision_id in self._ids:
            raise ValueError(f"decision {rec.decision_id} already recorded (records are immutable)")
        for ref in rec.evidence_references:
            if self.evidence.resolve(ref.ref_id) != ref:
                raise ValueError(f"evidence {ref.ref_id} does not match its registration")
        self.journal.append("decision", {"decision_id": rec.decision_id, "record_hash": rec.record_hash, "record": rec})
        self._ids.add(rec.decision_id)
        self.evidence.register("decision", rec.decision_id, rec.timestamp, rec.final_decision.value)
        self.last = rec
        return rec.record_hash

    def __contains__(self, decision_id: object) -> bool:
        return decision_id in self._ids
