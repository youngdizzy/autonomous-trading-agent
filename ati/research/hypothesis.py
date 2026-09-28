"""Hypothesis engine: OBSERVATION → HYPOTHESIS → PRE-REGISTRATION → TEST → FALSIFICATION →
REPLICATION → OUT-OF-SAMPLE → FINDING.

Success criteria are locked (hashed and journaled) *before* any result is recorded. A result can
only be recorded against a locked pre-registration, and a pre-registration can never be edited —
redefining success after seeing results requires a new hypothesis, which increases the recorded
count of hypotheses tested (and therefore the multiple-testing penalty).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ati.core.canonical import sha256_hex
from ati.core.errors import LifecycleError
from ati.core.time import ensure_utc
from ati.ledger.journal import Journal
from ati.research.metrics import Metrics

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_AUTOMATED = "NOT_AUTOMATED"


@dataclass(frozen=True)
class Criterion:
    metric: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"unsupported operator {self.op}")
        if self.metric not in Metrics.__dataclass_fields__:
            raise ValueError(f"unknown metric {self.metric}")

    def evaluate(self, metrics: Metrics) -> Verdict:
        value = getattr(metrics, self.metric)
        if value is None:
            return Verdict.INSUFFICIENT_EVIDENCE
        return Verdict.PASS if _OPS[self.op](value, self.threshold) else Verdict.FAIL


@dataclass(frozen=True)
class PreRegistration:
    hypothesis_id: str
    statement: str
    observation_refs: tuple[str, ...]
    strategy_key: str
    strategy_hash: str
    dev_dataset_id: str
    criteria: tuple[Criterion, ...]
    min_trades: int
    locked_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "locked_at", ensure_utc(self.locked_at))
        if not self.criteria:
            raise ValueError("a hypothesis must declare falsification criteria")
        if not self.statement.strip():
            raise ValueError("statement required")

    @property
    def prereg_hash(self) -> str:
        return sha256_hex(self)

    def evaluate(self, metrics: Metrics) -> tuple[Verdict, list[tuple[str, str]]]:
        if metrics.n_trades < self.min_trades:
            return Verdict.INSUFFICIENT_EVIDENCE, [("n_trades", f"{metrics.n_trades} < {self.min_trades}")]
        details = [(f"{c.metric} {c.op} {c.threshold}", c.evaluate(metrics).value) for c in self.criteria]
        verdicts = {v for _, v in details}
        if Verdict.FAIL.value in verdicts:
            return Verdict.FAIL, details
        if Verdict.INSUFFICIENT_EVIDENCE.value in verdicts:
            return Verdict.INSUFFICIENT_EVIDENCE, details
        return Verdict.PASS, details


class ResearchLog:
    """Journal-backed record of every pre-registration and every experiment, including failures."""

    def __init__(self, journal: Journal):
        self.journal = journal
        self._prereg: dict[str, PreRegistration] = {}
        self._experiments: list[dict] = []

    def preregister(self, prereg: PreRegistration) -> str:
        existing = self._prereg.get(prereg.hypothesis_id)
        if existing is not None:
            if existing.prereg_hash != prereg.prereg_hash:
                raise LifecycleError(f"{prereg.hypothesis_id} already pre-registered with different criteria")
            return existing.prereg_hash
        self.journal.append("preregistration", {"prereg": prereg, "prereg_hash": prereg.prereg_hash})
        self._prereg[prereg.hypothesis_id] = prereg
        return prereg.prereg_hash

    def record_experiment(self, hypothesis_id: str, stage: str, evidence_hash: str, metrics: Metrics,
                          dataset_id: str) -> Verdict:
        prereg = self._prereg.get(hypothesis_id)
        if prereg is None:
            raise LifecycleError(f"no pre-registration for {hypothesis_id}; criteria must be locked before testing")
        verdict, details = prereg.evaluate(metrics)
        row = {"hypothesis_id": hypothesis_id, "prereg_hash": prereg.prereg_hash, "stage": stage,
               "evidence_hash": evidence_hash, "dataset_id": dataset_id, "metrics": metrics,
               "verdict": verdict, "details": details}
        self.journal.append("experiment", row)
        self._experiments.append(row)
        return verdict

    def get(self, hypothesis_id: str) -> PreRegistration:
        return self._prereg[hypothesis_id]

    @property
    def hypotheses_tested(self) -> int:
        return len(self._prereg)

    @property
    def experiments(self) -> tuple[dict, ...]:
        return tuple(self._experiments)
