"""Deterministic health gate.

Each component gets exactly one explicit status — never a blended score:

  PASS       the component is readable, intact and permits action
  BLOCKED    intact, but a safety state forbids action (kill switch, halted execution, pause, stale data)
  FAIL       integrity problem (corruption, contradiction, invalid or mixed data) — needs an operator
  NOT_READY  prerequisites are missing (e.g. no fresh data yet, too little history)
  NOT_RUN    the check could not be performed this cycle

Data availability is classified into exactly one ``DataState``:

  REAL_DATA_AVAILABLE    market-category data, fresh, provenance re-derived from archived payloads,
                         and enough bars for the champion's lookback
  INSUFFICIENT_REAL_DATA market-category data, fresh and verified, but too little history
  REAL_DATA_UNAVAILABLE  market-category system without fresh data (provider unreachable, stale, empty)
  MOCK_DATA_ONLY         non-market system (MOCK/SYNTHETIC): usable for mechanics, never market evidence
  MIXED_DATA             more than one data category encountered — fail closed, never pick one
  INVALID_DATA           data failed validation / provenance / history consistency — fail closed
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from ati.config import LIVE_TRADING, OperatingMode
from ati.core.errors import (DataIntegrityError, HistoricalConflictError, MalformedResponse, ModeMismatch,
                             ProvenanceError, ProviderUnavailable, RateLimited)
from ati.execution.broker import TERMINAL
from ati.ledger.journal import Journal
from ati.market.models import MARKET_EVIDENCE_STATUSES
from ati.memory.store import MemoryStore
from ati.research.hypothesis import ResearchLog
from ati.risk.engine import ReconState, RiskLimits


class Status(str, Enum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"
    FAIL = "FAIL"
    NOT_READY = "NOT_READY"
    NOT_RUN = "NOT_RUN"


class DataState(str, Enum):
    REAL_DATA_UNAVAILABLE = "REAL_DATA_UNAVAILABLE"
    INSUFFICIENT_REAL_DATA = "INSUFFICIENT_REAL_DATA"
    MOCK_DATA_ONLY = "MOCK_DATA_ONLY"
    REAL_DATA_AVAILABLE = "REAL_DATA_AVAILABLE"
    MIXED_DATA = "MIXED_DATA"
    INVALID_DATA = "INVALID_DATA"


COMPONENTS = ("data", "research", "memory", "risk", "execution", "ledger", "persistence", "company")


@dataclass(frozen=True)
class Check:
    component: str
    status: Status
    detail: str


@dataclass(frozen=True)
class HealthReport:
    checks: tuple[Check, ...]
    data_state: DataState

    def status(self, component: str) -> Status:
        return next(c.status for c in self.checks if c.component == component)

    def as_dict(self) -> dict:
        return {"data_state": self.data_state.value,
                "checks": {c.component: {"status": c.status.value, "detail": c.detail} for c in self.checks}}

    def gate(self, action: str, paused: bool) -> list[str]:
        """Reasons the action may not proceed now (empty = allowed). Read-only actions and PAUSE are always
        allowed while the company journal itself is intact: inspection and stopping never need permission."""
        reasons = []
        if self.status("company") is Status.FAIL:
            return ["company state is not trustworthy (company journal FAIL)"]
        if action in ("NO_TRADE", "PAUSE", "REQUEST_DATA", "REVIEW_POSITION", "REVIEW_RISK", "REVIEW_SYSTEM"):
            return []
        if paused:
            reasons.append("company is PAUSED: only read-only actions and PAUSE are allowed")
        if action == "TRADE_PROPOSAL":
            needed = ("data", "memory", "risk", "execution", "ledger", "persistence")
            ok_data = {DataState.REAL_DATA_AVAILABLE, DataState.MOCK_DATA_ONLY}
        elif action == "RESEARCH_REQUEST":
            needed = ("data", "research", "memory", "ledger", "persistence")
            ok_data = {DataState.REAL_DATA_AVAILABLE, DataState.MOCK_DATA_ONLY}
        else:
            return [f"unknown action {action!r}"]
        for component in needed:
            if self.status(component) is not Status.PASS:
                c = next(c for c in self.checks if c.component == component)
                reasons.append(f"{component} {c.status.value}: {c.detail}")
        if self.data_state not in ok_data:
            reasons.append(f"data state {self.data_state.value} does not permit {action}")
        return reasons


def _journal_ok(journal: Journal) -> str | None:
    try:
        journal.verify()
        return None
    except Exception as exc:  # any doubt about a journal is an integrity failure
        return f"{type(exc).__name__}: {exc}"


def assess(system, loop, datasets: dict, company_journal: Journal, paused: bool, conflicts=None,
           extra_journals: tuple = ()) -> HealthReport:
    s = system
    checks: list[Check] = []

    # --- data -------------------------------------------------------------------------------------
    market = s.data_status in MARKET_EVIDENCE_STATUSES
    err = getattr(loop, "last_data_error", None)
    champions = [c for sym in s.symbols if (c := s.strategies.champion(sym, s.timeframe)) is not None]
    need_bars = max((c.lookback for c in champions), default=1)
    categories = {c.status for sym in s.symbols for c in s.store.series(s.provider.name, sym, s.timeframe)}
    historical = s.archive.conflicts()
    if conflicts is not None and conflicts.open:
        ids = sorted(conflicts.open)[:3]
        state, check = DataState.INVALID_DATA, Check("data", Status.FAIL,
                                                     f"DATA_CONFLICT: {len(conflicts.open)} open source conflict(s) {ids}")
    elif historical:
        state, check = DataState.INVALID_DATA, Check("data", Status.FAIL, f"DATA_CONFLICT: {len(historical)} provider "
                                                     "payload(s) contradicted recorded history (unacknowledged)")
    elif len(categories) > 1 or (categories and categories != {s.data_status}):
        state, check = DataState.MIXED_DATA, Check("data", Status.FAIL, f"categories {sorted(c.value for c in categories)}")
    elif isinstance(err, ModeMismatch) or (isinstance(err, DataIntegrityError) and err.code == "STATUS_MIX"):
        state, check = DataState.MIXED_DATA, Check("data", Status.FAIL, str(err)[:300])
    elif isinstance(err, (HistoricalConflictError, ProvenanceError, MalformedResponse)) or \
            (isinstance(err, DataIntegrityError) and err.code not in ("STALE", "EMPTY")):
        state, check = DataState.INVALID_DATA, Check("data", Status.FAIL, f"{type(err).__name__}: {str(err)[:300]}")
    elif err is not None or len(datasets) != len(s.symbols):
        why = f"{type(err).__name__}: {str(err)[:300]}" if err else "no fresh data"
        blocked = isinstance(err, (ProviderUnavailable, RateLimited)) or \
            (isinstance(err, DataIntegrityError) and err.code == "STALE")
        if market:
            state = DataState.REAL_DATA_UNAVAILABLE
        else:
            state = DataState.MOCK_DATA_ONLY
        check = Check("data", Status.BLOCKED if blocked else Status.NOT_READY, why)
    elif not market:
        state, check = DataState.MOCK_DATA_ONLY, Check(
            "data", Status.PASS, f"{s.data_status.value} data: mechanics only, never market evidence")
    else:
        try:
            for ds in datasets.values():
                s.archive.verify_market_provenance(ds)
            short = {sym: len(ds) for sym, ds in datasets.items() if len(ds) < need_bars}
            if short:
                state, check = DataState.INSUFFICIENT_REAL_DATA, Check("data", Status.NOT_READY, f"bars {short} < {need_bars}")
            else:
                state, check = DataState.REAL_DATA_AVAILABLE, Check("data", Status.PASS, "fresh, provenance verified")
        except ProvenanceError as exc:
            state, check = DataState.INVALID_DATA, Check("data", Status.FAIL, f"provenance: {exc}"[:300])
    checks.append(check)

    # --- research ---------------------------------------------------------------------------------
    problem = _journal_ok(s.research_journal)
    if problem is None:
        try:
            log = ResearchLog(s.research_journal)
            checks.append(Check("research", Status.PASS, f"{log.hypotheses_tested} hypotheses tested"))
        except Exception as exc:
            checks.append(Check("research", Status.FAIL, f"{type(exc).__name__}: {exc}"[:300]))
    else:
        checks.append(Check("research", Status.FAIL, problem[:300]))

    # --- memory (and the evidence registry it is bound to) ------------------------------------------
    problem = _journal_ok(s.memory.journal) or _journal_ok(s.evidence.journal)
    if problem is None:
        try:
            reloaded = MemoryStore(Journal(s.memory.journal.path, kind="memory", attrs=s.memory.journal.attrs,
                                           clock=s.clock, guard=s.guard), s.evidence)
            q = len(reloaded.quarantined)
            checks.append(Check("memory", Status.PASS, f"{len(reloaded)} entries; {q} quarantined (excluded)"))
        except Exception as exc:
            checks.append(Check("memory", Status.FAIL, f"{type(exc).__name__}: {exc}"[:300]))
    else:
        checks.append(Check("memory", Status.FAIL, problem[:300]))

    # --- risk -------------------------------------------------------------------------------------
    ks = s.kill_switch.state()
    if not isinstance(s.limits, RiskLimits):
        checks.append(Check("risk", Status.FAIL, "risk limits unreadable"))
    elif ks.get("at") is None and ks["engaged"]:
        checks.append(Check("risk", Status.FAIL, ks["reason"]))
    elif ks["engaged"]:
        checks.append(Check("risk", Status.BLOCKED, f"kill switch engaged: {ks['reason']}"))
    else:
        checks.append(Check("risk", Status.PASS, f"limits {s.limits.limits_hash[:12]}; kill switch released"))

    # --- execution --------------------------------------------------------------------------------
    ex = s.execution
    problem = _journal_ok(ex.journal)
    open_orders = [o.client_order_id for o in ex.orders.values() if o.status not in TERMINAL]
    if ex.mode is not OperatingMode.PAPER or LIVE_TRADING:
        checks.append(Check("execution", Status.FAIL, f"mode {ex.mode.value} / LIVE_TRADING={LIVE_TRADING}"))
    elif problem:
        checks.append(Check("execution", Status.FAIL, problem[:300]))
    elif ex.halted or open_orders or ex.recon_state is not ReconState.OK:
        checks.append(Check("execution", Status.BLOCKED,
                            f"halted={ex.halted_reason!r} unresolved={open_orders} reconciliation={ex.recon_state.value}"))
    else:
        checks.append(Check("execution", Status.PASS, "PAPER; reconciled; no unresolved orders"))

    # --- ledger (fills + decisions) ------------------------------------------------------------------
    problem = _journal_ok(s.decisions.journal) or _journal_ok(ex.journal)
    checks.append(Check("ledger", Status.FAIL, problem[:300]) if problem else
                  Check("ledger", Status.PASS, f"{len(ex.account.positions)} position records; journals intact"))

    # --- persistence (every other durable store: evidence, loop, learning, conflicts; state dir writable) ----
    journals = [s.evidence.journal, getattr(loop, "journal", None), *extra_journals]
    problem = next((f"{j.path.name}: {p}" for j in journals if j is not None for p in [_journal_ok(j)] if p), None)
    if problem is None and not os.access(s.state_dir, os.W_OK):
        problem = f"state directory {s.state_dir} is not writable"
    checks.append(Check("persistence", Status.FAIL, problem[:300]) if problem else
                  Check("persistence", Status.PASS, f"{sum(1 for j in journals if j is not None)} journals verify; "
                                                    "state directory writable"))

    # --- company ------------------------------------------------------------------------------------
    problem = _journal_ok(company_journal)
    if problem:
        checks.append(Check("company", Status.FAIL, problem[:300]))
    elif paused:
        checks.append(Check("company", Status.BLOCKED, "PAUSED"))
    else:
        checks.append(Check("company", Status.PASS, "active"))
    return HealthReport(tuple(checks), state)
