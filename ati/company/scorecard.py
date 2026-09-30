"""Company scorecard: independent dimensions, each with its own state and evidence. There is deliberately
no aggregate score — a strong dimension can never compensate for a failing one.

States: PASS | WEAK | FAIL | NOT_AVAILABLE | INSUFFICIENT_EVIDENCE. Everything is derived from journals and
live deterministic state; nothing here is Claude's opinion.
"""

from __future__ import annotations

from ati.company import budget
from ati.company.health import DataState, HealthReport, Status
from ati.ledger.journal import decode
from ati.market.models import MARKET_EVIDENCE_STATUSES
from ati.research.hypothesis import ResearchLog

DIMENSIONS = ("DATA_QUALITY", "RESEARCH_QUALITY", "EVIDENCE_QUALITY", "REPRODUCIBILITY", "ROBUSTNESS",
              "RISK_DISCIPLINE", "EXECUTION_DISCIPLINE", "LEARNING_QUALITY", "OVERFITTING_RISK", "SYSTEM_RELIABILITY")


def _d(state: str, detail: str) -> dict:
    return {"state": state, "detail": detail[:300]}


def build(system, health: HealthReport, company_journal, learning=None) -> dict:
    s = system
    card: dict[str, dict] = {}
    market = s.data_status in MARKET_EVIDENCE_STATUSES
    ds = health.data_state
    card["DATA_QUALITY"] = _d("PASS" if ds is DataState.REAL_DATA_AVAILABLE else
                              "NOT_AVAILABLE" if ds in (DataState.REAL_DATA_UNAVAILABLE, DataState.MOCK_DATA_ONLY) else
                              "INSUFFICIENT_EVIDENCE" if ds is DataState.INSUFFICIENT_REAL_DATA else "FAIL",
                              f"{ds.value}; data category {s.data_status.value}")
    try:
        log = ResearchLog(s.research_journal)
    except Exception as exc:  # unreadable research history is a failure, never "no research"
        log = None
        card["RESEARCH_QUALITY"] = _d("FAIL", f"research journal unreadable: {type(exc).__name__}")
    if log is not None:
        n = len(log.experiments)
        card["RESEARCH_QUALITY"] = _d("INSUFFICIENT_EVIDENCE" if n == 0 else "PASS",
                                      f"{n} pre-registered experiments over {log.hypotheses_tested} hypotheses; "
                                      "all criteria locked before testing")
    card["EVIDENCE_QUALITY"] = _d("PASS" if market else "NOT_AVAILABLE",
                                  "market-category evidence" if market else
                                  f"{s.data_status.value} evidence only: mechanics, never market evidence")
    bad = [name for name, j in (("research", s.research_journal), ("decisions", s.decisions.journal),
                                ("memory", s.memory.journal), ("company", company_journal)) if not _verifies(j)]
    card["REPRODUCIBILITY"] = _d("FAIL" if bad else "PASS",
                                 f"journals failing verification: {bad}" if bad else "hash chains verify; datasets content-addressed")
    advs = [decode(e.payload)["report"] for e in s.research_journal.entries("adversarial_report")]
    if not advs:
        card["ROBUSTNESS"] = _d("NOT_AVAILABLE", "no adversarial review has run")
    else:
        blocking = sum(1 for a in advs if any(o["verdict"] in ("FAIL", "INSUFFICIENT_EVIDENCE") for o in a["objections"]))
        card["ROBUSTNESS"] = _d("WEAK" if blocking else ("PASS" if market else "INSUFFICIENT_EVIDENCE"),
                                f"{blocking}/{len(advs)} adversarial reviews blocking"
                                + ("" if market else "; non-market data"))
    risk = health.status("risk")
    card["RISK_DISCIPLINE"] = _d("PASS" if risk is Status.PASS else "FAIL" if risk is Status.FAIL else "WEAK",
                                 f"risk {risk.value}; limits {s.limits.limits_hash[:12]} (code-owned)")
    ex = health.status("execution")
    card["EXECUTION_DISCIPLINE"] = _d("PASS" if ex is Status.PASS else "FAIL" if ex is Status.FAIL else "WEAK",
                                      f"execution {ex.value}; PAPER only")
    if learning is None:
        card["LEARNING_QUALITY"] = _d("NOT_AVAILABLE", "learning ledger not attached")
    else:
        states: dict[str, int] = {}
        for c in learning.candidates.values():
            states[c.state.value] = states.get(c.state.value, 0) + 1
        card["LEARNING_QUALITY"] = _d("INSUFFICIENT_EVIDENCE" if not learning.candidates else "PASS",
                                      f"{len(learning.outcomes)} outcomes; candidates by state {states}; "
                                      "no candidate changes a rule")
    if log is None:
        card["OVERFITTING_RISK"] = _d("FAIL", "multiple-testing exposure unknown: research journal unreadable")
    else:
        u = budget.usage(s, company_journal, s.clock.now())
        frac = u["root_hypotheses_tested"] / budget.POLICY.max_root_hypotheses
        card["OVERFITTING_RISK"] = _d("FAIL" if frac >= 1 else "WEAK" if frac >= 0.5 else "PASS",
                                      f"{u['root_hypotheses_tested']}/{budget.POLICY.max_root_hypotheses} hypotheses; "
                                      f"{u['holdout_evaluations']}/{budget.POLICY.max_holdout_evaluations} holdouts used")
    ends = [e.payload["status"] for e in company_journal.entries("cycle_end")]
    failed = sum(1 for x in ends if x == "FAILED")
    fails = [c.component for c in health.checks if c.status is Status.FAIL]
    card["SYSTEM_RELIABILITY"] = _d("FAIL" if fails else "INSUFFICIENT_EVIDENCE" if not ends else
                                    "WEAK" if failed * 4 > len(ends) else "PASS",
                                    f"failing components {fails}; {failed}/{len(ends)} cycles FAILED")
    return card


def _verifies(journal) -> bool:
    try:
        journal.verify()
        return True
    except Exception:
        return False
