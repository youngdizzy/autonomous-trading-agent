"""Company readiness report — read-only.

Assembled from the journals and live deterministic state. It fetches nothing, appends nothing, and never
upgrades a status: anything that has not happened is NOT_AVAILABLE / NOT_RUN, anything too thin is
INSUFFICIENT_EVIDENCE. It exists so an operator (or a future scheduler) can see, in one place, why the company
is or is not ready for more responsibility.
"""

from __future__ import annotations

from ati.company import autonomy, budget, factory
from ati.company.health import assess
from ati.config import LIVE_TRADING
from ati.data.dataset import Dataset
from ati.ledger.journal import decode
from ati.market.models import MARKET_EVIDENCE_STATUSES
from ati.memory.store import MemoryKind
from ati.research.adversarial import required_t
from ati.research.hypothesis import ResearchLog


def _datasets(system) -> dict:
    s = system
    out = {}
    for sym in s.symbols:
        series = s.store.series(s.provider.name, sym, s.timeframe)
        if series:
            out[sym] = Dataset.build(series, data_version="readiness-view",
                                     realization=getattr(s.provider, "realization", "observed"))
    return out


def report(cp) -> dict:
    s = cp.s
    now = s.clock.now()
    datasets = _datasets(s)
    health = assess(s, cp.loop, datasets, cp.journal, cp.paused, cp.conflicts,
                    extra_journals=(cp.learning.journal, cp.conflicts.journal))
    try:
        log = ResearchLog(s.research_journal)
    except Exception as exc:   # an unreadable research history is reported, never hidden
        log = None
        research_error = f"{type(exc).__name__}: {exc}"[:300]
    champion = s.strategies.champion()

    from ati.market.accumulate import SERIES
    from ati.market.health import scorecard, verify_sealed_holdouts

    from ati.market.health import readiness_table
    from ati.research.protocols import REGISTRY

    data = {"readiness_table": readiness_table(s, SERIES),
            "protocols": [p.describe() for p in REGISTRY.values()],
            "scorecard": scorecard(s, SERIES, champion.definition_hash if champion else None),
            "sealed_holdouts": verify_sealed_holdouts(s),
            "category": s.data_status.value, "market_evidence": s.data_status in MARKET_EVIDENCE_STATUSES,
            "provider": s.provider.name, "health": health.as_dict()["checks"]["data"], "data_state": health.data_state.value,
            "open_conflicts": sorted(cp.conflicts.open),
            "datasets": {sym: {"dataset_hash": ds.dataset_id, "bars": len(ds),
                               "first": ds.candles[0].open_time.isoformat(), "last_close": ds.candles[-1].close_time.isoformat(),
                               "status": ds.identity.status.value} for sym, ds in datasets.items()} or "NOT_AVAILABLE"}

    strategy = {"champion": {"key": champion.key, "fingerprint": champion.definition_hash, "version": champion.version,
                             "params": champion.param_dict} if champion else "NOT_AVAILABLE",
                "registry": {k: s.strategies.state(k).value for k in s.strategies.keys()} or "NOT_AVAILABLE",
                "candidates": [factory.stages(s, e.payload["fingerprint"])
                               for e in s.research_journal.entries("candidate_lineage")] or "NOT_AVAILABLE"}

    if log is None:
        research = {"status": "FAIL", "detail": research_error}
    else:
        tested = {e["hypothesis_id"] for e in log.experiments}
        roots = sorted({h.split(":", 1)[0] for h in tested})
        prereg = [e.payload["prereg"]["hypothesis_id"] for e in s.research_journal.entries("preregistration")]
        u = budget.usage(s, cp.journal, now)
        research = {
            "active_hypotheses": [h for h in prereg if ":" not in h and h not in tested] or "NONE",
            "completed_experiments": len(log.experiments),
            "rejected": [h for h in roots if log.status(h) in ("TESTED:FAIL", "TESTED:INSUFFICIENT_EVIDENCE")] or "NONE",
            "multiple_testing": {"root_hypotheses_tested": log.hypotheses_tested,
                                 "bonferroni_t_for_next_single_config_test": round(required_t(log.hypotheses_tested + 1, 0.05), 3),
                                 "budget": {k: v for k, v in u.items() if k not in ("policy",)}}}

    counts: dict[str, int] = {}
    for c in cp.learning.candidates.values():
        counts[c.state.value] = counts.get(c.state.value, 0) + 1
    learning = {"outcomes": len(cp.learning.outcomes), "candidates_by_state": counts or "NONE",
                "recurring_mistakes": [c["statement"] for c in cp.learning.summary(s, limit=50)
                                       if c["class"] == "RECURRING_PATTERN"] or "NONE",
                "supported_findings": [c.statement for c in cp.learning.candidates.values()
                                       if c.state.value == "SUPPORTED"] or "NONE",
                "validated_findings": [m.statement for m in s.memory.query(now, MemoryKind.VALIDATED_FINDING)]
                or "NONE (requires holdout PASS and approved promotion on market data)"}

    latest = strategy["candidates"][-1] if isinstance(strategy["candidates"], list) else None
    validation = {"latest_candidate": latest["fingerprint"] if latest else "NOT_AVAILABLE",
                  "backtest": latest["stages"]["DEVELOPMENT"] if latest else "NOT_RUN",
                  "wfo": latest["stages"]["WFO"] if latest else "NOT_RUN",
                  "robustness": latest["stages"]["ROBUSTNESS"] if latest else "NOT_RUN",
                  "adversarial": latest["stages"]["ADVERSARIAL"] if latest else "NOT_RUN",
                  "holdout": latest["stages"]["HOLDOUT"] if latest else "NOT_RUN",
                  "promotion": latest["stages"]["VALIDATION"] if latest else "NOT_RUN"}

    violations = []
    for e in s.decisions.journal.entries("decision"):
        rec = decode(e.payload)["record"]
        failed = [c["name"] for c in rec["risk_constraints"] if not c["passed"]]
        if failed:
            violations.append({"decision_id": rec["decision_id"], "failed": failed})
    risk = {"state": health.as_dict()["checks"]["risk"], "kill_switch": s.kill_switch.state(),
            "constraints": {k: str(v) for k, v in s.limits.__dict__.items() if k != "allowed_data"},
            "limits_hash": s.limits.limits_hash, "owner": "code (RiskLimits); Claude cannot change",
            "blocked_by_constraint": violations[-10:] or "NONE"}

    ex = s.execution
    failures = [o.status.value for o in ex.orders.values() if o.status.value in ("REJECTED", "UNKNOWN", "NOT_FOUND")]
    execution = {"mode": ex.mode.value, "live_trading": LIVE_TRADING, "autonomy": cp.autonomy.name,
                 "maximum_autonomy": autonomy.maximum_permitted().name,
                 "reconciliation": ex.recon_state.value, "halted": ex.halted_reason or "NO",
                 "orders": len(ex.orders), "failures": failures or "NONE",
                 "health": health.as_dict()["checks"]["execution"]}
    return {"generated_at": now.isoformat(), "company": {"state": cp.state.value, "paused": cp.paused},
            "DATA": data, "STRATEGY": strategy, "RESEARCH": research, "LEARNING": learning,
            "VALIDATION": validation, "RISK": risk, "EXECUTION": execution}
