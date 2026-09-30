"""Daily company intelligence report — deterministic and read-only.

Answers, from journals and live deterministic state only: COMPANY (state, cycles, blocked reasons), STRATEGY
(champion, fingerprint, challengers), RESEARCH (hypotheses, experiments, failures, budget), LEARNING
(observations, recurring patterns, hypotheses generated, validated findings) and TRADING (paper trades,
wins/losses, P&L, drawdown). Every number carries its data label (MOCK / PAPER / REAL). Profitability is never
implied: without enough market-category trades the report states INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from ati.company import budget
from ati.ledger.journal import decode
from ati.market.models import MARKET_EVIDENCE_STATUSES
from ati.memory.store import MemoryKind
from ati.research.hypothesis import ResearchLog
from ati.strategies.registry import Lifecycle

MIN_TRADES_FOR_A_CLAIM = 30


def report(cp, day: datetime | None = None) -> dict:
    s = cp.s
    now = s.clock.now()
    day_s = (day or now).date().isoformat()
    has_data = any(s.store.series(s.provider.name, sym, s.timeframe) for sym in s.symbols)
    label = (f"{s.data_status.value} data / {s.execution.mode.value} execution" if has_data else
             f"NO DATA (system bound to {s.data_status.value}) / {s.execution.mode.value} execution")
    market = s.data_status in MARKET_EVIDENCE_STATUSES

    ends = [e for e in cp.journal.entries("cycle_end")]
    today = [e for e in ends if e.at[:10] == day_s]
    blocked = {}
    for e in ends:
        if e.payload["status"] in ("BLOCKED", "FAILED"):
            d = decode(e.payload)["detail"]
            reason = (d.get("reason") or "; ".join(d.get("reasons", [])) or "unspecified")[:120]
            blocked[reason] = blocked.get(reason, 0) + 1
    by_status: dict[str, int] = {}
    for e in ends:
        by_status[e.payload["status"]] = by_status.get(e.payload["status"], 0) + 1
    company = {"state": cp.state.value, "paused": cp.paused, "autonomy": cp.autonomy.name,
               "cycles_total": len(ends), "cycles_today": len(today), "cycles_by_status": by_status or "NONE",
               "blocked_or_failed_reasons": dict(sorted(blocked.items(), key=lambda kv: -kv[1])[:8]) or "NONE"}

    champion = s.strategies.champion()
    strategy = {"champion": champion.key if champion else "NONE",
                "fingerprint": champion.definition_hash if champion else "NOT_AVAILABLE",
                "challenger_count": len(s.strategies.keys(Lifecycle.CHALLENGER)),
                "rejected_strategies": len(s.strategies.keys(Lifecycle.REJECTED))}

    try:
        log = ResearchLog(s.research_journal)
        tested = {e["hypothesis_id"] for e in log.experiments}
        prereg = [e.payload["prereg"]["hypothesis_id"] for e in s.research_journal.entries("preregistration")]
        u = budget.usage(s, cp.journal, now)
        research = {"active_hypotheses": [h for h in prereg if ":" not in h and h not in tested] or "NONE",
                    "experiments": len(log.experiments),
                    "failed_experiments": sum(1 for e in log.experiments if str(getattr(e["verdict"], "value", e["verdict"])) != "PASS"),
                    "budget": {k: u[k] for k in ("root_hypotheses_tested", "experiments_run", "candidates_generated",
                                                 "holdout_evaluations", "tests_of_this_idea", "research_runs_today")},
                    "label": s.data_status.value}
    except Exception as exc:   # an unreadable research history is reported, never hidden
        research = {"status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"[:300]}

    classes: dict[str, int] = {}
    for c in cp.learning.candidates.values():
        k = cp.learning.pattern_class(c, s).value
        classes[k] = classes.get(k, 0) + 1
    learning = {"observations": len(cp.learning.outcomes), "candidates_by_class": classes or "NONE",
                "recurring_patterns": [cp.learning.fact(c) for c in cp.learning.candidates.values()
                                       if cp.learning.pattern_class(c, s).value == "RECURRING_PATTERN"][:8] or "NONE",
                "hypotheses_generated": sorted({c.hypothesis_id for c in cp.learning.candidates.values() if c.hypothesis_id})
                or "NONE",
                "validated_findings": [m.statement for m in s.memory.query(now, MemoryKind.VALIDATED_FINDING)]
                or "NONE",
                "label": s.data_status.value}

    pnls = []
    for e in cp.loop.journal.entries("trade_review"):
        f = decode(e.payload)["facts"]
        if None not in (f.get("entry_price"), f.get("exit_price"), f.get("qty")):
            pnls.append((f["exit_price"] - f["entry_price"]) * f["qty"] - (f.get("fees") or Decimal(0)))
    equity, peak, worst = Decimal(0), Decimal(0), Decimal(0)
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    enough = market and len(pnls) >= MIN_TRADES_FOR_A_CLAIM
    trading = {"label": label, "paper_trades_closed": len(pnls), "wins": sum(1 for p in pnls if p > 0),
               "losses": sum(1 for p in pnls if p <= 0), "net_pnl": str(sum(pnls, Decimal(0))),
               "max_drawdown_of_closed_trade_pnl": str(worst), "open_positions": sum(1 for p in s.execution.account.positions.values() if p.qty),
               "profitability": "UNDER EVALUATION (paper)" if enough else
               f"INSUFFICIENT_EVIDENCE ({len(pnls)} closed {s.data_status.value} paper trades; a claim needs "
               f">= {MIN_TRADES_FOR_A_CLAIM} on market data plus validated research)"}
    return {"report_date": day_s, "generated_at": now.isoformat(), "data_label": label, "COMPANY": company,
            "STRATEGY": strategy, "RESEARCH": research, "LEARNING": learning, "TRADING": trading}
