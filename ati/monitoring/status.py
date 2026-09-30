"""Compact operational status. Plain facts from the journals and live components; no decoration."""

from __future__ import annotations

from decimal import Decimal

from ati.config import LIVE_TRADING
from ati.execution.broker import TERMINAL, OrderStatus
from ati.execution.policy import live_state
from ati.ledger.journal import decode
from ati.strategies.registry import Lifecycle


def build_status(system, loop=None) -> dict:
    s = system
    series = {sym: s.store.series(s.provider.name, sym, s.timeframe) for sym in s.symbols}
    marks = {sym: c[-1].close for sym, c in series.items() if c}
    try:
        equity = s.execution.account.equity(marks)
    except KeyError:
        equity = None
    exposure = sum((p.qty * marks.get(sym, Decimal(0)) for sym, p in s.execution.account.positions.items()), Decimal(0))
    day_start = loop.day_start_equity if loop else None
    peak = loop.peak_equity if loop else None
    experiments = list(s.research_journal.entries("experiment"))
    preregs = list(s.research_journal.entries("preregistration"))
    last_decision = s.decisions.last
    if last_decision is None:
        entries = list(s.decisions.journal.entries("decision"))
        last_decision_view = decode(entries[-1].payload)["record"] if entries else None
    else:
        last_decision_view = {"decision_id": last_decision.decision_id, "final_decision": last_decision.final_decision.value,
                              "symbol": last_decision.symbol, "reason": last_decision.reason}
    freshness = {sym: (s.clock.now() - c[-1].close_time) if c else None for sym, c in series.items()}
    return {
        "agent": {
            "operating_mode": f"{s.mode.value} (LIVE_TRADING={LIVE_TRADING})",
            "data_status": s.data_status.value,
            "reasoning_client": getattr(s.reasoning, "label", "unknown"),
            "current_state": loop.current_stage if loop else "IDLE",
            "ticks": loop.ticks if loop else 0,
            "last_decision": last_decision_view and {k: last_decision_view.get(k) for k in ("decision_id", "final_decision", "symbol", "reason")},
            "last_action": loop.last_action if loop else "none",
        },
        "portfolio": {
            "equity": equity, "cash": s.execution.account.cash,
            "positions": {sym: p.qty for sym, p in s.execution.account.positions.items() if p.qty},
            "exposure": exposure,
            "daily_pnl": (equity - day_start) if equity is not None and day_start is not None else None,
            "drawdown": (1 - equity / peak) if equity is not None and peak else None,
            "realized_pnl": s.execution.account.realized_pnl, "fees_paid": s.execution.account.fees_paid,
        },
        "research": {
            "current_hypothesis": decode(preregs[-1].payload)["prereg"]["statement"] if preregs else None,
            "latest_experiment": ({k: decode(experiments[-1].payload)[k] for k in ("hypothesis_id", "stage", "verdict")}
                                  if experiments else None),
            "champions": {f"{sym} {s.timeframe.value}": (c.registry_key if (c := s.strategies.champion(sym, s.timeframe))
                                                         else None) for sym in s.symbols},
            "challengers": {sym: s.strategies.keys(Lifecycle.CHALLENGER, sym) for sym in s.symbols},
            "rejected": {sym: s.strategies.keys(Lifecycle.REJECTED, sym) for sym in s.symbols},
            "memory_entries": len(s.memory),
        },
        "safety": {
            "kill_switch": s.kill_switch.state(),
            "reconciliation": s.execution.recon_state.value,
            "execution_halted": s.execution.halted_reason,
            "data_freshness": {k: (str(v) if v is not None else "NO DATA") for k, v in freshness.items()},
            "open_orders": [o.client_order_id for o in s.execution.orders.values() if o.status not in TERMINAL],
        },
        "execution": execution_view(s),
    }


def execution_view(system) -> dict:
    """Broker execution boundary, read-only. The last decision is shown as three separate facts — what Claude
    proposed, what TradeTown authorized, what the broker executed — so a proposal is never displayed as a trade."""
    s = system
    ex = s.execution
    try:
        health = ex.broker.health_check()
        connection = "HEALTHY" if health.ok else "UNHEALTHY: " + "; ".join(f"{n}: {d}" for n, ok, d in health.checks if not ok)
    except Exception as exc:                               # a status view never raises on a broken venue
        connection = f"UNHEALTHY: {type(exc).__name__}"
    orders = list(ex.orders.values())
    fills = [f for o in orders for f in o.fills]
    last_order = orders[-1] if orders else None
    rejected = [o for o in orders if o.status is OrderStatus.REJECTED]
    decisions = list(s.decisions.journal.entries("decision"))
    last = decode(decisions[-1].payload)["record"] if decisions else None
    chain = None
    if last is not None:
        linked = [o for o in orders if o.decision_id == last["decision_id"]]
        chain = {"claude_proposal": ({"thesis": last["thesis"][:160], "proposed_size": last["proposed_size"],
                                      "confidence": last["confidence"]} if last["thesis"] else "none (deterministic)"),
                 "tradetown_authorization": {"final_decision": last["final_decision"],
                                             "approved_size": last["approved_size"], "reason": last["reason"][:160]},
                 "broker_execution": ({"order": linked[0].client_order_id, "status": linked[0].status.value,
                                       "filled": linked[0].filled_qty} if linked else "no order sent")}
    return {"broker": type(ex.broker).__name__, "connection_status": connection,
            "account": {"cash": ex.account.cash, "reconciliation": ex.recon_state.value,
                        "last_reconciled_at": ex.last_reconciled_at},
            "execution_mode": ex.policy.mode.value, "venue_mode": ex.mode.value, "live": live_state(),
            "risk_status": "HALTED: " + ex.halted_reason if ex.halted else "OK",
            "kill_switch": s.kill_switch.state(),
            "open_positions": {sym: p.qty for sym, p in ex.account.positions.items() if p.qty},
            "open_orders": [{"order": o.client_order_id, "symbol": o.symbol, "status": o.status.value}
                            for o in orders if o.status not in TERMINAL],
            "last_order": last_order and {"order": last_order.client_order_id, "status": last_order.status.value,
                                          "symbol": last_order.symbol, "side": last_order.side.value,
                                          "qty": last_order.intended_qty},
            "last_fill": {"fill": fills[-1].fill_id, "qty": fills[-1].qty, "price": fills[-1].price} if fills else None,
            "last_rejection": (ex.last_refusal and {k: ex.last_refusal[k] for k in ("decision_id", "rule", "reason")})
            or ({"order": rejected[-1].client_order_id, "reason": rejected[-1].message} if rejected else None),
            "last_decision": chain}


def render(status: dict) -> str:
    lines = []
    for section, values in status.items():
        lines.append(section.upper())
        for key, value in values.items():
            lines.append(f"  {key:<22} {value}")
    return "\n".join(lines)
