"""End-to-end paper loop on MOCK data with MOCK (scripted) reasoning."""

from datetime import timedelta
from decimal import Decimal

from ati.agent.loop import AutonomousLoop
from ati.decision.records import FinalDecision
from ati.execution.broker import OrderStatus
from ati.ledger.journal import decode
from ati.monitoring.status import build_status, render
from ati.risk.engine import ReconState
from tests.rig import entered, install_champion, make_system, run_until


def test_no_champion_means_no_trading(tmp_path):
    s, clock = make_system(tmp_path / "st")
    loop = AutonomousLoop(s)
    for _ in range(5):
        clock.advance(timedelta(hours=1))
        rep = loop.tick()
        assert rep.reconciliation == "OK" and "no champion" in rep.actions[0]
    assert not s.execution.orders


def test_full_cycle_entry_exit_review(tmp_path):
    s, clock = make_system(tmp_path / "st")
    install_champion(s)
    loop = AutonomousLoop(s)
    run_until(loop, clock, entered)
    rec = s.decisions.last
    assert rec.final_decision is FinalDecision.EXECUTE
    assert rec.thesis and rec.invalidation_condition and rec.available_information_cutoff <= rec.timestamp
    assert rec.data_status == "MOCK" and rec.mode == "PAPER"
    assert rec.approved_size > 0 and rec.expected_risk <= Decimal("100000") * s.limits.max_risk_per_trade_fraction
    order = next(iter(s.execution.orders.values()))
    assert order.decision_id == rec.decision_id and order.status is OrderStatus.FILLED
    assert "BTC/USD" in loop.stops
    # the decision was recorded before the order intent was created
    decision_at = next(s.decisions.journal.entries("decision")).at
    intent_at = next(s.execution.journal.entries("order_intent")).at
    assert decision_at <= intent_at

    run_until(loop, clock, lambda r: any("exit" in a for a in r.actions))
    assert s.execution.account.position_qty("BTC/USD") == 0
    assert list(loop.journal.entries("trade_review"))
    assert s.execution.reconcile().state is ReconState.OK
    status = build_status(s, loop)
    assert status["agent"]["operating_mode"].startswith("PAPER (LIVE_TRADING=False)")
    assert status["safety"]["reconciliation"] == "OK"
    assert "PORTFOLIO" in render(status)


def test_every_tick_is_bracketed_and_reconciled(tmp_path):
    s, clock = make_system(tmp_path / "st")
    install_champion(s)
    loop = AutonomousLoop(s)
    run_until(loop, clock, entered)
    starts = list(loop.journal.entries("tick_start"))
    ends = list(loop.journal.entries("tick_end"))
    assert len(starts) == len(ends) == loop.ticks
    assert len(list(s.execution.journal.entries("reconciliation"))) >= loop.ticks


def test_decisions_are_not_repeated_for_same_information(tmp_path):
    s, clock = make_system(tmp_path / "st")
    install_champion(s)
    loop = AutonomousLoop(s)
    run_until(loop, clock, entered)
    n = len(list(s.decisions.journal.entries("decision")))
    # re-running a tick without new information (same clock) produces no new decision
    loop.tick()
    assert len(list(s.decisions.journal.entries("decision"))) == n
