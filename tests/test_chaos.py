"""Chaos suite. Each test injects one failure and asserts the deterministic, safe outcome:
UNKNOWN → STOP → RECONCILE → VERIFY → RESUME. No test accepts an uncertain state as safe."""

import json
from datetime import timedelta
from decimal import Decimal

import pytest

from ati.agent.loop import AutonomousLoop
from ati.core.errors import ExecutionHalted, HistoricalConflictError, MalformedResponse, ProviderUnavailable
from ati.core.time import FixedClock
from ati.core.types import Side
from ati.decision.records import FinalDecision
from ati.execution.broker import OrderStatus
from ati.market.mock import MockProvider
from ati.market.models import Timeframe
from ati.risk.engine import ReconState
from tests.helpers import T0
from tests.rig import entered, install_champion, make_system, packet, run_until, script_with
from tests.test_execution_ledger import Rig


class FlakyProvider:
    """Wraps a provider; ``mode`` selects the injected failure."""

    def __init__(self, inner):
        self.inner, self.name, self.mode = inner, inner.name, None
        self.data_status = inner.data_status

    def fetch_candles(self, symbol, tf, start, end):
        if self.mode == "down":
            raise ProviderUnavailable("API disappeared (injected)")
        if self.mode == "malformed":
            raise MalformedResponse("corrupt payload (injected)")
        candles = self.inner.fetch_candles(symbol, tf, start, end)
        if self.mode == "rewrite_history":
            from dataclasses import replace
            candles[5] = replace(candles[5], close=candles[5].close + Decimal("1"),
                                 high=max(candles[5].high, candles[5].close + Decimal("1")))
        return candles


def loop_with_flaky(tmp_path):
    clock = FixedClock(T0 + timedelta(hours=400))
    flaky = FlakyProvider(MockProvider(11, clock, epoch=T0))
    s, _ = make_system(tmp_path / "st", provider=flaky, clock=clock)
    install_champion(s)
    return s, clock, flaky, AutonomousLoop(s)


# 1. API disappears ------------------------------------------------------------------------------
def test_api_disappears(tmp_path):
    s, clock, flaky, loop = loop_with_flaky(tmp_path)
    flaky.mode = "down"
    for _ in range(30):
        clock.advance(timedelta(hours=1))
        rep = loop.tick()
        assert not rep.data_ok and rep.stopped_at == "DATA" and "ProviderUnavailable" in rep.errors[0]
    assert not s.execution.orders and s.decisions.last is None


# 2. API returns corrupt data / rewrites history -------------------------------------------------
def test_corrupt_or_rewritten_data_fails_closed(tmp_path):
    s, clock, flaky, loop = loop_with_flaky(tmp_path)
    clock.advance(timedelta(hours=1))
    assert loop.tick().data_ok
    flaky.mode = "malformed"
    clock.advance(timedelta(hours=1))
    assert not loop.tick().data_ok
    flaky.mode = "rewrite_history"
    clock.advance(timedelta(hours=1))
    rep = loop.tick()
    assert not rep.data_ok and "HistoricalConflictError" in rep.errors[0]
    assert not s.execution.orders


# 3. Broker times out (not executed) -------------------------------------------------------------
def test_broker_timeout_before_execution(tmp_path):
    rig = Rig(tmp_path)
    rig.exe.reconcile()
    rig.broker.faults.add("timeout_before_execute")
    order = rig.exe.submit(rig.verdict())
    assert order.status is OrderStatus.UNKNOWN and rig.exe.halted
    with pytest.raises(ExecutionHalted):
        rig.exe.submit(rig.verdict())
    rig.broker.faults.clear()
    assert rig.exe.reconcile().state is ReconState.OK
    assert order.status is OrderStatus.NOT_FOUND and not rig.exe.halted
    assert rig.exe.account.position_qty("BTC/USD") == 0


# 4. Order succeeds but response disappears -------------------------------------------------------
def test_order_executed_but_response_lost(tmp_path):
    rig = Rig(tmp_path)
    rig.exe.reconcile()
    rig.broker.faults.add("timeout_after_execute")
    v = rig.verdict()
    order = rig.exe.submit(v)
    assert order.status is OrderStatus.UNKNOWN and rig.exe.halted
    assert rig.exe.account.position_qty("BTC/USD") == 0  # nothing assumed
    rig.broker.faults.clear()
    assert rig.exe.reconcile().state is ReconState.OK
    assert order.status is OrderStatus.FILLED and rig.exe.account.position_qty("BTC/USD") == v.qty
    # the same decision can never be sent again
    assert rig.exe.submit(v) is order and rig.broker.submissions == 1


# 5. Response says failure but broker executed ----------------------------------------------------
def test_false_rejection_detected(tmp_path):
    rig = Rig(tmp_path)
    rig.exe.reconcile()
    rig.broker.faults.add("false_reject_after_execute")
    v = rig.verdict()
    order = rig.exe.submit(v)
    assert order.status is OrderStatus.FILLED  # venue confirmed fills despite the "failure"
    assert rig.exe.halted and "reported failure but order executed" in rig.exe.halted_reason
    rig.broker.faults.clear()
    assert rig.exe.reconcile().state is ReconState.OK and not rig.exe.halted
    assert rig.exe.account.position_qty("BTC/USD") == v.qty


# 6. Database write fails -------------------------------------------------------------------------
def test_journal_failure_before_submission_sends_nothing(tmp_path, monkeypatch):
    rig = Rig(tmp_path)
    rig.exe.reconcile()
    v = rig.verdict()
    monkeypatch.setattr(rig.exe.journal, "_write_line", lambda line: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(ExecutionHalted):
        rig.exe.submit(v)
    assert rig.broker.submissions == 0 and rig.exe.halted


def test_journal_failure_after_execution_recovers_via_reconciliation(tmp_path, monkeypatch):
    rig = Rig(tmp_path)
    rig.exe.reconcile()
    v = rig.verdict()
    original = rig.exe.journal._write_line
    calls = {"n": 0}

    def fail_after_intent(line):
        calls["n"] += 1
        if '"type":"fill"' in line:
            raise OSError("disk full")
        original(line)

    monkeypatch.setattr(rig.exe.journal, "_write_line", fail_after_intent)
    with pytest.raises(ExecutionHalted):
        rig.exe.submit(v)
    assert rig.broker.submissions == 1  # the venue did execute
    monkeypatch.undo()
    restarted = rig.new_engine()  # process restart: rebuild from the durable journal
    assert restarted.halted and restarted.account.position_qty("BTC/USD") == 0
    assert restarted.reconcile().state is ReconState.OK
    assert restarted.account.position_qty("BTC/USD") == v.qty and not restarted.halted


# 7. Claude returns malformed output --------------------------------------------------------------
@pytest.mark.parametrize("output", ["I think we should buy!", '{"action": "BUY_NOW"}', "", "{" * 50])
def test_malformed_reasoning_output_never_trades(tmp_path, output):
    s, clock = make_system(tmp_path / "st", script=script_with(output))
    install_champion(s)
    loop = AutonomousLoop(s)
    run_until(loop, clock, lambda r: s.decisions.last is not None)
    assert s.decisions.last.final_decision is FinalDecision.INVALID_REASONING_OUTPUT
    assert not s.execution.orders


# 8. Claude proposes impossible size ---------------------------------------------------------------
def test_impossible_size_is_capped_by_risk_engine(tmp_path):
    def huge(prompt):
        p = packet(prompt)
        return json.dumps({"action": "PROPOSE_TRADE", "symbol": p["symbol"], "side": "BUY",
                           "strategy_key": p["strategy"]["key"], "entry_price": p["recent_closes"][-1],
                           "stop_price": p["signal"]["stop"], "proposed_qty": "1000000000", "thesis": "t",
                           "invalidation_condition": "i", "confidence": 1.0})
    s, clock = make_system(tmp_path / "st", script=script_with(huge))
    install_champion(s)
    loop = AutonomousLoop(s)
    run_until(loop, clock, entered)
    rec = s.decisions.last
    assert rec.proposed_size == Decimal("1000000000")
    assert rec.approved_size < Decimal("10")
    assert rec.expected_risk <= Decimal("100000") * s.limits.max_risk_per_trade_fraction
    order = next(iter(s.execution.orders.values()))
    assert order.intended_qty == rec.approved_size


def test_claude_cannot_loosen_stop(tmp_path):
    def loose(prompt):
        p = packet(prompt)
        stop = Decimal(p["recent_closes"][-1]) * Decimal("0.85")
        return json.dumps({"action": "PROPOSE_TRADE", "symbol": p["symbol"], "side": "BUY",
                           "strategy_key": p["strategy"]["key"], "entry_price": p["recent_closes"][-1],
                           "stop_price": str(stop), "thesis": "t", "invalidation_condition": "i", "confidence": 0.9})
    s, clock = make_system(tmp_path / "st", script=script_with(loose))
    install_champion(s)
    loop = AutonomousLoop(s)
    run_until(loop, clock, entered)
    strategy_stop = Decimal(s.decisions.last.signal.split("stop=")[1].split(" ")[0])
    assert loop.stops["BTC/USD"][0] == strategy_stop


# 9. Market data becomes stale -----------------------------------------------------------------------
def test_stale_data_blocks_trading(tmp_path):
    clock = FixedClock(T0 + timedelta(hours=400))
    lagging = MockProvider(11, FixedClock(T0 + timedelta(hours=400)), epoch=T0)  # its clock never advances
    s, _ = make_system(tmp_path / "st", provider=lagging, clock=clock)
    install_champion(s)
    loop = AutonomousLoop(s)
    clock.advance(timedelta(hours=5))
    rep = loop.tick()
    assert not rep.data_ok and any("STALE" in e for e in rep.errors)
    assert rep.stopped_at == "DATA" and not s.execution.orders


# 10. Account balance unexpectedly changes ----------------------------------------------------------
def test_unexpected_balance_change_halts_until_explained(tmp_path):
    rig = Rig(tmp_path)
    rig.exe.reconcile()
    rig.broker.external_cash_change(Decimal("-5000"))
    result = rig.exe.reconcile()
    assert result.state is ReconState.MISMATCH and rig.exe.halted
    v = rig.verdict()
    assert not v.approved
    with pytest.raises(PermissionError):
        rig.exe.record_external_adjustment(Decimal("-5000"), "withdrawal", "ok")
    rig.exe.record_external_adjustment(Decimal("-5000"), "verified withdrawal", "OPERATOR: external adjustment verified")
    assert rig.exe.reconcile().state is ReconState.OK and not rig.exe.halted
    after = rig.verdict()
    assert {"reconciliation", "account_state_known"}.isdisjoint(c.name for c in after.failed)
    # Known limitation (fail-safe direction): a withdrawal is not rebased out of the day's P&L, so the
    # risk engine sees it as a 5% daily loss and refuses new risk until the next day / operator review.
    assert not after.approved and "daily_loss_limit" in {c.name for c in after.failed}


# 11. Network disconnects during execution ------------------------------------------------------------
def test_network_loss_during_execution(tmp_path):
    rig = Rig(tmp_path)
    rig.exe.reconcile()
    # the venue passes its health check, then the connection drops during submission (a venue that is already
    # down fails the pre-submission health gate and nothing is sent — see test_broker_execution_boundary.py)
    rig.broker.faults.add("disconnect_on_submit")
    order = rig.exe.submit(rig.verdict())
    assert order.status is OrderStatus.UNKNOWN and rig.exe.halted
    rig.broker.faults.add("unavailable")                    # and stays down
    assert rig.exe.reconcile().state is ReconState.UNKNOWN  # still cannot see the venue: stay stopped
    assert rig.exe.halted
    rig.broker.faults.clear()
    assert rig.exe.reconcile().state is ReconState.OK and not rig.exe.halted


# 12. Process restarts after a fill ----------------------------------------------------------------
def test_process_restart_after_fill(tmp_path):
    state = tmp_path / "st"
    s, clock = make_system(state)
    install_champion(s)
    loop = AutonomousLoop(s)
    run_until(loop, clock, entered)
    qty = s.execution.account.position_qty("BTC/USD")
    stop = loop.stops["BTC/USD"]
    assert qty > 0

    # new process: new objects, new approval key, same durable state
    s2, _ = make_system(state, clock=clock, provider=s.provider)
    assert s2.execution.account.position_qty("BTC/USD") == qty
    assert s2.strategies.champion("BTC/USD", Timeframe.H1).key == s.strategies.champion("BTC/USD", Timeframe.H1).key
    loop2 = AutonomousLoop(s2)
    assert loop2.stops["BTC/USD"] == stop and loop2.ticks == loop.ticks
    clock.advance(timedelta(hours=1))
    rep = loop2.tick()
    assert rep.reconciliation == "OK" and not rep.resumed_after_interruption
    assert s2.execution.account.cash == s.execution.account.cash


def test_crash_mid_tick_is_detected_and_reconciled_first(tmp_path):
    state = tmp_path / "st"
    s, clock = make_system(state)
    install_champion(s)
    loop = AutonomousLoop(s)
    clock.advance(timedelta(hours=1))
    loop.tick()
    loop._log("tick_start", {"tick": loop.ticks + 1, "resumed_after_interruption": False})  # crash before tick_end
    s2, _ = make_system(state, clock=clock, provider=s.provider)
    loop2 = AutonomousLoop(s2)
    assert loop2.open_tick
    clock.advance(timedelta(hours=1))
    rep = loop2.tick()
    assert rep.resumed_after_interruption and rep.stages[:3] == ["HEALTH", "DATA", "RECONCILE"]
    assert rep.reconciliation == "OK"


def test_approval_from_previous_process_rejected(tmp_path):
    from ati.core.errors import ApprovalInvalid
    rig = Rig(tmp_path)
    rig.exe.reconcile()
    v = rig.verdict()
    from ati.risk.engine import ApprovalAuthority
    rig.exe.authority = ApprovalAuthority()  # restarted process has a fresh key
    with pytest.raises(ApprovalInvalid):
        rig.exe.submit(v)
    assert rig.broker.submissions == 0


def test_kill_switch_file_corruption_blocks_new_risk(tmp_path):
    rig = Rig(tmp_path)
    rig.exe.reconcile()
    (tmp_path / "kill.json").write_text("corrupted")
    v = rig.verdict()
    assert not v.approved and "kill_switch" in {c.name for c in v.failed}


def test_unknown_account_state_refuses_proposals(tmp_path):
    rig = Rig(tmp_path)  # never reconciled
    v = rig.verdict()
    assert not v.approved
    sell = rig.verdict(side=Side.SELL)
    assert not sell.approved
