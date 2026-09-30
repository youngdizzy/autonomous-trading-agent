"""Broker Execution Boundary 1.0 — Claude proposes, TradeTown authorizes, the broker executes, the journal remembers.

Everything runs against the deterministic ``MockBrokerAdapter`` (scripted acknowledgements, fills, rejections,
timeouts, cancels, injected external activity) or the existing ``PaperBroker``. No real broker, credential or
network is used; LIVE_TRADING stays False. Market data is MOCK.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ati.agent.schema import COMPANY_ACTIONS
from ati.config import LIVE_TRADING, OperatingMode
from ati.core.errors import (ApprovalInvalid, ExecutionHalted, LiveTradingDisabled, ModeMismatch, SecretLeakError,
                             SubmissionRefused)
from ati.core.time import FixedClock
from ati.core.types import Side
from ati.execution.broker import OrderStatus
from ati.execution.engine import ExecutionEngine, client_order_id
from ati.execution.live import create_live_broker
from ati.execution.mock import MockBrokerAdapter
from ati.execution.policy import APPROVAL_ACK, ExecutionMode, ExecutionPolicy, live_state
from ati.ledger.journal import decode
from ati.market.models import DataStatus, Timeframe
from ati.market.universe import default_universe
from ati.monitoring.status import execution_view
from ati.risk.engine import ApprovalAuthority, MarketSnapshot, OrderRequest, ReconState, RiskEngine, RiskLimits
from ati.risk.killswitch import KillSwitch
from ati.security.secrets import SecretGuard, SecretValue
from tests.helpers import T0
from tests.test_company_control import plane, reply, trade_payload, until_signal
from tests.rig import install_champion

NOW = T0 + timedelta(days=3)
CASH = Decimal("100000")
PRICE = Decimal("30000")
SECRET = "mock-broker-secret-7f3a9c1e5b"
ROOT = Path(__file__).resolve().parents[1]


class Rig:
    """Production wiring (risk engine → signed approval → execution engine → adapter) around the mock venue."""

    def __init__(self, tmp: Path, policy: ExecutionPolicy = ExecutionPolicy()):
        self.tmp = tmp
        self.clock = FixedClock(NOW)
        self.guard = SecretGuard()
        self.secret = self.guard.register(SecretValue("MOCK_BROKER_SECRET", SECRET))
        self.broker = MockBrokerAdapter(self.clock, {"BTC/USD": PRICE, "ETH/USD": Decimal("2000")}, CASH, self.secret)
        self.kill = KillSwitch(tmp / "kill.json", self.clock)
        self.authority = ApprovalAuthority()
        self.risk = RiskEngine(RiskLimits(), default_universe(), self.kill, self.authority, self.clock)
        self.policy = policy
        self.exe = self.engine()
        self.n = 0

    def engine(self) -> ExecutionEngine:
        return ExecutionEngine(self.broker, self.tmp / "exec.jsonl", self.authority, self.clock, mode=OperatingMode.PAPER,
                               data_status=DataStatus.MOCK, initial_cash=CASH, kill_switch=self.kill, policy=self.policy,
                               guard=self.guard)

    def market(self, symbol="BTC/USD", age=timedelta(minutes=5), price=None):
        return MarketSnapshot(symbol, Timeframe.H1, price or self.broker.prices[symbol], self.clock.now() - age,
                              DataStatus.MOCK, Decimal("500"), Decimal("0.0002"), Decimal("0.0003"))

    def verdict(self, side=Side.BUY, symbol="BTC/USD", proposed_qty=None, age=timedelta(minutes=5), decision_id=None):
        self.n += 1
        price = self.broker.prices[symbol]
        pf = self.exe.portfolio_snapshot({s: p for s, p in self.broker.prices.items()}, CASH, CASH)
        req = OrderRequest(decision_id or f"d{self.n}", "trend@v1", symbol, side, price,
                           price * Decimal("0.98") if side is Side.BUY else None, proposed_qty)
        return self.risk.evaluate(req, pf, self.market(symbol, age))

    def records(self, type_=None):
        return [(e.type, decode(e.payload)) for e in self.exe.journal.entries(type_)]

    def statuses(self, coid):
        return [p["status"] for t, p in self.records("order_status") if p["client_order_id"] == coid]


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    assert r.exe.reconcile().state is ReconState.OK
    return r


def refused(rig, rule):
    last = rig.records("submission_refused")[-1][1]
    assert last["rule"] == rule, last
    assert {"decision_id", "rule", "reason", "at", "state", "execution_mode", "authorized_qty"} <= set(last)
    return last


# ----------------------------------------------------------------------------------------------- A: happy paths
class TestPaperPath:
    def test_A_claude_proposal_reaches_the_paper_venue_through_the_existing_path(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st")
        install_champion(s)
        until_signal(cp, s, clock)
        s.reasoning.script["company"] = reply("TRADE_PROPOSAL", trade_payload)
        out = cp.run_cycle()
        assert out.status == "COMPLETED" and out.detail["final_decision"] == "EXECUTE", out.detail
        order = next(iter(s.execution.orders.values()))
        assert order.status is OrderStatus.FILLED and order.mode is OperatingMode.PAPER
        view = execution_view(s)["last_decision"]                       # three facts, never "Claude traded"
        assert view["claude_proposal"]["thesis"].startswith("[MOCK]")
        assert view["tradetown_authorization"]["final_decision"] == "EXECUTE"
        assert view["broker_execution"]["status"] == "FILLED"

    def test_O_full_fill_updates_position_and_reconciles(self, rig):
        v = rig.verdict()
        order = rig.exe.submit(v)
        assert order.status is OrderStatus.FILLED and order.broker_order_id.startswith("B")
        assert rig.exe.account.position_qty("BTC/USD") == v.qty == rig.broker.positions["BTC/USD"]
        assert rig.exe.reconcile().state is ReconState.OK
        intent = rig.records("order_intent")[-1][1]
        assert intent["intended_qty"] == v.qty and intent["execution_mode"] == "PAPER"

    def test_ack_and_fills_are_separate_facts(self, rig):
        rig.broker.script = ["ack"]
        order = rig.exe.submit(rig.verdict())
        assert order.status is OrderStatus.ACKNOWLEDGED and not order.fills
        rig.broker.advance_fill(order.client_order_id)
        assert rig.exe.reconcile().state is ReconState.OK
        assert rig.statuses(order.client_order_id) == ["ACKNOWLEDGED", "FILLED"]
        assert [p["fill"]["fill_id"] for _, p in rig.records("fill")] == [f"{order.client_order_id}-f1"]

    def test_N_partial_fill_then_completion(self, rig):
        rig.broker.script = ["partial:0.4"]
        v = rig.verdict()
        order = rig.exe.submit(v)
        part = (v.qty * Decimal("0.4")).quantize(Decimal("0.00000001"))
        assert order.status is OrderStatus.PARTIALLY_FILLED and rig.exe.account.position_qty("BTC/USD") == part
        assert rig.exe.reconcile().state is ReconState.OK                   # a venue-confirmed working order is fine
        with pytest.raises(SubmissionRefused):
            rig.exe.submit(rig.verdict())                                    # no stacking while one is working
        refused(rig, "open_order_exists")
        rig.broker.advance_fill(order.client_order_id)
        assert rig.exe.reconcile().state is ReconState.OK
        assert order.status is OrderStatus.FILLED and rig.exe.account.position_qty("BTC/USD") == v.qty

    def test_expiry_is_recorded(self, rig):
        rig.broker.script = ["ack"]
        order = rig.exe.submit(rig.verdict())
        rig.broker.expire(order.client_order_id)
        assert rig.exe.reconcile().state is ReconState.OK and order.status is OrderStatus.EXPIRED


# ----------------------------------------------------------------------------------------------- B–I: authorization
class TestAuthorization:
    def test_B_oversized_proposal_is_cut_to_the_risk_contract(self, rig):
        v = rig.verdict(proposed_qty=Decimal("1000"))                          # $30M proposed
        assert v.approved and v.qty * PRICE <= RiskLimits().max_order_notional
        order = rig.exe.submit(v)
        assert order.intended_qty == v.qty < Decimal("1000")                 # the broker received the authorized size

    def test_B_oversized_after_risk_is_impossible(self, rig):
        v = rig.verdict()
        with pytest.raises(ApprovalInvalid):
            rig.exe.submit(replace(v, qty=v.qty * 100))                       # editing the approval breaks its signature
        refused(rig, "risk_approval")
        assert "submit_order" not in rig.broker.calls

    def test_C_live_is_disabled_at_every_layer(self, rig, tmp_path):
        assert LIVE_TRADING is False and live_state() == "LIVE_DISABLED"
        with pytest.raises(LiveTradingDisabled):
            create_live_broker()

        class LiveVenue:
            mode = OperatingMode.LIVE
        with pytest.raises(LiveTradingDisabled):
            ExecutionEngine(LiveVenue(), tmp_path / "live.jsonl", rig.authority, rig.clock, mode=OperatingMode.LIVE,
                            data_status=DataStatus.REAL, initial_cash=CASH, kill_switch=rig.kill)
        live = replace(rig.verdict(), mode=OperatingMode.LIVE)
        with pytest.raises(ApprovalInvalid):
            rig.exe.submit(live)                                             # cannot be forged into a live approval
        pf = rig.exe.portfolio_snapshot({"BTC/USD": PRICE}, CASH, CASH)
        denied = rig.risk.evaluate(OrderRequest("dl", "trend@v1", "BTC/USD", Side.BUY, PRICE, PRICE * Decimal("0.98")),
                                   replace(pf, mode=OperatingMode.LIVE), rig.market())
        assert not denied.approved and denied.failed[0].name == "mode_permitted"
        paper_ok = rig.verdict()
        rig.exe.mode = OperatingMode.LIVE                                   # even a mutated engine refuses
        with pytest.raises((SubmissionRefused, ModeMismatch, ApprovalInvalid)):
            rig.exe.submit(paper_ok)
        with pytest.raises((SubmissionRefused, ModeMismatch, ApprovalInvalid)):
            rig.exe.submit(rig.verdict())                                    # risk refuses a LIVE portfolio outright
        assert "submit_order" not in rig.broker.calls

    def test_D_T_kill_switch_blocks_new_risk_even_after_approval(self, rig):
        rig.broker.script = ["ack"]
        working = rig.exe.submit(rig.verdict(symbol="ETH/USD"))
        v = rig.verdict()                                                      # approved before the stop
        assert v.approved
        rig.kill.engage("operator emergency stop")
        with pytest.raises(SubmissionRefused):
            rig.exe.submit(v)
        assert refused(rig, "kill_switch")["state"]["kill_switch"]["engaged"] is True
        assert not rig.verdict().approved                                      # and nothing new is approved
        assert working.client_order_id in [o["order"] for o in execution_view_min(rig)["open_orders"]]   # visible
        rig.exe.cancel(working.client_order_id, "emergency stop: operator cancel")
        assert working.status is OrderStatus.CANCELED
        assert rig.broker.submissions == 1

    def test_H_stale_market_data_is_rejected_by_risk(self, rig):
        v = rig.verdict(age=timedelta(hours=3))
        assert not v.approved and "data_fresh" in {c.name for c in v.failed}
        with pytest.raises(ApprovalInvalid):
            rig.exe.submit(v)

    def test_I_stale_account_state_is_refused(self, rig):
        rig.clock.advance(timedelta(minutes=6))                                # reconciliation now older than policy
        v = rig.verdict()                                                      # fresh, valid risk approval
        assert v.approved
        with pytest.raises(SubmissionRefused):
            rig.exe.submit(v)
        refused(rig, "account_state_fresh")
        assert "submit_order" not in rig.broker.calls
        rig.exe.reconcile()
        assert rig.exe.submit(rig.verdict()).status is OrderStatus.FILLED     # fresh again → allowed

    def test_I_unknown_account_after_restart_is_refused(self, rig):
        v = rig.verdict()                                                      # approved before the restart
        rig.exe = rig.engine()                                                 # restart: not reconciled yet
        with pytest.raises(SubmissionRefused):
            rig.exe.submit(v)
        refused(rig, "account_state_fresh")
        assert not rig.verdict().approved                                      # and risk refuses an UNKNOWN account

    def test_every_refusal_is_journaled_with_its_rule_and_state(self, rig):
        rig.broker.down = True
        v = rig.verdict()
        with pytest.raises(SubmissionRefused) as info:
            rig.exe.submit(v)
        rec = refused(rig, "broker_health")
        assert info.value.rule == "broker_health" and rec["decision_id"] == v.decision_id and rec["at"] == rig.clock.now()
        assert any("venue_reachable" in c for c in rec["state"]["checks"])


def execution_view_min(rig):
    class S:                                                                  # the status view's needs, nothing else
        execution = rig.exe
        kill_switch = rig.kill

        class decisions:
            class journal:
                @staticmethod
                def entries(_):
                    return iter(())
    return execution_view(S)


# ----------------------------------------------------------------------------------------------- E–G, W, X: Claude
class TestClaudeBoundary:
    def _propose(self, tmp_path, **over):
        cp, s, clock = plane(tmp_path / "st")
        install_champion(s)
        until_signal(cp, s, clock)
        s.reasoning.script["company"] = reply("TRADE_PROPOSAL", lambda p: trade_payload(p, **over))
        out = cp.run_cycle()
        assert not s.execution.orders and out.status in ("FAILED", "BLOCKED"), out.detail
        return out

    def test_E_unknown_strategy(self, tmp_path):
        assert "not active" in self._propose(tmp_path, strategy_key="ghost@v9").detail["reason"]

    def test_F_protocol_or_other_fields_are_not_claudes_to_set(self, tmp_path):
        self._propose(tmp_path / "p", protocol_id="REAL-PROTOCOL-002")
        self._propose(tmp_path / "m", execution_mode="LIVE")
        self._propose(tmp_path / "q", max_order_notional="99999999")

    def test_G_wrong_symbol_or_timeframe(self, tmp_path):
        self._propose(tmp_path / "s", symbol="DOGE/USD")
        self._propose(tmp_path / "t", timeframe="4h")

    def test_W_claude_quantity_can_only_lower_risk(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st")
        install_champion(s)
        until_signal(cp, s, clock)
        s.reasoning.script["company"] = reply("TRADE_PROPOSAL", lambda p: trade_payload(p, proposed_qty="500"))
        out = cp.run_cycle()
        rec = s.decisions.last
        if rec.final_decision.value == "EXECUTE":
            order = next(iter(s.execution.orders.values()))
            assert order.intended_qty == rec.approved_size < Decimal("500")
            assert order.intended_qty * order.intended_price <= s.limits.max_order_notional
        assert out.status == "COMPLETED"

    def test_X_claude_has_no_broker_or_infrastructure_actions(self):
        assert set(COMPANY_ACTIONS) == {"NO_TRADE", "TRADE_PROPOSAL", "RESEARCH_REQUEST", "PAUSE", "REQUEST_DATA",
                                        "REVIEW_POSITION", "REVIEW_RISK", "REVIEW_SYSTEM"}
        for path in list((ROOT / "ati" / "agent").glob("*.py")) + list((ROOT / "ati" / "company").glob("*.py")):
            text = path.read_text()
            for call in ("submit_order(", "cancel_order(", "list_orders(", ".broker.", "health_check("):
                assert call not in text, (path.name, call)

    def test_X_unlisted_action_is_rejected(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("SUBMIT_ORDER", {"symbol": "BTC/USD"})})
        out = cp.run_cycle()
        assert out.status == "FAILED" and not s.execution.orders

    def test_W_only_the_execution_engine_talks_to_a_broker(self):
        callers = []
        for path in (ROOT / "ati").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in ("submit_order", "cancel_order"):
                    if isinstance(node.ctx, ast.Load) and path.name != "broker.py":
                        callers.append(path.relative_to(ROOT).as_posix())
        assert sorted(set(callers)) == ["ati/execution/engine.py"], callers


# ----------------------------------------------------------------------------------------------- J–M: venue failures
class TestVenueFailures:
    def test_J_unavailable_or_unauthenticated_broker_gets_no_order(self, rig):
        for fault in ("down", "session_valid", "order_entry_enabled"):
            setattr(rig.broker, fault, fault == "down")
            with pytest.raises(SubmissionRefused):
                rig.exe.submit(rig.verdict())
            refused(rig, "broker_health")
            setattr(rig.broker, fault, fault != "down")
        assert "submit_order" not in rig.broker.calls and not rig.exe.orders

    def test_K_timeout_after_execution_requires_reconciliation(self, rig):
        rig.broker.script = ["timeout_after"]
        v = rig.verdict()
        order = rig.exe.submit(v)
        assert order.status is OrderStatus.UNKNOWN and rig.exe.halted
        assert rig.exe.account.position_qty("BTC/USD") == 0                  # nothing assumed
        with pytest.raises(ExecutionHalted):
            rig.exe.submit(rig.verdict())
        assert rig.exe.reconcile().state is ReconState.OK and not rig.exe.halted
        assert order.status is OrderStatus.FILLED and rig.exe.account.position_qty("BTC/USD") == v.qty

    def test_K_timeout_before_execution_resolves_to_not_found(self, rig):
        rig.broker.script = ["timeout_before"]
        order = rig.exe.submit(rig.verdict())
        assert order.status is OrderStatus.UNKNOWN and rig.exe.halted
        assert rig.exe.reconcile().state is ReconState.OK and order.status is OrderStatus.NOT_FOUND

    def test_L_duplicate_intent_is_one_logical_order(self, rig):
        rig.broker.script = ["timeout_after"]
        v = rig.verdict()
        first = rig.exe.submit(v)
        rig.exe.reconcile()
        again = rig.exe.submit(v)                                             # retry of the same decision
        assert again is first and rig.broker.submissions == 1
        assert rig.broker.calls.count("submit_order") == 1
        assert len(rig.records("order_intent")) == 1
        assert first.client_order_id == client_order_id(v.decision_id)

    def test_M_broker_rejection_is_recorded(self, rig):
        rig.broker.script = ["reject"]
        order = rig.exe.submit(rig.verdict())
        assert order.status is OrderStatus.REJECTED and "rejected by venue" in order.message
        assert not rig.exe.halted and rig.exe.reconcile().state is ReconState.OK
        assert execution_view_min(rig)["last_rejection"]["reason"] == "rejected by venue"


# ----------------------------------------------------------------------------------------------- P–S: lifecycle
class TestCancelAndReconcile:
    def test_P_cancel_succeeds(self, rig):
        rig.broker.script = ["partial:0.5"]
        order = rig.exe.submit(rig.verdict())
        rig.exe.cancel(order.client_order_id, "operator")
        assert order.status is OrderStatus.PARTIAL_CANCELED
        assert rig.statuses(order.client_order_id) == ["PARTIALLY_FILLED", "CANCEL_REQUESTED", "PARTIAL_CANCELED"]
        assert rig.exe.reconcile().state is ReconState.OK
        assert rig.exe.account.position_qty("BTC/USD") == order.filled_qty > 0

    def test_Q_cancel_failure_is_unknown_until_reconciled(self, rig):
        rig.broker.script = ["ack"]
        order = rig.exe.submit(rig.verdict())
        rig.broker.cancel_script = ["timeout"]
        rig.exe.cancel(order.client_order_id, "operator")
        assert order.status is OrderStatus.UNKNOWN and rig.exe.halted
        assert rig.exe.reconcile().state is ReconState.OK                   # venue: still working → truth established
        assert order.status is OrderStatus.ACKNOWLEDGED

    def test_Q_unprocessed_cancel_keeps_trading_halted(self, rig):
        rig.broker.script = ["ack"]
        order = rig.exe.submit(rig.verdict())
        rig.broker.cancel_script = ["ignore"]
        rig.exe.cancel(order.client_order_id, "operator")
        assert order.status is OrderStatus.CANCEL_REQUESTED
        assert rig.exe.reconcile().state is ReconState.MISMATCH and rig.exe.halted
        rig.broker.cancel_script = ["ok"]
        rig.broker.cancel_order(order.client_order_id)                        # the venue processes it later
        assert rig.exe.reconcile().state is ReconState.OK and order.status is OrderStatus.CANCELED

    def test_R_unexpected_venue_order_is_detected(self, rig):
        rig.broker.inject_order("manual-1", "BTC/USD", Side.BUY, Decimal("0.01"))
        result = rig.exe.reconcile()
        assert result.state is ReconState.MISMATCH and rig.exe.halted
        assert any("unexpected venue order manual-1" in i for i in result.issues)
        with pytest.raises(ExecutionHalted):
            rig.exe.submit(rig.verdict())

    def test_S_unexpected_or_repriced_fill_is_detected(self, rig):
        order = rig.exe.submit(rig.verdict())
        rig.broker.inject_fill(order.client_order_id, Decimal("0.001"))
        issues = rig.exe.reconcile().issues
        assert any(i.startswith("unexpected fill") for i in issues) and rig.exe.halted
        rig2 = Rig(rig.tmp / "b")
        rig2.exe.reconcile()
        o2 = rig2.exe.submit(rig2.verdict())
        rig2.broker.reprice_fill(o2.client_order_id, 0, PRICE + 1)
        assert any("differs" in i for i in rig2.exe.reconcile().issues)


# ----------------------------------------------------------------------------------------------- modes
class TestModes:
    def test_observe_sends_nothing(self, tmp_path):
        r = Rig(tmp_path, ExecutionPolicy(mode=ExecutionMode.OBSERVE))
        r.exe.reconcile()
        with pytest.raises(SubmissionRefused):
            r.exe.submit(r.verdict())
        refused(r, "execution_mode")
        assert "submit_order" not in r.broker.calls

    def test_assisted_requires_operator_approval_for_new_risk(self, tmp_path):
        r = Rig(tmp_path, ExecutionPolicy(mode=ExecutionMode.ASSISTED))
        r.exe.reconcile()
        v = r.verdict()
        with pytest.raises(SubmissionRefused):
            r.exe.submit(v)
        refused(r, "operator_approval_required")
        with pytest.raises(PermissionError):
            r.exe.approve(v.decision_id, "yes please")
        r.exe.approve(v.decision_id, APPROVAL_ACK)
        order = r.exe.submit(v)
        assert order.status is OrderStatus.FILLED
        r.exe.reconcile()
        sell = r.verdict(side=Side.SELL)                                      # exits never wait for a human
        assert r.exe.submit(sell).status is OrderStatus.FILLED
        assert v.decision_id in r.engine().approvals                          # approvals are journaled

    def test_autonomous_limited_enforces_tighter_limits(self, tmp_path):
        policy = ExecutionPolicy(mode=ExecutionMode.AUTONOMOUS_LIMITED, autonomous_symbols=frozenset({"BTC/USD"}),
                                 autonomous_max_order_notional=Decimal("5000"), autonomous_max_orders_per_day=1)
        r = Rig(tmp_path, policy)
        r.exe.reconcile()
        with pytest.raises(SubmissionRefused):
            r.exe.submit(r.verdict(symbol="ETH/USD", proposed_qty=Decimal("1")))
        assert "not in autonomous symbols" in refused(r, "autonomous_limit")["reason"]
        with pytest.raises(SubmissionRefused):
            r.exe.submit(r.verdict(proposed_qty=Decimal("1")))                 # $30,000 > $5,000
        ok = r.exe.submit(r.verdict(proposed_qty=Decimal("0.1")))              # $3,000
        assert ok.status is OrderStatus.FILLED
        r.exe.reconcile()
        with pytest.raises(SubmissionRefused):
            r.exe.submit(r.verdict(proposed_qty=Decimal("0.01")))              # daily count reached
        assert "orders today" in refused(r, "autonomous_limit")["reason"]


# ----------------------------------------------------------------------------------------------- U, V: credentials
class TestCredentials:
    def _everything(self, rig):
        rig.broker.script = ["ack", "reject", "timeout_after"]
        o = rig.exe.submit(rig.verdict())
        rig.broker.cancel_script = ["timeout"]
        rig.exe.cancel(o.client_order_id, "x")
        rig.exe.reconcile()
        rig.exe.cancel(o.client_order_id, "x")
        rig.exe.reconcile()
        for _ in range(2):
            try:
                rig.exe.submit(rig.verdict())
            except Exception as exc:                                          # noqa: BLE001 - collect messages
                assert SECRET not in str(exc)
            rig.exe.reconcile()
        rig.broker.down = True
        try:
            rig.exe.submit(rig.verdict())
        except Exception as exc:                                              # noqa: BLE001
            assert SECRET not in str(exc)

    def test_U_credentials_never_appear_in_journals_reports_or_views(self, rig):
        self._everything(rig)
        blob = rig.tmp.joinpath("exec.jsonl").read_text()
        assert SECRET not in blob and "MOCK_BROKER_SECRET" not in blob
        rig.broker.down = False
        texts = [repr(rig.broker), repr(rig.secret), str(rig.broker.health_check()),
                 json.dumps(execution_view_min(rig), default=str)]
        texts += [str(r) for r in rig.broker.list_orders()]
        assert all(SECRET not in t for t in texts)
        with pytest.raises(SecretLeakError):
            rig.exe.journal.append("note", {"text": f"token={SECRET}"})    # the guard refuses to persist it

    def test_V_claude_context_never_contains_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MOCK_BROKER_SECRET", SECRET)
        cp, s, clock = plane(tmp_path / "st")
        from ati.security.secrets import load_secret
        load_secret("MOCK_BROKER_SECRET", s.guard)
        install_champion(s)
        prompts = []
        s.reasoning.script["company"] = lambda prompt: prompts.append(prompt) or reply("NO_TRADE")(prompt)
        cp.run_cycle()
        assert prompts and all(SECRET not in p for p in prompts)
        ctx = json.loads(prompts[-1].split("\n\nPACKET:\n\n", 1)[1].split("\n\n<<<UNTRUSTED_DATA", 1)[0])
        assert ctx["RISK_STATE"]["execution"]["live"] == "LIVE_DISABLED"
        assert "broker" not in json.dumps(ctx).lower().replace("broker_", "")      # no adapter object or handle
        with pytest.raises(SecretLeakError):
            s.guard.scan(f"prompt with {SECRET}", "prompt")


# ----------------------------------------------------------------------------------------------- Y, Z: restart
class TestRestart:
    def test_Y_Z_restart_preserves_and_reconciles_execution_state(self, tmp_path):
        r = Rig(tmp_path, ExecutionPolicy(mode=ExecutionMode.ASSISTED))
        r.exe.reconcile()
        a = r.verdict()
        r.exe.approve(a.decision_id, APPROVAL_ACK)
        r.broker.script = ["partial:0.25"]
        working = r.exe.submit(a)
        with pytest.raises(SubmissionRefused):
            r.exe.submit(r.verdict(symbol="ETH/USD"))                        # refused: no approval
        before = {k: (o.status, o.filled_qty, o.broker_order_id) for k, o in r.exe.orders.items()}
        cash, pos = r.exe.account.cash, r.exe.account.position_qty("BTC/USD")

        restarted = r.engine()                                               # same journal, same venue
        assert {k: (o.status, o.filled_qty, o.broker_order_id) for k, o in restarted.orders.items()} == before
        assert restarted.account.cash == cash and restarted.account.position_qty("BTC/USD") == pos
        assert a.decision_id in restarted.approvals and restarted.last_refusal["rule"] == "operator_approval_required"
        assert restarted.halted and restarted.recon_state is ReconState.UNKNOWN   # must reconcile before trading
        assert restarted.reconcile().state is ReconState.OK and not restarted.halted
        assert restarted.orders[working.client_order_id].status is OrderStatus.PARTIALLY_FILLED
        r.broker.advance_fill(working.client_order_id)
        again = r.engine()
        again.reconcile()
        assert again.orders[working.client_order_id].status is OrderStatus.FILLED
        assert again.account.position_qty("BTC/USD") == r.broker.positions["BTC/USD"]


def test_paper_broker_contract_methods(tmp_path):
    """The existing PaperBroker satisfies the extended contract (health, order list, cancel of a final order)."""
    from ati.execution.paper import PaperBroker, Quote
    from ati.research.costs import CostModel

    clock = FixedClock(NOW)
    b = PaperBroker(lambda s: Quote(s, PRICE, clock.now(), Decimal("100"), DataStatus.MOCK), CostModel(), CASH,
                    DataStatus.MOCK, clock)
    assert b.health_check().ok and b.list_orders() == ()
    assert b.cancel_order("nope").status is OrderStatus.NOT_FOUND
    b.faults.add("unavailable")
    assert not b.health_check().ok
