from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from ati.config import OperatingMode
from ati.core.errors import (ApprovalInvalid, ExecutionHalted, JournalCorruption, JournalWriteError,
                             LiveTradingDisabled, ModeMismatch, SecretLeakError)
from ati.core.time import FixedClock
from ati.core.types import Side
from ati.execution.broker import OrderStatus
from ati.execution.engine import ExecutionEngine, client_order_id
from ati.execution.live import create_live_broker
from ati.execution.paper import PaperBroker, Quote
from ati.ledger.accounting import Account, Fill
from ati.ledger.journal import Journal
from ati.market.models import DataStatus, Timeframe
from ati.market.universe import default_universe
from ati.research.costs import CostModel
from ati.risk.engine import (ApprovalAuthority, MarketSnapshot, OrderRequest, ReconState, RiskEngine, RiskLimits)
from ati.risk.killswitch import KillSwitch
from ati.security.secrets import SecretGuard, SecretValue
from tests.helpers import T0

NOW = T0 + timedelta(days=3)
CASH = Decimal("100000")


class Rig:
    """Wires risk + execution + paper venue exactly as production does. MOCK quotes."""

    def __init__(self, tmp_path, clock=None, volume=Decimal("100")):
        self.tmp = tmp_path
        self.clock = clock or FixedClock(NOW)
        self.price = Decimal("30000")
        self.volume = volume
        self.authority = ApprovalAuthority()
        self.broker = PaperBroker(self.quote, CostModel(), CASH, DataStatus.MOCK, self.clock)
        self.kill = KillSwitch(tmp_path / "kill.json", self.clock)
        self.risk = RiskEngine(RiskLimits(), default_universe(), self.kill, self.authority, self.clock)
        self.exe = self.new_engine()
        self.n = 0

    def new_engine(self):
        return ExecutionEngine(self.broker, self.tmp / "exec.jsonl", self.authority, self.clock, mode=OperatingMode.PAPER,
                               data_status=DataStatus.MOCK, initial_cash=CASH, kill_switch=self.kill)

    def quote(self, symbol):
        return Quote(symbol, self.price, self.clock.now(), self.volume, DataStatus.MOCK)

    def market(self):
        return MarketSnapshot("BTC/USD", Timeframe.H1, self.price, self.clock.now() - timedelta(minutes=5), DataStatus.MOCK,
                              Decimal("500"), Decimal("0.0002"), Decimal("0.0003"))

    def verdict(self, side=Side.BUY, decision_id=None, **kw):
        self.n += 1
        pf = self.exe.portfolio_snapshot({"BTC/USD": self.price}, CASH, CASH)
        r = OrderRequest(decision_id or f"d{self.n}", "trend@v1", "BTC/USD", side, self.price,
                         self.price * Decimal("0.98") if side is Side.BUY else None, **kw)
        return self.risk.evaluate(r, pf, self.market())


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    assert r.exe.reconcile().state is ReconState.OK
    return r


class TestJournal:
    def test_chain_and_tamper_detection(self, tmp_path, clock):
        j = Journal(tmp_path / "j.jsonl", kind="t", attrs={"mode": "PAPER"}, clock=clock)
        j.append("a", {"x": Decimal("1.5")})
        j.append("b", {"y": 2})
        Journal(tmp_path / "j.jsonl", kind="t", attrs={"mode": "PAPER"}, clock=clock).verify()
        text = (tmp_path / "j.jsonl").read_text().replace('"y":2', '"y":3')
        (tmp_path / "j.jsonl").write_text(text)
        with pytest.raises(JournalCorruption):
            Journal(tmp_path / "j.jsonl", kind="t", attrs={"mode": "PAPER"}, clock=clock)

    def test_deletion_detected(self, tmp_path, clock):
        j = Journal(tmp_path / "j.jsonl", kind="t", attrs={}, clock=clock)
        for i in range(3):
            j.append("e", {"i": i})
        lines = (tmp_path / "j.jsonl").read_text().splitlines(keepends=True)
        (tmp_path / "j.jsonl").write_text("".join(lines[:2] + lines[3:]))
        with pytest.raises(JournalCorruption):
            Journal(tmp_path / "j.jsonl", kind="t", attrs={}, clock=clock)

    def test_truncated_write_detected(self, tmp_path, clock):
        j = Journal(tmp_path / "j.jsonl", kind="t", attrs={}, clock=clock)
        j.append("e", {"i": 1})
        with open(tmp_path / "j.jsonl", "a") as fh:
            fh.write('{"seq": 2, "partial')
        with pytest.raises(JournalCorruption):
            Journal(tmp_path / "j.jsonl", kind="t", attrs={}, clock=clock)

    def test_mode_binding(self, tmp_path, clock):
        Journal(tmp_path / "j.jsonl", kind="execution", attrs={"mode": "PAPER"}, clock=clock)
        with pytest.raises(ModeMismatch):
            Journal(tmp_path / "j.jsonl", kind="execution", attrs={"mode": "LIVE"}, clock=clock)

    def test_failed_write_breaks_journal(self, tmp_path, clock, monkeypatch):
        j = Journal(tmp_path / "j.jsonl", kind="t", attrs={}, clock=clock)
        monkeypatch.setattr(j, "_write_line", lambda line: (_ for _ in ()).throw(OSError("disk")))
        with pytest.raises(JournalWriteError):
            j.append("e", {})
        monkeypatch.undo()
        with pytest.raises(JournalWriteError):
            j.append("e", {})
        assert len(j) == 0

    def test_secrets_never_persisted(self, tmp_path, clock):
        guard = SecretGuard()
        guard.register(SecretValue("API_KEY", "super-secret-value-123"))
        j = Journal(tmp_path / "j.jsonl", kind="t", attrs={}, clock=clock, guard=guard)
        with pytest.raises(SecretLeakError):
            j.append("e", {"note": "key is super-secret-value-123"})
        with pytest.raises(SecretLeakError):
            j.append("e", {"note": "sk-" + "ant-" + "abcdefghijklmnop"})  # credential shape, built at runtime
        assert "super-secret" not in (tmp_path / "j.jsonl").read_text()


class TestAccounting:
    def fill(self, fid, side, qty, price, fee="1", mode=OperatingMode.PAPER, status=DataStatus.MOCK):
        return Fill(fid, "c", "BTC/USD", side, Decimal(qty), Decimal(price), Decimal(fee), NOW, mode, status)

    def test_pnl_fees_and_duplicate_fills(self):
        a = Account(OperatingMode.PAPER, DataStatus.MOCK, Decimal("1000"))
        assert a.apply_fill(self.fill("f1", Side.BUY, "2", "100"))
        assert not a.apply_fill(self.fill("f1", Side.BUY, "2", "100"))
        a.apply_fill(self.fill("f2", Side.SELL, "1", "110"))
        assert a.cash == Decimal("1000") - 201 + 109
        assert a.realized_pnl == Decimal("10") - 2
        assert a.position_qty("BTC/USD") == 1
        assert a.equity({"BTC/USD": Decimal("120")}) == a.cash + 120

    def test_no_phantom_short(self):
        a = Account(OperatingMode.PAPER, DataStatus.MOCK, Decimal("1000"))
        with pytest.raises(ValueError):
            a.apply_fill(self.fill("f1", Side.SELL, "1", "100"))

    def test_paper_and_live_never_mix(self):
        a = Account(OperatingMode.PAPER, DataStatus.MOCK, Decimal("1000"))
        with pytest.raises(ModeMismatch):
            a.apply_fill(self.fill("f1", Side.BUY, "1", "100", mode=OperatingMode.LIVE))
        with pytest.raises(ModeMismatch):
            a.apply_fill(self.fill("f2", Side.BUY, "1", "100", status=DataStatus.SYNTHETIC))

    def test_invalid_fill_values(self):
        with pytest.raises(ValueError):
            self.fill("f", Side.BUY, "0", "100")
        with pytest.raises(ValueError):
            self.fill("f", Side.BUY, "1", "NaN")
        with pytest.raises(ValueError):
            self.fill("f", Side.BUY, "1", "100", fee="-1")


class TestExecution:
    def test_happy_path_records_everything(self, rig):
        v = rig.verdict()
        order = rig.exe.submit(v)
        assert order.status is OrderStatus.FILLED and order.filled_qty == v.qty
        assert order.fees > 0 and order.slippage > 0
        assert order.client_order_id == client_order_id(v.decision_id)
        assert rig.exe.reconcile().state is ReconState.OK
        types = [e.type for e in rig.exe.journal.entries()]
        assert types.index("order_intent") < types.index("fill")

    def test_duplicate_submission_is_idempotent(self, rig):
        v = rig.verdict()
        a = rig.exe.submit(v)
        b = rig.exe.submit(v)
        assert a is b and rig.broker.submissions == 1

    def test_requires_signed_unexpired_approval(self, rig):
        v = rig.verdict()
        with pytest.raises(ApprovalInvalid):
            rig.exe.submit(replace(v, qty=v.qty * 2))
        rejected = rig.verdict(proposed_qty=Decimal("-1"))
        with pytest.raises(ApprovalInvalid):
            rig.exe.submit(rejected)
        rig.clock.advance(timedelta(minutes=5))
        with pytest.raises(ApprovalInvalid):
            rig.exe.submit(v)
        assert rig.broker.submissions == 0

    def test_partial_fill(self, tmp_path):
        rig = Rig(tmp_path, volume=Decimal("1"))
        rig.exe.reconcile()
        v = rig.verdict()
        order = rig.exe.submit(v)
        assert order.status is OrderStatus.PARTIAL_CANCELED
        assert order.filled_qty == Decimal("0.1") < v.qty
        assert rig.exe.account.position_qty("BTC/USD") == Decimal("0.1")

    def test_round_trip_pnl(self, rig):
        rig.exe.submit(rig.verdict())
        rig.price = Decimal("30300")
        sell = rig.exe.submit(rig.verdict(side=Side.SELL))
        assert sell.status is OrderStatus.FILLED
        assert rig.exe.account.position_qty("BTC/USD") == 0
        assert rig.exe.reconcile().state is ReconState.OK

    def test_live_disabled(self):
        with pytest.raises(LiveTradingDisabled):
            create_live_broker()

    def test_broker_mode_must_match(self, tmp_path):
        rig = Rig(tmp_path)
        rig.broker.mode = OperatingMode.BACKTEST
        with pytest.raises(ModeMismatch):
            rig.new_engine()

    def test_no_reconcile_no_trading(self, tmp_path):
        rig = Rig(tmp_path)
        v = rig.verdict()
        # Before the first reconciliation the account state is not trusted: rejected at the first gate.
        assert not v.approved and {c.name for c in v.failed} & {"account_state_known", "reconciliation"}
