from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from ati.config import OperatingMode
from ati.core.errors import ApprovalInvalid
from ati.core.time import FixedClock
from ati.core.types import Side
from ati.market.models import DataStatus, Timeframe
from ati.market.universe import default_universe
from ati.risk.engine import (ApprovalAuthority, MarketSnapshot, OrderRequest, PortfolioSnapshot, ReconState,
                             RiskEngine, RiskLimits)
from ati.risk.killswitch import RELEASE_ACK, KillSwitch
from tests.helpers import T0

NOW = T0 + timedelta(days=10)


@pytest.fixture
def clock():
    return FixedClock(NOW)


@pytest.fixture
def ks(tmp_path, clock):
    return KillSwitch(tmp_path / "kill.json", clock)


@pytest.fixture
def engine(ks, clock):
    return RiskEngine(RiskLimits(), default_universe(), ks, ApprovalAuthority(), clock)


def pf(**kw):
    base = dict(as_of=NOW, mode=OperatingMode.PAPER, equity=Decimal("100000"), cash=Decimal("100000"), positions=(),
                day_start_equity=Decimal("100000"), peak_equity=Decimal("100000"), reconciliation=ReconState.OK,
                account_state_known=True)
    return PortfolioSnapshot(**(base | kw))


def mkt(**kw):
    base = dict(symbol="BTC/USD", timeframe=Timeframe.H1, last_price=Decimal("30000"), last_close_time=NOW - timedelta(minutes=10),
                status=DataStatus.MOCK, recent_volume=Decimal("500"), est_spread_rate=Decimal("0.0002"),
                est_slippage_rate=Decimal("0.0003"))
    return MarketSnapshot(**(base | kw))


def req(**kw):
    base = dict(decision_id="d1", strategy_key="trend@v1", symbol="BTC/USD", side=Side.BUY, entry_price=Decimal("30000"),
                stop_price=Decimal("29400"), proposed_qty=None)
    return OrderRequest(**(base | kw))


def failed(v):
    return {c.name for c in v.failed}


class TestSizing:
    def test_approved_size_respects_max_trade_risk(self, engine):
        v = engine.evaluate(req(), pf(), mkt())
        assert v.approved and v.token
        limit = Decimal("100000") * RiskLimits().max_risk_per_trade_fraction
        assert v.max_loss <= limit
        assert v.qty > 0

    def test_proposed_qty_can_only_lower_size(self, engine):
        base = engine.evaluate(req(), pf(), mkt()).qty
        assert engine.evaluate(req(proposed_qty=base * 10), pf(), mkt()).qty == base
        assert engine.evaluate(req(proposed_qty=Decimal("0.01")), pf(), mkt()).qty == Decimal("0.01")

    def test_absolute_max_trade_loss(self, engine):
        v = engine.evaluate(req(), pf(equity=Decimal("10000000"), cash=Decimal("10000000"),
                                      day_start_equity=Decimal("10000000"), peak_equity=Decimal("10000000")), mkt())
        assert v.max_loss <= RiskLimits().max_trade_loss

    def test_max_order_notional(self, engine):
        v = engine.evaluate(req(stop_price=Decimal("29990")), pf(equity=Decimal("10000000"), cash=Decimal("10000000"),
                            day_start_equity=Decimal("10000000"), peak_equity=Decimal("10000000")), mkt())
        assert v.qty * Decimal("30000") <= RiskLimits().max_order_notional

    def test_symbol_exposure_cap(self, engine):
        held = pf(positions=(("BTC/USD", Decimal("1.6"), Decimal("30000")),), cash=Decimal("52000"))
        v = engine.evaluate(req(stop_price=Decimal("29990")), held, mkt())
        assert not v.approved or (Decimal("1.6") + v.qty) * 30000 <= Decimal("50000") + 1

    def test_correlated_exposure_cap(self, engine):
        held = pf(positions=(("ETH/USD", Decimal("40"), Decimal("1800")),), cash=Decimal("28000"))
        v = engine.evaluate(req(stop_price=Decimal("29990")), held, mkt())
        assert v.qty * 30000 + 40 * 1800 <= 75000 + 1

    def test_leverage_and_cash(self, engine):
        v = engine.evaluate(req(stop_price=Decimal("29990")), pf(cash=Decimal("3000")), mkt())
        assert v.qty * Decimal("30000") <= Decimal("3000")

    def test_liquidity_cap(self, engine):
        v = engine.evaluate(req(stop_price=Decimal("29990")), pf(), mkt(recent_volume=Decimal("2")))
        assert v.qty <= Decimal("2") * RiskLimits().max_participation

    def test_size_below_minimum_rejected(self, engine):
        v = engine.evaluate(req(), pf(), mkt(recent_volume=Decimal("0.0001")))
        assert not v.approved and "sizing" in failed(v)


class TestRejections:
    @pytest.mark.parametrize("r,p,m,name", [
        (dict(), dict(mode=OperatingMode.LIVE), dict(), "mode_permitted"),
        (dict(), dict(account_state_known=False), dict(), "account_state_known"),
        (dict(), dict(reconciliation=ReconState.MISMATCH), dict(), "reconciliation"),
        (dict(), dict(reconciliation=ReconState.UNKNOWN), dict(), "reconciliation"),
        (dict(symbol="DOGE/USD"), dict(), dict(), "symbol_known"),
        (dict(), dict(), dict(status=DataStatus.UNKNOWN), "data_status"),
        (dict(), dict(), dict(last_close_time=NOW - timedelta(hours=5)), "data_fresh"),
        (dict(), dict(), dict(last_close_time=NOW + timedelta(hours=1)), "data_fresh"),
        (dict(entry_price=Decimal("33000")), dict(), dict(), "price_sane"),
        (dict(entry_price=Decimal("-1")), dict(), dict(), "price_positive"),
        (dict(), dict(), dict(est_slippage_rate=Decimal("0.01")), "slippage"),
        (dict(stop_price=None), dict(), dict(), "stop_present"),
        (dict(stop_price=Decimal("30100")), dict(), dict(), "stop_valid"),
        (dict(proposed_qty=Decimal("-1")), dict(), dict(), "proposed_qty_valid"),
        (dict(side=Side.SELL), dict(), dict(), "reduces_position"),
    ])
    def test_rejection(self, engine, r, p, m, name):
        v = engine.evaluate(req(**r), pf(**p), mkt(**m))
        assert not v.approved and name in failed(v) and v.qty == 0 and v.token == ""

    def test_live_mode_rejected_even_with_real_data(self, engine):
        v = engine.evaluate(req(), pf(mode=OperatingMode.LIVE), mkt(status=DataStatus.REAL))
        assert not v.approved

    def test_mock_data_never_allowed_live(self):
        assert "MOCK" not in RiskLimits().allowed_for(OperatingMode.LIVE)


class TestLimitsAndKillSwitch:
    def test_daily_loss_limit_blocks_and_engages(self, engine, ks):
        v = engine.evaluate(req(), pf(equity=Decimal("97900"), peak_equity=Decimal("100000")), mkt())
        assert not v.approved and "daily_loss_limit" in failed(v)
        assert ks.engaged

    def test_drawdown_limit_blocks_and_engages(self, engine, ks):
        v = engine.evaluate(req(), pf(equity=Decimal("89000"), day_start_equity=Decimal("89000"),
                                      peak_equity=Decimal("100000")), mkt())
        assert not v.approved and "drawdown_limit" in failed(v) and ks.engaged

    def test_daily_budget_shrinks_size(self, engine):
        full = engine.evaluate(req(), pf(), mkt())
        partial = engine.evaluate(req(), pf(equity=Decimal("98300")), mkt())
        assert partial.max_loss <= Decimal("300") + Decimal("1e-6") < full.max_loss

    def test_kill_switch_blocks_new_risk(self, engine, ks):
        ks.engage("test")
        v = engine.evaluate(req(), pf(), mkt())
        assert not v.approved and "kill_switch" in failed(v)

    def test_kill_switch_permits_reducing_when_reconciled(self, engine, ks):
        ks.engage("test")
        held = pf(positions=(("BTC/USD", Decimal("0.5"), Decimal("30000")),))
        v = engine.evaluate(req(side=Side.SELL, stop_price=None), held, mkt())
        assert v.approved and v.qty == Decimal("0.5")
        v2 = engine.evaluate(req(side=Side.SELL, stop_price=None), replace(held, reconciliation=ReconState.UNKNOWN), mkt())
        assert not v2.approved

    def test_sell_never_exceeds_position(self, engine):
        held = pf(positions=(("BTC/USD", Decimal("0.5"), Decimal("30000")),))
        v = engine.evaluate(req(side=Side.SELL, proposed_qty=Decimal("3")), held, mkt())
        assert v.qty == Decimal("0.5")

    def test_kill_switch_persists_and_fails_closed(self, tmp_path, clock):
        path = tmp_path / "k.json"
        KillSwitch(path, clock).engage("x")
        assert KillSwitch(path, clock).engaged
        path.write_text("{garbage")
        assert KillSwitch(path, clock).engaged
        path.write_text('{"engaged": "no"}')
        assert KillSwitch(path, clock).engaged

    def test_kill_switch_release_requires_operator_ack(self, ks):
        ks.engage("x")
        with pytest.raises(PermissionError):
            ks.release("please")
        ks.release(RELEASE_ACK)
        assert not ks.engaged


class TestApprovalIntegrity:
    def test_forged_or_altered_approval_rejected(self, engine):
        v = engine.evaluate(req(), pf(), mkt())
        engine.authority.verify(v)
        with pytest.raises(ApprovalInvalid):
            engine.authority.verify(replace(v, qty=v.qty * 100))
        with pytest.raises(ApprovalInvalid):
            engine.authority.verify(replace(v, token="0" * 64))
        with pytest.raises(ApprovalInvalid):
            ApprovalAuthority().verify(v)  # different process / authority
        with pytest.raises(ApprovalInvalid):
            engine.authority.verify({"approved": True})

    def test_every_check_recorded(self, engine):
        v = engine.evaluate(req(), pf(), mkt())
        names = [c.name for c in v.checks]
        for required in ("mode_permitted", "reconciliation", "data_fresh", "kill_switch", "daily_loss_limit",
                         "drawdown_limit", "stop_valid", "sizing", "max_trade_loss"):
            assert required in names
