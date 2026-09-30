"""Deterministic risk engine — the safety authority.

Claude (or any strategy) supplies an ``OrderRequest``. The engine computes whether it is
mechanically permitted and, if so, the *permitted* quantity. A proposed quantity can only lower
the size, never raise it. The result is a ``RiskVerdict`` signed by the ``ApprovalAuthority``;
the execution engine accepts nothing else. Every check is recorded, pass or fail.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from ati.config import OperatingMode, assert_mode_permitted
from ati.core.canonical import canonical_json, sha256_hex
from ati.core.errors import ApprovalInvalid, LiveTradingDisabled
from ati.core.time import Clock, ensure_utc
from ati.core.types import Side
from ati.market.models import DataStatus, Timeframe
from ati.market.universe import Universe
from ati.risk.killswitch import KillSwitch
from ati.risk.sizing import floor_to_step, per_unit_risk


class ReconState(str, Enum):
    OK = "OK"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PortfolioSnapshot:
    as_of: datetime
    mode: OperatingMode
    equity: Decimal
    cash: Decimal
    positions: tuple[tuple[str, Decimal, Decimal], ...]  # (symbol, qty, mark)
    day_start_equity: Decimal
    peak_equity: Decimal
    reconciliation: ReconState
    account_state_known: bool

    def qty(self, symbol: str) -> Decimal:
        return next((q for s, q, _ in self.positions if s == symbol), Decimal(0))

    def exposure(self, symbols: set[str] | None = None) -> Decimal:
        return sum((abs(q) * m for s, q, m in self.positions if symbols is None or s in symbols), Decimal(0))


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    timeframe: Timeframe
    last_price: Decimal
    last_close_time: datetime
    status: DataStatus
    recent_volume: Decimal
    est_spread_rate: Decimal
    est_slippage_rate: Decimal


@dataclass(frozen=True)
class OrderRequest:
    decision_id: str
    strategy_key: str
    symbol: str
    side: Side
    entry_price: Decimal
    stop_price: Decimal | None
    proposed_qty: Decimal | None = None


@dataclass(frozen=True)
class RiskLimits:
    max_risk_per_trade_fraction: Decimal = Decimal("0.005")
    max_trade_loss: Decimal = Decimal("1000")
    daily_loss_limit_fraction: Decimal = Decimal("0.02")
    max_drawdown_fraction: Decimal = Decimal("0.10")
    max_portfolio_exposure_fraction: Decimal = Decimal("1.0")
    max_symbol_exposure_fraction: Decimal = Decimal("0.5")
    max_correlated_exposure_fraction: Decimal = Decimal("0.75")
    max_leverage: Decimal = Decimal("1.0")
    max_order_notional: Decimal = Decimal("25000")
    max_slippage_rate: Decimal = Decimal("0.0025")
    max_participation: Decimal = Decimal("0.05")
    max_price_deviation: Decimal = Decimal("0.02")
    fee_rate: Decimal = Decimal("0.0026")
    max_data_age: timedelta = timedelta(hours=2)
    approval_ttl: timedelta = timedelta(seconds=60)
    allowed_data: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("LIVE", ("REAL",)),
        ("PAPER", ("REAL", "DELAYED", "MOCK")),
        ("BACKTEST", ("REAL", "HISTORICAL", "DELAYED", "MOCK", "SYNTHETIC")),
    )

    def allowed_for(self, mode: OperatingMode) -> tuple[str, ...]:
        return dict(self.allowed_data).get(mode.value, ())

    @property
    def limits_hash(self) -> str:
        return sha256_hex({k: (v.total_seconds() if isinstance(v, timedelta) else v)
                           for k, v in self.__dict__.items()})


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RiskVerdict:
    decision_id: str
    strategy_key: str
    symbol: str
    side: Side
    approved: bool
    qty: Decimal
    entry_price: Decimal
    stop_price: Decimal | None
    max_loss: Decimal
    checks: tuple[Check, ...]
    mode: OperatingMode
    evaluated_at: datetime
    valid_until: datetime
    limits_hash: str
    token: str = field(default="", repr=False)

    def signing_payload(self) -> str:
        return canonical_json(replace(self, token=""))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


class ApprovalAuthority:
    """Holds a per-process signing key shared only by the risk engine (signer) and the execution
    engine (verifier). Approvals therefore cannot be forged or edited by any other layer and do
    not survive a restart (a restarted process must re-evaluate risk)."""

    def __init__(self) -> None:
        self.__key = secrets.token_bytes(32)

    def _mac(self, verdict: RiskVerdict) -> str:
        return hmac.new(self.__key, verdict.signing_payload().encode(), hashlib.sha256).hexdigest()

    def sign(self, verdict: RiskVerdict) -> RiskVerdict:
        return replace(verdict, token=self._mac(verdict))

    def verify(self, verdict: object) -> RiskVerdict:
        if not isinstance(verdict, RiskVerdict):
            raise ApprovalInvalid("execution requires a RiskVerdict")
        if not verdict.token or not hmac.compare_digest(self._mac(verdict), verdict.token):
            raise ApprovalInvalid("risk approval signature invalid (forged or altered)")
        return verdict


class RiskEngine:
    def __init__(self, limits: RiskLimits, universe: Universe, kill_switch: KillSwitch, authority: ApprovalAuthority,
                 clock: Clock):
        self.limits = limits
        self.universe = universe
        self.kill_switch = kill_switch
        self.authority = authority
        self.clock = clock

    def evaluate(self, req: OrderRequest, pf: PortfolioSnapshot, mkt: MarketSnapshot) -> RiskVerdict:
        L = self.limits
        now = self.clock.now()
        checks: list[Check] = []

        def check(name: str, ok: bool, detail: str) -> bool:
            checks.append(Check(name, bool(ok), detail))
            return bool(ok)

        def verdict(qty: Decimal = Decimal(0), max_loss: Decimal = Decimal(0)) -> RiskVerdict:
            approved = all(c.passed for c in checks) and qty > 0
            v = RiskVerdict(req.decision_id, req.strategy_key, req.symbol, req.side, approved,
                            qty if approved else Decimal(0), req.entry_price, req.stop_price,
                            max_loss if approved else Decimal(0), tuple(checks), pf.mode, now,
                            now + L.approval_ttl, L.limits_hash)
            return self.authority.sign(v) if approved else v

        # --- structural gates (any failure → immediate rejection) ----------------------------
        try:
            assert_mode_permitted(pf.mode)
            check("mode_permitted", True, pf.mode.value)
        except LiveTradingDisabled as exc:
            check("mode_permitted", False, str(exc))
            return verdict()
        if not check("account_state_known", pf.account_state_known,
                     "account state known" if pf.account_state_known else "UNKNOWN account state"):
            return verdict()
        if not check("reconciliation", pf.reconciliation is ReconState.OK, pf.reconciliation.value):
            return verdict()
        if not check("symbol_known", req.symbol in self.universe and mkt.symbol == req.symbol, req.symbol):
            return verdict()
        inst = self.universe.get(req.symbol)
        if not check("data_status", mkt.status.value in L.allowed_for(pf.mode), f"{mkt.status.value} data in {pf.mode.value}"):
            return verdict()
        age = now - ensure_utc(mkt.last_close_time)
        if not check("data_fresh", timedelta(0) <= age <= L.max_data_age, f"latest data age {age}"):
            return verdict()
        if not check("price_positive", req.entry_price > 0 and mkt.last_price > 0, f"entry {req.entry_price}"):
            return verdict()
        deviation = abs(req.entry_price / mkt.last_price - 1)
        if not check("price_sane", deviation <= L.max_price_deviation, f"entry deviates {deviation:.4%} from last price"):
            return verdict()
        est_cost = mkt.est_spread_rate + mkt.est_slippage_rate
        if not check("slippage", est_cost <= L.max_slippage_rate, f"estimated spread+slippage {est_cost}"):
            return verdict()
        if req.proposed_qty is not None and not check("proposed_qty_valid", req.proposed_qty > 0, f"{req.proposed_qty}"):
            return verdict()

        held = pf.qty(req.symbol)

        # --- risk-reducing orders (permitted under kill switch, but only when reconciled) ----
        if req.side is Side.SELL:
            if not check("reduces_position", held > 0, f"position {held}; shorting not permitted"):
                return verdict()
            qty = min(held, req.proposed_qty) if req.proposed_qty else held
            qty = floor_to_step(qty, inst.lot_step)
            check("kill_switch", True, "risk-reducing order permitted while reconciled")
            return verdict(qty, Decimal(0))

        # --- new risk -------------------------------------------------------------------------
        ks = self.kill_switch.state()
        if not check("kill_switch", not ks["engaged"], f"engaged: {ks['reason']}" if ks["engaged"] else "released"):
            return verdict()
        if pf.peak_equity > 0:
            dd = 1 - pf.equity / pf.peak_equity
            if not check("drawdown_limit", dd < L.max_drawdown_fraction,
                         f"drawdown {dd:.2%} (limit {L.max_drawdown_fraction:.0%})"):
                self.kill_switch.engage(f"drawdown limit breached: {dd:.2%}")
                return verdict()
        day_loss = pf.day_start_equity - pf.equity
        day_limit = pf.day_start_equity * L.daily_loss_limit_fraction
        if not check("daily_loss_limit", day_loss < day_limit, f"today's loss {day_loss:.2f} (limit {day_limit:.2f})"):
            self.kill_switch.engage(f"daily loss limit breached: {day_loss:.2f}")
            return verdict()
        if not check("stop_present", req.stop_price is not None, "stop required for new risk"):
            return verdict()
        if not check("stop_valid", Decimal(0) < req.stop_price < req.entry_price,
                     f"stop {req.stop_price} vs entry {req.entry_price}"):
            return verdict()

        round_trip = 2 * (L.fee_rate + mkt.est_spread_rate + mkt.est_slippage_rate)
        unit_risk = per_unit_risk(req.entry_price, req.stop_price, round_trip)
        budget = min(pf.equity * L.max_risk_per_trade_fraction, L.max_trade_loss, day_limit - day_loss)
        group = {s for s in self.universe.symbols if self.universe.get(s).correlation_group == inst.correlation_group}
        gross = pf.exposure()
        entry = req.entry_price
        caps = {
            "risk_budget": budget / unit_risk,
            "max_order_notional": L.max_order_notional / entry,
            "symbol_exposure": (pf.equity * L.max_symbol_exposure_fraction - pf.exposure({req.symbol})) / entry,
            "portfolio_exposure": (pf.equity * L.max_portfolio_exposure_fraction - gross) / entry,
            "correlated_exposure": (pf.equity * L.max_correlated_exposure_fraction - pf.exposure(group)) / entry,
            "leverage": (pf.equity * L.max_leverage - gross) / entry,
            "cash": pf.cash / (entry * (1 + L.fee_rate + mkt.est_spread_rate + mkt.est_slippage_rate)),
            "liquidity": mkt.recent_volume * L.max_participation,
        }
        if req.proposed_qty is not None:
            caps["proposed"] = req.proposed_qty
        binding = min(caps, key=caps.get)
        qty = floor_to_step(max(caps[binding], Decimal(0)), inst.lot_step)
        check("sizing", qty >= inst.min_qty,
              f"qty {qty} (binding: {binding}; min {inst.min_qty}); caps: "
              + ", ".join(f"{k}={max(v, Decimal(0)):.6f}" for k, v in sorted(caps.items())))
        max_loss = qty * unit_risk
        check("max_trade_loss", max_loss <= min(L.max_trade_loss, budget) + Decimal("1e-9"),
              f"loss at stop incl. costs {max_loss:.2f} (budget {budget:.2f})")
        return verdict(qty, max_loss)
