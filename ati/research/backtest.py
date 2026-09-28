"""Event-driven bar backtester with explicit temporal discipline.

Timeline for bar i:
  1. At bar i OPEN: fill orders decided at the close of bar i-1-delay (never earlier than decided).
  2. During bar i: if long, a stop at S triggers when low <= S; the exit fills at min(open, S)
     (gap-aware) less spread and slippage.
  3. At bar i CLOSE (= decision time T): the strategy sees ``dataset.view_at(T)`` — only bars
     closed at or before T — and may request an entry/exit for a later open.

Fills may be partial: a single fill takes at most ``max_participation`` of the fill bar's volume
(the unfilled remainder is cancelled). Sizing uses ``ati.risk.sizing`` with information at T only.
A leakage audit asserts for every decision that no visible data was available after T and every
fill happened strictly after its decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from ati.core.errors import LookaheadError
from ati.core.types import Side
from ati.data.dataset import Dataset
from ati.research.costs import CostModel
from ati.risk.sizing import floor_to_step, risk_based_qty
from ati.strategies.base import StrategyDefinition, Target


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: Decimal = Decimal("100000")
    costs: CostModel = CostModel()
    risk_fraction: Decimal = Decimal("0.01")      # equity risked per trade if the stop is hit
    max_notional_fraction: Decimal = Decimal("1")  # 1 = no leverage
    lot_step: Decimal = Decimal("0.00001")
    entry_delay_bars: int = 0                     # timing robustness: extra bars between decision and fill
    sizing: str = "risk"                          # "risk" | "unit" (fixed notional, for attribution)
    unit_notional: Decimal = Decimal("10000")


@dataclass(frozen=True)
class Trade:
    decided_at: datetime
    entry_time: datetime
    entry_mid: Decimal
    entry_price: Decimal
    qty: Decimal
    requested_qty: Decimal
    initial_stop: Decimal
    exit_decided_at: datetime | None
    exit_time: datetime
    exit_mid: Decimal
    exit_price: Decimal
    exit_reason: str
    gross_pnl: Decimal       # at mid prices, before any cost
    spread_slip_cost: Decimal
    fees: Decimal
    net_pnl: Decimal
    initial_risk: Decimal    # qty * per-unit risk at decision

    @property
    def r_multiple(self) -> float:
        return float(self.net_pnl / self.initial_risk) if self.initial_risk > 0 else 0.0

    @property
    def net_return(self) -> float:
        return float(self.net_pnl / (self.entry_mid * self.qty))


@dataclass
class BacktestResult:
    strategy_key: str
    strategy_hash: str
    dataset_id: str
    config: BacktestConfig
    trades: list[Trade]
    equity_curve: list[tuple[datetime, Decimal]]
    exposure_bars: int
    bars: int
    decisions: int
    partial_fills: int
    audit: dict = field(default_factory=dict)


@dataclass
class _Pending:
    side: Side
    decided_at: datetime
    fill_index: int
    qty: Decimal
    stop: Decimal | None
    reason: str
    initial_risk: Decimal = Decimal(0)


def run_backtest(strategy: StrategyDefinition, dataset: Dataset, config: BacktestConfig = BacktestConfig(),
                 *, trade_start: datetime | None = None, allow_holdout: bool = False) -> BacktestResult:
    if not allow_holdout:
        dataset.require_not_holdout("run_backtest")
    dataset.verify()
    candles = dataset.candles
    costs = config.costs
    lookback = strategy.lookback
    cash = config.initial_equity
    qty_held = Decimal(0)
    pos: dict | None = None
    pending: _Pending | None = None
    trades: list[Trade] = []
    curve: list[tuple[datetime, Decimal]] = []
    exposure = decisions = partial = 0
    max_seen_vs_cutoff = None

    def close_position(i: int, mid: Decimal, reason: str, exit_decided_at: datetime | None) -> None:
        nonlocal cash, qty_held, pos
        price = costs.fill_price(mid, Side.SELL)
        fee = price * qty_held * costs.fee_rate
        cash += price * qty_held - fee
        gross = (mid - pos["entry_mid"]) * qty_held
        slip = (pos["entry_price"] - pos["entry_mid"]) * qty_held + (mid - price) * qty_held
        fees = pos["entry_fee"] + fee
        trades.append(Trade(
            decided_at=pos["decided_at"], entry_time=pos["entry_time"], entry_mid=pos["entry_mid"],
            entry_price=pos["entry_price"], qty=qty_held, requested_qty=pos["requested"], initial_stop=pos["initial_stop"],
            exit_decided_at=exit_decided_at, exit_time=candles[i].open_time, exit_mid=mid, exit_price=price,
            exit_reason=reason, gross_pnl=gross, spread_slip_cost=slip, fees=fees, net_pnl=gross - slip - fees,
            initial_risk=pos["initial_risk"],
        ))
        qty_held = Decimal(0)
        pos = None

    for i, bar in enumerate(candles):
        # 1. fills at the open
        if pending is not None and pending.fill_index == i:
            if bar.open_time < pending.decided_at:
                raise LookaheadError("fill before decision")
            if pending.side is Side.BUY and pos is None:
                cap = bar.volume * costs.max_participation
                fill_qty = floor_to_step(min(pending.qty, cap), config.lot_step)
                if fill_qty < pending.qty:
                    partial += 1
                if fill_qty > 0:
                    price = costs.fill_price(bar.open, Side.BUY)
                    fee = price * fill_qty * costs.fee_rate
                    cash -= price * fill_qty + fee
                    qty_held = fill_qty
                    pos = {"decided_at": pending.decided_at, "entry_time": bar.open_time, "entry_mid": bar.open,
                           "entry_price": price, "entry_fee": fee, "stop": pending.stop, "initial_stop": pending.stop,
                           "requested": pending.qty,
                           "initial_risk": pending.initial_risk * (fill_qty / pending.qty)}
            elif pending.side is Side.SELL and pos is not None:
                close_position(i, bar.open, pending.reason, pending.decided_at)
            pending = None
        # 2. intrabar stop
        if pos is not None and bar.low <= pos["stop"]:
            close_position(i, min(bar.open, pos["stop"]), "stop", None)
        # 3. decision at the close
        cutoff = bar.close_time
        in_trading = trade_start is None or cutoff >= trade_start
        if i >= lookback - 1 and pending is None:
            view = dataset.view_at(cutoff, lookback)
            seen = view.max_available_at()
            if seen is not None and seen > cutoff:
                raise LookaheadError("view exposed data after cutoff")
            max_seen_vs_cutoff = seen if max_seen_vs_cutoff is None else max(max_seen_vs_cutoff, seen)
            signal = strategy.signal(view, pos is not None)
            decisions += 1
            fill_index = i + 1 + config.entry_delay_bars
            if fill_index < len(candles):
                if pos is None and signal.target is Target.LONG and in_trading and signal.stop_price and signal.stop_price < bar.close:
                    equity_now = cash
                    if config.sizing == "risk":
                        budget = equity_now * config.risk_fraction
                        qty = risk_based_qty(budget, bar.close, signal.stop_price, costs.round_trip_rate)
                    else:
                        qty = config.unit_notional / bar.close
                    qty = min(qty, equity_now * config.max_notional_fraction / (bar.close * (1 + costs.round_trip_rate)))
                    qty = floor_to_step(qty, config.lot_step)
                    if qty > 0:
                        per_unit = (bar.close - signal.stop_price) + bar.close * costs.round_trip_rate
                        pending = _Pending(Side.BUY, cutoff, fill_index, qty, signal.stop_price, signal.reason, qty * per_unit)
                elif pos is not None and signal.target is Target.FLAT:
                    pending = _Pending(Side.SELL, cutoff, fill_index, qty_held, None, "signal_exit")
                elif pos is not None and signal.stop_price and signal.stop_price > pos["stop"]:
                    pos["stop"] = signal.stop_price  # ratchet only upward
        if pos is not None:
            exposure += 1
        curve.append((cutoff, cash + qty_held * bar.close))

    if pos is not None:
        # Mark the open position closed at the final close so trade statistics are complete; flagged.
        last = len(candles) - 1
        mid = candles[last].close
        price = costs.fill_price(mid, Side.SELL)
        fee = price * qty_held * costs.fee_rate
        gross = (mid - pos["entry_mid"]) * qty_held
        slip = (pos["entry_price"] - pos["entry_mid"]) * qty_held + (mid - price) * qty_held
        fees = pos["entry_fee"] + fee
        cash += price * qty_held - fee
        trades.append(Trade(pos["decided_at"], pos["entry_time"], pos["entry_mid"], pos["entry_price"], qty_held,
                            pos["requested"], pos["initial_stop"], None, candles[last].close_time, mid, price,
                            "end_of_data", gross, slip, fees, gross - slip - fees, pos["initial_risk"]))
        curve[-1] = (curve[-1][0], cash)

    return BacktestResult(
        strategy_key=strategy.key, strategy_hash=strategy.definition_hash, dataset_id=dataset.dataset_id,
        config=config, trades=trades, equity_curve=curve, exposure_bars=exposure, bars=len(candles),
        decisions=decisions, partial_fills=partial,
        audit={"max_visible_available_at": max_seen_vs_cutoff, "lookahead_violations": 0,
               "data_status": dataset.identity.status.value},
    )
