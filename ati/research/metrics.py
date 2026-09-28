"""Research metrics. Each metric states the question it answers. Where a sample is too small to
answer, the metric is ``None`` and callers must report INSUFFICIENT EVIDENCE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from ati.research.backtest import BacktestResult

MIN_TRADES_FOR_STATS = 20


@dataclass(frozen=True)
class Metrics:
    n_trades: int                      # How much evidence is there?
    net_pnl: float                     # Did it make money after all modeled costs?
    net_return: float                  # ...relative to starting equity?
    max_drawdown: float                # What is the worst peak-to-trough loss we would have endured?
    expectancy_r: float | None         # Average outcome per trade in units of risk taken.
    win_rate: float | None             # Is the edge from frequency or from payoff asymmetry?
    profit_factor: float | None        # Gross wins / gross losses.
    sharpe_annualized: float | None    # Return per unit of volatility (bar returns).
    mean_trade_return: float | None
    t_stat_trade_return: float | None  # Is mean trade return distinguishable from zero?
    exposure: float                    # Fraction of time capital was at risk.
    total_fees: float
    total_spread_slippage: float       # What did execution cost?
    cost_share_of_gross: float | None  # How much of the gross edge do costs consume?
    top5_profit_share: float | None    # Is the result driven by a handful of trades?
    partial_fills: int

    @property
    def sufficient(self) -> bool:
        return self.n_trades >= MIN_TRADES_FOR_STATS


def _max_drawdown(values: list[float]) -> float:
    peak, worst = -math.inf, 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak)
    return worst


def compute_metrics(result: BacktestResult, bars_per_year: float) -> Metrics:
    trades = result.trades
    initial = float(result.config.initial_equity)
    equity = [float(v) for _, v in result.equity_curve]
    net = float(sum((t.net_pnl for t in trades), Decimal(0)))
    gross = float(sum((t.gross_pnl for t in trades), Decimal(0)))
    fees = float(sum((t.fees for t in trades), Decimal(0)))
    slip = float(sum((t.spread_slip_cost for t in trades), Decimal(0)))
    n = len(trades)
    enough = n >= MIN_TRADES_FOR_STATS
    rets = [t.net_return for t in trades]
    wins = [float(t.net_pnl) for t in trades if t.net_pnl > 0]
    losses = [-float(t.net_pnl) for t in trades if t.net_pnl <= 0]
    mean = sum(rets) / n if n else None
    t_stat = None
    if enough:
        sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (n - 1))
        t_stat = mean / (sd / math.sqrt(n)) if sd > 0 else None
    bar_rets = [(b / a - 1) for a, b in zip(equity, equity[1:]) if a > 0]
    sharpe = None
    if enough and len(bar_rets) > 2:
        m = sum(bar_rets) / len(bar_rets)
        sd = math.sqrt(sum((r - m) ** 2 for r in bar_rets) / (len(bar_rets) - 1))
        sharpe = (m / sd) * math.sqrt(bars_per_year) if sd > 0 else None
    top5 = None
    if net > 0 and n >= 5:
        top5 = sum(sorted((float(t.net_pnl) for t in trades), reverse=True)[:5]) / net
    return Metrics(
        n_trades=n,
        net_pnl=net,
        net_return=net / initial,
        max_drawdown=_max_drawdown([initial] + equity),
        expectancy_r=(sum(t.r_multiple for t in trades) / n) if enough else None,
        win_rate=(len(wins) / n) if enough else None,
        profit_factor=(sum(wins) / sum(losses)) if enough and sum(losses) > 0 else None,
        sharpe_annualized=sharpe,
        mean_trade_return=mean if enough else None,
        t_stat_trade_return=t_stat,
        exposure=result.exposure_bars / result.bars if result.bars else 0.0,
        total_fees=fees,
        total_spread_slippage=slip,
        cost_share_of_gross=((fees + slip) / gross) if gross > 0 else None,
        top5_profit_share=top5,
        partial_fills=result.partial_fills,
    )
