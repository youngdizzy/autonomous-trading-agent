"""P&L attribution. Separates where results came from by re-running the same strategy on the same
data under controlled variations:

  signal     = P&L with frictionless fills and fixed notional per trade (pure direction/timing)
  sizing     = frictionless risk-sized P&L − signal (what position sizing added or removed)
  execution  = −(spread + slippage) in the real-cost run
  fees       = −fees in the real-cost run
  interaction= net − (signal + sizing + execution + fees)

Because costs change position sizes and therefore paths, the components do not sum exactly; the
residual is reported explicitly as ``interaction`` instead of being hidden. Regime contribution is
net P&L grouped by the point-in-time regime at decision time.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from ati.data.dataset import Dataset
from ati.research.backtest import BacktestConfig, run_backtest
from ati.research.costs import CostModel
from ati.research.robustness import regime_breakdown


def attribute(defn, ds: Dataset, cfg: BacktestConfig) -> dict:
    ds.require_not_holdout("attribution")
    free = CostModel.frictionless()
    unit = run_backtest(defn, ds, replace(cfg, costs=free, sizing="unit"))
    sized = run_backtest(defn, ds, replace(cfg, costs=free, sizing="risk"))
    real = run_backtest(defn, ds, cfg)

    def total(res, attr):
        return float(sum((getattr(t, attr) for t in res.trades), Decimal(0)))

    signal = total(unit, "net_pnl")
    sizing = total(sized, "net_pnl") - signal
    execution = -total(real, "spread_slip_cost")
    fees = -total(real, "fees")
    net = total(real, "net_pnl")
    return {
        "signal": signal, "sizing": sizing, "execution": execution, "fees": fees,
        "interaction": net - (signal + sizing + execution + fees), "net": net,
        "unit_notional": float(cfg.unit_notional),
        "regime": regime_breakdown(defn, ds, cfg),
    }
