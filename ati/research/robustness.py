"""Robustness probes. Each answers: does the result survive a plausible change in assumptions?"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from ati.data.dataset import Dataset
from ati.research.backtest import BacktestConfig, run_backtest
from ati.research.metrics import compute_metrics
from ati.strategies.base import StrategyDefinition, StrategyLogic
from ati.temporal.features import realized_vol, sma


@dataclass(frozen=True)
class ProbeResult:
    name: str
    variant: str
    n_trades: int
    net_pnl: float
    expectancy_r: float | None


def _run(defn, ds, cfg, name, variant) -> ProbeResult:
    m = compute_metrics(run_backtest(defn, ds, cfg), defn.timeframe.bars_per_year)
    return ProbeResult(name, variant, m.n_trades, m.net_pnl, m.expectancy_r)


def parameter_perturbation(defn: StrategyDefinition, ds: Dataset, cfg: BacktestConfig, rel: float = 0.2) -> list[ProbeResult]:
    """Does a small change to any single parameter destroy the result? (overfitting to a point)"""
    ds.require_not_holdout("parameter_perturbation")
    out = []
    logic: type[StrategyLogic] = defn.logic
    for name, value in defn.params:
        spec = logic.params[name]
        for direction in (-1, 1):
            if spec.kind is int:
                new = int(round(value * (1 + direction * rel)))
                if new == value:
                    new = value + direction
            else:
                new = round(value * (1 + direction * rel), 6)
            if not (spec.lo <= new <= spec.hi):
                continue
            params = defn.param_dict | {name: new}
            try:
                logic.validate_params(params)
            except ValueError:
                continue
            variant = StrategyDefinition.create(defn.strategy_id, defn.version, defn.kind, params, defn.timeframe,
                                                defn.created_at, defn.parent_hash, "perturbation")
            out.append(_run(variant, ds, cfg, "param", f"{name}={new}"))
    return out


def cost_stress(defn, ds, cfg: BacktestConfig, factors=(Decimal("1"), Decimal("2"), Decimal("3"))) -> list[ProbeResult]:
    """Does the edge survive higher fees/spread/slippage than assumed?"""
    ds.require_not_holdout("cost_stress")
    return [_run(defn, ds, replace(cfg, costs=cfg.costs.scaled(f)), "costs", f"x{f}") for f in factors]


def timing_stress(defn, ds, cfg: BacktestConfig, delays=(0, 1, 2)) -> list[ProbeResult]:
    """Does the edge depend on unrealistically immediate execution?"""
    ds.require_not_holdout("timing_stress")
    return [_run(defn, ds, replace(cfg, entry_delay_bars=d), "timing", f"delay={d}") for d in delays]


def liquidity_stress(defn, ds, cfg: BacktestConfig, caps=(Decimal("0.10"), Decimal("0.02"), Decimal("0.005"))) -> list[ProbeResult]:
    """Does the result depend on taking a large share of available volume?"""
    ds.require_not_holdout("liquidity_stress")
    return [_run(defn, ds, replace(cfg, costs=replace(cfg.costs, max_participation=c)), "liquidity", f"participation={c}")
            for c in caps]


def regime_labels(ds: Dataset, vol_window: int = 48, trend_window: int = 100) -> dict:
    """Point-in-time regime label for each bar close: (HIGH_VOL|LOW_VOL, UP|DOWN).
    Volatility is compared to the *expanding* median of past volatility, so labels use no future
    information."""
    labels = {}
    past_vols: list[float] = []
    for bar in ds.candles:
        view = ds.view_at(bar.close_time, max(vol_window + 1, trend_window))
        vol = realized_vol(view, vol_window)
        trend = sma(view, trend_window)
        if vol is None or trend is None:
            labels[bar.close_time] = None
            if vol is not None:
                past_vols.append(vol)
            continue
        median = sorted(past_vols)[len(past_vols) // 2] if past_vols else vol
        labels[bar.close_time] = ("HIGH_VOL" if vol > median else "LOW_VOL", "UP" if bar.close >= trend else "DOWN")
        past_vols.append(vol)
    return labels


def regime_breakdown(defn, ds, cfg) -> dict[str, dict]:
    """Is the edge concentrated in one market regime?"""
    ds.require_not_holdout("regime_breakdown")
    labels = regime_labels(ds)
    res = run_backtest(defn, ds, cfg)
    out: dict[str, dict] = {}
    for t in res.trades:
        label = labels.get(t.decided_at)
        key = "UNLABELED" if label is None else f"{label[0]}/{label[1]}"
        row = out.setdefault(key, {"n": 0, "net_pnl": 0.0})
        row["n"] += 1
        row["net_pnl"] += float(t.net_pnl)
    return out
