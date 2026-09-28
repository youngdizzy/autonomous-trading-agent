"""Walk-forward validation: select parameters on a development window, evaluate on the following
unseen window, roll forward. Only out-of-sample windows count as evidence. The number of
configurations tried is recorded because it determines how much a good result should be
discounted (multiple testing)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from ati.core.canonical import sha256_hex
from ati.data.dataset import Dataset
from ati.research.backtest import BacktestConfig, Trade, run_backtest
from ati.research.metrics import MIN_TRADES_FOR_STATS, Metrics, compute_metrics
from ati.strategies.base import StrategyDefinition


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: datetime
    train_end: datetime
    test_end: datetime
    selected_params: tuple
    in_sample_score: float | None
    oos_trades: int
    oos_net_pnl: float


@dataclass(frozen=True)
class WalkForwardResult:
    strategy_id: str
    kind: str
    dataset_id: str
    grid_size: int
    configs_tried: int
    objective: str
    folds: tuple[Fold, ...]
    oos_metrics: Metrics
    oos_trades: tuple[Trade, ...]
    positive_fold_fraction: float

    @property
    def evidence_hash(self) -> str:
        return sha256_hex({
            "kind": "walk_forward", "strategy_id": self.strategy_id, "dataset_id": self.dataset_id,
            "grid_size": self.grid_size, "configs_tried": self.configs_tried, "objective": self.objective,
            "folds": [(f.index, f.selected_params, f.oos_trades, round(f.oos_net_pnl, 6)) for f in self.folds],
        })


def _score(metrics: Metrics, objective: str) -> float | None:
    if metrics.n_trades < max(5, MIN_TRADES_FOR_STATS // 4):
        return None  # too few trades to select on
    if objective == "net_pnl":
        return metrics.net_pnl
    if objective == "expectancy":
        return metrics.expectancy_r if metrics.expectancy_r is not None else metrics.net_pnl / max(metrics.n_trades, 1)
    raise ValueError(f"unknown objective {objective}")


def walk_forward(base: StrategyDefinition, dataset: Dataset, grid: list[dict], *, train_bars: int, test_bars: int,
                 config: BacktestConfig = BacktestConfig(), objective: str = "net_pnl") -> WalkForwardResult:
    dataset.require_not_holdout("walk_forward")
    if not grid:
        raise ValueError("empty parameter grid")
    candidates = [StrategyDefinition.create(base.strategy_id, base.version, base.kind, p, base.timeframe,
                                            base.created_at, base.parent_hash, "walk-forward candidate") for p in grid]
    lookback = max(c.lookback for c in candidates)
    candles = dataset.candles
    bpy = base.timeframe.bars_per_year
    folds: list[Fold] = []
    oos_trades: list[Trade] = []
    start = 0
    index = 0
    while start + train_bars + test_bars <= len(candles):
        train = _subset(dataset, start, start + train_bars)
        best, best_score = None, None
        for cand in candidates:
            score = _score(compute_metrics(run_backtest(cand, train, config), bpy), objective)
            if score is not None and (best_score is None or score > best_score):
                best, best_score = cand, score
        test_lo = start + train_bars
        test_hi = test_lo + test_bars
        if best is None:
            folds.append(Fold(index, candles[start].open_time, candles[test_lo].open_time, candles[test_hi - 1].close_time,
                              (), None, 0, 0.0))
        else:
            # Warm-up bars before the test window are past data; entries are only allowed inside it.
            warm = _subset(dataset, max(0, test_lo - lookback), test_hi)
            res = run_backtest(best, warm, config, trade_start=candles[test_lo].open_time)
            trades = [t for t in res.trades if t.decided_at >= candles[test_lo].open_time]
            oos_trades.extend(trades)
            folds.append(Fold(index, candles[start].open_time, candles[test_lo].open_time, candles[test_hi - 1].close_time,
                              best.params, best_score, len(trades), float(sum((t.net_pnl for t in trades), Decimal(0)))))
        start += test_bars
        index += 1
    if not folds:
        raise ValueError("dataset too short for requested walk-forward windows")
    pseudo = _pseudo_result(base, dataset, config, oos_trades)
    oos = compute_metrics(pseudo, bpy)
    active = [f for f in folds if f.oos_trades > 0]
    positive = sum(1 for f in active if f.oos_net_pnl > 0) / len(active) if active else 0.0
    return WalkForwardResult(base.strategy_id, base.kind, dataset.dataset_id, len(grid), len(grid) * len(folds),
                             objective, tuple(folds), oos, tuple(oos_trades), positive)


def _subset(dataset: Dataset, lo: int, hi: int) -> Dataset:
    return Dataset.build(dataset.candles[lo:hi], data_version=dataset.identity.data_version,
                         realization=dataset.identity.realization, partition=dataset.partition)


def _pseudo_result(base, dataset, config, trades):
    """Sequential equity path of OOS trades only (for OOS drawdown and trade statistics)."""
    from ati.research.backtest import BacktestResult

    equity = config.initial_equity
    curve = []
    for t in trades:
        equity += t.net_pnl
        curve.append((t.exit_time, equity))
    return BacktestResult(base.key, base.definition_hash, dataset.dataset_id, config, list(trades), curve,
                          exposure_bars=0, bars=max(len(curve), 1), decisions=0, partial_fills=0)
