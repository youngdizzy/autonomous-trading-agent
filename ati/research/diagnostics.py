"""Development-only diagnostic experiments: REGIME, EXECUTION and RISK.

They answer research questions about the *baseline* without creating a candidate strategy:

  REGIME     does the baseline's out-of-sample edge exist inside one point-in-time regime label?
  EXECUTION  does it survive harsher execution (cost multiplier >= 1, or entry delay)? Costs can only be
             stressed, never made cheaper than the model — an "improvement" by assuming cheaper fills is refused.
  RISK       how do backtest outcomes respond to a different per-trade risk fraction? Backtest sizing only:
             the live RiskEngine limits are code-owned and never read or written here.

Every diagnostic is pre-registered and recorded in the ResearchLog like any experiment (so it counts for
multiple testing). It runs on a DEVELOPMENT partition only (``require_not_holdout``), never touches the
holdout, and can never produce a challenger or a promotion: development evidence alone is not validation.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from ati.data.dataset import Dataset
from ati.research.backtest import BacktestConfig
from ati.research.metrics import Metrics, compute_metrics
from ati.research.robustness import regime_labels
from ati.research.walkforward import WalkForwardResult, _pseudo_result, walk_forward
from ati.strategies.base import StrategyDefinition

REGIME_LABELS = ("HIGH_VOL/UP", "HIGH_VOL/DOWN", "LOW_VOL/UP", "LOW_VOL/DOWN")
EXECUTION_BOUNDS = {"cost_multiplier": (1.0, 5.0), "entry_delay_bars": (0, 5)}
RISK_BOUNDS = {"risk_fraction": (0.001, 0.02)}


def _wf(base: StrategyDefinition, dev: Dataset, cfg: BacktestConfig, train_bars: int, test_bars: int) -> WalkForwardResult:
    dev.require_not_holdout("diagnostic experiment")
    return walk_forward(base, dev, [base.param_dict], train_bars=train_bars, test_bars=test_bars, config=cfg)


def regime(base, dev, label: str, *, train_bars: int, test_bars: int) -> tuple[Metrics, str, dict]:
    if label not in REGIME_LABELS:
        raise ValueError(f"unknown regime {label!r}")
    wf = _wf(base, dev, BacktestConfig(), train_bars, test_bars)
    labels = regime_labels(dev)
    subset = [t for t in wf.oos_trades if labels.get(t.decided_at) is not None
              and "/".join(labels[t.decided_at]) == label]
    m = compute_metrics(_pseudo_result(base, dev, BacktestConfig(), subset), base.timeframe.bars_per_year)
    return m, wf.evidence_hash, {"regime": label, "regime_trades": len(subset), "all_oos_trades": len(wf.oos_trades)}


def execution(base, dev, condition: dict, *, train_bars: int, test_bars: int) -> tuple[Metrics, str, dict]:
    cfg = BacktestConfig()
    if "cost_multiplier" in condition:
        k = Decimal(str(condition["cost_multiplier"]))
        if not Decimal(1) <= k <= Decimal(5):
            raise ValueError("cost_multiplier must be within [1, 5]: costs may be stressed, never reduced")
        c = cfg.costs
        cfg = replace(cfg, costs=replace(c, fee_rate=c.fee_rate * k, half_spread_rate=c.half_spread_rate * k,
                                         slippage_rate=c.slippage_rate * k))
    elif "entry_delay_bars" in condition:
        delay = condition["entry_delay_bars"]
        if not isinstance(delay, int) or not 0 <= delay <= 5:
            raise ValueError("entry_delay_bars must be an integer within [0, 5]")
        cfg = replace(cfg, entry_delay_bars=delay)
    else:
        raise ValueError("EXECUTION condition must be cost_multiplier or entry_delay_bars")
    wf = _wf(base, dev, cfg, train_bars, test_bars)
    return wf.oos_metrics, wf.evidence_hash, dict(condition)


def risk(base, dev, condition: dict, *, train_bars: int, test_bars: int) -> tuple[Metrics, str, dict]:
    f = condition.get("risk_fraction")
    lo, hi = RISK_BOUNDS["risk_fraction"]
    if set(condition) != {"risk_fraction"} or not isinstance(f, (int, float)) or not lo <= f <= hi:
        raise ValueError(f"RISK condition must be risk_fraction within [{lo}, {hi}] (backtest sizing only)")
    wf = _wf(base, dev, replace(BacktestConfig(), risk_fraction=Decimal(str(f))), train_bars, test_bars)
    return wf.oos_metrics, wf.evidence_hash, {"risk_fraction": f, "scope": "backtest sizing only; live limits unchanged"}
