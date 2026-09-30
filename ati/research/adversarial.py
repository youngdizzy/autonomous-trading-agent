"""Adversarial research: every promising result is attacked before it is believed.

Each question is answered by computation where possible. Questions that cannot be computed are
reported as NOT_AUTOMATED (they go to the Claude adversarial reviewer and the record), never as
PASS. Any FAIL blocks promotion; any INSUFFICIENT_EVIDENCE also blocks promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

from ati.core.canonical import sha256_hex
from ati.data.dataset import Dataset, Partition
from ati.market.models import MARKET_EVIDENCE_STATUSES
from ati.research.backtest import BacktestConfig
from ati.research.bootstrap import block_bootstrap_mean
from ati.research.hypothesis import Verdict
from ati.research.robustness import cost_stress, parameter_perturbation, regime_breakdown, timing_stress
from ati.research.walkforward import WalkForwardResult
from ati.strategies.base import StrategyDefinition


@dataclass(frozen=True)
class AdversarialPolicy:
    min_oos_trades: int = 30
    min_param_survival: float = 0.6
    min_positive_folds: float = 0.5
    max_top5_share: float = 0.8
    max_prob_mean_le_zero: float = 0.10
    alpha: float = 0.05
    allow_non_market_data: bool = False  # True only for exercising mechanics on MOCK data


@dataclass(frozen=True)
class Objection:
    question: str
    verdict: Verdict
    detail: str


@dataclass(frozen=True)
class AdversarialReport:
    strategy_key: str
    strategy_hash: str
    dataset_id: str
    data_status: str
    objections: tuple[Objection, ...]

    @property
    def blocking(self) -> bool:
        return any(o.verdict in (Verdict.FAIL, Verdict.INSUFFICIENT_EVIDENCE) for o in self.objections)

    @property
    def evidence_hash(self) -> str:
        return sha256_hex(self)


def required_t(trials: int, alpha: float) -> float:
    """Bonferroni-adjusted two-sided critical value for ``trials`` tests."""
    return NormalDist().inv_cdf(1 - alpha / (2 * max(trials, 1)))


def challenge(defn: StrategyDefinition, dev: Dataset, cfg: BacktestConfig, wf: WalkForwardResult,
              hypotheses_tested: int, policy: AdversarialPolicy = AdversarialPolicy()) -> AdversarialReport:
    dev.require_not_holdout("adversarial challenge")
    o: list[Objection] = []
    m = wf.oos_metrics

    status = dev.identity.status
    if status in MARKET_EVIDENCE_STATUSES:
        o.append(Objection("Is the data real market data?", Verdict.PASS, status.value))
    elif policy.allow_non_market_data:
        o.append(Objection("Is the data real market data?", Verdict.NOT_AUTOMATED,
                           f"{status.value} data: mechanics exercise only; NOT evidence about markets"))
    else:
        o.append(Objection("Is the data real market data?", Verdict.INSUFFICIENT_EVIDENCE,
                           f"{status.value} data cannot support a market finding"))

    o.append(Objection("Could this be data leakage?",
                       Verdict.PASS if dev.partition is not Partition.HOLDOUT else Verdict.FAIL,
                       "point-in-time views + fills after decisions enforced structurally by the backtester"))

    enough = m.n_trades >= policy.min_oos_trades
    o.append(Objection("Is the sample large enough?", Verdict.PASS if enough else Verdict.INSUFFICIENT_EVIDENCE,
                       f"{m.n_trades} OOS trades (min {policy.min_oos_trades})"))

    o.append(Objection("Is there any out-of-sample edge at all?", Verdict.PASS if m.net_pnl > 0 else Verdict.FAIL,
                       f"OOS net P&L {m.net_pnl:.2f}"))

    ci = block_bootstrap_mean([t.net_return for t in wf.oos_trades], seed=17)
    if ci is None:
        o.append(Objection("Is the mean trade return distinguishable from noise?", Verdict.INSUFFICIENT_EVIDENCE, "too few trades"))
    else:
        ok = ci.prob_mean_le_zero <= policy.max_prob_mean_le_zero
        o.append(Objection("Is the mean trade return distinguishable from noise?", Verdict.PASS if ok else Verdict.FAIL,
                           f"bootstrap P(mean<=0)={ci.prob_mean_le_zero:.3f}, 95% CI [{ci.lo:.5f}, {ci.hi:.5f}]"))

    trials = wf.configs_tried * max(hypotheses_tested, 1)
    need = required_t(trials, policy.alpha)
    t = m.t_stat_trade_return
    o.append(Objection(f"How many hypotheses/configs were tried before this one? ({trials})",
                       Verdict.INSUFFICIENT_EVIDENCE if t is None else (Verdict.PASS if t >= need else Verdict.FAIL),
                       f"t={t if t is None else round(t, 3)} vs Bonferroni-required {need:.3f}"))

    folds_ok = wf.positive_fold_fraction >= policy.min_positive_folds
    o.append(Objection("Does it survive different time periods?", Verdict.PASS if folds_ok else Verdict.FAIL,
                       f"{wf.positive_fold_fraction:.0%} of active walk-forward folds positive"))

    if m.top5_profit_share is None:
        o.append(Objection("Is the result driven by a few trades?", Verdict.INSUFFICIENT_EVIDENCE if m.net_pnl > 0 else Verdict.FAIL,
                           "no positive net result to attribute" if m.net_pnl <= 0 else "too few trades"))
    else:
        o.append(Objection("Is the result driven by a few trades?",
                           Verdict.PASS if m.top5_profit_share <= policy.max_top5_share else Verdict.FAIL,
                           f"top-5 trades = {m.top5_profit_share:.0%} of net"))

    costs = cost_stress(defn, dev, cfg)
    doubled = next(p for p in costs if p.variant == "x2")
    o.append(Objection("Does the edge survive doubled costs and slippage?", Verdict.PASS if doubled.net_pnl > 0 else Verdict.FAIL,
                       ", ".join(f"{p.variant}: {p.net_pnl:.0f}" for p in costs)))

    perturbed = parameter_perturbation(defn, dev, cfg)
    if not perturbed:
        o.append(Objection("Does it survive parameter changes?", Verdict.INSUFFICIENT_EVIDENCE, "no valid neighbours"))
    else:
        share = sum(1 for p in perturbed if p.net_pnl > 0) / len(perturbed)
        o.append(Objection("Does it survive parameter changes?", Verdict.PASS if share >= policy.min_param_survival else Verdict.FAIL,
                           f"{share:.0%} of {len(perturbed)} ±20% neighbours profitable"))

    timing = timing_stress(defn, dev, cfg, delays=(1,))
    o.append(Objection("Does it survive a one-bar execution delay?", Verdict.PASS if timing[0].net_pnl > 0 else Verdict.FAIL,
                       f"delay=1 net {timing[0].net_pnl:.0f}"))

    regimes = {k: v for k, v in regime_breakdown(defn, dev, cfg).items() if k != "UNLABELED" and v["n"] >= 5}
    if len(regimes) < 2:
        o.append(Objection("Does it survive different regimes?", Verdict.INSUFFICIENT_EVIDENCE, f"regimes with >=5 trades: {len(regimes)}"))
    else:
        positive = [k for k, v in regimes.items() if v["net_pnl"] > 0]
        o.append(Objection("Does it survive different regimes?", Verdict.PASS if len(positive) >= 2 else Verdict.FAIL,
                           "; ".join(f"{k}: n={v['n']} net={v['net_pnl']:.0f}" for k, v in sorted(regimes.items()))))

    for question in ("Could this be selection or survivorship bias in the instrument choice?",
                     "What alternative explanation exists for the result?",
                     "Was this hypothesis selected because it happened to work?"):
        o.append(Objection(question, Verdict.NOT_AUTOMATED, "requires adversarial reviewer judgement; recorded, not assumed"))

    return AdversarialReport(defn.registry_key, defn.definition_hash, dev.dataset_id, status.value, tuple(o))
