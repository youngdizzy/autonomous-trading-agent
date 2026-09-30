"""Objective contract: what "better" means, stated before any result is seen.

No single metric decides anything. A comparison reports every dimension separately, applies hard
constraints and failure conditions first, and only then states a conclusion — and the strongest conclusion
it can ever reach is IMPROVED_ON_DEVELOPMENT. Promotion remains the promotion gate's decision; this
contract never promotes, and progress toward a goal never overrides a constraint.

The contract is code (hashed into every comparison record). Claude cannot edit it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ati.core.canonical import sha256_hex
from ati.research.metrics import Metrics


class Direction(str, Enum):
    HIGHER = "HIGHER"
    LOWER = "LOWER"


@dataclass(frozen=True)
class Dimension:
    name: str
    metric: str          # attribute of research.metrics.Metrics
    better: Direction
    tolerance: float     # differences within tolerance are SAME


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("return", "net_return", Direction.HIGHER, 0.002),
    Dimension("risk_adjusted_return", "sharpe_annualized", Direction.HIGHER, 0.05),
    Dimension("expectancy", "expectancy_r", Direction.HIGHER, 0.01),
    Dimension("drawdown", "max_drawdown", Direction.LOWER, 0.005),
    Dimension("consistency", "win_rate", Direction.HIGHER, 0.01),
    Dimension("transaction_costs", "cost_share_of_gross", Direction.LOWER, 0.02),
    Dimension("tail_concentration", "top5_profit_share", Direction.LOWER, 0.02),
    Dimension("statistical_strength", "t_stat_trade_return", Direction.HIGHER, 0.05),
)


@dataclass(frozen=True)
class ObjectiveContract:
    primary_objective: str = "positive risk-adjusted expectancy after all modelled costs, out of sample"
    # Constraints: a candidate violating any of these is NOT_AN_IMPROVEMENT regardless of other dimensions.
    max_drawdown: float = 0.20
    max_top5_profit_share: float = 0.80
    max_cost_share_of_gross: float = 0.60
    min_trades: int = 30
    # Failure conditions: unacceptable outcomes.
    require_positive_net: bool = True
    require_positive_expectancy: bool = True
    # Evidence requirements before any conclusion stronger than INCONCLUSIVE.
    required_verdicts: tuple[str, ...] = ("walk_forward_oos:PASS", "adversarial:NON_BLOCKING", "holdout:PASS")

    @property
    def contract_hash(self) -> str:
        return sha256_hex(self)


CONTRACT = ObjectiveContract()


class Conclusion(str, Enum):
    NOT_AN_IMPROVEMENT = "NOT_AN_IMPROVEMENT"
    INCONCLUSIVE = "INCONCLUSIVE"
    MIXED = "MIXED"
    IMPROVED_ON_DEVELOPMENT = "IMPROVED_ON_DEVELOPMENT"   # never "validated", never "promote"


def _cmp(dim: Dimension, base: float | None, cand: float | None) -> str:
    if base is None or cand is None:
        return "UNKNOWN"
    delta = cand - base if dim.better is Direction.HIGHER else base - cand
    if abs(delta) <= dim.tolerance:
        return "SAME"
    return "BETTER" if delta > 0 else "WORSE"


def constraint_violations(m: Metrics, contract: ObjectiveContract = CONTRACT) -> list[str]:
    out = []
    if m.n_trades < contract.min_trades:
        out.append(f"INSUFFICIENT_EVIDENCE: {m.n_trades} trades < {contract.min_trades}")
    if m.max_drawdown > contract.max_drawdown:
        out.append(f"drawdown {m.max_drawdown:.3f} > {contract.max_drawdown}")
    if m.top5_profit_share is not None and m.top5_profit_share > contract.max_top5_profit_share:
        out.append(f"top-5 trades {m.top5_profit_share:.2f} of profit > {contract.max_top5_profit_share}")
    if m.cost_share_of_gross is not None and m.cost_share_of_gross > contract.max_cost_share_of_gross:
        out.append(f"costs {m.cost_share_of_gross:.2f} of gross > {contract.max_cost_share_of_gross}")
    if contract.require_positive_net and m.net_pnl <= 0:
        out.append("failure condition: net P&L <= 0")
    if contract.require_positive_expectancy and (m.expectancy_r is None or m.expectancy_r <= 0):
        out.append("failure condition: expectancy not positive (or not measurable)")
    return out


# Dimensions computed outside Metrics, passed as (baseline, candidate) values.
EXTRA_DIMENSIONS = {"wfo_stability": Dimension("wfo_stability", "positive_fold_fraction", Direction.HIGHER, 0.05)}
# Dimensions the champion/challenger comparison should cover but this build cannot measure honestly.
NOT_MEASURED = ("volatility_of_returns", "regime_behavior", "parameter_sensitivity")


def compare(baseline: Metrics, candidate: Metrics, verdicts: dict[str, str],
            contract: ObjectiveContract = CONTRACT, extra: dict[str, tuple] | None = None) -> dict:
    """Baseline vs candidate on the *development* partition, plus the recorded stage verdicts.
    Returns a per-dimension profile and a conclusion; never a single score. Robustness, adversarial and holdout
    performance enter only through the recorded verdicts (evidence requirements), never as a metric trade-off."""
    dims = {d.name: _cmp(d, getattr(baseline, d.metric), getattr(candidate, d.metric)) for d in DIMENSIONS}
    for name, (b, c) in (extra or {}).items():
        dims[name] = _cmp(EXTRA_DIMENSIONS[name], b, c)
    violations = constraint_violations(candidate, contract)
    missing = [r for r in contract.required_verdicts
               if verdicts.get(r.split(":")[0]) != r.split(":")[1]]
    if violations:
        conclusion = Conclusion.NOT_AN_IMPROVEMENT
    elif missing or "UNKNOWN" in dims.values():
        conclusion = Conclusion.INCONCLUSIVE
    elif "WORSE" in dims.values():
        conclusion = Conclusion.MIXED
    elif "BETTER" in dims.values():
        conclusion = Conclusion.IMPROVED_ON_DEVELOPMENT
    else:
        conclusion = Conclusion.INCONCLUSIVE
    return {"contract_hash": contract.contract_hash, "dimensions": dims, "not_measured": list(NOT_MEASURED),
            "constraint_violations": violations,
            "missing_evidence": missing, "conclusion": conclusion.value,
            "note": "development-partition comparison; promotion is decided only by the promotion gate"}


def goal_progress(candidate: Metrics | None, contract: ObjectiveContract = CONTRACT) -> dict:
    """OBJECTIVE → CURRENT STATE → GAP → NEXT TEST. Informational; never overrides a constraint."""
    if candidate is None:
        return {"objective": contract.primary_objective, "current_state": "NOT_AVAILABLE",
                "gap": "no evaluated candidate", "next_test": "run a pre-registered experiment"}
    violations = constraint_violations(candidate, contract)
    return {"objective": contract.primary_objective,
            "current_state": {"net_return": candidate.net_return, "max_drawdown": candidate.max_drawdown,
                              "trades": candidate.n_trades, "expectancy_r": candidate.expectancy_r},
            "gap": violations or "no constraint violated on development data",
            "next_test": "out-of-sample evidence on an unused holdout period" if not violations
            else "a new pre-registered hypothesis addressing the first violation"}
