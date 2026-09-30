"""Champion / challenger promotion gate.

Promotion is a deterministic function of pre-declared policy and recorded evidence. Claude's
confidence is not an input. Every decision — approved or denied — produces an immutable
``PromotionRecord`` appended to the journal, so failed challengers are never hidden.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime

from ati.core.canonical import canonical_json, sha256_hex
from ati.core.time import ensure_utc
from ati.ledger.journal import Journal
from ati.research.adversarial import AdversarialReport
from ati.research.hypothesis import Verdict
from ati.research.walkforward import WalkForwardResult
from ati.strategies.base import StrategyDefinition
from ati.validation.holdout import HoldoutEvaluation


@dataclass(frozen=True)
class PromotionPolicy:
    min_oos_trades: int = 30
    min_holdout_trades: int = 20
    min_positive_fold_fraction: float = 0.5
    require_beat_champion_oos: bool = True
    allow_mock_evidence: bool = False


@dataclass(frozen=True)
class PromotionRecord:
    challenger_key: str
    challenger_hash: str
    champion_key: str | None
    approved: bool
    reasons: tuple[str, ...]
    evidence: tuple[tuple[str, str], ...]  # (kind, evidence_hash)
    data_status: str
    policy: PromotionPolicy
    decided_at: datetime
    symbol: str = ""        # research dimension the decision belongs to (part of the integrity hash)
    timeframe: str = ""
    record_hash: str = ""

    def _body(self) -> str:
        return canonical_json({k: getattr(self, k) for k in (
            "challenger_key", "challenger_hash", "champion_key", "approved", "reasons", "evidence",
            "data_status", "policy", "decided_at", "symbol", "timeframe")})

    def verify_integrity(self) -> bool:
        return hmac.compare_digest(self.record_hash, hashlib.sha256(self._body().encode()).hexdigest())


def decide_promotion(challenger: StrategyDefinition, champion: StrategyDefinition | None,
                     wf: WalkForwardResult, adversarial: AdversarialReport, holdout: HoldoutEvaluation,
                     champion_wf: WalkForwardResult | None, policy: PromotionPolicy, decided_at: datetime,
                     journal: Journal, symbol: str = "") -> PromotionRecord:
    """``symbol`` names the research dimension (with the challenger's timeframe). A record without one can be
    journaled but never applied: champions exist per (symbol, timeframe) only."""
    reasons: list[str] = []
    if champion is not None and champion.timeframe is not challenger.timeframe:
        reasons.append("champion and challenger belong to different timeframes: not comparable")
    h = challenger.definition_hash
    if adversarial.strategy_hash != h or holdout.strategy_hash != h:
        reasons.append("evidence does not belong to this exact challenger definition")
    if wf.strategy_id != challenger.strategy_id or wf.kind != challenger.kind:
        reasons.append("walk-forward evidence is for a different strategy")
    if challenger.params not in [f.selected_params for f in wf.folds]:
        reasons.append("challenger parameters were never selected in walk-forward development")
    if wf.oos_metrics.n_trades < policy.min_oos_trades:
        reasons.append(f"INSUFFICIENT EVIDENCE: {wf.oos_metrics.n_trades} OOS trades < {policy.min_oos_trades}")
    if wf.positive_fold_fraction < policy.min_positive_fold_fraction:
        reasons.append(f"only {wf.positive_fold_fraction:.0%} of folds positive")
    if adversarial.blocking:
        failed = [f"{o.question} → {o.verdict.value}" for o in adversarial.objections
                  if o.verdict in (Verdict.FAIL, Verdict.INSUFFICIENT_EVIDENCE)]
        reasons.append("adversarial review blocking: " + "; ".join(failed))
    if holdout.verdict is not Verdict.PASS:
        reasons.append(f"holdout verdict {holdout.verdict.value}")
    if holdout.metrics.n_trades < policy.min_holdout_trades:
        reasons.append(f"INSUFFICIENT EVIDENCE: {holdout.metrics.n_trades} holdout trades < {policy.min_holdout_trades}")
    if adversarial.data_status not in ("REAL", "HISTORICAL", "DELAYED") and not policy.allow_mock_evidence:
        reasons.append(f"evidence is {adversarial.data_status} data, not market data")
    if champion is not None and policy.require_beat_champion_oos:
        if champion_wf is None or champion_wf.dataset_id != wf.dataset_id:
            reasons.append("no like-for-like champion walk-forward on the same development data")
        elif wf.oos_metrics.net_pnl <= champion_wf.oos_metrics.net_pnl:
            reasons.append("challenger does not beat champion out-of-sample on the same data")
    evidence = (("walk_forward", wf.evidence_hash), ("adversarial", adversarial.evidence_hash),
                ("holdout", holdout.evidence_hash))
    record = PromotionRecord(challenger.registry_key, h, champion.registry_key if champion else None, not reasons,
                             tuple(reasons) if reasons else ("all promotion criteria met",), evidence,
                             adversarial.data_status, policy, ensure_utc(decided_at), symbol,
                             challenger.timeframe.value)
    object.__setattr__(record, "record_hash", hashlib.sha256(record._body().encode()).hexdigest())
    journal.append("promotion_decision", {"record": record})
    return record
