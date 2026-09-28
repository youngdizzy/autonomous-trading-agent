"""One complete research cycle, wired end to end:

    observation (memory) → pre-registered hypothesis → walk-forward (development only)
    → falsification check → adversarial challenge → holdout (once) → promotion gate
    → memory (finding or rejected hypothesis) — every step journaled and registered as evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from ati.data.dataset import Dataset
from ati.market.models import MARKET_EVIDENCE_STATUSES
from ati.memory.store import MemoryEntry, MemoryKind
from ati.research.adversarial import AdversarialPolicy, challenge
from ati.research.backtest import BacktestConfig
from ati.research.hypothesis import Criterion, PreRegistration, ResearchLog, Verdict
from ati.research.walkforward import walk_forward
from ati.strategies.base import StrategyDefinition
from ati.strategies.registry import Lifecycle
from ati.validation.holdout import HoldoutVault
from ati.validation.promotion import PromotionPolicy, PromotionRecord, decide_promotion


@dataclass(frozen=True)
class CycleResult:
    hypothesis_id: str
    dev_verdict: Verdict
    challenger_key: str | None
    adversarial_blocking: bool | None
    holdout_verdict: Verdict | None
    promotion: PromotionRecord | None
    memory_entry_id: str


def run_research_cycle(system, full: Dataset, boundary: datetime, *, hypothesis_id: str, statement: str,
                       base: StrategyDefinition, grid: list[dict], criteria: tuple[Criterion, ...],
                       train_bars: int, test_bars: int, config: BacktestConfig = BacktestConfig(),
                       adversarial_policy: AdversarialPolicy = AdversarialPolicy(),
                       promotion_policy: PromotionPolicy = PromotionPolicy()) -> CycleResult:
    s = system
    now = s.clock.now()
    journal = s.research_journal
    log = ResearchLog(journal)
    mock = full.identity.status not in MARKET_EVIDENCE_STATUSES
    tag = f"[{full.identity.status.value}] " if mock else ""

    prior = s.memory.prior_rejections(statement, base.kind)
    if prior:
        journal.append("research_note", {"hypothesis_id": hypothesis_id, "note": "idea previously rejected",
                                         "prior": [p.entry_id for p in prior]})

    s.evidence.register("dataset", full.dataset_id, full.identity.temporal_boundary, "research dataset")
    vault = HoldoutVault(full, boundary, journal, max_evaluations=1)
    dev = vault.development
    s.evidence.register("dataset", dev.dataset_id, dev.identity.temporal_boundary, "development partition")

    prereg = PreRegistration(hypothesis_id, statement, (), base.key, base.definition_hash, dev.dataset_id, criteria,
                             adversarial_policy.min_oos_trades, now)
    log.preregister(prereg)

    wf = walk_forward(base, dev, grid, train_bars=train_bars, test_bars=test_bars, config=config)
    s.evidence.register("walk_forward", wf.evidence_hash, now, f"{base.kind} walk-forward on {dev.dataset_id}")
    dev_verdict = log.record_experiment(hypothesis_id, "walk_forward_oos", wf.evidence_hash, wf.oos_metrics, dev.dataset_id)

    def remember(kind: MemoryKind, text: str, refs: list[str], confidence: float) -> str:
        entry = MemoryEntry(kind, tag + text, s.clock.now(), "system:research_workflow",
                            tuple(s.evidence.resolve(r) for r in refs), confidence, (dev.dataset_id,), (base.key,), base.kind)
        return s.memory.add(entry)

    if dev_verdict is not Verdict.PASS:
        mid = remember(MemoryKind.REJECTED_HYPOTHESIS, f"{statement} — walk-forward {dev_verdict.value}",
                       [wf.evidence_hash], 0.8)
        return CycleResult(hypothesis_id, dev_verdict, None, None, None, None, mid)

    chosen = Counter(f.selected_params for f in wf.folds if f.selected_params).most_common(1)[0][0]
    challenger = s.strategies.register(StrategyDefinition.create(
        base.strategy_id, base.version, base.kind, dict(chosen), base.timeframe, now, base.parent_hash,
        f"selected by walk-forward for {hypothesis_id}"))
    if s.strategies.state(challenger.key) is Lifecycle.CANDIDATE:
        s.strategies.transition(challenger.key, Lifecycle.CHALLENGER, f"survived development for {hypothesis_id}")

    adv = challenge(challenger, dev, config, wf, log.hypotheses_tested, adversarial_policy)
    s.evidence.register("adversarial", adv.evidence_hash, now, "adversarial challenge")
    journal.append("adversarial_report", {"report": adv, "evidence_hash": adv.evidence_hash})

    hold_prereg = PreRegistration(hypothesis_id + ":holdout", statement, (), challenger.key, challenger.definition_hash,
                                  dev.dataset_id, criteria, promotion_policy.min_holdout_trades, now)
    log.preregister(hold_prereg)
    hold = vault.evaluate(challenger, hold_prereg, config)
    s.evidence.register("holdout", hold.evidence_hash, now, "holdout evaluation")
    log.record_experiment(hold_prereg.hypothesis_id, "holdout", hold.evidence_hash, hold.metrics, hold.holdout_dataset_id)

    champion = s.strategies.champion()
    record = decide_promotion(challenger, champion, wf, adv, hold, None, promotion_policy, now, journal)
    s.evidence.register("promotion", record.record_hash, now, "approved" if record.approved else "denied")
    refs = [wf.evidence_hash, adv.evidence_hash, hold.evidence_hash]
    if record.approved:
        s.strategies.apply_promotion(record)
        if mock:
            mid = remember(MemoryKind.HYPOTHESIS, f"{statement} — passed all gates on non-market data; mechanics only, "
                                                  "NOT evidence about real markets", refs, 0.1)
        else:
            mid = remember(MemoryKind.VALIDATED_FINDING, statement, refs, 0.6)
    else:
        s.strategies.transition(challenger.key, Lifecycle.REJECTED, "; ".join(record.reasons)[:300])
        mid = remember(MemoryKind.REJECTED_HYPOTHESIS, f"{statement} — promotion denied: {'; '.join(record.reasons)}"[:1900],
                       refs, 0.8)
    return CycleResult(hypothesis_id, dev_verdict, challenger.key, adv.blocking, hold.verdict, record, mid)
