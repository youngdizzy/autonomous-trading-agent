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
    dev_verdict: Verdict | None
    challenger_key: str | None
    adversarial_blocking: bool | None
    holdout_verdict: Verdict | None
    promotion: PromotionRecord | None
    memory_entry_id: str | None
    status: str = "COMPLETED"          # COMPLETED | NOT_RUN
    reasons: tuple[str, ...] = ()


def _resolve_challenger(registry, base: StrategyDefinition, params: dict, now: datetime,
                        hypothesis_id: str) -> StrategyDefinition:
    """Identity by fingerprint. An identical registered definition is *re-tested* (same definition hash);
    different parameters are a *different* definition and get their own version (lineage recorded via
    ``derive``). A registered definition is never mutated."""
    candidate = StrategyDefinition.create(base.strategy_id, base.version, base.kind, params, base.timeframe, now,
                                          base.parent_hash, f"selected by walk-forward for {hypothesis_id}")
    try:
        registered = registry.get(candidate.key)
    except KeyError:
        return registry.register(candidate)
    if registered.definition_hash == candidate.definition_hash:
        return registered
    for key in registry.keys():
        d = registry.get(key)
        if (d.strategy_id, d.kind, d.timeframe, d.code_hash, d.params) == \
                (candidate.strategy_id, candidate.kind, candidate.timeframe, candidate.code_hash, candidate.params):
            return d
    return registry.derive(candidate.key, params, now, f"selected by walk-forward for {hypothesis_id}")


def research_preconditions(system, full: Dataset, base: StrategyDefinition, *, hypothesis_id: str, min_candles: int,
                           require_market_data: bool) -> list[str]:
    """Everything that must hold before a research cycle may touch a dataset. Returns failures."""
    s = system
    failures: list[str] = []
    ident = full.identity
    try:
        full.verify()
    except Exception as exc:  # any integrity doubt → not run
        failures.append(f"dataset integrity: {exc}")
    if require_market_data:
        if ident.status not in MARKET_EVIDENCE_STATUSES:
            failures.append(f"{ident.status.value} data where market data is required")
        else:
            try:
                s.archive.verify_market_provenance(full)
            except Exception as exc:
                failures.append(f"provenance: {exc}")
    if ident.status is not s.data_status:
        failures.append(f"dataset is {ident.status.value}; this system is bound to {s.data_status.value}")
    if ident.symbol not in s.universe:
        failures.append(f"symbol {ident.symbol} not in universe")
    if ident.timeframe is not base.timeframe:
        failures.append(f"timeframe {ident.timeframe.value} != strategy timeframe {base.timeframe.value}")
    if ident.partition.value != "FULL":
        failures.append(f"partition {ident.partition.value}; a research cycle starts from a FULL dataset")
    if not all(c.is_closed for c in full.candles):
        failures.append("forming candles present")
    if ident.n_candles < min_candles:
        failures.append(f"INSUFFICIENT DATA: {ident.n_candles} candles < {min_candles} required by the protocol")
    try:
        full.require_not_holdout("research cycle")
    except Exception as exc:
        failures.append(str(exc))
    if ResearchLog(s.research_journal).was_tested(hypothesis_id):  # reconstructed from the journal
        failures.append(f"{hypothesis_id} was already tested; a hypothesis is run exactly once")
    return failures


def run_research_cycle(system, full: Dataset, boundary: datetime, *, hypothesis_id: str, statement: str,
                       base: StrategyDefinition, grid: list[dict], criteria: tuple[Criterion, ...],
                       train_bars: int, test_bars: int, config: BacktestConfig = BacktestConfig(),
                       adversarial_policy: AdversarialPolicy = AdversarialPolicy(),
                       promotion_policy: PromotionPolicy = PromotionPolicy(), min_candles: int = 0,
                       observation_refs: tuple[str, ...] = ()) -> CycleResult:
    s = system
    now = s.clock.now()
    journal = s.research_journal
    require_market = not (adversarial_policy.allow_non_market_data and promotion_policy.allow_mock_evidence)
    failures = research_preconditions(s, full, base, hypothesis_id=hypothesis_id, min_candles=min_candles,
                                      require_market_data=require_market)
    if failures:
        journal.append("research_not_run", {"hypothesis_id": hypothesis_id, "dataset_id": full.dataset_id,
                                            "reasons": failures})
        return CycleResult(hypothesis_id, None, None, None, None, None, None, "NOT_RUN", tuple(failures))
    log = ResearchLog(journal)
    mock = full.identity.status not in MARKET_EVIDENCE_STATUSES
    tag = f"[{full.identity.status.value}] " if mock else ""

    prior = s.memory.prior_rejections(statement, base.kind)
    if prior:
        journal.append("research_note", {"hypothesis_id": hypothesis_id, "note": "idea previously rejected",
                                         "prior": [p.entry_id for p in prior]})

    s.evidence.register("dataset", full.dataset_id, full.identity.temporal_boundary, "research dataset",
                        dataset_id=full.dataset_id)
    vault = HoldoutVault(full, boundary, journal, max_evaluations=1)
    dev = vault.development
    s.evidence.register("dataset", dev.dataset_id, dev.identity.temporal_boundary, "development partition",
                        dataset_id=dev.dataset_id)

    # observation_refs: what motivated the hypothesis (e.g. learning evidence) — locked into the pre-registration.
    prereg = PreRegistration(hypothesis_id, statement, tuple(observation_refs), base.key, base.definition_hash,
                             dev.dataset_id, criteria,
                             adversarial_policy.min_oos_trades, now)
    log.preregister(prereg)

    wf = walk_forward(base, dev, grid, train_bars=train_bars, test_bars=test_bars, config=config)
    s.evidence.register("walk_forward", wf.evidence_hash, now, f"{base.kind} walk-forward on {dev.dataset_id}",
                        dataset_id=wf.dataset_id, strategy=base.strategy_id)
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
    challenger = _resolve_challenger(s.strategies, base, dict(chosen), now, hypothesis_id)
    state = s.strategies.state(challenger.key)
    if state in (Lifecycle.CANDIDATE, Lifecycle.REJECTED):
        why = "survived development" if state is Lifecycle.CANDIDATE else \
            "re-admitted: survived development in a new retest; prior rejection remains in history"
        s.strategies.transition(challenger.key, Lifecycle.CHALLENGER, f"{why} for {hypothesis_id}")
    elif state is not Lifecycle.CHALLENGER:
        # CHAMPION / RETIRED have no legal path through challenger evaluation in this workflow. Stop before
        # the holdout is consumed; the development experiment above is already recorded.
        reason = f"{challenger.key} is {state.value}; not eligible for challenger evaluation (holdout not used)"
        journal.append("research_note", {"hypothesis_id": hypothesis_id, "note": reason})
        return CycleResult(hypothesis_id, dev_verdict, challenger.key, None, None, None, None, "COMPLETED", (reason,))

    adv = challenge(challenger, dev, config, wf, log.hypotheses_tested, adversarial_policy)
    s.evidence.register("adversarial", adv.evidence_hash, now, "adversarial challenge",
                        dataset_id=adv.dataset_id, strategy=challenger.definition_hash)
    journal.append("adversarial_report", {"report": adv, "evidence_hash": adv.evidence_hash})

    hold_prereg = PreRegistration(hypothesis_id + ":holdout", statement, tuple(observation_refs), challenger.key,
                                  challenger.definition_hash,
                                  dev.dataset_id, criteria, promotion_policy.min_holdout_trades, now)
    log.preregister(hold_prereg)
    hold = vault.evaluate(challenger, hold_prereg, config)
    s.evidence.register("holdout", hold.evidence_hash, now, "holdout evaluation",
                        dataset_id=hold.holdout_dataset_id, strategy=challenger.definition_hash)
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
