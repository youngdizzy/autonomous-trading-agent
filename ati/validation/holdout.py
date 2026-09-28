"""Holdout vault.

Structural guarantees (each covered by tests):

1. The holdout candles live only in the vault's private slot. The vault's public surface exposes
   the DEVELOPMENT dataset and the holdout *identity* (a hash commitment) — never holdout candles.
2. Every research/optimization function refuses a dataset whose partition is HOLDOUT.
3. ``evaluate`` requires a locked pre-registration for exactly the strategy being evaluated, and
   returns only metrics and a verdict — nothing that could be used to tune on individual bars.
4. Each strategy lineage (strategy_id) may be evaluated on a given holdout at most once. A tweaked
   version of a strategy that has already seen the holdout cannot be re-tested on it.
5. Total evaluations are budgeted. When the budget is spent the vault is BURNED and refuses all
   further use: a fresh, never-seen holdout period is required.
6. Every access is journaled *before* computation (write-ahead). If the journal write fails, no
   evaluation happens.

Honest limit: Python cannot stop in-process code that deliberately reaches into private attributes.
The boundary protects against every path the agent layer can reach (Claude never runs code; it can
only emit schema-validated proposals), and against accidental misuse by research code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ati.core.canonical import sha256_hex
from ati.core.errors import HoldoutViolation
from ati.data.dataset import Dataset, DatasetIdentity, Partition, seal_holdout_range
from ati.ledger.journal import Journal
from ati.research.backtest import BacktestConfig, run_backtest
from ati.research.hypothesis import PreRegistration, Verdict
from ati.research.metrics import Metrics, compute_metrics
from ati.strategies.base import StrategyDefinition


_VAULT_TOKEN = object()  # capability passed to the backtester only from HoldoutVault.evaluate


@dataclass(frozen=True)
class HoldoutEvaluation:
    strategy_key: str
    strategy_hash: str
    holdout_dataset_id: str
    prereg_hash: str
    metrics: Metrics
    verdict: Verdict
    details: tuple
    evaluation_number: int

    @property
    def evidence_hash(self) -> str:
        return sha256_hex(self)


class HoldoutVault:
    __slots__ = ("__holdout", "_development", "_journal", "_max_evaluations", "_evaluated_lineages",
                 "_evaluations", "_boundary")

    def __init__(self, full: Dataset, boundary: datetime, journal: Journal, max_evaluations: int = 3):
        if full.partition is not Partition.FULL:
            raise HoldoutViolation("vault must be built from a FULL dataset")
        dev, hold = full.split(boundary)
        seal_holdout_range(hold.identity)  # from now on, nothing overlapping this period is research input
        self.__holdout = hold
        self._development = dev
        self._journal = journal
        self._max_evaluations = max_evaluations
        self._evaluated_lineages: set[str] = set()
        self._evaluations = 0
        self._boundary = boundary
        journal.append("holdout_sealed", {"development": dev.identity, "holdout_identity": hold.identity,
                                          "holdout_commitment": hold.identity.content_sha256,
                                          "holdout_dataset_id": hold.dataset_id, "boundary": boundary,
                                          "max_evaluations": max_evaluations})

    @property
    def development(self) -> Dataset:
        return self._development

    @property
    def holdout_identity(self) -> DatasetIdentity:
        return self.__holdout.identity

    @property
    def burned(self) -> bool:
        return self._evaluations >= self._max_evaluations

    @property
    def evaluations_used(self) -> int:
        return self._evaluations

    def evaluate(self, strategy: StrategyDefinition, prereg: PreRegistration,
                 config: BacktestConfig = BacktestConfig()) -> HoldoutEvaluation:
        if self.burned:
            raise HoldoutViolation("holdout budget spent (BURNED); a fresh holdout period is required")
        if prereg.strategy_hash != strategy.definition_hash:
            raise HoldoutViolation("pre-registration does not lock this exact strategy definition")
        if prereg.dev_dataset_id != self._development.dataset_id:
            raise HoldoutViolation("pre-registration was made against a different development dataset")
        if strategy.lineage_root in self._evaluated_lineages:
            raise HoldoutViolation(f"lineage {strategy.lineage_root} has already seen this holdout")
        number = self._evaluations + 1
        # Write-ahead: the access is recorded before any holdout computation happens.
        self._journal.append("holdout_access", {"strategy_key": strategy.key, "strategy_hash": strategy.definition_hash,
                                                "prereg_hash": prereg.prereg_hash, "evaluation_number": number})
        self._evaluations = number
        self._evaluated_lineages.add(strategy.lineage_root)
        hold = self.__holdout
        hold.verify()
        warm = self._development.candles[-strategy.lookback:]
        combined = Dataset.build(warm + hold.candles, data_version=hold.identity.data_version,
                                 realization=hold.identity.realization, partition=Partition.HOLDOUT)
        result = run_backtest(strategy, combined, config, trade_start=hold.identity.start, holdout_token=_VAULT_TOKEN)
        result.trades[:] = [t for t in result.trades if t.decided_at >= hold.identity.start]
        metrics = compute_metrics(result, strategy.timeframe.bars_per_year)
        verdict, details = prereg.evaluate(metrics)
        evaluation = HoldoutEvaluation(strategy.key, strategy.definition_hash, hold.dataset_id, prereg.prereg_hash,
                                       metrics, verdict, tuple(tuple(d) for d in details), number)
        self._journal.append("holdout_result", {"evaluation": evaluation, "evidence_hash": evaluation.evidence_hash})
        return evaluation
