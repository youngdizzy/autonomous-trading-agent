# Mission, Philosophy, Operating Principles

## Mission

Build autonomous trading intelligence capable of discovering and validating durable market edges —
and of earning, through evidence, the right to trade real capital.

The product is not a bot that trades. It is a research → decision → risk → execution → learning
system whose every action can be explained, reproduced, and audited.

## Philosophy

- Evidence over confidence.
- Research over guessing.
- Risk before profit.
- Reproducibility over impressive results.
- Survival before scale.

## Operating principles

Each principle is enforced by code where code can enforce it. The enforcing component is named.

| # | Principle | Enforced by |
|---|---|---|
| 1 | Never fabricate evidence. | Evidence references are content hashes of stored artifacts; decision/memory records reject references that do not resolve (`ati/decision`, `ati/memory`). Every data point carries a `DataStatus`. |
| 2 | Never hide failed experiments. | Experiments, rejected hypotheses, denied promotions are appended to the journal; memory has no delete (`ati/memory/store.py`, `ati/validation/promotion.py`). |
| 3 | Never silently change historical records. | Append-only hash-chained journal; dataset content hashes; candle store fails closed on disagreement (`ati/ledger/journal.py`, `ati/data/dataset.py`, `ati/market/store.py`). |
| 4 | Never use future information. | `PointInTimeView` only contains closed candles with `close_time <= cutoff`; `InformationSet` refuses future items; backtester fills strictly after decisions (`ati/temporal`, `ati/research/backtest.py`). |
| 5 | Never optimize the holdout. | Holdout candles exist only inside `HoldoutVault`; research functions refuse HOLDOUT partitions; one evaluation per strategy lineage; evaluation budget (`ati/validation/holdout.py`). |
| 6 | Never bypass deterministic risk. | Execution accepts only HMAC-signed, unexpired `RiskVerdict`s produced by the `RiskEngine` (`ati/risk`, `ati/execution/engine.py`). |
| 7 | Never confuse paper results with live results. | Journals and accounts are bound to one `OperatingMode` and one `DataStatus` at creation; mixing raises. LIVE is hard-disabled (`ati/config.py`). |
| 8 | Never confuse a hypothesis with a validated finding. | Memory kinds have evidence requirements; `VALIDATED_FINDING` requires holdout evidence; confidence is capped by kind (`ati/memory/models.py`). |
| 9 | Never allow Claude confidence to replace evidence. | Claude's `confidence` is recorded but never used by risk sizing or promotion; promotion gates are purely metric-based (`ati/validation/promotion.py`). |
| 10 | Never scale capital faster than evidence justifies. | Risk limits are static configuration; nothing in the agent layer can raise them. Capital scaling is NOT IMPLEMENTED by design in Foundation 1.0. |

## What the company must be able to answer

For any trade: why it was taken, what information existed at that instant, what evidence
supported it, what would have invalidated it, what happened, whether the decision was good given
the information available at the time, what was learned, whether that lesson survived out-of-sample
testing, and what evidence justifies any change. The `DecisionRecord`, the journal, and the memory
store exist to make those answers mechanical rather than narrative.
