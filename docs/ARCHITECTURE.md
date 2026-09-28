# Architecture — Foundation 1.0

## Authority split

```
            PROPOSES (untrusted text)                     DECIDES (deterministic)
  ┌───────────────────────────────────┐      ┌─────────────────────────────────────────────┐
  │ Claude                            │      │ Data validity · temporal cutoffs · identity  │
  │  primary decision · adversarial   │ ───► │ Risk engine (sizing, limits, kill switch)    │
  │  review · post-trade review ·     │schema│ Execution (idempotent, reconciled)           │
  │  research questions               │ gate │ Ledger · journal · holdout · promotion gate  │
  └───────────────────────────────────┘      └─────────────────────────────────────────────┘
```

Claude never holds an object that can reach a broker. Its output is text; the only thing that
text can become is one of three schema-validated values (`TradeProposal`, `NoTrade`,
`ResearchRequest`). A `TradeProposal` is a *request* to the risk engine, which may shrink it to zero.

## Data flow

```
Claude ─► Market ─► Research ─► Decision ─► Risk ─► Execution ─► Ledger ─► Memory ─► Claude
  ▲          │           │          │          │          │            │          │
  │   ati/market   ati/research  ati/decision ati/risk ati/execution ati/ledger ati/memory
  │   ati/data     ati/validation ati/agent                                       │
  └────────────── point-in-time context packet (ati/agent/pipeline.py) ◄──────────┘
```

| Layer | Package | Responsibility |
|---|---|---|
| Core | `ati/core`, `ati/config.py` | canonical hashing, UTC discipline, error taxonomy, `LIVE_TRADING=False` |
| Market | `ati/market` | typed candles with invariants, provider protocol, MOCK provider, Kraken adapter, fail-closed store |
| Dataset | `ati/data` | content-derived identity, mutation detection, partitions, sealed holdout periods |
| Temporal | `ati/temporal` | `PointInTimeView`, `InformationSet`, point-in-time features |
| Strategy | `ati/strategies` | immutable content-addressed definitions (hash covers code), registry, lifecycle |
| Research | `ati/research` | backtester, costs, metrics, walk-forward, robustness, bootstrap, attribution, hypotheses, adversarial challenge, workflow |
| Validation | `ati/validation` | holdout vault, promotion gate |
| Decision | `ati/decision` | immutable `DecisionRecord`, deterministic decision ids |
| Risk | `ati/risk` | sizing, limits, kill switch, signed approvals |
| Execution | `ati/execution` | idempotent write-ahead engine, reconciliation, paper venue, live boundary (disabled) |
| Ledger | `ati/ledger` | hash-chained journal, fill accounting |
| Memory | `ati/memory` | evidence registry, structured memory |
| Agent | `ati/agent` | roles, reasoning clients, output schema, decision pipeline, autonomous loop |
| Security | `ati/security` | secrets, untrusted-content fencing |
| Monitoring | `ati/monitoring` | compact status |

## Key invariants and where they are enforced

| Invariant | Mechanism | Tests |
|---|---|---|
| No future information at decision time | `PointInTimeView` holds only candles with `close_time <= cutoff`; construction re-checks; backtester fills at next open | `test_dataset_temporal.py`, `test_research.py::test_future_data_cannot_change_past_decisions` |
| History cannot be silently rewritten | append-only `CandleStore` raises `HistoricalConflictError`; dataset hashes; hash-chained journal | `test_market.py`, `test_dataset_temporal.py`, `test_execution_ledger.py::TestJournal`, `test_chaos.py` |
| Real and simulated data never mix | `DataStatus` on every candle; series, stores, datasets, accounts, journals refuse mixing | `STATUS_MIX` tests, `test_paper_and_live_never_mix` |
| Holdout cannot leak | vault-private candles, partition guards, sealed-period overlap guard, per-lineage single evaluation, budget burn, write-ahead access log, vault-only backtest token | `test_validation.py::TestHoldoutStructure` |
| Promotion requires evidence | deterministic gate; tamper-evident record; registry accepts only intact approved records | `test_validation.py::TestPromotion` |
| No risk bypass | execution verifies HMAC-signed, unexpired approvals from a per-process key | `test_risk.py::TestApprovalIntegrity`, `test_execution_ledger.py` |
| Uncertain state stops trading | UNKNOWN order → halt; reconciliation required; account unknown → proposals refused | `test_chaos.py` |
| Research criteria stay locked across restarts; multiple-testing count is historical | `ResearchLog` is rebuilt from the research journal (hash-verified); conflicting locks fail closed; count = distinct root hypotheses with ≥1 experiment | `test_research_memory_integrity.py::TestResearchPersistence`, `test_multiple_testing_penalty_uses_persistent_history` |
| A rejection is evidence, not a ban; a retest is new evidence | challenger identity by fingerprint (identical = retest, different params = new version via `derive`); REJECTED → CHALLENGER only when a new locked hypothesis survives development; prior rejection stays in lifecycle history and promotion records; CHAMPION/RETIRED stop before the holdout | `test_research_retest.py` |
| Non-market data never becomes doctrine; evidence is unique and independent | `MemoryStore` category gate (doctrine only in REAL/HISTORICAL/DELAYED stores), canonical-record checks, unique refs, holdout dataset ≠ development datasets, supersession needs new testing evidence, violating journal entries quarantined on reload | `test_research_memory_integrity.py` |
| Claude output cannot command | closed schema, unknown keys rejected, no execution path | `test_claude_contract_security.py` |

## Decision pipeline (one trade candidate)

1. Champion strategy computes a signal from a `PointInTimeView` (deterministic).
2. If flat and LONG: build a point-in-time packet (prices, features, signal, portfolio summary,
   memory as of the cutoff, prior rejected hypotheses). External text is fenced as untrusted data.
3. Claude primary role → schema validation. Invalid/pending/over-budget → no trade.
4. Claude adversarial role → may BLOCK. Invalid → no trade.
5. Effective stop = the tighter of Claude's and the strategy's. Claude's quantity only lowers size.
6. Risk engine evaluates; the `DecisionRecord` is journaled (always, before any order).
7. Only an approved, signed verdict reaches `ExecutionEngine.submit`.

Exits (stop hit, signal flat) are deterministic and never wait for reasoning.

## Autonomous loop (resumable tick)

```
WAKE → HEALTH (journal verification) → DATA (fetch, validate, store, freshness)
     → RECONCILE (venue vs ledger; not OK ⇒ stop)
     → RISK STATE (day start, peak equity) → MONITOR (stops) → SCAN (champion)
     → DECIDE (pipeline) → EXIT → POST-TRADE REVIEW → MEMORY → WAIT
```

Each tick is bracketed by journaled `tick_start`/`tick_end`. A missing `tick_end` on startup is
detected and reported; since every tick reconciles before adding risk, recovery is the normal path.
The loop owns no long-lived process state, so it can be driven by any scheduler (see below).

## Claude-native operation

No server, container, or VPS is required. The deterministic system is a Python package with a
stdlib-only runtime and file-based journals. Two ways to supply reasoning:

- **File exchange** (`FileExchangeClient`): the loop writes `requests/<role>__<id>.md`; a Claude
  session (for example one woken hourly by a Routine) reads it and writes
  `responses/<role>__<id>.json`; the next tick consumes it. While pending, no risk is taken.
- **API client**: same `ReasoningClient` protocol; NOT IMPLEMENTED (no project API key).

## Hierarchical reasoning (quality per token)

| Tier | When | Who |
|---|---|---|
| NONE | every tick: health, data, reconciliation, stops, signal evaluation, exits | deterministic code |
| LIGHT | after an exit: post-trade process review | one short role call |
| DEEP | only when a champion signal creates an entry candidate: primary decision + adversarial review | two role calls |

A per-tick call budget is enforced; exhausting it defaults to no trade. Decision ids are derived
from (strategy, symbol, information cutoff), so the same information is never reasoned about twice.
