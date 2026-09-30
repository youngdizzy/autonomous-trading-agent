# Capability Status — Foundation 1.0

Vocabulary is exact. Nothing below is upgraded from one state to another.

- **IMPLEMENTED** — built and verified locally by tests.
- **IMPLEMENTED — EXTERNAL VERIFICATION BLOCKED** — built and contract-tested; the real external
  system could not be reached from this environment.
- **BLOCKED** — cannot be done in this environment.
- **NOT IMPLEMENTED** — deliberately out of scope for this milestone.
- **INSUFFICIENT EVIDENCE** — a claim that the evidence does not support.

| Capability | Status |
|---|---|
| Market data provider abstraction, typed candles, integrity checks | IMPLEMENTED |
| MOCK market data provider | IMPLEMENTED (all output labelled MOCK) |
| Kraken public OHLC adapter (pair/order/`last`/interval checks, transport-owned status) | IMPLEMENTED — EXTERNAL VERIFICATION BLOCKED (api.kraken.com denied by egress policy, re-confirmed 2026-09-28) |
| Raw provider payload archive (write-ahead, replay, provenance re-derivation) | IMPLEMENTED (verified with MOCK payloads only) |
| REAL provenance gate (REAL only from the network transport; files/fixtures/Claude cannot mint REAL) | IMPLEMENTED |
| Research preconditions → NOT_RUN; one run per hypothesis | IMPLEMENTED |
| Pre-declared REAL research protocol (`REAL-PROTOCOL-001`) | IMPLEMENTED; REAL RESEARCH RUN = NOT_RUN (no REAL data) |
| `ati ingest-kraken` / `ati research-real` | IMPLEMENTED — EXTERNAL VERIFICATION BLOCKED |
| Incremental accumulation BTC/USD + ETH/USD × 1h + 4h (`ati accumulate`: per-series isolation, overlap re-check, durable run records) | IMPLEMENTED — EXTERNAL VERIFICATION BLOCKED (api.kraken.com proxy 403, re-confirmed 2026-09-30); verified with a Kraken-shaped MOCK feed |
| Durable historical-conflict state (series blocked until operator acknowledgement; history never overwritten) | IMPLEMENTED |
| Concurrency safety (state-directory lock; journals refuse appends after outside growth) | IMPLEMENTED |
| Series health, gap reporting, sealed-holdout commitment re-verification, dataset scorecard (`ati data-health`, readiness) | IMPLEMENTED |
| Research on gapped data | Refused (DATA_GAP); research windows are contiguous, unsealed runs |
| REAL candles accumulated | NONE — REAL_DATA_UNAVAILABLE |
| Research protocol registry (`ati/research/protocols.py`: REAL-PROTOCOL-001 BTC/USD 1h, 002 BTC/USD 4h, 003 ETH/USD 1h, 004 ETH/USD 4h; deterministic protocol hashes; runs record protocol id + hash) | IMPLEMENTED — all four routable (Phase 4B); none has REAL data |
| Timeframe-aware research identity (registry key `strategy_id@vN/<timeframe>`; lifecycle and champion state per (symbol, timeframe); promotion records carry their dimension; legacy unscoped records kept as LEGACY_UNSCOPED, never guessed) | IMPLEMENTED |
| Readiness contract (REAL_DATA_UNAVAILABLE · BLOCKED · INSUFFICIENT_REAL_CANDLES · VALIDATION_PENDING · INSUFFICIENT_EVIDENCE · VALIDATION_FAILED · VALIDATION_PASSED · NOT_APPLICABLE_MOCK) | IMPLEMENTED; validation states derived only from recorded protocol runs |
| REAL research readiness (3,000-candle protocol minimum) | REAL_DATA_UNAVAILABLE; Kraken's OHLC endpoint returns ≤ 720 bars, so reaching 3,000 1h bars needs ~95 days of uninterrupted hourly accumulation (deep backfill NOT IMPLEMENTED) |
| stdlib HTTPS transport | IMPLEMENTED — EXTERNAL VERIFICATION BLOCKED |
| Deep historical backfill (Kraken Trades endpoint or other) | NOT IMPLEMENTED |
| Fail-closed historical store | IMPLEMENTED |
| Dataset identity, mutation detection, save/load | IMPLEMENTED |
| Point-in-time views, information sets, lookahead tests | IMPLEMENTED |
| Immutable versioned strategies (code-hashed), lifecycle, registry replay | IMPLEMENTED |
| Backtester (fees, spread, slippage, gap-aware stops, partial fills, risk sizing) | IMPLEMENTED |
| Walk-forward, robustness (params, costs, timing, liquidity, regimes) | IMPLEMENTED |
| Block bootstrap, trade-order Monte Carlo, attribution | IMPLEMENTED |
| Pre-registered hypotheses, experiment log, failed-experiment retention | IMPLEMENTED |
| Adversarial challenge with multiple-testing penalty | IMPLEMENTED (3 judgement questions are NOT_AUTOMATED by design) |
| Holdout vault (seal, overlap guard, lineage limit, budget, write-ahead log) | IMPLEMENTED |
| Champion/challenger promotion gate | IMPLEMENTED |
| Decision records | IMPLEMENTED |
| Deterministic risk engine + kill switch | IMPLEMENTED |
| Idempotent execution, reconciliation, recovery | IMPLEMENTED (against paper venue) |
| Paper venue | IMPLEMENTED (prices from whatever data feeds it; currently MOCK only) |
| Paper trading on REAL market data | BLOCKED — ENVIRONMENT CAPABILITY (no market data reachable); the same adapter→archive→loop→paper path is verified with a Kraken-shaped MOCK feed |
| Live broker adapter | NOT IMPLEMENTED; LIVE_TRADING = false |
| Structured memory + evidence registry (1.1: category gate, unique + independent evidence, quarantine on reload) | IMPLEMENTED |
| Research log persistence (1.1: locked criteria and hypothesis count survive restarts) | IMPLEMENTED |
| Claude output schema, roles, MOCK and file-exchange reasoning clients | IMPLEMENTED |
| Direct Claude API reasoning client | NOT IMPLEMENTED (BLOCKED — CREDENTIALS) |
| Autonomous loop (resumable tick) | IMPLEMENTED |
| Company control plane (Phase 1: closed actions, health gate, pause, recovery, idempotent cycles) | IMPLEMENTED (MOCK data and REAL-unavailable paths verified; REAL-available path untested — no REAL data) |
| Outcome learning / LEARN stage (outcome records, deterministic detectors, learning candidates, evidence-quality ladder, holdout-derived isolation) | IMPLEMENTED (MOCK outcomes only; no candidate has market evidence) |
| Learning-candidate contract (FACT / INTERPRETATION / PROPOSED_QUESTION, provenance, status OBSERVED→ANALYZED→HYPOTHESIS_CANDIDATE→PROMOTED_TO_HYPOTHESIS, REJECTED on contradiction) | IMPLEMENTED |
| Learning → memory (HYPOTHESIS kind only, through MemoryStore evidence gates, once per recurring trade pattern) | IMPLEMENTED |
| Learning → hypothesis bridge (learning evidence locked into the existing pre-registration) | IMPLEMENTED |
| Candidate attempts accounting ("how many attempts before this result") | IMPLEMENTED |
| Daily company intelligence report (`ati company report`) | IMPLEMENTED |
| Crash recovery across the ten learning/research stages | IMPLEMENTED (tested); a run interrupted after a promotion decision is reported for review, never applied or reversed by recovery |
| Experiment design contract (all six types; design, rationale, baseline and experiment id recorded before pre-registration) | IMPLEMENTED |
| SINGLE_VARIABLE / INTERACTION / STRUCTURAL candidate experiments (full WFO → adversarial/robustness → holdout → promotion gate) | IMPLEMENTED (STRUCTURAL limited to already-registered strategy logic; buy-and-hold cannot pass walk-forward selection) |
| REGIME / EXECUTION / RISK diagnostic experiments (pre-registered, development partition only, never a candidate) | IMPLEMENTED |
| Research windows that never reuse or overlap a used holdout | IMPLEMENTED |
| Candidate lineage record + derived evaluation trail (HYPOTHESIS → … → PROMOTION_REVIEW) | IMPLEMENTED |
| Idea saturation, compute and dataset-usage budgets | IMPLEMENTED |
| Future-condition assumption monitors (volatility, trend/range, liquidity, distribution, structural break, correlation, execution cost) | IMPLEMENTED (correlation NOT_AVAILABLE with one symbol; execution cost NOT_AVAILABLE without paper fills) |
| Read-only readiness report (`ati company readiness`) | IMPLEMENTED |
| Scheduler entry point (`ati company run`, bounded, stops at the first cycle needing Claude) | IMPLEMENTED; no scheduler is configured |
| End-to-end MOCK/PAPER company traversal (trade → outcome → learning → hypothesis → experiment → memory) | IMPLEMENTED — demonstrated in tests with a scripted MOCK Claude only |
| Objective contract (constraints, failure conditions, evidence requirements; no single score) | IMPLEMENTED |
| Research budget (hypotheses, variants per baseline, holdouts, runs per day — derived from journals) | IMPLEMENTED |
| Autonomy levels (RESEARCH_AUTONOMY and PAPER_AUTONOMY selectable; SUPERVISED_LIVE / FULL_LIVE impossible) | IMPLEMENTED |
| Company scorecard (10 independent dimensions, no aggregate) | IMPLEMENTED |
| Multi-source DATA_CONFLICT (comparator, persistent register, health gate, operator-only resolution) | IMPLEMENTED (exercised with two MOCK sources; a second real source is NOT IMPLEMENTED) |
| Text/news evidence layering (raw / verbatim facts / model interpretation; one source per source id) | IMPLEMENTED (structure only); news ingestion NOT IMPLEMENTED |
| Validated doctrine from learning | INSUFFICIENT EVIDENCE — requires holdout PASS + approved promotion on market data; none exists |
| Claude company decisions via file exchange | IMPLEMENTED — manual: a Claude session must write each response file |
| Scheduled autonomous operation (Routine driving the loop) | NOT IMPLEMENTED (mechanism available, not configured) |
| Status view | IMPLEMENTED (CLI demo prints it) |
| Real-money readiness | INSUFFICIENT EVIDENCE — no real data, no real fills, no out-of-sample record |
| Any validated market edge | INSUFFICIENT EVIDENCE — the only research run used MOCK data and was denied promotion |

## Real-money readiness criteria (none are met)

Real data across multiple regimes; realistic execution measured against real fills; positive
out-of-sample and holdout results after costs with sufficient trades; robustness and adversarial
review passing on real data; an extended paper period on real data with clean reconciliation; a
reviewed live adapter passing the same contract and chaos suites; a reviewed code change enabling
LIVE mode. Passing tests, profitable MOCK backtests, and Claude's confidence are explicitly not
criteria.
