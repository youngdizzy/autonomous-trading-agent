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
