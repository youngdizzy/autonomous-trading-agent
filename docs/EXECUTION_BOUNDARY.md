# Broker Execution Boundary 1.0

The model thinks. TradeTown enforces. The broker executes. The journal remembers. Status vocabulary: `docs/STATUS.md`.

## Call graph (repository truth)

```
Claude (file exchange)                    ati/agent/reasoning.py        text only; no tools, credentials or venue handle
  → schema-validated action               ati/agent/schema.py           closed actions: NO_TRADE · TRADE_PROPOSAL · RESEARCH_REQUEST ·
                                                                        PAUSE · REQUEST_DATA · REVIEW_*; closed TRADE_PROPOSAL payload
  → control plane                         ati/company/control.py        proposal must cite its symbol's champion; entry signal required
  → adversarial review                    ati/agent/pipeline.py         can only BLOCK or annotate
  → RiskContract (RiskLimits) + RiskEngine ati/risk/engine.py           mode permitted, account known, reconciled, symbol, data category,
                                                                        data freshness, price sanity, slippage, kill switch, drawdown,
                                                                        daily loss, stop, sizing caps (risk budget, notional, symbol/
                                                                        portfolio/correlated exposure, leverage, cash, liquidity);
                                                                        Claude's quantity can only lower the size; HMAC-signed verdict
  → DecisionRecord (write-ahead)          ati/decision/records.py
  → execution gates                       ati/execution/engine.py       signature · expiry · venue mode · idempotency (client order id
                                                                        from decision id) · execution mode · LIVE_TRADING · kill switch
                                                                        · account freshness · open order · operator approval (ASSISTED)
                                                                        · autonomous limits · broker health
  → order intent (write-ahead) → Broker   ati/execution/{paper,mock}.py  (live adapter: ati/execution/live.py, NOT IMPLEMENTED)
Broker → order status / fills (separate records) → reconciliation → execution journal → outcome → learning → Claude context
```

The repository has no components named Gatekeeper, Nexus or Emergency Stop: the gatekeeping is the execution-gate stage
above and the emergency stop is the persistent `KillSwitch` (`ati/risk/killswitch.py`; unreadable state reads as
ENGAGED; release needs the operator acknowledgement). Under the kill switch no new risk is approved or submitted;
risk-reducing sells stay permitted (existing policy); open orders and positions stay visible; the operator can cancel.

## Modes (`ati/execution/policy.py`, operator configuration at system construction)

OBSERVE (nothing is sent) · PAPER (default) · ASSISTED (new risk needs `ExecutionEngine.approve(decision_id, ack)`,
journaled) · AUTONOMOUS_LIMITED (new risk must fit configured symbol/notional/daily-count limits, tighter only).
LIVE is not a mode: `ati.config.LIVE_TRADING` is a code constant (False); a LIVE execution engine cannot be
constructed, the risk engine refuses LIVE portfolios, and the execution engine re-checks the constant at submission.

## Lifecycle and reconciliation

PENDING_SUBMIT → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED; CANCEL_REQUESTED → CANCELED / PARTIAL_CANCELED; REJECTED;
EXPIRED; NOT_FOUND; UNKNOWN. Any uncertain outcome (timeout, lost response, unconfirmed cancel, malformed report) is
UNKNOWN and halts new submissions. Reconciliation queries every unresolved order, lists all venue orders and compares
cash, positions, fill ids, fill quantities and prices; unexpected orders or fills are reported (MISMATCH), never
auto-applied. Venue-confirmed working orders are tracked, not errors, but block new orders for the symbol.

## Live status

**PAPER_READY.** No broker is configured in this repository, no credentials exist, and no broker API contract has
been verified from this environment. A real adapter must implement `ati.execution.broker.Broker` (idempotent submit on
client order id, order lookup, order list, cancel, account snapshot, health check), hold its credential as a
`SecretValue` registered with the `SecretGuard`, pass `test_broker_execution_boundary.py`, `test_chaos.py` and
`test_execution_ledger.py` against a broker sandbox, and requires an explicit owner-reviewed change of `LIVE_TRADING`
plus an external authorization mechanism (NOT IMPLEMENTED) before any live order.
