# Company Control Plane — Company 1.0 Phase 1

Claude decides what it wants to do; the control plane decides whether that is a valid, currently permitted
action; the existing deterministic systems decide whether it is safe; only existing mechanisms perform it.

```
Claude ──(file exchange: one JSON response per request)──► parse_company_response   (ati/agent/schema.py)
                                                               │ closed vocabulary, bound to request+cycle
                                                               ▼
                                          CompanyControlPlane.run_cycle            (ati/company/control.py)
     deterministic duties ─ refresh data · reconcile · risk state · stop/signal exits   (AutonomousLoop stages)
     health gate ───────── PASS / BLOCKED / FAIL / NOT_READY / NOT_RUN per component    (ati/company/health.py)
     routing ────────────── TRADE_PROPOSAL → DecisionPipeline.route_proposal → RiskEngine → ExecutionEngine
                            RESEARCH_REQUEST → run_research_cycle (ResearchLog locks, holdout vault, memory)
                            PAUSE → persistent pause · REVIEW_* / REQUEST_DATA / NO_TRADE → read-only
```

## Operating one cycle (no daemon, no loop)

```bash
python -m ati company cycle  --state-dir DIR --data mock|kraken   # one bounded cycle
python -m ati company status --state-dir DIR --data mock|kraken
python -m ati company pause  --state-dir DIR --data mock|kraken
python -m ati company resume --state-dir DIR --data mock|kraken --ack "OPERATOR: company state reviewed; resume autonomous activity"
```

A cycle that needs Claude writes `DIR/exchange/requests/company__<hash>.md` and returns `AWAITING_CLAUDE`.
Claude writes `DIR/exchange/responses/company__<hash>.json` (same stem, a regular file ≤ 64 KiB). The next
`cycle` invocation consumes it. The response must be exactly:

```json
{"request_id": "<from the request packet>", "cycle_id": "<from the request packet>",
 "action": "NO_TRADE|TRADE_PROPOSAL|RESEARCH_REQUEST|PAUSE|REQUEST_DATA|REVIEW_POSITION|REVIEW_RISK|REVIEW_SYSTEM",
 "reason": "concise reason (≤ 600 chars)", "payload": { ... closed per action ... }}
```

## Rules the code enforces

- **Closed vocabulary.** Unknown actions, unknown envelope/payload fields, wrong request/cycle id, malformed
  JSON, and text that looks executable (shell, code, filesystem paths) → `CLAUDE_RESPONSE_REJECTED`, nothing
  executes. There is no RESUME action; resuming is operator-only.
- **Gate per action.** Read-only actions and PAUSE are always allowed while the company journal is intact.
  TRADE_PROPOSAL needs data, memory, risk, execution and ledger PASS, data state REAL_DATA_AVAILABLE or
  MOCK_DATA_ONLY, and not paused. RESEARCH_REQUEST needs data, research, memory and ledger PASS and not paused.
- **Trade proposals** are BUY-side entries for the current champion only, and only when the champion signals an
  entry at the current cutoff. They take exactly the existing path: adversarial review → RiskEngine →
  DecisionRecord → ExecutionEngine (PAPER). Claude may tighten the stop and lower the size, nothing else.
- **Research requests** name a pre-declared protocol; the protocol's strategy, grid, windows and minimum
  criteria are fixed. Claude may only add stricter criteria. A hypothesis id runs once; a registered id is
  locked.
- **Cycles.** Company state ∈ {IDLE, RUNNING, PAUSED, BLOCKED, FAILED, COMPLETED} with a fixed transition table.
  The cycle id is derived from observable company state: the same state is REPLAYed, never re-run. If state
  changes before an action starts, the pending response is stale and the cycle ends BLOCKED.
- **Recovery.** Every step is journaled once; a restart resumes after the last step, re-reads the response and
  requires its hash to match, re-applies the health gate, and never re-executes a recorded result. Trades
  recover by decision id (and re-register the position's stop); research recovers from the research journal.
- **Pause** is persistent; deterministic risk-reducing exits still run while paused.

## Data states

REAL_DATA_AVAILABLE · INSUFFICIENT_REAL_DATA · REAL_DATA_UNAVAILABLE · MOCK_DATA_ONLY · MIXED_DATA ·
INVALID_DATA — defined in `ati/company/health.py`. In this environment a Kraken-bound company reports
REAL_DATA_UNAVAILABLE (egress policy) and a MOCK company reports MOCK_DATA_ONLY: mechanics only, never
market evidence.

## Self-improvement engine

The cycle ends with a mandatory **LEARN** step (journaled once per cycle, idempotent) before FINALIZE.

    OUTCOME (journals) → OutcomeRecord → deterministic detector → LEARNING CANDIDATE
      → RESEARCH_REQUEST citing it (pre-registered) → experiment → walk-forward → adversarial → holdout
      → promotion gate (unchanged)

- **Outcomes** are read from the loop journal (trade reviews, data failures), the research journal
  (experiments, promotion decisions) and the company journal (rejected responses) — `learning.jsonl`.
- **Learning candidates** (`ati/company/learning.py`) are records, never rules. States:
  OBSERVED → ANALYZED → HYPOTHESIS_CANDIDATE → PREREGISTERED → TESTING → SUPPORTED | REJECTED | INCONCLUSIVE;
  terminal states stay visible forever. After PREREGISTERED, state follows `ResearchLog` facts only.
- **Evidence quality**: OBSERVATION (1 event) · WEAK (2) · REPEATED (≥3) · SUPPORTED (dev + holdout PASS) ·
  VALIDATED (SUPPORTED on market data with an approved promotion). A source event counts once.
- **Holdout isolation**: holdout and promotion outcomes are recorded but `holdout_derived`; they never become
  research-eligible, and holdout/promotion evidence refs cannot motivate a RESEARCH_REQUEST.
- **Experiment design** (optional `experiment` in RESEARCH_REQUEST): type, independent variables, dependent
  variable, controls, failure and stopping criteria, candidate params. SINGLE_VARIABLE changes exactly one
  baseline parameter and INTERACTION changes two or more. Both are executed. STRUCTURAL, REGIME, EXECUTION and
  RISK are BLOCKED (their executors are NOT IMPLEMENTED). The design and the named baseline are written to the
  research journal before pre-registration. The baseline is registered under its own key, and the candidate is
  derived as a child version, so it never replaces the baseline.
- **Comparison** (`ati/company/objectives.py`): the baseline and the candidate are compared on the recorded
  development partition, dimension by dimension. Constraints come first, then evidence requirements. The
  strongest conclusion is IMPROVED_ON_DEVELOPMENT, and only the promotion gate can promote.
- **Budget** (`ati/company/budget.py`) and **autonomy** (`ati/company/autonomy.py`): code-owned, derived from
  journals; the ceiling is PAPER_AUTONOMY.
- **Scorecard** (`ati/company/scorecard.py`): ten independent dimensions, each with its own state and no
  aggregate.
- **DATA_CONFLICT** (`ati/market/conflict.py`): if two sources disagree, the conflict is recorded in
  `conflicts.jsonl`. The health check reports INVALID_DATA and research and trading are blocked until an
  operator resolves the conflict.
- **Text evidence** (`ati/memory/textual.py`): keeps the raw source, verbatim facts and model interpretation
  apart. Any number of interpretations of one source count as one source.

The context package also carries the autonomy level, the objective contract, the research budget, the scorecard,
learning candidates, failed experiments and validated findings. These inform Claude but authorize nothing.
