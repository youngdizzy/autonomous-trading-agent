# Real Market Evidence Activation 1.0

Outcome: **ARCHITECTURE = VERIFIED · CODE = VERIFIED (offline) · LIVE CONNECTIVITY = BLOCKED · REAL EVIDENCE = NOT AVAILABLE.**
No REAL candle, dataset, research result or paper fill exists in this repository.

## Path from Kraken to evidence (as implemented)

```
UrllibTransport (only source of DataStatus.REAL)
  → KrakenPublicOHLC.fetch_candles
      → PayloadArchive.record        raw response text + receipt journaled to evidence.jsonl BEFORE parsing
      → parse_ohlc                   envelope, pair, strictly increasing times, `last`, interval, candle invariants
  → PayloadArchive.ingest → CandleStore   closed candles only; disagreement → payload marked rejected, FAIL CLOSED
  → Dataset.build                    content hash (values + provider/symbol/timeframe/status), identity
  → research_preconditions           provenance re-derivation, category, symbol, timeframe, closed, size, holdout, run-once
  → run_research_cycle / AutonomousLoop → paper execution only
```

## Provenance rules

1. A transport declares the category of the bytes it delivers. Only `UrllibTransport` declares REAL
   (enforced by a source-scan test). The adapter labels candles with its transport's category.
2. A REAL/HISTORICAL/DELAYED candle must carry a provider payload hash whose source matches its
   provider (construction-time gate).
3. The binding gate: `PayloadArchive.verify_market_provenance` re-parses the archived payload each
   candle cites and requires identical values, a closed candle, and a payload recorded with the same
   category. `Dataset.load` refuses market-evidence datasets without this verification, so a copied
   MOCK file relabelled REAL — even with every hash recomputed — is rejected (tested).
4. A system's data category must equal its provider's declared category; a MOCK provider cannot feed
   a REAL system (tested).
5. Nothing in `ati/agent` can construct candles, provenance, or statuses (source-scan test); Claude's
   output schema rejects any status/provenance field.

## Closed-candle rule

A Kraken row is closed only if it is not the final row (documented as the uncommitted frame), its
open time is ≤ `result.last`, and its close time is ≤ local receipt time. Anything else is a forming
candle, which datasets, the store and research refuse. Tested at the exact receipt-time boundary.
Unverified against real responses (BLOCKED).

## Audit defects found and fixed in this milestone

| Defect | Fix |
|---|---|
| `Dataset.load` trusted the file's own `status`; REAL could be minted by editing a JSON file | market-evidence datasets load only through archive re-derivation |
| Kraken adapter hard-coded REAL regardless of transport (a MOCK fixture produced REAL candles in tests) | status comes from the transport; fixture tests now yield MOCK |
| Adapter accepted any pair key, and the store silently sorted out-of-order/duplicate rows | pair must match request; rows must be strictly increasing |
| `result.last` ignored | rows after `last` are never closed; missing `last` fails closed |
| (found in self-review) archive keyed receipts by bytes only; identical bytes received later could re-derive differently after restart | raw bytes stored once, every receipt journaled; candles re-derived from their exact receipt |

## First REAL research run

`REAL-PROTOCOL-001` (`ati/research/protocol.py`) fixes the strategy, grid, criteria, windows and a
3,000-bar minimum before any real data exists. Result in this environment:

```
REAL RESEARCH RUN = NOT_RUN — no archived REAL data (ingestion BLOCKED by egress policy)
```

Even once Kraken is reachable, the public OHLC endpoint returns ≤ 720 bars, so the protocol stays
NOT_RUN (INSUFFICIENT DATA) until ~125 days of hourly payloads have been archived.

## Resource measurements (MOCK Kraken-shaped payloads, this container)

| Measure | Value |
|---|---|
| 720-row fetch → archive → parse → store (includes MOCK feed generation) | 204 ms, 1.6 MB peak allocation |
| Raw payload / evidence journal after one fetch | 63,508 B / 73,005 B |
| Restart replay of one payload | 12.6 ms |
| Provenance re-derivation of 720 candles | 6.9 ms |
| Research cycle (3,000 bars, 6-config grid, adversarial, holdout, promotion) | 6.0 s, 35 MB max RSS |
| Primary decision prompt / response | 1,999 / 371 characters |

Not a bottleneck. Future concern before automation: polling with a 300-bar window archives ~64 KB per
call; incremental `since` polling is needed before an hourly schedule.


## Phase 3 — evidence accumulation (operational)

```bash
python -m ati accumulate  --state-dir ./real --data kraken   # BTC/USD, ETH/USD × 1h, 4h; exit 3 if any series failed
python -m ati data-health --state-dir ./real --data kraken   # read-only scorecard, gaps, conflicts, sealed holdouts
python -m ati company readiness --state-dir ./real --data kraken
python -m ati verify --state-dir ./real
```

- **Canonical persistence** is the payload archive: every raw provider response is journaled write-ahead
  in `evidence.jsonl`. The candle store is rebuilt at startup by re-parsing those exact bytes, in journal
  order, with their original receipt times. Candles, provenance, dataset identities, ordering and conflict
  state therefore reconstruct exactly. Sealed holdout periods are restored from `research.jsonl`, and each is
  re-verified against its sealed content commitment.
- **Runs after the first** re-request a fixed overlap (24 × 1h, 6 × 4h) before the last stored bar. Recent
  history is re-checked for conflicts every run, and the archive grows by ≈ 11 KiB per run for all four
  series, instead of ≈ 440 KB with full-window polling.
- **Conflicts**: a provider answer that contradicts recorded history is archived and rejected, and the series
  is marked DATA_CONFLICT. It stays blocked, even across restarts, until an operator acknowledges it with the
  exact phrase in `ati.market.archive.CONFLICT_ACK`. The recorded history is never replaced.
- **Gaps** are reported with their locations and never filled. Research windows are contiguous runs outside
  every sealed holdout, and the research preconditions refuse any dataset that contains a gap.
- **Blocked here.** api.kraken.com is denied by this environment's network policy (proxy 403 on CONNECT).
  That is an environment permission, not a code change: allow `api.kraken.com` in the cloud environment's
  network settings.
