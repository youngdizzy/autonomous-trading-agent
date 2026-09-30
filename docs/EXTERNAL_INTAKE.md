# External Strategy Intake (Phase 5)

External corpora supply hypotheses; TradeTown supplies proof. An external strategy can only ever become a
*research candidate*, and only through the existing research workflow. Status vocabulary: `docs/STATUS.md`.

## Trust boundary

```
external git repository @ exact commit
  → SourceArtifact        ati/intake/source.py     bytes whose git blob id must equal the commit's tree listing; SHA-256
  → parse (text only)     ati/intake/vault.py      sections, arguments, Pine structure, security scan, EXTERNAL_CLAIMs
  → normalize/classify    ati/intake/normalize.py  compatibility state; COMPATIBLE only by an exact mapping rule
  → journal               ati/intake/corpus.py     external_source · external_intake · research_universe (research.jsonl)
  → research_external     → ati.research.workflow.run_research_cycle (unchanged): pre-registration (intake id in
                            observation_refs), walk-forward on development, adversarial, sealed holdout, promotion gate
```

Never possible (tested structurally): external code executed or imported; intake importing execution, broker,
risk, agent or control-plane modules; validation, promotion, readiness or learning code reading intake records;
external claims registered as evidence; an external candidate choosing its own holdout boundary or thresholds.

## Identity

`external_strategy_id = "ext_" + sha256(repository, commit, path, source_hash)[:24]`. Re-importing the same bytes at the
same commit/path is a no-op; the same location with different bytes raises `ExternalSourceError`; a changed file at a
later commit is a new identity and the old one stays. Raw third-party source is not copied into the journal: the
corpus commit is the archive, and `ExternalIntake.verify(corpus)` re-derives every recorded hash from it.

## Compatibility states (first failing check wins; every reason is recorded)

PARSE_FAILED · UNSAFE · UNSUPPORTED_LANGUAGE · INCOMPATIBLE · UNSUPPORTED_EXECUTION_MODEL · UNSUPPORTED_MARKET ·
UNSUPPORTED_TIMEFRAME · UNSUPPORTED_INDICATOR · SEMANTICS_UNCERTAIN · PARTIALLY_COMPATIBLE · COMPATIBLE.

Mapping rule `map-1` is the only route to COMPATIBLE: the Pine form of TradeTown's `ma_crossover` (state-based
SMA(fast) > SMA(slow), exit on fast <= slow, stop close − k·ATR ratcheted while long, ATR = *simple mean* of true range
— Pine's `atr()` is Wilder's RMA and is therefore UNSUPPORTED_INDICATOR). The source block is the specification of
record; descriptions are metadata. External sizing/capital/commission settings are recorded, never adopted.

## Multiple testing

Each selection batch is a `research_universe` record (corpus commit and size, selection method and salt, every
considered id and its state). `factory.attempts()` reports `external_universes_before` and
`external_candidates_considered_before`, so a candidate's search history travels with it. Pilot selection is a
deterministic sha256(salt + path) order stratified by language — blind to names, descriptions and claims.

## Operation (manual)

The system never shells out, so the operator runs git (outside the system) and hands over the commit's tree listing:

```bash
git clone https://github.com/brainbrick-trades/the-quant-trading-vault <corpus>          # read-only corpus checkout
git -C <corpus> checkout <sha> && git -C <corpus> ls-tree -r -z <sha> > <manifest>
python -m ati intake-vault --state-dir <state> --data kraken --corpus <corpus> --manifest <manifest> --commit <sha>
```

Every file's git blob id is re-derived from its bytes and must equal the listing (a modified, re-encoded or
other-commit checkout is refused). The commit id itself is operator-declared; the recorded blob ids let anyone
re-verify against the public repository. Security-scan signatures are data (`ati/intake/signatures.json`), so the
system's code never names shell or dynamic-execution APIs. Pilots are capped at 25 files.

Research of a COMPATIBLE record is an explicit call to `ati.intake.corpus.research_external(system, intake, id, full,
protocol=...)`; it needs the protocol's 3,000 REAL candles like any other candidate (no REAL data exists yet).

## Pilot result (brainbrick-trades/The-Quant-Trading-Vault @ c9d6fa49486855899a92fea65004f440533049aa)

Corpus: 5,806 strategy files (Pine 5,283 · JavaScript 362 · Python 131 · MyLanguage 27 · C++ 3). Pilot: 24 files
(Pine 14 · JS 4 · Python 3 · MyLanguage 2 · C++ 1). COMPATIBLE 0 · researched 0. Primary states:
UNSUPPORTED_EXECUTION_MODEL 11, UNSUPPORTED_LANGUAGE 10, INCOMPATIBLE 2, UNSUPPORTED_MARKET 1.
