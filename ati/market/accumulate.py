"""Incremental accumulation of market evidence: BTC/USD and ETH/USD × 1h and 4h.

Every series is accumulated through the system's own provider and payload archive — raw bytes are journaled
write-ahead, parsed by the provider adapter, validated by the candle gate, and stored append-only:

  first load        new closed candles are stored
  exact overlap     already-known candles are ignored (no duplicates)
  partial overlap   only genuinely new candles are appended; runs after the first request only from a fixed
                    overlap (24 × 1h / 6 × 4h bars) before the last stored bar, so recent history is re-verified
                    every run while archived raw bytes stay small
  conflict          same open time, different values → HistoricalConflictError: the payload is kept archived
                    and rejected, the series is marked DATA_CONFLICT (durable) and blocked until an operator
                    acknowledges it. Recorded history is never overwritten.
  provider failure  nothing is written for that series; evidence already stored is untouched
  open candle       the forming bar is never stored as closed evidence

Series are isolated: one series failing (provider, conflict, malformed data) never affects another. Each run
appends one ``accumulation_run`` record per series to the evidence journal (the archive's own journal), so
first/latest accumulation times and outcomes survive restarts. The state-directory lock is held by the CLI for
the whole run, so two accumulations cannot interleave.

No fabricated data: a failed series is reported as failed. There is no fallback to another category.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ati.core.errors import (DataIntegrityError, HistoricalConflictError, MalformedResponse, ProviderError)
from ati.data.dataset import Dataset
from ati.market.models import Timeframe

SERIES: tuple[tuple[str, Timeframe], ...] = (("BTC/USD", Timeframe.H1), ("BTC/USD", Timeframe.H4),
                                             ("ETH/USD", Timeframe.H1), ("ETH/USD", Timeframe.H4))
WINDOW_BARS = 720      # Kraken's OHLC endpoint returns at most 720 most-recent bars per request
OVERLAP_BARS = {Timeframe.H1: 24, Timeframe.H4: 6}   # recent history re-checked for conflicts on every run


@dataclass(frozen=True)
class SeriesResult:
    symbol: str
    timeframe: str
    status: str            # ACCUMULATED | NO_NEW_DATA | PROVIDER_FAILURE | REJECTED | DATA_CONFLICT | BLOCKED
    received: int
    closed_received: int
    added: int
    candles_before: int
    candles_after: int
    dataset_before: str | None
    dataset_after: str | None
    detail: str = ""


def _dataset_id(system, symbol, tf) -> tuple[int, str | None]:
    series = system.store.series(system.provider.name, symbol, tf)
    if not series:
        return 0, None
    return len(series), Dataset.build(series, data_version="accumulated", realization=_realization(system)).dataset_id


def _realization(system) -> str:
    return getattr(system.provider, "realization", "observed")


def accumulate(system, now: datetime | None = None, series=SERIES, window_bars: int = WINDOW_BARS) -> list[SeriesResult]:
    s = system
    now = now or s.clock.now()
    results: list[SeriesResult] = []
    for symbol, tf in series:
        n_before, ds_before = _dataset_id(s, symbol, tf)
        base = dict(symbol=symbol, timeframe=tf.value, candles_before=n_before, dataset_before=ds_before)
        if symbol not in s.universe:
            results.append(SeriesResult(**base, status="BLOCKED", received=0, closed_received=0, added=0,
                                        candles_after=n_before, dataset_after=ds_before, detail="symbol not in universe"))
            continue
        if s.archive.series_conflicted(s.provider.name, symbol, tf):
            results.append(SeriesResult(**base, status="BLOCKED", received=0, closed_received=0, added=0,
                                        candles_after=n_before, dataset_after=ds_before,
                                        detail="DATA_CONFLICT unresolved: series blocked until an operator acknowledges"))
            continue
        start = now - tf.delta * window_bars
        start = start - timedelta(microseconds=start.microsecond, seconds=int(start.timestamp()) % tf.seconds)
        stored = s.store.series(s.provider.name, symbol, tf)
        if stored:
            # Incremental: re-request a fixed overlap before the last stored bar (so every run re-checks recent
            # history for conflicts) instead of the whole 720-bar window — the archive keeps every response's raw
            # bytes, so full-window polling would grow the evidence journal by ~110 KB per series per run.
            start = max(start, stored[-1].open_time - tf.delta * OVERLAP_BARS[tf])
        received = closed = added = 0
        try:
            candles = s.provider.fetch_candles(symbol, tf, start, now)
            received, closed = len(candles), sum(c.is_closed for c in candles)
            if candles and candles[0].status is not s.data_status:
                raise DataIntegrityError("STATUS_MIX", f"provider returned {candles[0].status.value}, system is "
                                                       f"{s.data_status.value}")
            added = s.archive.ingest(s.store, candles)
            status, detail = ("ACCUMULATED" if added else "NO_NEW_DATA"), ""
        except HistoricalConflictError as exc:
            status, detail = "DATA_CONFLICT", str(exc)[:300]
        except (MalformedResponse, DataIntegrityError) as exc:
            status, detail = "REJECTED", f"{type(exc).__name__}: {exc}"[:300]
        except ProviderError as exc:
            status, detail = "PROVIDER_FAILURE", f"{type(exc).__name__}: {exc}"[:300]
        n_after, ds_after = _dataset_id(s, symbol, tf)
        result = SeriesResult(**base, status=status, received=received, closed_received=closed, added=added,
                              candles_after=n_after, dataset_after=ds_after, detail=detail)
        s.evidence.journal.append("accumulation_run", {"at": now, "provider": s.provider.name, **result.__dict__,
                                                       "data_status": s.data_status.value})
        results.append(result)
    return results


def history(system, symbol: str, tf: Timeframe) -> list[dict]:
    """Durable accumulation records for one series (from the evidence journal)."""
    from ati.ledger.journal import decode

    return [decode(e.payload) for e in system.evidence.journal.entries("accumulation_run")
            if e.payload["symbol"] == symbol and e.payload["timeframe"] == tf.value
            and e.payload["provider"] == system.provider.name]
