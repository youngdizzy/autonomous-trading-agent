"""Historical candle store. Append-only: once a closed candle is recorded, a different value for
the same (provider, symbol, timeframe, open_time) is a historical conflict and fails closed."""

from __future__ import annotations

from typing import Iterable

from ati.core.errors import DataIntegrityError, HistoricalConflictError
from ati.market.models import Candle, DataStatus, Timeframe

_Key = tuple[str, str, Timeframe, object]


class CandleStore:
    def __init__(self) -> None:
        self._candles: dict[_Key, Candle] = {}
        self._status: dict[tuple[str, str, Timeframe], DataStatus] = {}

    def ingest(self, candles: Iterable[Candle]) -> int:
        """Record closed candles. Returns the number of new candles. Open candles are refused:
        history only contains final values. Validates the whole batch before writing any of it."""
        batch = list(candles)
        staged: dict[_Key, Candle] = {}
        for candle in batch:
            if not candle.is_closed:
                raise DataIntegrityError("INCOMPLETE", f"refusing open candle {candle.symbol}@{candle.open_time}")
            series = (candle.provider, candle.symbol, candle.timeframe)
            known_status = self._status.get(series)
            if known_status is not None and known_status is not candle.status:
                raise DataIntegrityError("STATUS_MIX", f"{series} is {known_status.value}, got {candle.status.value}")
            key = (*series, candle.open_time)
            existing = self._candles.get(key) or staged.get(key)
            if existing is not None and existing.content_key() != candle.content_key():
                raise HistoricalConflictError(
                    f"history disagreement for {candle.symbol} {candle.timeframe.value} @ {candle.open_time}: "
                    f"recorded {existing.content_key()[1:6]} vs received {candle.content_key()[1:6]}"
                )
            if existing is None:
                staged[key] = candle
        for key, candle in staged.items():
            self._candles[key] = candle
            self._status[key[:3]] = candle.status
        return len(staged)

    def series(self, provider: str, symbol: str, timeframe: Timeframe) -> list[Candle]:
        items = [c for (p, s, tf, _), c in self._candles.items() if (p, s, tf) == (provider, symbol, timeframe)]
        return sorted(items, key=lambda c: c.open_time)
