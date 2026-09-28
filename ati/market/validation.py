"""Series-level market data integrity. Every check fails closed with a stable code."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

from ati.core.errors import DataIntegrityError
from ati.core.time import ensure_utc
from ati.market.models import Candle, Timeframe


def validate_series(
    candles: Sequence[Candle],
    *,
    symbol: str,
    timeframe: Timeframe,
    provider: str | None = None,
    require_closed: bool = True,
) -> None:
    if not candles:
        raise DataIntegrityError("EMPTY", "no candles")
    first = candles[0]
    prev: Candle | None = None
    for candle in candles:
        if not isinstance(candle, Candle):
            raise DataIntegrityError("TYPE", f"expected Candle, got {type(candle).__name__}")
        if candle.symbol != symbol:
            raise DataIntegrityError("SYMBOL", f"expected {symbol}, got {candle.symbol}")
        if candle.timeframe is not timeframe:
            raise DataIntegrityError("TIMEFRAME", f"expected {timeframe.value}, got {candle.timeframe.value}")
        if provider is not None and candle.provider != provider:
            raise DataIntegrityError("PROVIDER", f"expected {provider}, got {candle.provider}")
        if candle.provider != first.provider:
            raise DataIntegrityError("PROVIDER", "mixed providers in one series")
        if candle.status is not first.status:
            raise DataIntegrityError(
                "STATUS_MIX", f"mixed data status {first.status.value} and {candle.status.value}; never silently combined"
            )
        if require_closed and not candle.is_closed:
            raise DataIntegrityError("INCOMPLETE", f"open candle at {candle.open_time} where closed data is required")
        if prev is not None:
            if candle.open_time == prev.open_time:
                raise DataIntegrityError("DUPLICATE", f"duplicate timestamp {candle.open_time}")
            if candle.open_time < prev.open_time:
                raise DataIntegrityError("NON_MONOTONIC", f"{candle.open_time} after {prev.open_time}")
        prev = candle


def count_gaps(candles: Sequence[Candle]) -> int:
    """Missing bars between consecutive candles. Gaps are recorded, not filled: fabricating a bar
    would be fabricating evidence."""
    gaps = 0
    for a, b in zip(candles, candles[1:]):
        missing = int((b.open_time - a.open_time).total_seconds() // a.timeframe.seconds) - 1
        gaps += max(missing, 0)
    return gaps


def check_freshness(latest: Candle, now: datetime, max_age: timedelta) -> None:
    """The latest *closed* candle must have closed within ``max_age`` of ``now``."""
    now = ensure_utc(now, "now")
    if not latest.is_closed:
        raise DataIntegrityError("INCOMPLETE", "freshness must be judged on a closed candle")
    if latest.close_time > now:
        raise DataIntegrityError("FUTURE", f"candle closes at {latest.close_time}, after now {now}")
    age = now - latest.close_time
    if age > max_age:
        raise DataIntegrityError("STALE", f"latest closed candle is {age} old (max {max_age})")
