"""Typed market observations.

A ``Candle`` cannot be constructed in a malformed state: price/volume sanity, finiteness,
timezone discipline and closed-before-end impossibility are enforced at construction. Series-level
checks (ordering, duplicates, staleness, symbol/timeframe consistency) live in ``validation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from ati.core.errors import DataIntegrityError
from ati.core.time import ensure_utc


class DataStatus(str, Enum):
    """What kind of data this is. Never silently converted from one to another."""

    REAL = "REAL"                # live observation from a real venue
    HISTORICAL = "HISTORICAL"    # real venue data retrieved after the fact
    DELAYED = "DELAYED"          # real venue data with known delivery delay
    SIMULATED = "SIMULATED"      # produced by a simulator of real mechanics (e.g. paper fills)
    MOCK = "MOCK"                # hand-made or generated test data; never evidence about markets
    SYNTHETIC = "SYNTHETIC"      # generated to study a statistical property
    UNKNOWN = "UNKNOWN"


#: Categories that describe the real market. Only these can ever support market evidence.
MARKET_EVIDENCE_STATUSES = frozenset({DataStatus.REAL, DataStatus.HISTORICAL, DataStatus.DELAYED})


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        return {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[self.value]

    @property
    def delta(self) -> timedelta:
        return timedelta(seconds=self.seconds)

    @property
    def bars_per_year(self) -> float:
        return 365.0 * 86400 / self.seconds


@dataclass(frozen=True)
class Provenance:
    """Where an observation came from. ``raw_sha256`` hashes the raw provider payload when one
    exists, so the parsed values can be traced to the exact bytes received."""

    source: str
    method: str
    retrieved_at: datetime
    raw_sha256: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieved_at", ensure_utc(self.retrieved_at, "provenance.retrieved_at"))
        if not self.source or not self.method:
            raise DataIntegrityError("PROVENANCE", "provenance requires source and method")


def _finite_decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise DataIntegrityError("TYPE", f"{name} must be Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise DataIntegrityError("NONFINITE", f"{name} is not finite: {value}")
    return value


@dataclass(frozen=True)
class Candle:
    provider: str
    symbol: str
    timeframe: Timeframe
    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    is_closed: bool
    status: DataStatus
    received_at: datetime
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.timeframe, Timeframe):
            raise DataIntegrityError("TIMEFRAME", f"invalid timeframe {self.timeframe!r}")
        if not isinstance(self.status, DataStatus):
            raise DataIntegrityError("STATUS", f"invalid data status {self.status!r}")
        if not isinstance(self.is_closed, bool):
            raise DataIntegrityError("TYPE", "is_closed must be bool")
        if not self.provider or not self.symbol:
            raise DataIntegrityError("IDENTITY", "provider and symbol are required")
        if self.status in MARKET_EVIDENCE_STATUSES:
            # First gate on market-evidence labels: they must trace to a provider payload. The binding
            # gate is ati.market.archive.verify_market_provenance, which re-derives candles from the
            # journaled raw payload.
            raw = self.provenance.raw_sha256 if isinstance(self.provenance, Provenance) else None
            if not (isinstance(raw, str) and len(raw) == 64 and all(ch in "0123456789abcdef" for ch in raw)):
                raise DataIntegrityError("PROVENANCE", f"{self.status.value} candle without a provider payload hash")
            if self.provenance.source != self.provider:
                raise DataIntegrityError("PROVENANCE", "provenance source does not match provider")
        try:
            object.__setattr__(self, "open_time", ensure_utc(self.open_time, "open_time"))
            object.__setattr__(self, "received_at", ensure_utc(self.received_at, "received_at"))
        except (TypeError, ValueError) as exc:
            raise DataIntegrityError("TIMESTAMP", str(exc)) from exc
        if int(self.open_time.timestamp()) % self.timeframe.seconds != 0 or self.open_time.microsecond:
            raise DataIntegrityError("ALIGNMENT", f"open_time {self.open_time} not aligned to {self.timeframe.value}")
        o, h, l, c = (_finite_decimal(getattr(self, n), n) for n in ("open", "high", "low", "close"))
        v = _finite_decimal(self.volume, "volume")
        if min(o, h, l, c) <= 0:
            raise DataIntegrityError("PRICE", f"non-positive price in {self.symbol}@{self.open_time}")
        if h < max(o, c, l) or l > min(o, c, h):
            raise DataIntegrityError("OHLC", f"malformed OHLC o={o} h={h} l={l} c={c}")
        if v < 0:
            raise DataIntegrityError("VOLUME", f"negative volume {v}")
        if self.is_closed and self.received_at < self.close_time:
            raise DataIntegrityError(
                "CLOSED_BEFORE_END",
                f"candle claims closed at receipt {self.received_at} before its end {self.close_time}",
            )

    @property
    def close_time(self) -> datetime:
        return self.open_time + self.timeframe.delta

    @property
    def available_at(self) -> datetime:
        """Earliest instant at which this candle's final values were knowable."""
        return self.close_time

    def content_key(self) -> tuple:
        """Market content only. Excludes receipt time and provenance so that re-fetching identical
        history yields identical content identity."""
        return (self.open_time, self.open, self.high, self.low, self.close, self.volume, self.is_closed)
