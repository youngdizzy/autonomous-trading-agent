"""UTC time discipline. Naive datetimes are rejected everywhere: an ambiguous timestamp is an
unknown timestamp, and unknown temporal state is unsafe."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol

UTC = timezone.utc


def ensure_utc(value: datetime, name: str = "timestamp") -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{name} must be timezone-aware (got naive datetime)")
    return value.astimezone(UTC)


def parse_utc(text: str) -> datetime:
    if not isinstance(text, str):
        raise TypeError("timestamp text must be str")
    return ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))


def to_iso(value: datetime) -> str:
    return ensure_utc(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def from_epoch(seconds: int | float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=UTC)


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """Deterministic clock for tests and replay."""

    def __init__(self, start: datetime):
        self._now = ensure_utc(start)

    def now(self) -> datetime:
        return self._now

    def set(self, value: datetime) -> None:
        self._now = ensure_utc(value)

    def advance(self, delta: timedelta) -> datetime:
        self._now = self._now + delta
        return self._now
