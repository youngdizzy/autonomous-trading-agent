"""Point-in-time access.

At historical decision time T the decision process may only see information whose
``available_at <= T``. For candles, ``available_at`` is the close time: a candle that opened
before T but closes after T is *not* visible — its close price did not exist yet.

``PointInTimeView`` holds only an already-filtered tuple (never a reference to the full dataset),
and re-validates the filter on construction, so a view cannot be built that contains the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Iterable, Protocol, Sequence

from ati.core.errors import LookaheadError
from ati.core.time import ensure_utc
from ati.market.models import Candle


class PointInTimeView:
    __slots__ = ("_candles", "_cutoff", "dataset_id")

    def __init__(self, candles: Sequence[Candle], cutoff: datetime, dataset_id: str):
        cutoff = ensure_utc(cutoff, "cutoff")
        for candle in candles:
            if not candle.is_closed or candle.available_at > cutoff:
                raise LookaheadError(
                    f"candle {candle.symbol}@{candle.open_time} available at {candle.available_at} is after cutoff {cutoff}"
                )
        self._candles = tuple(candles)
        self._cutoff = cutoff
        self.dataset_id = dataset_id

    @property
    def cutoff(self) -> datetime:
        return self._cutoff

    @property
    def candles(self) -> tuple[Candle, ...]:
        return self._candles

    def __len__(self) -> int:
        return len(self._candles)

    @property
    def latest(self) -> Candle:
        if not self._candles:
            raise LookaheadError("no information available at cutoff")
        return self._candles[-1]

    def closes(self, n: int | None = None) -> list[Decimal]:
        rows = self._candles if n is None else self._candles[-n:]
        return [c.close for c in rows]

    def max_available_at(self) -> datetime | None:
        return self._candles[-1].available_at if self._candles else None


class Timestamped(Protocol):
    @property
    def available_at(self) -> datetime: ...


@dataclass(frozen=True)
class InformationItem:
    """Any non-candle information (news, research result, model output, regime label)."""

    item_id: str
    kind: str
    available_at: datetime
    content: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_at", ensure_utc(self.available_at, "available_at"))


class InformationSet:
    """Everything the decision process may know at ``cutoff``. Items from the future are excluded
    at construction; asking for one by id raises ``LookaheadError`` rather than returning None,
    so a leak attempt is loud."""

    def __init__(self, cutoff: datetime, items: Iterable[InformationItem]):
        self.cutoff = ensure_utc(cutoff, "cutoff")
        self._visible: dict[str, InformationItem] = {}
        self._future_ids: set[str] = set()
        for item in items:
            if item.available_at <= self.cutoff:
                self._visible[item.item_id] = item
            else:
                self._future_ids.add(item.item_id)

    def get(self, item_id: str) -> InformationItem:
        if item_id in self._future_ids:
            raise LookaheadError(f"{item_id} is not available until after cutoff {self.cutoff}")
        return self._visible[item_id]

    def items(self, kind: str | None = None) -> list[InformationItem]:
        return [i for i in self._visible.values() if kind is None or i.kind == kind]


def assert_available(timestamps: Iterable[datetime], cutoff: datetime, what: str = "reference") -> None:
    cutoff = ensure_utc(cutoff)
    for ts in timestamps:
        if ensure_utc(ts) > cutoff:
            raise LookaheadError(f"{what} available at {ts} is after cutoff {cutoff}")
