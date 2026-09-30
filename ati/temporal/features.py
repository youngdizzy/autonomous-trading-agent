"""Features computed strictly from a ``PointInTimeView``. They cannot see beyond the view."""

from __future__ import annotations

import math
from decimal import Decimal

from ati.temporal.pit import PointInTimeView


def sma(view: PointInTimeView, n: int) -> Decimal | None:
    if n <= 0 or len(view) < n:
        return None
    closes = view.closes(n)
    return sum(closes, Decimal(0)) / n


def atr(view: PointInTimeView, n: int) -> Decimal | None:
    candles = view.candles
    if n <= 0 or len(candles) < n + 1:
        return None
    trs = []
    for prev, cur in zip(candles[-n - 1:-1], candles[-n:]):
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    return sum(trs, Decimal(0)) / n


def realized_vol(view: PointInTimeView, n: int) -> float | None:
    closes = view.closes(n + 1)
    if len(closes) < n + 1 or n < 2:
        return None
    rets = [math.log(float(b) / float(a)) for a, b in zip(closes, closes[1:])]
    mean = sum(rets) / len(rets)
    return math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
