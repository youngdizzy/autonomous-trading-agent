"""MOCK market data. Every candle produced here carries ``DataStatus.MOCK`` and a provenance that
names the generator and seed. MOCK data is for exercising mechanics; it is never evidence about
real markets."""

from __future__ import annotations

import math
import random
from datetime import datetime
from decimal import Decimal

from ati.core.clock_util import require_aligned
from ati.core.time import Clock, ensure_utc
from ati.market.models import Candle, DataStatus, Provenance, Timeframe

_Q = Decimal("0.01")


class MockProvider:
    """Deterministic regime-switching random walk. Same (seed, params) → same candles."""

    name = "mock"

    def __init__(
        self,
        seed: int,
        clock: Clock,
        *,
        start_price: float = 30000.0,
        vol_per_bar: float = 0.006,
        regime_drift: float = 0.0015,
        regime_len: int = 120,
        epoch: datetime | None = None,
    ):
        self.seed = seed
        self.clock = clock
        self.start_price = start_price
        self.vol = vol_per_bar
        self.drift = regime_drift
        self.regime_len = regime_len
        self.epoch = ensure_utc(epoch) if epoch else None

    @property
    def realization(self) -> str:
        return f"mock.random_walk(seed={self.seed},vol={self.vol},drift={self.drift},regime_len={self.regime_len})"

    def fetch_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> list[Candle]:
        start, end = ensure_utc(start), ensure_utc(end)
        require_aligned(start, timeframe)
        epoch = self.epoch or start
        require_aligned(epoch, timeframe)
        now = self.clock.now()
        # Generate the whole path from the epoch so any window of the same realization is consistent.
        rng = random.Random(f"{self.seed}:{symbol}:{timeframe.value}")
        price = self.start_price
        n_total = int((end - epoch).total_seconds() // timeframe.seconds)
        out: list[Candle] = []
        drift = 0.0
        for i in range(n_total):
            if i % self.regime_len == 0:
                drift = rng.choice((-self.drift, 0.0, self.drift)) * rng.uniform(0.3, 1.0)
            open_time = epoch + timeframe.delta * i
            ret = drift + rng.gauss(0.0, self.vol)
            o = price
            c = max(o * math.exp(ret), 0.01)
            wick = abs(rng.gauss(0.0, self.vol * 0.5))
            h = max(o, c) * (1 + wick)
            l = min(o, c) * (1 - abs(rng.gauss(0.0, self.vol * 0.5)))
            vol = abs(rng.gauss(50.0, 15.0)) + 1.0
            price = c
            if open_time < start:
                continue
            close_time = open_time + timeframe.delta
            if open_time > now:
                break
            closed = close_time <= now
            od, cd = Decimal(repr(o)).quantize(_Q), Decimal(repr(c)).quantize(_Q)
            hd = max(Decimal(repr(h)).quantize(_Q), od, cd)
            ld = max(min(Decimal(repr(l)).quantize(_Q), od, cd), _Q)
            out.append(
                Candle(
                    provider=self.name,
                    symbol=symbol,
                    timeframe=timeframe,
                    open_time=open_time,
                    open=od,
                    high=hd,
                    low=ld,
                    close=cd,
                    volume=Decimal(repr(vol)).quantize(Decimal("0.0001")),
                    is_closed=closed,
                    status=DataStatus.MOCK,
                    received_at=now,
                    provenance=Provenance(source="mock", method=self.realization, retrieved_at=now, note="MOCK"),
                )
            )
        return out
