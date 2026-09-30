"""Test helpers. All data produced here is MOCK."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from ati.core.time import UTC, FixedClock
from ati.data.dataset import Dataset
from ati.market.mock import MockProvider
from ati.market.models import Candle, DataStatus, Provenance, Timeframe

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def make_candle(i: int = 0, *, symbol="BTC/USD", tf=Timeframe.H1, o="100", h="101", l="99", c="100.5", v="10",
                closed=True, status=DataStatus.MOCK, provider="mock", received_at=None, open_time=None) -> Candle:
    open_time = open_time or (T0 + tf.delta * i)
    received = received_at or (open_time + tf.delta if closed else open_time)
    return Candle(
        provider=provider, symbol=symbol, timeframe=tf, open_time=open_time,
        open=Decimal(o), high=Decimal(h), low=Decimal(l), close=Decimal(c), volume=Decimal(v),
        is_closed=closed, status=status, received_at=received,
        provenance=Provenance(source=provider, method="test-fixture", retrieved_at=received, note="MOCK"),
    )


def mock_dataset(n: int = 500, seed: int = 7, tf: Timeframe = Timeframe.H1, symbol: str = "BTC/USD", **kw) -> Dataset:
    end = T0 + tf.delta * n
    clock = FixedClock(end)
    provider = MockProvider(seed, clock, **kw)
    candles = provider.fetch_candles(symbol, tf, T0, end)
    return Dataset.build(candles, data_version="mock-v1", realization=provider.realization)
