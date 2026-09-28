from datetime import timedelta
from decimal import Decimal

import pytest

from ati.core.errors import DataIntegrityError, HistoricalConflictError
from ati.core.time import FixedClock
from ati.market.mock import MockProvider
from ati.market.models import DataStatus, Timeframe
from ati.market.store import CandleStore
from ati.market.validation import check_freshness, count_gaps, validate_series
from tests.helpers import T0, make_candle


def codes(fn):
    with pytest.raises(DataIntegrityError) as exc:
        fn()
    return exc.value.code


class TestCandleConstruction:
    def test_valid(self):
        c = make_candle()
        assert c.close_time == T0 + timedelta(hours=1)
        assert c.available_at == c.close_time

    @pytest.mark.parametrize("kw,code", [
        (dict(h="99.5"), "OHLC"),              # high below open
        (dict(l="100.8"), "OHLC"),             # low above close
        (dict(o="0", l="0"), "PRICE"),         # impossible price
        (dict(o="-5", l="-6"), "PRICE"),
        (dict(v="-1"), "VOLUME"),
        (dict(c="NaN"), "NONFINITE"),
        (dict(h="Infinity"), "NONFINITE"),
    ])
    def test_malformed_rejected(self, kw, code):
        assert codes(lambda: make_candle(**kw)) == code

    def test_naive_timestamp_rejected(self):
        naive = T0.replace(tzinfo=None)
        assert codes(lambda: make_candle(open_time=naive, received_at=T0 + timedelta(hours=2))) == "TIMESTAMP"

    def test_misaligned_open_time_rejected(self):
        assert codes(lambda: make_candle(open_time=T0 + timedelta(minutes=7))) == "ALIGNMENT"

    def test_claimed_closed_before_its_end_rejected(self):
        assert codes(lambda: make_candle(received_at=T0 + timedelta(minutes=30))) == "CLOSED_BEFORE_END"

    def test_float_prices_rejected(self):
        from dataclasses import replace
        with pytest.raises(DataIntegrityError):
            replace(make_candle(), close=100.5)

    def test_provenance_required(self):
        from ati.market.models import Provenance
        with pytest.raises(DataIntegrityError):
            Provenance(source="", method="x", retrieved_at=T0)


class TestSeriesValidation:
    def series(self, n=5, **kw):
        return [make_candle(i, **kw) for i in range(n)]

    def test_valid(self):
        validate_series(self.series(), symbol="BTC/USD", timeframe=Timeframe.H1)

    def test_duplicate(self):
        s = self.series()
        s.insert(2, s[2])
        assert codes(lambda: validate_series(s, symbol="BTC/USD", timeframe=Timeframe.H1)) == "DUPLICATE"

    def test_non_monotonic(self):
        s = self.series()
        s[1], s[2] = s[2], s[1]
        assert codes(lambda: validate_series(s, symbol="BTC/USD", timeframe=Timeframe.H1)) == "NON_MONOTONIC"

    def test_wrong_symbol(self):
        s = self.series() + [make_candle(9, symbol="ETH/USD")]
        assert codes(lambda: validate_series(s, symbol="BTC/USD", timeframe=Timeframe.H1)) == "SYMBOL"

    def test_wrong_timeframe(self):
        s = self.series()
        assert codes(lambda: validate_series(s, symbol="BTC/USD", timeframe=Timeframe.M5)) == "TIMEFRAME"

    def test_incomplete_candle_when_closed_required(self):
        s = self.series() + [make_candle(5, closed=False)]
        assert codes(lambda: validate_series(s, symbol="BTC/USD", timeframe=Timeframe.H1)) == "INCOMPLETE"
        validate_series(s, symbol="BTC/USD", timeframe=Timeframe.H1, require_closed=False)

    def test_status_mix_never_silently_combined(self):
        s = self.series() + [make_candle(5, status=DataStatus.REAL, provider="mock")]
        assert codes(lambda: validate_series(s, symbol="BTC/USD", timeframe=Timeframe.H1)) == "STATUS_MIX"

    def test_mixed_provider(self):
        s = self.series() + [make_candle(5, provider="kraken")]
        assert codes(lambda: validate_series(s, symbol="BTC/USD", timeframe=Timeframe.H1)) == "PROVIDER"

    def test_gaps_counted_not_filled(self):
        s = [make_candle(0), make_candle(3), make_candle(4)]
        assert count_gaps(s) == 2


class TestFreshness:
    def test_fresh(self):
        c = make_candle(0)
        check_freshness(c, c.close_time + timedelta(minutes=5), timedelta(hours=1))

    def test_stale(self):
        c = make_candle(0)
        assert codes(lambda: check_freshness(c, c.close_time + timedelta(hours=3), timedelta(hours=1))) == "STALE"

    def test_future_candle(self):
        c = make_candle(0)
        assert codes(lambda: check_freshness(c, c.close_time - timedelta(minutes=1), timedelta(hours=1))) == "FUTURE"


class TestStore:
    def test_idempotent_reingest(self):
        store = CandleStore()
        assert store.ingest([make_candle(0), make_candle(1)]) == 2
        assert store.ingest([make_candle(0)]) == 0

    def test_historical_disagreement_fails_closed(self):
        store = CandleStore()
        store.ingest([make_candle(0)])
        with pytest.raises(HistoricalConflictError):
            store.ingest([make_candle(0, c="100.6")])
        # store unchanged
        assert store.series("mock", "BTC/USD", Timeframe.H1)[0].close == Decimal("100.5")

    def test_batch_is_atomic(self):
        store = CandleStore()
        store.ingest([make_candle(0)])
        with pytest.raises(HistoricalConflictError):
            store.ingest([make_candle(1), make_candle(0, c="100.6")])
        assert len(store.series("mock", "BTC/USD", Timeframe.H1)) == 1

    def test_open_candles_refused(self):
        with pytest.raises(DataIntegrityError):
            CandleStore().ingest([make_candle(0, closed=False)])

    def test_status_cannot_change_for_series(self):
        store = CandleStore()
        store.ingest([make_candle(0)])
        with pytest.raises(DataIntegrityError):
            store.ingest([make_candle(1, status=DataStatus.REAL)])


class TestMockProvider:
    def test_everything_labeled_mock_and_deterministic(self):
        clock = FixedClock(T0 + timedelta(hours=50))
        a = MockProvider(1, clock).fetch_candles("BTC/USD", Timeframe.H1, T0, T0 + timedelta(hours=40))
        b = MockProvider(1, clock).fetch_candles("BTC/USD", Timeframe.H1, T0, T0 + timedelta(hours=40))
        assert [x.content_key() for x in a] == [x.content_key() for x in b]
        assert all(c.status is DataStatus.MOCK and "seed=1" in c.provenance.method for c in a)

    def test_never_returns_future_and_marks_forming_candle_open(self):
        now = T0 + timedelta(hours=10, minutes=30)
        out = MockProvider(1, FixedClock(now)).fetch_candles("BTC/USD", Timeframe.H1, T0, T0 + timedelta(hours=40))
        assert len(out) == 11
        assert out[-1].is_closed is False and all(c.is_closed for c in out[:-1])
