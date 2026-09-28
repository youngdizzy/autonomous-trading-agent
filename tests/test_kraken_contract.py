"""Contract tests for the Kraken adapter against a MOCK fixture. These prove parsing and failure
mapping only. Live connectivity is BLOCKED in this environment and is NOT verified here."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ati.core.errors import DataIntegrityError, MalformedResponse, ProviderUnavailable, RateLimited
from ati.core.time import UTC, FixedClock
from ati.market.kraken import KrakenPublicOHLC, parse_ohlc
from ati.market.models import DataStatus, Timeframe

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "kraken_ohlc_MOCK.json"
RECEIVED = datetime(2024, 1, 1, 3, 30, tzinfo=UTC)


def load():
    raw = FIXTURE.read_bytes()
    return json.loads(raw), raw


class FakeTransport:
    def __init__(self, payload, raw=b"{}"):
        self.payload, self.raw, self.calls = payload, raw, []

    def get_json(self, url, params, timeout_s):
        self.calls.append((url, params))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload, self.raw


def test_fixture_is_labeled_mock():
    payload, _ = load()
    assert "MOCK" in payload["_MOCK_NOTICE"]


def test_parse_shape_and_forming_candle():
    payload, raw = load()
    candles = parse_ohlc(payload, raw, "BTC/USD", Timeframe.H1, RECEIVED)
    assert len(candles) == 4
    assert [c.is_closed for c in candles] == [True, True, True, False]
    assert all(c.status is DataStatus.REAL and c.provider == "kraken" for c in candles)
    assert candles[0].provenance.raw_sha256 and len(candles[0].provenance.raw_sha256) == 64
    assert str(candles[1].close) == "42250.0"
    assert str(candles[0].volume) == "12.50000000"


@pytest.mark.parametrize("mutate,exc", [
    (lambda p: p.update(error=["EAPI:Rate limit exceeded"]), RateLimited),
    (lambda p: p.update(error=["EService:Unavailable"]), ProviderUnavailable),
    (lambda p: p.update(error=["EGeneral:Invalid arguments"]), MalformedResponse),
    (lambda p: p.pop("result"), MalformedResponse),
    (lambda p: p["result"]["XXBTZUSD"].append([1, 2]), MalformedResponse),
    (lambda p: p["result"]["XXBTZUSD"][0].__setitem__(1, "abc"), MalformedResponse),
    (lambda p: p["result"]["XXBTZUSD"][0].__setitem__(1, 42000.0), MalformedResponse),  # floats refused
    (lambda p: p["result"].__setitem__("XETHZUSD", []), MalformedResponse),
])
def test_malformed_and_error_mapping(mutate, exc):
    payload, raw = load()
    mutate(payload)
    with pytest.raises(exc):
        parse_ohlc(payload, raw, "BTC/USD", Timeframe.H1, RECEIVED)


def test_impossible_prices_rejected_by_candle_invariants():
    payload, raw = load()
    payload["result"]["XXBTZUSD"][0][2] = "1.0"  # high below open
    with pytest.raises(DataIntegrityError):
        parse_ohlc(payload, raw, "BTC/USD", Timeframe.H1, RECEIVED)


def test_non_envelope_rejected():
    for bad in (None, [], "x", {"result": {}}):
        with pytest.raises(MalformedResponse):
            parse_ohlc(bad, b"", "BTC/USD", Timeframe.H1, RECEIVED)


def test_adapter_uses_transport_and_filters_range():
    payload, raw = load()
    t = FakeTransport(payload, raw)
    adapter = KrakenPublicOHLC(t, FixedClock(RECEIVED))
    start = datetime(2024, 1, 1, 1, tzinfo=UTC)
    out = adapter.fetch_candles("BTC/USD", Timeframe.H1, start, start + timedelta(hours=2))
    assert [c.open_time.hour for c in out] == [1, 2]
    assert t.calls[0][1]["pair"] == "XBTUSD" and t.calls[0][1]["interval"] == "60"


def test_outage_propagates_never_substitutes_data():
    adapter = KrakenPublicOHLC(FakeTransport(ProviderUnavailable("down")), FixedClock(RECEIVED))
    with pytest.raises(ProviderUnavailable):
        adapter.fetch_candles("BTC/USD", Timeframe.H1, RECEIVED - timedelta(hours=3), RECEIVED)


def test_unknown_symbol_rejected():
    adapter = KrakenPublicOHLC(FakeTransport({}), FixedClock(RECEIVED))
    with pytest.raises(DataIntegrityError):
        adapter.fetch_candles("DOGE/USD", Timeframe.H1, RECEIVED, RECEIVED)
