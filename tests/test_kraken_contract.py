"""Contract tests for the Kraken adapter against MOCK payloads.

These prove parsing, closed-candle semantics, failure mapping and provenance labelling only.
Live connectivity is BLOCKED in this environment and is NOT verified here. Because the fixture
arrives through a test transport that declares MOCK, every candle produced here is MOCK — the
adapter never labels data REAL on its own.
"""

import copy
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ati.core.errors import DataIntegrityError, MalformedResponse, ProviderUnavailable, RateLimited
from ati.core.time import UTC, FixedClock
from ati.market.kraken import KrakenPublicOHLC, parse_ohlc
from ati.market.models import DataStatus, Timeframe
from ati.market.provider import UrllibTransport

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "kraken_ohlc_MOCK.json"
RECEIVED = datetime(2024, 1, 1, 3, 30, tzinfo=UTC)
H = 3600


def load():
    raw = FIXTURE.read_bytes()
    return json.loads(raw), raw


def parse(payload, raw=b"{}", received=RECEIVED, status=DataStatus.MOCK, symbol="BTC/USD"):
    return parse_ohlc(payload, raw, symbol, Timeframe.H1, received, status)


def envelope(times, last, key="XXBTZUSD"):
    rows = [[t, "100.0", "101.0", "99.0", "100.5", "100.2", "1.0", 3] for t in times]
    return {"error": [], "result": {key: rows, "last": last}}


class FakeTransport:
    data_status = DataStatus.MOCK

    def __init__(self, payload, raw=None):
        self.payload, self.calls = payload, []
        self.raw = raw if raw is not None or isinstance(payload, Exception) else json.dumps(payload).encode()

    def get_json(self, url, params, timeout_s):
        self.calls.append((url, params))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload, self.raw


def test_fixture_is_labeled_mock():
    payload, _ = load()
    assert "MOCK" in payload["_MOCK_NOTICE"]


class TestShapeAndClosedSemantics:
    def test_parse_shape(self):
        payload, raw = load()
        candles = parse(payload, raw)
        assert len(candles) == 4
        assert all(c.status is DataStatus.MOCK and c.provider == "kraken" for c in candles)
        assert candles[0].provenance.raw_sha256 and len(candles[0].provenance.raw_sha256) == 64
        assert str(candles[1].close) == "42250.0" and str(candles[0].volume) == "12.50000000"
        assert [c.open_time for c in candles] == sorted(c.open_time for c in candles)

    def test_final_row_is_forming_even_if_old(self):
        """Kraken documents the final row as the uncommitted frame: never closed, whatever the clock says."""
        c = parse(envelope([0, H, 2 * H], last=2 * H), received=datetime(2030, 1, 1, tzinfo=UTC))
        assert [x.is_closed for x in c] == [True, True, False]

    def test_rows_after_last_committed_marker_are_open(self):
        c = parse(envelope([0, H, 2 * H, 3 * H], last=H), received=datetime(2030, 1, 1, tzinfo=UTC))
        assert [x.is_closed for x in c] == [True, True, False, False]

    def test_close_time_boundary_against_receipt(self):
        at_close = datetime.fromtimestamp(2 * H, tz=UTC)
        c = parse(envelope([0, H, 2 * H], last=2 * H), received=at_close)
        assert [x.is_closed for x in c] == [True, True, False]  # row H closes exactly at receipt: closed
        c = parse(envelope([0, H, 2 * H], last=2 * H), received=at_close - timedelta(seconds=1))
        assert [x.is_closed for x in c] == [True, False, False]  # one second earlier: still forming

    def test_forming_candle_is_excluded_from_datasets_and_history(self):
        from ati.data.dataset import Dataset
        from ati.market.store import CandleStore
        payload, raw = load()
        candles = parse(payload, raw)
        with pytest.raises(DataIntegrityError):
            Dataset.build(candles, data_version="v", realization="r")
        with pytest.raises(DataIntegrityError):
            CandleStore().ingest(candles)
        Dataset.build([c for c in candles if c.is_closed], data_version="v", realization="r")


class TestFailClosed:
    @pytest.mark.parametrize("mutate,exc", [
        (lambda p: p.update(error=["EAPI:Rate limit exceeded"]), RateLimited),
        (lambda p: p.update(error=["EService:Unavailable"]), ProviderUnavailable),
        (lambda p: p.update(error=["EGeneral:Invalid arguments"]), MalformedResponse),
        (lambda p: p.update(error="oops"), MalformedResponse),
        (lambda p: p.pop("result"), MalformedResponse),
        (lambda p: p["result"].pop("last"), MalformedResponse),
        (lambda p: p["result"].__setitem__("last", "1704074400"), MalformedResponse),
        (lambda p: p["result"]["XXBTZUSD"].append([1, 2]), MalformedResponse),
        (lambda p: p["result"]["XXBTZUSD"][0].__setitem__(1, "abc"), MalformedResponse),
        (lambda p: p["result"]["XXBTZUSD"][0].__setitem__(1, 42000.0), MalformedResponse),  # floats refused
        (lambda p: p["result"]["XXBTZUSD"][0].__setitem__(0, 1704067200.5), MalformedResponse),
        (lambda p: p["result"].__setitem__("XETHZUSD", []), MalformedResponse),             # two pairs
        (lambda p: p["result"].__setitem__("XETHZUSD", p["result"].pop("XXBTZUSD")), MalformedResponse),  # wrong pair
        (lambda p: p["result"]["XXBTZUSD"].insert(1, list(p["result"]["XXBTZUSD"][0])), MalformedResponse),  # duplicate
        (lambda p: p["result"]["XXBTZUSD"].reverse(), MalformedResponse),                   # non-monotonic
    ])
    def test_malformed_and_error_mapping(self, mutate, exc):
        payload, raw = load()
        mutate(payload)
        with pytest.raises(exc):
            parse(payload, raw)

    @pytest.mark.parametrize("field,value,code", [
        (2, "1.0", "OHLC"),            # high below open
        (3, "99999.0", "OHLC"),        # low above close
        (1, "0", "PRICE"),
        (1, "-5", "PRICE"),
        (6, "-1", "VOLUME"),
        (4, "NaN", "NONFINITE"),
        (2, "Infinity", "NONFINITE"),
    ])
    def test_candle_invariants_reject_provider_values(self, field, value, code):
        payload, raw = load()
        payload["result"]["XXBTZUSD"][0][field] = value
        with pytest.raises(DataIntegrityError) as err:
            parse(payload, raw)
        assert err.value.code == code

    def test_misaligned_time_rejected(self):
        with pytest.raises(DataIntegrityError):
            parse(envelope([0, H + 60, 2 * H], last=H))

    def test_wrong_interval_rejected(self):
        with pytest.raises(MalformedResponse):
            parse(envelope([i * 4 * H for i in range(12)], last=40 * H))  # 4h rows for a 1h request

    def test_non_envelope_rejected(self):
        for bad in (None, [], "x", {"result": {}}):
            with pytest.raises(MalformedResponse):
                parse(bad)

    def test_invalid_utf8_rejected(self):
        payload, _ = load()
        with pytest.raises(MalformedResponse):
            parse(payload, b"\xff\xfe")


class TestAdapter:
    def test_uses_transport_and_filters_range(self):
        payload, raw = load()
        t = FakeTransport(payload, raw)
        adapter = KrakenPublicOHLC(t, FixedClock(RECEIVED))
        start = datetime(2024, 1, 1, 1, tzinfo=UTC)
        out = adapter.fetch_candles("BTC/USD", Timeframe.H1, start, start + timedelta(hours=2))
        assert [c.open_time.hour for c in out] == [1, 2]
        assert t.calls[0][1]["pair"] == "XBTUSD" and t.calls[0][1]["interval"] == "60"

    def test_status_comes_from_transport_not_adapter(self):
        payload, raw = load()
        out = KrakenPublicOHLC(FakeTransport(payload, raw), FixedClock(RECEIVED)).fetch_candles(
            "BTC/USD", Timeframe.H1, RECEIVED - timedelta(hours=4), RECEIVED)
        assert out and all(c.status is DataStatus.MOCK for c in out)

    def test_transport_without_declared_status_is_unknown(self):
        class Anonymous:
            def get_json(self, url, params, timeout_s):
                return load()
        out = KrakenPublicOHLC(Anonymous(), FixedClock(RECEIVED)).fetch_candles(
            "BTC/USD", Timeframe.H1, RECEIVED - timedelta(hours=4), RECEIVED)
        assert all(c.status is DataStatus.UNKNOWN for c in out)

    def test_only_the_network_transport_may_declare_real(self):
        assert UrllibTransport.data_status is DataStatus.REAL
        uses = {}
        for p in (ROOT / "ati").rglob("*.py"):
            lines = [ln.strip() for ln in p.read_text().splitlines() if re.search(r"DataStatus\.REAL\b", ln)]
            if lines:
                uses[str(p.relative_to(ROOT))] = lines
        assert set(uses) == {"ati/market/models.py", "ati/market/provider.py"}
        assert all(ln.startswith("MARKET_EVIDENCE_STATUSES") for ln in uses["ati/market/models.py"])  # definition only
        assert all(ln.startswith("data_status = DataStatus.REAL") for ln in uses["ati/market/provider.py"])

    def test_outage_and_timeout_propagate_never_substitute(self):
        for exc in (ProviderUnavailable("down"), ProviderUnavailable("timed out")):
            adapter = KrakenPublicOHLC(FakeTransport(exc), FixedClock(RECEIVED))
            with pytest.raises(ProviderUnavailable):
                adapter.fetch_candles("BTC/USD", Timeframe.H1, RECEIVED - timedelta(hours=3), RECEIVED)

    def test_real_transport_timeout_maps_to_provider_unavailable(self, monkeypatch):
        import urllib.request

        def boom(*a, **k):
            raise TimeoutError("timed out")
        monkeypatch.setattr(urllib.request, "urlopen", boom)
        with pytest.raises(ProviderUnavailable):
            UrllibTransport().get_json("https://api.kraken.com/0/public/Time", {}, 1.0)

    def test_unknown_symbol_and_timeframe_rejected(self):
        adapter = KrakenPublicOHLC(FakeTransport({}), FixedClock(RECEIVED))
        with pytest.raises(DataIntegrityError):
            adapter.fetch_candles("DOGE/USD", Timeframe.H1, RECEIVED, RECEIVED)
