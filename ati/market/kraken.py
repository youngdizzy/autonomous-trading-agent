"""Kraken public OHLC adapter.

STATUS: IMPLEMENTED — EXTERNAL VERIFICATION BLOCKED. The egress policy of the Foundation 1.0
environment denies ``api.kraken.com``. Parsing is contract-tested against a hand-constructed
fixture that follows Kraken's documented response shape; it has never parsed a real response.

Documented shape (GET /0/public/OHLC?pair=XBTUSD&interval=60&since=<unix>):
    {"error": [], "result": {"<PAIRNAME>": [[time, open, high, low, close, vwap, volume, count], ...],
                             "last": <unix>}}
Kraken returns the currently forming candle as the final row; it is therefore marked not closed.
Known limitation: the OHLC endpoint returns at most 720 recent bars; deep history needs a
different ingestion path (NOT IMPLEMENTED).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from ati.core.canonical import sha256_text
from ati.core.errors import DataIntegrityError, MalformedResponse, ProviderUnavailable, RateLimited
from ati.core.time import Clock, ensure_utc, from_epoch
from ati.market.models import Candle, DataStatus, Provenance, Timeframe
from ati.market.provider import HttpTransport

OHLC_URL = "https://api.kraken.com/0/public/OHLC"
PAIR_QUERY = {"BTC/USD": "XBTUSD", "ETH/USD": "ETHUSD"}
INTERVAL_MINUTES = {Timeframe.M1: 1, Timeframe.M5: 5, Timeframe.M15: 15, Timeframe.H1: 60, Timeframe.H4: 240, Timeframe.D1: 1440}


def _dec(value: Any, field: str) -> Decimal:
    if not isinstance(value, (str, int)):
        raise MalformedResponse(f"{field}: expected string/int, got {type(value).__name__}")
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise MalformedResponse(f"{field}: not a number: {value!r}") from exc


def parse_ohlc(payload: Any, raw: bytes, symbol: str, timeframe: Timeframe, received_at: datetime) -> list[Candle]:
    received_at = ensure_utc(received_at)
    if not isinstance(payload, dict) or "error" not in payload:
        raise MalformedResponse("response is not a Kraken envelope")
    errors = payload["error"]
    if not isinstance(errors, list):
        raise MalformedResponse("error field is not a list")
    if errors:
        text = "; ".join(map(str, errors))
        if "Rate limit" in text or "EAPI:Rate" in text:
            raise RateLimited(text)
        if "EService" in text:
            raise ProviderUnavailable(text)
        raise MalformedResponse(f"Kraken error: {text}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise MalformedResponse("missing result")
    pair_keys = [k for k in result if k != "last"]
    if len(pair_keys) != 1:
        raise MalformedResponse(f"expected exactly one pair in result, got {pair_keys}")
    rows = result[pair_keys[0]]
    if not isinstance(rows, list):
        raise MalformedResponse("OHLC rows is not a list")
    provenance = Provenance(
        source="kraken",
        method=f"GET {OHLC_URL} pair={PAIR_QUERY.get(symbol)} interval={INTERVAL_MINUTES[timeframe]}",
        retrieved_at=received_at,
        raw_sha256=sha256_text(raw.decode("utf-8", errors="replace")),
    )
    candles: list[Candle] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 8:
            raise MalformedResponse(f"row {index} malformed: {row!r}")
        if not isinstance(row[0], int):
            raise MalformedResponse(f"row {index} time is not an integer")
        open_time = from_epoch(row[0])
        is_last = index == len(rows) - 1
        closed = (not is_last) and (open_time + timeframe.delta <= received_at)
        try:
            candles.append(
                Candle(
                    provider="kraken",
                    symbol=symbol,
                    timeframe=timeframe,
                    open_time=open_time,
                    open=_dec(row[1], "open"),
                    high=_dec(row[2], "high"),
                    low=_dec(row[3], "low"),
                    close=_dec(row[4], "close"),
                    volume=_dec(row[6], "volume"),
                    is_closed=closed,
                    status=DataStatus.REAL,
                    received_at=received_at,
                    provenance=provenance,
                )
            )
        except DataIntegrityError:
            raise
    return candles


class KrakenPublicOHLC:
    name = "kraken"

    def __init__(self, transport: HttpTransport, clock: Clock, timeout_s: float = 10.0):
        self.transport = transport
        self.clock = clock
        self.timeout_s = timeout_s

    def fetch_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> list[Candle]:
        if symbol not in PAIR_QUERY:
            raise DataIntegrityError("SYMBOL", f"no Kraken mapping for {symbol}")
        if timeframe not in INTERVAL_MINUTES:
            raise DataIntegrityError("TIMEFRAME", f"Kraken does not serve {timeframe.value}")
        start, end = ensure_utc(start), ensure_utc(end)
        params = {"pair": PAIR_QUERY[symbol], "interval": str(INTERVAL_MINUTES[timeframe]), "since": str(int(start.timestamp()) - 1)}
        payload, raw = self.transport.get_json(OHLC_URL, params, self.timeout_s)
        candles = parse_ohlc(payload, raw, symbol, timeframe, self.clock.now())
        return [c for c in candles if start <= c.open_time < end]
