"""Kraken public OHLC adapter.

STATUS: IMPLEMENTED — EXTERNAL VERIFICATION BLOCKED. The egress policy of this environment denies
``api.kraken.com`` (proxy 403 on CONNECT, re-confirmed 2026-09-28). Parsing is contract-tested only
against hand-constructed MOCK payloads that follow Kraken's documented response shape; it has
never parsed a real response.

Documented shape (GET /0/public/OHLC?pair=XBTUSD&interval=60&since=<unix>):
    {"error": [], "result": {"<PAIRNAME>": [[time, open, high, low, close, vwap, volume, count], ...],
                             "last": <unix>}}

Closed-candle rule (conservative; every condition must hold):
  1. the row is not the final row — Kraken documents the final row as the current, not yet
     committed frame;
  2. the row's open time is <= ``result.last`` — ``last`` marks committed data;
  3. the row's close time is <= the local receipt time.
A row failing any condition is kept as an open (forming) candle, which research and datasets refuse.

Response checks (fail closed, never repaired): error envelope, exactly one pair key matching the
requested pair, integer times strictly increasing (no duplicates, no reordering), numeric strings
for prices and volume, candle invariants, and — for responses with at least 10 rows — at least one
step equal to the requested interval.

Provenance: candles carry the *transport's* ``data_status`` (only the real network transport
declares REAL) and the SHA-256 of the raw response text. When an archive is attached, the raw
payload is journaled before parsing (write-ahead), so a REAL dataset can later be re-derived from
the exact bytes received.

Known limitation: the OHLC endpoint returns at most 720 recent bars; deep history needs a different
ingestion path (NOT IMPLEMENTED).
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
# Result keys Kraken is documented to use for these pairs (legacy "X...Z..." name or the altname).
PAIR_RESULT_KEYS = {"BTC/USD": frozenset({"XXBTZUSD", "XBTUSD"}), "ETH/USD": frozenset({"XETHZUSD", "ETHUSD"})}
INTERVAL_MINUTES = {Timeframe.M1: 1, Timeframe.M5: 5, Timeframe.M15: 15, Timeframe.H1: 60, Timeframe.H4: 240, Timeframe.D1: 1440}


def _dec(value: Any, field: str) -> Decimal:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise MalformedResponse(f"{field}: expected numeric string, got {type(value).__name__}")
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise MalformedResponse(f"{field}: not a number") from exc


def raw_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MalformedResponse("response is not valid UTF-8") from exc


def parse_ohlc(payload: Any, raw: bytes, symbol: str, timeframe: Timeframe, received_at: datetime,
               status: DataStatus) -> list[Candle]:
    received_at = ensure_utc(received_at)
    if not isinstance(status, DataStatus):
        raise TypeError("status must be the transport's DataStatus")
    if not isinstance(payload, dict) or "error" not in payload:
        raise MalformedResponse("response is not a Kraken envelope")
    errors = payload["error"]
    if not isinstance(errors, list):
        raise MalformedResponse("error field is not a list")
    if errors:
        text = "; ".join(map(str, errors))[:300]
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
        raise MalformedResponse(f"expected exactly one pair in result, got {len(pair_keys)}")
    if pair_keys[0] not in PAIR_RESULT_KEYS.get(symbol, frozenset()):
        raise MalformedResponse(f"response pair does not match requested {symbol}")
    last = result.get("last")
    if not isinstance(last, int) or isinstance(last, bool):
        raise MalformedResponse("result.last missing or not an integer")
    rows = result[pair_keys[0]]
    if not isinstance(rows, list):
        raise MalformedResponse("OHLC rows is not a list")
    times: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 8:
            raise MalformedResponse(f"row {index} malformed")
        if not isinstance(row[0], int) or isinstance(row[0], bool):
            raise MalformedResponse(f"row {index} time is not an integer")
        if times and row[0] <= times[-1]:
            raise MalformedResponse(f"row {index} time not strictly increasing (duplicate or reordered)")
        times.append(row[0])
    step = timeframe.seconds
    if len(times) >= 10 and not any(b - a == step for a, b in zip(times, times[1:])):
        raise MalformedResponse(f"no consecutive rows {step}s apart: response is not {timeframe.value} data")
    provenance = Provenance(
        source="kraken",
        method=f"GET {OHLC_URL} pair={PAIR_QUERY.get(symbol)} interval={INTERVAL_MINUTES[timeframe]}",
        retrieved_at=received_at,
        raw_sha256=sha256_text(raw_text(raw)),
    )
    candles: list[Candle] = []
    for index, row in enumerate(rows):
        open_time = from_epoch(row[0])
        closed = (index < len(rows) - 1) and row[0] <= last and (open_time + timeframe.delta <= received_at)
        candles.append(
            Candle(
                provider="kraken", symbol=symbol, timeframe=timeframe, open_time=open_time,
                open=_dec(row[1], "open"), high=_dec(row[2], "high"), low=_dec(row[3], "low"),
                close=_dec(row[4], "close"), volume=_dec(row[6], "volume"),
                is_closed=closed, status=status, received_at=received_at, provenance=provenance,
            )
        )
    return candles


class KrakenPublicOHLC:
    name = "kraken"

    def __init__(self, transport: HttpTransport, clock: Clock, timeout_s: float = 10.0, archive=None):
        self.transport = transport
        self.clock = clock
        self.timeout_s = timeout_s
        self.archive = archive  # ati.market.archive.PayloadArchive (optional)

    @property
    def data_status(self) -> DataStatus:
        status = getattr(self.transport, "data_status", DataStatus.UNKNOWN)
        return status if isinstance(status, DataStatus) else DataStatus.UNKNOWN

    def fetch_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> list[Candle]:
        if symbol not in PAIR_QUERY:
            raise DataIntegrityError("SYMBOL", f"no Kraken mapping for {symbol}")
        if timeframe not in INTERVAL_MINUTES:
            raise DataIntegrityError("TIMEFRAME", f"Kraken does not serve {timeframe.value}")
        start, end = ensure_utc(start), ensure_utc(end)
        params = {"pair": PAIR_QUERY[symbol], "interval": str(INTERVAL_MINUTES[timeframe]), "since": str(int(start.timestamp()) - 1)}
        payload, raw = self.transport.get_json(OHLC_URL, params, self.timeout_s)
        received_at = self.clock.now()
        status = self.data_status
        digest = None
        if self.archive is not None:
            digest = self.archive.record(provider=self.name, symbol=symbol, timeframe=timeframe, url=OHLC_URL,
                                         params=params, received_at=received_at, raw=raw, status=status)
        try:
            candles = parse_ohlc(payload, raw, symbol, timeframe, received_at, status)
        except (MalformedResponse, DataIntegrityError, RateLimited, ProviderUnavailable) as exc:
            if self.archive is not None and digest is not None:
                self.archive.reject(digest, f"{type(exc).__name__}: {exc}")
            raise
        return [c for c in candles if start <= c.open_time < end]
