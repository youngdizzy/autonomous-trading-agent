"""Provider abstraction. The rest of the system depends on this protocol, never on a vendor API."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Protocol

from ati.core.errors import MalformedResponse, ProviderUnavailable, RateLimited
from ati.market.models import Candle, DataStatus, Timeframe


class MarketDataProvider(Protocol):
    name: str

    def fetch_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> list[Candle]:
        """Return candles with ``start <= open_time < end``. Must label every candle's status and
        provenance truthfully and raise ``ProviderError`` subclasses on failure — never return
        substitute data."""
        ...


class HttpTransport(Protocol):
    #: The data category of bytes this transport delivers. Provenance is owned here, by the
    #: ingestion layer: an adapter labels candles with its transport's status, never with a
    #: constant. Only a transport that actually performs network I/O to the provider may say REAL.
    data_status: DataStatus

    def get_json(self, url: str, params: dict[str, str], timeout_s: float) -> tuple[Any, bytes]:
        """Return (parsed JSON, raw bytes)."""
        ...


class UrllibTransport:
    """Real HTTPS transport (stdlib). Honors HTTPS_PROXY via urllib's environment handling and
    the system trust store; TLS verification is never disabled.
    IMPLEMENTED — EXTERNAL VERIFICATION BLOCKED (api.kraken.com denied by the environment's egress
    policy as of 2026-09-28)."""

    data_status = DataStatus.REAL  # the only transport in the codebase allowed to declare REAL

    def get_json(self, url: str, params: dict[str, str], timeout_s: float) -> tuple[Any, bytes]:
        full = f"{url}?{urllib.parse.urlencode(params)}" if params else url
        request = urllib.request.Request(full, headers={"User-Agent": "ati/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 (https only)
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry = exc.headers.get("Retry-After") if exc.headers else None
                raise RateLimited(f"HTTP 429 from {url}", float(retry) if retry and retry.isdigit() else None) from exc
            raise ProviderUnavailable(f"HTTP {exc.code} from {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderUnavailable(f"transport failure for {url}: {exc}") from exc
        try:
            return json.loads(raw), raw
        except (ValueError, UnicodeDecodeError) as exc:
            raise MalformedResponse(f"non-JSON response from {url}") from exc
