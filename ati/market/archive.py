"""Raw provider payload archive — where market provenance is anchored.

Every response an attached adapter receives is appended, *before parsing*, to the system's existing
evidence journal (hash-chained, mode/data-status bound) as a ``provider_payload`` entry holding the
exact response text, request, receipt time, and the transport's data status. Responses that fail
parsing are additionally marked ``provider_payload_rejected``.

Two uses:
- ``replay`` rebuilds the candle store after a restart by re-parsing archived payloads — accumulated
  history is re-derived from what the provider actually said, not from a derived cache.
- ``verify_market_provenance`` is the binding check for REAL/HISTORICAL/DELAYED datasets: every
  candle must be re-derivable, value for value, from an archived payload that was received through
  a transport declaring that same status. Copying a fixture, editing a JSON file's ``status``, or
  constructing candles in code does not produce such a payload, so it cannot produce REAL.

Honest limit: the anchor is the local hash chain. Someone able to rewrite the whole journal file
can forge it; committing/pushing journals externally anchors them.
"""

from __future__ import annotations

from datetime import datetime

from ati.core.canonical import sha256_text
from ati.core.errors import HistoricalConflictError, MalformedResponse, ProvenanceError
from ati.core.time import ensure_utc
from ati.ledger.journal import Journal, decode
from ati.market.models import MARKET_EVIDENCE_STATUSES, DataStatus, Timeframe


CONFLICT_ACK = "OPERATOR: historical conflict reviewed; recorded history kept unchanged"


def _parser(provider: str):
    if provider == "kraken":
        from ati.market.kraken import parse_ohlc

        return parse_ohlc
    raise ProvenanceError(f"no archived-payload parser for provider {provider!r}")


class PayloadArchive:
    """Raw bytes are stored once per content hash (``provider_payload``); every *receipt* of those
    bytes is journaled (``provider_payload_receipt`` for repeats), because a candle's closed status
    depends on when it was received. Candles are re-derived from the exact receipt they came from."""

    def __init__(self, journal: Journal):
        self.journal = journal
        self._payloads: dict[str, dict] = {}                     # digest → first full payload entry
        self._receipts: dict[tuple[str, datetime], dict] = {}    # (digest, received_at) → receipt, journal order
        self._rejected: set[str] = set()
        # Historical conflicts are durable state: a payload that contradicted recorded history blocks its
        # series until an operator acknowledges it. Recorded history is never overwritten either way.
        self._conflicts: dict[str, dict] = {}          # digest → {series, reason}
        self._resolved: set[str] = set()
        for entry in journal.entries():
            if entry.type in ("provider_payload", "provider_payload_receipt"):
                p = decode(entry.payload)
                if entry.type == "provider_payload":
                    self._payloads.setdefault(p["raw_sha256"], p)
                self._receipts.setdefault((p["raw_sha256"], p["received_at"]), p)
            elif entry.type == "provider_payload_rejected":
                self._rejected.add(entry.payload["raw_sha256"])
                if entry.payload["reason"].startswith("historical conflict"):
                    self._conflicts[entry.payload["raw_sha256"]] = {"reason": entry.payload["reason"]}
            elif entry.type == "provider_conflict_acknowledged":
                self._resolved.add(entry.payload["raw_sha256"])

    def record(self, *, provider: str, symbol: str, timeframe: Timeframe, url: str, params: dict[str, str],
               received_at: datetime, raw: bytes, status: DataStatus) -> str:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MalformedResponse("response is not valid UTF-8") from exc
        digest = sha256_text(text)
        received_at = ensure_utc(received_at)
        if (digest, received_at) in self._receipts:
            return digest
        receipt = {"provider": provider, "symbol": symbol, "timeframe": timeframe.value, "url": url,
                   "params": dict(params), "received_at": received_at, "status": status.value, "raw_sha256": digest}
        if digest not in self._payloads:
            written = self.journal.append("provider_payload", receipt | {"raw": text})
            self._payloads[digest] = decode(written.payload)
        else:
            written = self.journal.append("provider_payload_receipt", receipt)
        self._receipts[(digest, received_at)] = decode(written.payload)
        return digest

    def reject(self, digest: str, reason: str) -> None:
        if digest not in self._rejected:
            self.journal.append("provider_payload_rejected", {"raw_sha256": digest, "reason": reason[:300]})
            self._rejected.add(digest)

    def conflicts(self, open_only: bool = True) -> list[dict]:
        """Historical conflicts (payloads that disagreed with recorded history), with their series."""
        out = []
        for digest, c in self._conflicts.items():
            if open_only and digest in self._resolved:
                continue
            p = self._payloads.get(digest, {})
            out.append({"raw_sha256": digest, "provider": p.get("provider"), "symbol": p.get("symbol"),
                        "timeframe": p.get("timeframe"), "reason": c["reason"], "acknowledged": digest in self._resolved})
        return out

    def series_conflicted(self, provider: str, symbol: str, timeframe: Timeframe) -> bool:
        return any((c["provider"], c["symbol"], c["timeframe"]) == (provider, symbol, timeframe.value)
                   for c in self.conflicts())

    def acknowledge_conflict(self, digest: str, acknowledgement: str) -> None:
        """Operator only. Unblocks the series; the disagreeing payload stays archived AND rejected, and the
        recorded history stays exactly as it was (never replaced by the disagreeing values)."""
        if acknowledgement != CONFLICT_ACK:
            raise PermissionError("acknowledging a historical conflict requires the exact operator phrase")
        if digest not in self._conflicts:
            raise ProvenanceError(f"no historical conflict for payload {digest[:12]}")
        if digest not in self._resolved:
            self.journal.append("provider_conflict_acknowledged", {"raw_sha256": digest, "source": "operator"})
            self._resolved.add(digest)

    def receipts_for(self, provider: str, symbol: str, timeframe: Timeframe) -> list[dict]:
        return [r for r in self._receipts.values()
                if (r["provider"], r["symbol"], r["timeframe"]) == (provider, symbol, timeframe.value)]

    def __len__(self) -> int:
        return len(self._payloads)

    @property
    def receipts(self) -> int:
        return len(self._receipts)

    def _derive(self, receipt: dict) -> list:
        import json

        text = self._payloads[receipt["raw_sha256"]]["raw"]
        parse = _parser(receipt["provider"])
        return parse(json.loads(text), text.encode("utf-8"), receipt["symbol"], Timeframe(receipt["timeframe"]),
                     receipt["received_at"], DataStatus(receipt["status"]))

    def replay(self, store) -> int:
        """Re-derive closed candles from every accepted receipt, in journal order. A historical
        disagreement between payloads raises (fail closed)."""
        added = 0
        for (digest, _), receipt in self._receipts.items():
            if digest not in self._rejected:
                added += store.ingest(c for c in self._derive(receipt) if c.is_closed)
        return added

    def ingest(self, store, candles) -> int:
        """Store closed candles. If they contradict recorded history, the payload they came from is
        marked rejected (it stays archived as evidence of the disagreement) and the error propagates."""
        batch = [c for c in candles if c.is_closed]
        try:
            return store.ingest(batch)
        except HistoricalConflictError as exc:
            for digest in {c.provenance.raw_sha256 for c in batch if c.provenance.raw_sha256}:
                if digest in self._payloads:
                    self.reject(digest, f"historical conflict: {exc}")
                    self._conflicts.setdefault(digest, {"reason": f"historical conflict: {exc}"[:300]})
            raise

    def verify_market_provenance(self, dataset) -> None:
        """Binding check for market-evidence datasets (no-op for other categories)."""
        if dataset.identity.status in MARKET_EVIDENCE_STATUSES:
            self.verify_derivation(dataset)

    def verify_derivation(self, dataset) -> None:
        """Every candle must be re-derivable, value for value, from an accepted archived payload
        received through a transport declaring the dataset's own data status."""
        status = dataset.identity.status
        derived: dict[tuple[str, datetime], dict] = {}
        for candle in dataset.candles:
            key = (candle.provenance.raw_sha256, candle.received_at)
            receipt = self._receipts.get(key)
            if receipt is None or key[0] in self._rejected:
                raise ProvenanceError(f"{status.value} candle {candle.open_time} has no accepted archived receipt")
            if receipt["status"] != status.value or receipt["provider"] != candle.provider \
                    or receipt["symbol"] != candle.symbol or receipt["timeframe"] != candle.timeframe.value:
                raise ProvenanceError(f"archived payload {key[0][:12]} does not support {status.value} "
                                      f"{candle.provider}/{candle.symbol}/{candle.timeframe.value}")
            if key not in derived:
                derived[key] = {c.open_time: c for c in self._derive(receipt)}
            source = derived[key].get(candle.open_time)
            if source is None or source.content_key() != candle.content_key() or not source.is_closed:
                raise ProvenanceError(f"candle {candle.open_time} differs from its archived provider payload")
