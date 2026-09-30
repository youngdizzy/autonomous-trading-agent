"""Multi-source data accuracy: DATA_CONFLICT.

When two independent sources report the same candle (provider, symbol, timeframe, open_time) and their
values disagree beyond tolerance, neither is trusted. The conflict is journaled (``conflicts.jsonl``,
existing Journal class) and stays open until an operator records which source is authoritative; while
any conflict is open the health gate reports INVALID_DATA (DATA_CONFLICT) and research and trading are
blocked. Nothing here averages, picks a "majority" or silently prefers a source.

Only one market source is wired in this build (Kraken, BLOCKED from this environment), so the comparator
is exercised on supplied series; a second live source is NOT IMPLEMENTED.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ati.core.canonical import sha256_hex
from ati.core.errors import DataConflictError
from ati.ledger.journal import Journal, decode
from ati.market.models import Candle

PRICE_TOLERANCE = Decimal("0.001")      # relative, per OHLC field
RESOLVE_ACK = "OPERATOR: sources compared manually; authoritative source recorded"


@dataclass(frozen=True)
class Conflict:
    conflict_id: str
    symbol: str
    timeframe: str
    open_time: str
    sources: tuple[str, str]
    fields: tuple[tuple[str, str, str], ...]   # (field, value_a, value_b)


def compare(a: list[Candle], b: list[Candle], tolerance: Decimal = PRICE_TOLERANCE) -> list[Conflict]:
    """Overlapping candles of two series from *different* sources; returns every disagreement."""
    if a and b and a[0].provider == b[0].provider:
        raise DataConflictError("a source cannot corroborate itself: compare two different providers")
    index = {(c.symbol, c.timeframe, c.open_time): c for c in b}
    out = []
    for ca in a:
        cb = index.get((ca.symbol, ca.timeframe, ca.open_time))
        if cb is None or not (ca.is_closed and cb.is_closed):
            continue
        diffs = []
        for f in ("open", "high", "low", "close"):
            va, vb = getattr(ca, f), getattr(cb, f)
            ref = max(abs(va), abs(vb))
            if ref and abs(va - vb) / ref > tolerance:
                diffs.append((f, str(va), str(vb)))
        if diffs:
            body = {"sym": ca.symbol, "tf": ca.timeframe.value, "t": ca.open_time, "src": [ca.provider, cb.provider],
                    "d": diffs}
            out.append(Conflict("dc_" + sha256_hex(body)[:20], ca.symbol, ca.timeframe.value, ca.open_time.isoformat(),
                                (ca.provider, cb.provider), tuple(diffs)))
    return out


class ConflictRegister:
    def __init__(self, journal: Journal):
        self.journal = journal
        self.open: dict[str, dict] = {}
        self.resolved: dict[str, dict] = {}
        for e in journal.entries():
            p = decode(e.payload)
            if e.type == "data_conflict":
                if p["conflict_id"] not in self.resolved:
                    self.open[p["conflict_id"]] = p
            elif e.type == "conflict_resolved":
                self.resolved[p["conflict_id"]] = self.open.pop(p["conflict_id"])

    def record(self, conflicts: list[Conflict]) -> int:
        new = 0
        for c in conflicts:
            if c.conflict_id in self.open or c.conflict_id in self.resolved:
                continue
            payload = {"conflict_id": c.conflict_id, "symbol": c.symbol, "timeframe": c.timeframe,
                       "open_time": c.open_time, "sources": list(c.sources), "fields": [list(f) for f in c.fields]}
            self.journal.append("data_conflict", payload)
            self.open[c.conflict_id] = payload
            new += 1
        return new

    def resolve(self, conflict_id: str, authoritative_source: str, acknowledgement: str) -> None:
        """Operator only. Claude's vocabulary has no action that reaches here."""
        if acknowledgement != RESOLVE_ACK:
            raise PermissionError("resolving a data conflict requires the exact operator acknowledgement")
        c = self.open.get(conflict_id)
        if c is None:
            raise DataConflictError(f"no open conflict {conflict_id}")
        if authoritative_source not in c["sources"]:
            raise DataConflictError(f"{authoritative_source!r} is not one of the conflicting sources {c['sources']}")
        self.journal.append("conflict_resolved", {"conflict_id": conflict_id, "authoritative": authoritative_source,
                                                  "source": "operator"})
        self.resolved[conflict_id] = self.open.pop(conflict_id)
