"""Datasets with deterministic, content-derived identity.

``content_sha256`` is computed from market content only, so the same history always hashes the
same and any change to any value changes it. ``dataset_id`` additionally binds provider, symbol,
timeframe, data status, version, realization, temporal boundary and partition.

A ``Dataset`` never hands out a mutable reference to its candles and can only be read point-in-time
through ``view_at`` (see ``ati.temporal``) or in full through ``candles`` for research code that is
itself bound by the backtester's temporal discipline.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Sequence

from ati.core.canonical import canonical_json, sha256_hex
from ati.core.errors import DatasetIntegrityError, HistoricalConflictError, HoldoutViolation, ProvenanceError
from ati.core.time import ensure_utc, parse_utc, to_iso
from ati.market.models import MARKET_EVIDENCE_STATUSES, Candle, DataStatus, Provenance, Timeframe
from ati.market.validation import count_gaps, validate_series


class Partition(str, Enum):
    FULL = "FULL"
    DEVELOPMENT = "DEVELOPMENT"
    HOLDOUT = "HOLDOUT"


@dataclass(frozen=True)
class DatasetIdentity:
    provider: str
    symbol: str
    timeframe: Timeframe
    status: DataStatus
    start: datetime          # first candle open_time
    end: datetime            # last candle close_time
    n_candles: int
    gaps: int
    content_sha256: str
    data_version: str
    realization: str
    temporal_boundary: datetime  # every candle's values were final at or before this instant
    partition: Partition

    @property
    def dataset_id(self) -> str:
        return "ds_" + sha256_hex(self)[:24]


def content_hash(candles: Sequence[Candle]) -> str:
    head = candles[0]
    return sha256_hex(
        {
            "provider": head.provider,
            "symbol": head.symbol,
            "timeframe": head.timeframe,
            "status": head.status,
            "rows": [c.content_key() for c in candles],
        }
    )


class Dataset:
    __slots__ = ("_candles", "_close_times", "identity")

    def __init__(self, candles: tuple[Candle, ...], identity: DatasetIdentity):
        self._candles = candles
        self._close_times = [c.close_time for c in candles]
        self.identity = identity

    @classmethod
    def build(
        cls,
        candles: Sequence[Candle],
        *,
        data_version: str,
        realization: str,
        partition: Partition = Partition.FULL,
    ) -> "Dataset":
        candles = tuple(candles)
        head = candles[0] if candles else None
        if head is None:
            raise DatasetIntegrityError("dataset cannot be empty")
        validate_series(candles, symbol=head.symbol, timeframe=head.timeframe, require_closed=True)
        identity = DatasetIdentity(
            provider=head.provider,
            symbol=head.symbol,
            timeframe=head.timeframe,
            status=head.status,
            start=head.open_time,
            end=candles[-1].close_time,
            n_candles=len(candles),
            gaps=count_gaps(candles),
            content_sha256=content_hash(candles),
            data_version=data_version,
            realization=realization,
            temporal_boundary=candles[-1].close_time,
            partition=partition,
        )
        return cls(candles, identity)

    # --- identity -------------------------------------------------------------------------
    @property
    def dataset_id(self) -> str:
        return self.identity.dataset_id

    @property
    def partition(self) -> Partition:
        return self.identity.partition

    def verify(self) -> None:
        """Recompute content identity. Detects any in-memory or on-disk mutation."""
        if len(self._candles) != self.identity.n_candles or content_hash(self._candles) != self.identity.content_sha256:
            raise DatasetIntegrityError(f"dataset {self.dataset_id} content no longer matches its identity")

    # --- access ---------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._candles)

    @property
    def candles(self) -> tuple[Candle, ...]:
        return self._candles

    def index_at(self, cutoff: datetime) -> int:
        """Number of candles whose close_time <= cutoff."""
        return bisect.bisect_right(self._close_times, ensure_utc(cutoff))

    def view_at(self, cutoff: datetime, lookback: int | None = None):
        from ati.temporal.pit import PointInTimeView

        end = self.index_at(cutoff)
        start = 0 if lookback is None else max(0, end - lookback)
        return PointInTimeView(self._candles[start:end], cutoff, self.dataset_id)

    def require_not_holdout(self, purpose: str) -> None:
        """Refuse HOLDOUT partitions *and* any dataset whose bars overlap a sealed holdout period,
        so keeping a reference to the original FULL dataset does not open a side door."""
        if self.partition is Partition.HOLDOUT:
            raise HoldoutViolation(f"{purpose} may not use HOLDOUT dataset {self.dataset_id}")
        sealed = overlapping_sealed_range(self.identity)
        if sealed is not None:
            raise HoldoutViolation(f"{purpose}: dataset {self.dataset_id} overlaps sealed holdout period {sealed}")

    def split(self, boundary: datetime) -> tuple["Dataset", "Dataset"]:
        """Split into (DEVELOPMENT: close_time <= boundary, HOLDOUT: open_time >= boundary).
        Only ``HoldoutVault`` should call this."""
        boundary = ensure_utc(boundary)
        dev = [c for c in self._candles if c.close_time <= boundary]
        hold = [c for c in self._candles if c.open_time >= boundary]
        if not dev or not hold:
            raise DatasetIntegrityError("split boundary leaves an empty partition")
        version = self.identity.data_version
        real = self.identity.realization
        return (
            Dataset.build(dev, data_version=version, realization=real, partition=Partition.DEVELOPMENT),
            Dataset.build(hold, data_version=version, realization=real, partition=Partition.HOLDOUT),
        )

    # --- persistence ----------------------------------------------------------------------
    def save(self, path: Path) -> None:
        doc = {
            "identity": json.loads(canonical_json(self.identity)),
            "dataset_id": self.dataset_id,
            "candles": [
                {
                    "open_time": to_iso(c.open_time),
                    "o": str(c.open), "h": str(c.high), "l": str(c.low), "c": str(c.close), "v": str(c.volume),
                    "received_at": to_iso(c.received_at),
                    "provenance": json.loads(canonical_json(c.provenance)),
                }
                for c in self._candles
            ],
        }
        Path(path).write_text(json.dumps(doc, sort_keys=True))

    @classmethod
    def load(cls, path: Path, expected_id: str | None = None, provenance_verifier=None) -> "Dataset":
        doc = json.loads(Path(path).read_text())
        ident = doc["identity"]
        tf = Timeframe(ident["timeframe"])
        status = DataStatus(ident["status"])
        candles = []
        for row in doc["candles"]:
            p = row["provenance"]
            candles.append(
                Candle(
                    provider=ident["provider"], symbol=ident["symbol"], timeframe=tf,
                    open_time=parse_utc(row["open_time"]),
                    open=Decimal(row["o"]), high=Decimal(row["h"]), low=Decimal(row["l"]),
                    close=Decimal(row["c"]), volume=Decimal(row["v"]),
                    is_closed=True, status=status, received_at=parse_utc(row["received_at"]),
                    provenance=Provenance(
                        source=p["source"], method=p["method"], retrieved_at=parse_utc(p["retrieved_at"]["$t"]),
                        raw_sha256=p.get("raw_sha256"), note=p.get("note", ""),
                    ),
                )
            )
        ds = cls.build(candles, data_version=ident["data_version"], realization=ident["realization"],
                       partition=Partition(ident["partition"]))
        if ds.identity.content_sha256 != ident["content_sha256"]:
            raise DatasetIntegrityError(f"{path}: stored content hash does not match content (historical mutation)")
        if ds.dataset_id != doc["dataset_id"] or (expected_id and ds.dataset_id != expected_id):
            raise DatasetIntegrityError(f"{path}: dataset id mismatch")
        if status in MARKET_EVIDENCE_STATUSES:
            # A file's own claim of REAL is never sufficient: the category must be re-derived from
            # archived provider payloads (ati.market.archive.PayloadArchive.verify_market_provenance).
            if provenance_verifier is None:
                raise ProvenanceError(f"{path}: {status.value} dataset cannot be loaded without provenance verification")
            provenance_verifier(ds)
        return ds

    def verify_extension_of(self, older: "Dataset") -> None:
        """Legitimate growth vs historical mutation. Passes only if this dataset is the same series
        and contains every candle of ``older`` with identical content (new bars may be appended)."""
        a, b = older.identity, self.identity
        if (a.provider, a.symbol, a.timeframe, a.status) != (b.provider, b.symbol, b.timeframe, b.status):
            raise HistoricalConflictError("not the same series")
        mine = {c.open_time: c.content_key() for c in self._candles}
        for candle in older.candles:
            if mine.get(candle.open_time) != candle.content_key():
                raise HistoricalConflictError(f"historical candle {candle.open_time} is missing or changed")


# --- sealed holdout periods ---------------------------------------------------------------------
# Process-wide register of holdout periods sealed by a HoldoutVault (restored from the research
# journal on startup by ``ati.system``). Keyed by series identity so unrelated series are unaffected.
_SEALED: list[tuple[str, str, Timeframe, str, datetime, datetime]] = []


def seal_holdout_range(identity: DatasetIdentity) -> None:
    key = (identity.provider, identity.symbol, identity.timeframe, identity.realization, identity.start, identity.end)
    if key not in _SEALED:
        _SEALED.append(key)


def overlapping_sealed_range(identity: DatasetIdentity) -> tuple | None:
    for provider, symbol, tf, realization, start, end in _SEALED:
        if (identity.provider, identity.symbol, identity.timeframe, identity.realization) == (provider, symbol, tf, realization) \
                and identity.start < end and identity.end > start:
            return (symbol, tf.value, start.isoformat(), end.isoformat())
    return None


def sealed_ranges(provider: str, symbol: str, timeframe: Timeframe, realization: str) -> list[tuple[datetime, datetime]]:
    """Read-only: sealed holdout periods [start, end) for one series, sorted. Used to choose research windows
    that never overlap a holdout that has already been used."""
    return sorted((start, end) for p, sym, tf, real, start, end in _SEALED
                  if (p, sym, tf, real) == (provider, symbol, timeframe, realization))


def _clear_sealed_ranges_for_tests() -> None:
    _SEALED.clear()


class DatasetRegistry:
    """Remembers every dataset identity seen. The same logical slice with different content is a
    historical conflict."""

    def __init__(self) -> None:
        self._by_id: dict[str, DatasetIdentity] = {}
        self._by_slice: dict[tuple, str] = {}

    def register(self, ds: Dataset) -> str:
        ds.verify()
        i = ds.identity
        slice_key = (i.provider, i.symbol, i.timeframe, i.start, i.end, i.data_version, i.realization, i.partition)
        known = self._by_slice.get(slice_key)
        if known is not None and self._by_id[known].content_sha256 != i.content_sha256:
            from ati.core.errors import HistoricalConflictError

            raise HistoricalConflictError(f"slice {slice_key} re-registered with different content")
        self._by_id[ds.dataset_id] = i
        self._by_slice[slice_key] = ds.dataset_id
        return ds.dataset_id

    def get(self, dataset_id: str) -> DatasetIdentity:
        return self._by_id[dataset_id]

    def __contains__(self, dataset_id: object) -> bool:
        return dataset_id in self._by_id
