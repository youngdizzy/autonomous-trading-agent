"""Append-only, hash-chained JSONL journal.

Every entry commits to the previous entry's hash, so any edit, deletion, reordering, or truncation
of history is detected by ``verify()`` (run on every open). A journal is bound at creation to a
``kind`` and attributes (e.g. operating mode and data status); reopening with different attributes
fails, so paper and live (or MOCK and REAL) records can never share a journal.

Write discipline: each append is written, flushed and fsync'ed before returning. If a write fails
the journal marks itself broken and refuses all further appends — the caller must treat the
action as not recorded and fail closed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

from ati.core.canonical import canonical_json, to_canonical
from ati.core.errors import JournalCorruption, JournalWriteError, ModeMismatch
from ati.core.time import Clock, parse_utc, to_iso
from ati.security.secrets import SecretGuard

GENESIS = "0" * 64


@dataclass(frozen=True)
class JournalEntry:
    seq: int
    at: str
    type: str
    payload: Any
    prev: str
    hash: str


def _entry_hash(seq: int, at: str, type_: str, payload: Any, prev: str) -> str:
    body = json.dumps({"seq": seq, "at": at, "type": type_, "payload": payload, "prev": prev},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def decode(obj: Any) -> Any:
    """Invert the canonical encoding of Decimals and datetimes."""
    if isinstance(obj, dict):
        if set(obj) == {"$d"}:
            return Decimal(obj["$d"])
        if set(obj) == {"$t"}:
            return parse_utc(obj["$t"])
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


class Journal:
    def __init__(self, path: Path | str, *, kind: str, attrs: dict[str, str], clock: Clock,
                 guard: SecretGuard | None = None, fsync: bool = True):
        self.path = Path(path)
        self.kind = kind
        self.attrs = dict(attrs)
        self.clock = clock
        self.guard = guard or SecretGuard()
        self.fsync = fsync
        self._entries: list[JournalEntry] = []
        self._broken = False
        if self.path.exists() and self.path.stat().st_size > 0:
            self._load()
            header = self._entries[0].payload
            if header.get("kind") != kind or header.get("attrs") != to_canonical(self.attrs):
                raise ModeMismatch(f"journal {self.path} is {header}, opened as kind={kind} attrs={self.attrs}")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._append_raw("header", {"kind": kind, "attrs": self.attrs})

    # --- reading ------------------------------------------------------------------------
    def _load(self) -> None:
        raw = self.path.read_bytes()
        if not raw.endswith(b"\n"):
            raise JournalCorruption(f"{self.path}: truncated final entry (interrupted write)")
        prev = GENESIS
        for n, line in enumerate(raw.decode("utf-8").splitlines()):
            try:
                doc = json.loads(line)
                entry = JournalEntry(doc["seq"], doc["at"], doc["type"], doc["payload"], doc["prev"], doc["hash"])
            except (ValueError, KeyError, TypeError) as exc:
                raise JournalCorruption(f"{self.path}: unreadable entry {n}") from exc
            if entry.seq != n or entry.prev != prev or _entry_hash(entry.seq, entry.at, entry.type, entry.payload, entry.prev) != entry.hash:
                raise JournalCorruption(f"{self.path}: hash chain broken at entry {n}")
            if n == 0 and entry.type != "header":
                raise JournalCorruption(f"{self.path}: missing header")
            self._entries.append(entry)
            prev = entry.hash
        if not self._entries:
            raise JournalCorruption(f"{self.path}: empty journal")

    def verify(self) -> None:
        """Re-read the file from disk and check it matches the in-memory chain."""
        disk = Journal.__new__(Journal)
        disk.path, disk._entries = self.path, []
        disk._load()
        if [e.hash for e in disk._entries] != [e.hash for e in self._entries]:
            raise JournalCorruption(f"{self.path}: on-disk history diverges from recorded history")

    def entries(self, type_: str | None = None) -> Iterator[JournalEntry]:
        for entry in self._entries[1:]:
            if type_ is None or entry.type == type_:
                yield entry

    @property
    def head_hash(self) -> str:
        return self._entries[-1].hash if self._entries else GENESIS

    def __len__(self) -> int:
        return len(self._entries) - 1

    # --- writing ------------------------------------------------------------------------
    def append(self, type_: str, payload: Any) -> JournalEntry:
        if type_ == "header":
            raise ValueError("header is reserved")
        return self._append_raw(type_, payload)

    def _append_raw(self, type_: str, payload: Any) -> JournalEntry:
        if self._broken:
            raise JournalWriteError(f"{self.path}: journal is broken after a failed write; refusing appends")
        canonical_payload = to_canonical(payload)
        text_payload = canonical_json(canonical_payload)
        self.guard.scan(text_payload, where=f"journal {self.path.name}:{type_}")
        seq = len(self._entries)
        at = to_iso(self.clock.now())
        prev = self.head_hash
        digest = _entry_hash(seq, at, type_, canonical_payload, prev)
        line = json.dumps({"seq": seq, "at": at, "type": type_, "payload": canonical_payload, "prev": prev, "hash": digest},
                          sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        try:
            self._write_line(line)
        except OSError as exc:
            self._broken = True
            raise JournalWriteError(f"{self.path}: write failed: {exc}") from exc
        entry = JournalEntry(seq, at, type_, canonical_payload, prev, digest)
        self._entries.append(entry)
        return entry

    def _write_line(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())
