"""Exclusive, non-blocking state-directory lock (stdlib ``fcntl``, POSIX).

Operations that append to shared journals from the CLI (accumulation, company cycles, ingestion) hold this
lock for their whole duration, so two invocations can never interleave writes. A second invocation fails
immediately with ``StateLocked`` — it never waits, never retries, and never writes. The lock is released by
the operating system if the holder dies, so a crash cannot leave the state permanently locked.

Journals additionally refuse to append when their file grew outside the writing object (see
``ati.ledger.journal``), which catches any writer that bypassed this lock.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from ati.core.errors import StateLocked


class StateLock:
    def __init__(self, state_dir: Path | str, name: str = "state"):
        self.path = Path(state_dir) / f".{name}.lock"
        self._fd: int | None = None

    def __enter__(self) -> "StateLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise StateLocked(f"{self.path.parent} is in use by another company/accumulation process") from None
        self._fd = fd
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
