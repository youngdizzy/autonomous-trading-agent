"""Persistent kill switch.

State lives in a small file so it survives restarts. Any doubt about the state (unreadable,
malformed) reads as ENGAGED. Releasing requires an explicit operator acknowledgement string; no
agent-facing action maps to ``release``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ati.core.time import Clock, to_iso

RELEASE_ACK = "OPERATOR: state reconciled and verified; release kill switch"


class KillSwitch:
    def __init__(self, path: Path | str, clock: Clock):
        self.path = Path(path)
        self.clock = clock
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write({"engaged": False, "reason": "initialized", "at": to_iso(clock.now())})

    def _write(self, state: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True))
        os.replace(tmp, self.path)

    def state(self) -> dict:
        try:
            doc = json.loads(self.path.read_text())
            if not isinstance(doc, dict) or not isinstance(doc.get("engaged"), bool):
                raise ValueError("malformed")
            return doc
        except (OSError, ValueError):
            return {"engaged": True, "reason": "kill switch state unreadable (fail closed)", "at": None}

    @property
    def engaged(self) -> bool:
        return self.state()["engaged"]

    def engage(self, reason: str) -> None:
        self._write({"engaged": True, "reason": reason, "at": to_iso(self.clock.now())})

    def release(self, acknowledgement: str) -> None:
        if acknowledgement != RELEASE_ACK:
            raise PermissionError("kill switch release requires the exact operator acknowledgement")
        self._write({"engaged": False, "reason": "operator release", "at": to_iso(self.clock.now())})
