from __future__ import annotations

from datetime import datetime

from ati.core.errors import DataIntegrityError


def require_aligned(ts: datetime, timeframe) -> None:
    if int(ts.timestamp()) % timeframe.seconds != 0 or ts.microsecond:
        raise DataIntegrityError("ALIGNMENT", f"{ts} is not aligned to {timeframe.value}")
