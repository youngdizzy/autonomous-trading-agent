"""Canonical serialization and content hashing.

Identity in this system is content: two artifacts with the same meaningful content hash equally,
and any change to content changes the hash. Decimals are normalized so ``1.0`` and ``1.00`` hash
the same; floats must be finite; datetimes must be UTC-aware.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from ati.core.time import to_iso


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError(f"non-finite Decimal cannot be canonicalized: {value}")
    text = format(value.normalize(), "f")
    return "0" if text in ("-0", "0") else text


def to_canonical(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float cannot be canonicalized: {obj}")
        return obj
    if isinstance(obj, Decimal):
        return {"$d": _decimal_text(obj)}
    if isinstance(obj, datetime):
        return {"$t": to_iso(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_canonical(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.repr}
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                key = str(to_canonical(key))
            out[key] = to_canonical(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [to_canonical(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((to_canonical(item) for item in obj), key=lambda x: json.dumps(x, sort_keys=True))
    raise TypeError(f"cannot canonicalize {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    return json.dumps(to_canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
