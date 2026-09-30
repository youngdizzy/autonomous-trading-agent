"""Strategy logic interface and immutable definitions.

A ``StrategyDefinition`` is data: (id, version, logic kind, parameters, timeframe). Its
``definition_hash`` also covers the *source code* of the logic class, so editing the code of a
registered strategy changes its identity rather than silently altering history.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import ClassVar

from ati.core.canonical import sha256_hex, sha256_text
from ati.core.time import ensure_utc
from ati.market.models import Timeframe
from ati.temporal.pit import PointInTimeView


class Target(str, Enum):
    LONG = "LONG"
    FLAT = "FLAT"


@dataclass(frozen=True)
class Signal:
    target: Target
    stop_price: Decimal | None
    reason: str
    as_of: datetime


@dataclass(frozen=True)
class ParamSpec:
    kind: type
    lo: float
    hi: float


class StrategyLogic(ABC):
    kind: ClassVar[str]
    params: ClassVar[dict[str, ParamSpec]]

    @classmethod
    def validate_params(cls, values: dict[str, object]) -> None:
        if set(values) != set(cls.params):
            raise ValueError(f"{cls.kind}: parameters must be exactly {sorted(cls.params)}, got {sorted(values)}")
        for name, spec in cls.params.items():
            value = values[name]
            if type(value) is not spec.kind:
                raise ValueError(f"{cls.kind}.{name} must be {spec.kind.__name__}")
            if not (spec.lo <= float(value) <= spec.hi):
                raise ValueError(f"{cls.kind}.{name}={value} outside [{spec.lo}, {spec.hi}]")
        cls.check_relations(values)

    @classmethod
    def check_relations(cls, values: dict[str, object]) -> None:
        return None

    @classmethod
    @abstractmethod
    def lookback(cls, values: dict[str, object]) -> int: ...

    @classmethod
    @abstractmethod
    def generate(cls, view: PointInTimeView, values: dict[str, object], in_position: bool) -> Signal: ...

    @classmethod
    def code_hash(cls) -> str:
        return sha256_text(inspect.getsource(cls))


LOGIC_REGISTRY: dict[str, type[StrategyLogic]] = {}


def register_logic(cls: type[StrategyLogic]) -> type[StrategyLogic]:
    if cls.kind in LOGIC_REGISTRY:
        raise ValueError(f"duplicate strategy kind {cls.kind}")
    LOGIC_REGISTRY[cls.kind] = cls
    return cls


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    version: int
    kind: str
    params: tuple[tuple[str, object], ...]
    timeframe: Timeframe
    created_at: datetime
    parent_hash: str | None = None
    description: str = ""
    code_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.strategy_id or not self.strategy_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid strategy_id {self.strategy_id!r}")
        if not isinstance(self.timeframe, Timeframe):
            raise ValueError(f"timeframe must be a Timeframe, got {self.timeframe!r} (no implicit default)")
        if not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be a positive int")
        if self.kind not in LOGIC_REGISTRY:
            raise ValueError(f"unknown strategy kind {self.kind!r}")
        object.__setattr__(self, "params", tuple(sorted(tuple(self.params))))
        object.__setattr__(self, "created_at", ensure_utc(self.created_at, "created_at"))
        self.logic.validate_params(self.param_dict)
        actual_code = self.logic.code_hash()
        if self.code_hash and self.code_hash != actual_code:
            raise ValueError(f"{self.key}: logic source changed since definition was recorded")
        object.__setattr__(self, "code_hash", actual_code)

    @classmethod
    def create(cls, strategy_id: str, version: int, kind: str, params: dict[str, object], timeframe: Timeframe,
               created_at: datetime, parent_hash: str | None = None, description: str = "") -> "StrategyDefinition":
        return cls(strategy_id, version, kind, tuple(params.items()), timeframe, created_at, parent_hash, description)

    @property
    def key(self) -> str:
        """Behavioural name (strategy id + version). NOT unique across timeframes — never a registry lookup key."""
        return f"{self.strategy_id}@v{self.version}"

    @property
    def registry_key(self) -> str:
        """Timeframe deployment identity: the strategy registry's unique key (``trend@v1/1h``)."""
        return f"{self.strategy_id}@v{self.version}/{self.timeframe.value}"

    @property
    def behavior_fingerprint(self) -> str:
        """What the rule *is*, independent of the timeframe it runs on: id, version, kind, params, code. Two
        deployments of one strategy on 1h and 4h share it. (``definition_hash`` — the strategy fingerprint used for
        research, promotion and decisions — additionally binds the timeframe and lineage.)"""
        return sha256_hex({"strategy_id": self.strategy_id, "version": self.version, "kind": self.kind,
                           "params": [list(p) for p in self.params], "code_hash": self.code_hash})

    @property
    def logic(self) -> type[StrategyLogic]:
        return LOGIC_REGISTRY[self.kind]

    @property
    def param_dict(self) -> dict[str, object]:
        return dict(self.params)

    @property
    def lookback(self) -> int:
        return self.logic.lookback(self.param_dict)

    @property
    def definition_hash(self) -> str:
        """Content identity: kind, params, timeframe, code, id/version, lineage. Excludes the
        free-text description and creation time."""
        return sha256_hex({
            "strategy_id": self.strategy_id, "version": self.version, "kind": self.kind,
            "params": [list(p) for p in self.params], "timeframe": self.timeframe,
            "code_hash": self.code_hash, "parent_hash": self.parent_hash,
        })

    @property
    def lineage_root(self) -> str:
        return self.strategy_id

    def signal(self, view: PointInTimeView, in_position: bool) -> Signal:
        sig = self.logic.generate(view, self.param_dict, in_position)
        if sig.as_of != view.cutoff:
            raise AssertionError("signal must be stamped with the view cutoff")
        return sig
