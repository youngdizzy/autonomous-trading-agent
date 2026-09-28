"""Deterministic accounting from fills. State is a pure fold over fills, so it can be rebuilt from
the journal at any time and must match exactly. Duplicate fills (same fill_id) are ignored.

Spot only: selling more than is held raises (no implicit shorting)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from ati.config import OperatingMode
from ati.core.errors import ModeMismatch
from ati.core.time import ensure_utc
from ati.core.types import Side
from ati.market.models import DataStatus


@dataclass(frozen=True)
class Fill:
    fill_id: str
    client_order_id: str
    symbol: str
    side: Side
    qty: Decimal
    price: Decimal
    fee: Decimal
    at: datetime
    mode: OperatingMode
    price_status: DataStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", ensure_utc(self.at, "fill.at"))
        for name in ("qty", "price"):
            v = getattr(self, name)
            if not isinstance(v, Decimal) or not v.is_finite() or v <= 0:
                raise ValueError(f"fill {name} must be a positive finite Decimal, got {v!r}")
        if not isinstance(self.fee, Decimal) or not self.fee.is_finite() or self.fee < 0:
            raise ValueError("fill fee must be a non-negative finite Decimal")


@dataclass
class Position:
    qty: Decimal = Decimal(0)
    avg_price: Decimal = Decimal(0)


@dataclass
class Account:
    mode: OperatingMode
    data_status: DataStatus
    cash: Decimal
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: Decimal = Decimal(0)
    fees_paid: Decimal = Decimal(0)
    _seen: set[str] = field(default_factory=set)

    def apply_fill(self, fill: Fill) -> bool:
        if fill.mode is not self.mode:
            raise ModeMismatch(f"{fill.mode.value} fill applied to {self.mode.value} account")
        if fill.price_status is not self.data_status:
            raise ModeMismatch(f"fill priced on {fill.price_status.value} data applied to {self.data_status.value} account")
        if fill.fill_id in self._seen:
            return False
        pos = self.positions.setdefault(fill.symbol, Position())
        notional = fill.qty * fill.price
        if fill.side is Side.BUY:
            new_qty = pos.qty + fill.qty
            pos.avg_price = (pos.avg_price * pos.qty + notional) / new_qty
            pos.qty = new_qty
            self.cash -= notional + fill.fee
        else:
            if fill.qty > pos.qty:
                raise ValueError(f"sell {fill.qty} exceeds position {pos.qty} in {fill.symbol} (no shorting)")
            self.realized_pnl += (fill.price - pos.avg_price) * fill.qty
            pos.qty -= fill.qty
            if pos.qty == 0:
                pos.avg_price = Decimal(0)
            self.cash += notional - fill.fee
        self.realized_pnl -= fill.fee
        self.fees_paid += fill.fee
        self._seen.add(fill.fill_id)
        return True

    def position_qty(self, symbol: str) -> Decimal:
        p = self.positions.get(symbol)
        return p.qty if p else Decimal(0)

    def equity(self, marks: dict[str, Decimal]) -> Decimal:
        total = self.cash
        for symbol, pos in self.positions.items():
            if pos.qty:
                if symbol not in marks:
                    raise KeyError(f"no mark price for open position {symbol}")
                total += pos.qty * marks[symbol]
        return total
