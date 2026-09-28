"""Broker protocol and order models. Every venue adapter (paper now, live later) implements
``Broker``. The execution engine is the only caller."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol

from ati.config import OperatingMode
from ati.core.types import Side
from ati.ledger.accounting import Fill


class OrderStatus(str, Enum):
    PENDING_SUBMIT = "PENDING_SUBMIT"  # intent durably recorded, not yet acknowledged
    FILLED = "FILLED"
    PARTIAL_CANCELED = "PARTIAL_CANCELED"  # immediate-or-cancel: partial fill, remainder canceled
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"
    NOT_FOUND = "NOT_FOUND"            # broker has no record of the client order id
    UNKNOWN = "UNKNOWN"                # outcome not known — halts new trading until reconciled


TERMINAL = frozenset({OrderStatus.FILLED, OrderStatus.PARTIAL_CANCELED, OrderStatus.REJECTED,
                      OrderStatus.CANCELED, OrderStatus.NOT_FOUND})


@dataclass(frozen=True)
class BrokerOrderReport:
    client_order_id: str
    status: OrderStatus
    filled_qty: Decimal
    fills: tuple[Fill, ...]
    message: str = ""


@dataclass(frozen=True)
class BrokerAccount:
    cash: Decimal
    positions: dict[str, Decimal]


@dataclass
class Order:
    client_order_id: str
    decision_id: str
    strategy_key: str
    symbol: str
    side: Side
    intended_qty: Decimal
    intended_price: Decimal
    created_at: datetime
    mode: OperatingMode
    status: OrderStatus = OrderStatus.PENDING_SUBMIT
    fills: list[Fill] = field(default_factory=list)
    message: str = ""

    @property
    def filled_qty(self) -> Decimal:
        return sum((f.qty for f in self.fills), Decimal(0))

    @property
    def fees(self) -> Decimal:
        return sum((f.fee for f in self.fills), Decimal(0))

    @property
    def avg_fill_price(self) -> Decimal | None:
        q = self.filled_qty
        return sum((f.qty * f.price for f in self.fills), Decimal(0)) / q if q else None

    @property
    def slippage(self) -> Decimal | None:
        """Signed adverse slippage per unit vs intended price (positive = worse than intended)."""
        avg = self.avg_fill_price
        if avg is None:
            return None
        return avg - self.intended_price if self.side is Side.BUY else self.intended_price - avg


class Broker(Protocol):
    mode: OperatingMode

    def submit_order(self, client_order_id: str, symbol: str, side: Side, qty: Decimal) -> BrokerOrderReport:
        """Must be idempotent on client_order_id. Raise BrokerTimeout/BrokerUnavailable when the
        outcome is unknown, BrokerRejected only for a verifiable rejection."""
        ...

    def get_order(self, client_order_id: str) -> BrokerOrderReport | None: ...

    def get_account(self) -> BrokerAccount: ...
