"""Broker protocol and order models. Every venue adapter (paper, deterministic mock, live later) implements
``Broker``. The execution engine is the only caller.

Order lifecycle (acknowledgement and fills are separate facts; "submitted" never means "filled")::

    PENDING_SUBMIT (intent journaled, request sent, no answer yet)
      → ACKNOWLEDGED (venue accepted, no fills) → PARTIALLY_FILLED (resting, some fills) → FILLED
      → CANCEL_REQUESTED → CANCELED | PARTIAL_CANCELED
      → REJECTED | EXPIRED | NOT_FOUND
      → UNKNOWN (outcome not established: halts new trading until reconciled)

Market orders only: the execution contract has no limit/stop orders and no ``replace`` (a changed order is a
cancel plus a new, separately authorized decision)."""

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
    ACKNOWLEDGED = "ACKNOWLEDGED"      # venue accepted the order; no fills yet
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # venue working the order; some fills
    CANCEL_REQUESTED = "CANCEL_REQUESTED"  # cancel sent; outcome not yet confirmed by the venue
    FILLED = "FILLED"
    PARTIAL_CANCELED = "PARTIAL_CANCELED"  # immediate-or-cancel: partial fill, remainder canceled
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"
    NOT_FOUND = "NOT_FOUND"            # broker has no record of the client order id
    EXPIRED = "EXPIRED"                # venue ended the order unfilled (or partly filled) by time
    UNKNOWN = "UNKNOWN"                # outcome not known — halts new trading until reconciled


TERMINAL = frozenset({OrderStatus.FILLED, OrderStatus.PARTIAL_CANCELED, OrderStatus.REJECTED,
                      OrderStatus.CANCELED, OrderStatus.NOT_FOUND, OrderStatus.EXPIRED})
#: Venue-confirmed working orders: tracked, not an inconsistency — but they block new orders for the symbol.
OPEN = frozenset({OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED})
#: States whose truth is not established: reconciliation must resolve them before trading resumes.
UNRESOLVED = frozenset({OrderStatus.PENDING_SUBMIT, OrderStatus.CANCEL_REQUESTED, OrderStatus.UNKNOWN})


@dataclass(frozen=True)
class BrokerOrderReport:
    client_order_id: str
    status: OrderStatus
    filled_qty: Decimal
    fills: tuple[Fill, ...]
    message: str = ""
    broker_order_id: str = ""          # the venue's own id, when it assigns one


@dataclass(frozen=True)
class BrokerHealth:
    """Adapter self-report. ``ok`` is False if any check failed; deterministic code, not Claude, reads it."""
    ok: bool
    checks: tuple[tuple[str, bool, str], ...]      # (name, passed, detail)


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
    broker_order_id: str = ""

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

    def list_orders(self) -> tuple[BrokerOrderReport, ...]:
        """Every order the venue holds for this account (reconciliation detects orders TradeTown did not send)."""
        ...

    def cancel_order(self, client_order_id: str) -> BrokerOrderReport:
        """Request cancellation. Raise BrokerTimeout/BrokerUnavailable when the outcome is unknown; return the
        venue's report (which may show fills that happened before the cancel took effect)."""
        ...

    def health_check(self) -> BrokerHealth:
        """Session, account and order-entry availability as the adapter can establish them. Never raises for an
        unhealthy venue: returns ``ok=False`` with the failing checks."""
        ...
