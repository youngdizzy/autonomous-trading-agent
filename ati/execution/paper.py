"""Paper broker: a simulated venue with its *own* state, separate from the system's internal ledger,
so reconciliation between the two is meaningful.

Fills use the latest quote from a market-data source, the same ``CostModel`` as research
(spread + slippage + fees), a participation cap (immediate-or-cancel partial fills) and a latency
assumption stamped on each fill. Fills carry ``mode=PAPER`` and the data status of the quote they
were priced from, so paper results on MOCK data can never be presented as anything else.

``faults`` is a test-only fault-injection set used by the chaos suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable

from ati.config import OperatingMode
from ati.core.errors import BrokerRejected, BrokerTimeout, BrokerUnavailable
from ati.core.time import Clock
from ati.core.types import Side
from ati.ledger.accounting import Fill
from ati.market.models import DataStatus
from ati.execution.broker import BrokerAccount, BrokerOrderReport, OrderStatus
from ati.research.costs import CostModel


@dataclass(frozen=True)
class Quote:
    symbol: str
    mid: Decimal
    at: datetime
    available_volume: Decimal
    status: DataStatus


class PaperBroker:
    mode = OperatingMode.PAPER

    def __init__(self, quotes: Callable[[str], Quote], costs: CostModel, initial_cash: Decimal, data_status: DataStatus,
                 clock: Clock, latency: timedelta = timedelta(milliseconds=500), max_quote_age: timedelta = timedelta(hours=2)):
        self.quotes = quotes
        self.costs = costs
        self.cash = initial_cash
        self.positions: dict[str, Decimal] = {}
        self.data_status = data_status
        self.clock = clock
        self.latency = latency
        self.max_quote_age = max_quote_age
        self.orders: dict[str, BrokerOrderReport] = {}
        self.faults: set[str] = set()
        self.submissions = 0  # number of times an order was actually executed at the venue

    def _check_up(self) -> None:
        if "unavailable" in self.faults:
            raise BrokerUnavailable("paper venue unavailable (injected)")

    def submit_order(self, client_order_id: str, symbol: str, side: Side, qty: Decimal) -> BrokerOrderReport:
        self._check_up()
        if "timeout_before_execute" in self.faults:
            raise BrokerTimeout("no response (injected, not executed)")
        if client_order_id in self.orders:
            return self.orders[client_order_id]  # venue-side idempotency
        report = self._execute(client_order_id, symbol, side, qty)
        self.orders[client_order_id] = report
        if "timeout_after_execute" in self.faults:
            raise BrokerTimeout("response lost (injected, executed)")
        if "false_reject_after_execute" in self.faults:
            raise BrokerRejected("venue reported failure (injected, executed)")
        return report

    def _execute(self, coid: str, symbol: str, side: Side, qty: Decimal) -> BrokerOrderReport:
        now = self.clock.now()
        quote = self.quotes(symbol)
        if quote.status is not self.data_status:
            return BrokerOrderReport(coid, OrderStatus.REJECTED, Decimal(0), (), f"quote is {quote.status.value}")
        if now - quote.at > self.max_quote_age:
            return BrokerOrderReport(coid, OrderStatus.REJECTED, Decimal(0), (), "stale quote")
        fill_qty = min(qty, quote.available_volume * self.costs.max_participation)
        price = self.costs.fill_price(quote.mid, side)
        fee = price * fill_qty * self.costs.fee_rate
        held = self.positions.get(symbol, Decimal(0))
        if side is Side.SELL and fill_qty > held:
            return BrokerOrderReport(coid, OrderStatus.REJECTED, Decimal(0), (), "insufficient position")
        if side is Side.BUY and price * fill_qty + fee > self.cash:
            return BrokerOrderReport(coid, OrderStatus.REJECTED, Decimal(0), (), "insufficient cash")
        if fill_qty <= 0:
            return BrokerOrderReport(coid, OrderStatus.CANCELED, Decimal(0), (), "no liquidity")
        if side is Side.BUY:
            self.cash -= price * fill_qty + fee
            self.positions[symbol] = held + fill_qty
        else:
            self.cash += price * fill_qty - fee
            self.positions[symbol] = held - fill_qty
        self.submissions += 1
        fill = Fill(f"{coid}-f1", coid, symbol, side, fill_qty, price, fee, now + self.latency,
                    OperatingMode.PAPER, quote.status)
        status = OrderStatus.FILLED if fill_qty == qty else OrderStatus.PARTIAL_CANCELED
        return BrokerOrderReport(coid, status, fill_qty, (fill,))

    def get_order(self, client_order_id: str) -> BrokerOrderReport | None:
        self._check_up()
        return self.orders.get(client_order_id)

    def get_account(self) -> BrokerAccount:
        self._check_up()
        return BrokerAccount(self.cash, {s: q for s, q in self.positions.items() if q})

    # --- chaos helpers -----------------------------------------------------------------------
    def external_cash_change(self, delta: Decimal) -> None:
        """Simulate an unexpected account change (deposit/withdrawal/venue adjustment)."""
        self.cash += delta
