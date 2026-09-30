"""Deterministic scripted broker adapter — the stand-in for a real venue's order lifecycle.

Unlike ``PaperBroker`` (immediate-or-cancel fills against the market-data quote), this adapter behaves like a
real broker API: ``submit_order`` returns an *acknowledgement*; fills arrive later and separately
(``advance_fill``); orders can rest, be cancelled, expire, time out or be lost. Every outcome is scripted — there
is no randomness — so each scenario is reproducible.

Script vocabulary (one entry consumed per *new* client order id; default ``fill``):
  fill              acknowledged and fully filled at once
  ack               acknowledged, no fill yet (resting)
  partial:<frac>    acknowledged with a first fill of <frac> × qty (resting)
  reject            venue rejects (BrokerRejected)
  timeout_before    no response and nothing recorded at the venue (BrokerTimeout)
  timeout_after     venue accepted and filled, response lost (BrokerTimeout)
Cancel script (one entry per cancel; default ``ok``): ``ok`` | ``timeout`` (BrokerTimeout) | ``ignore`` (the venue
answers but has not processed the cancel: the order stays working).

The adapter owns its (mock) credential: it is held as a ``SecretValue``, used only to build a private session
token, and never appears in any report, message, health detail or repr. It is ``mode = PAPER``: a simulator can
never be wired into a LIVE execution engine.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Decimal

from ati.config import OperatingMode
from ati.core.errors import BrokerRejected, BrokerTimeout, BrokerUnavailable
from ati.core.time import Clock
from ati.core.types import Side
from ati.execution.broker import OPEN, BrokerAccount, BrokerHealth, BrokerOrderReport, OrderStatus
from ati.ledger.accounting import Fill
from ati.market.models import DataStatus
from ati.security.secrets import SecretValue


class MockBrokerAdapter:
    mode = OperatingMode.PAPER
    name = "mock-broker"

    def __init__(self, clock: Clock, prices: dict[str, Decimal], cash: Decimal, credential: SecretValue,
                 fee_rate: Decimal = Decimal("0.001"), data_status: DataStatus = DataStatus.MOCK):
        if not isinstance(credential, SecretValue):
            raise TypeError("the adapter takes its credential as a SecretValue")
        self.clock = clock
        self.prices = dict(prices)
        self.cash = cash
        self.positions: dict[str, Decimal] = {}
        self.fee_rate = fee_rate
        self.data_status = data_status
        self.__credential = credential
        self._session = hashlib.sha256(b"session:" + credential.reveal().encode()).hexdigest()   # private
        self.script: list[str] = []
        self.cancel_script: list[str] = []
        self.down = False                 # every call raises BrokerUnavailable
        self.session_valid = True         # authenticated session available
        self.order_entry_enabled = True   # venue accepts new orders
        self._orders: dict[str, dict] = {}
        self.calls: list[str] = []        # method names only (never arguments) — lets tests prove what was called
        self.submissions = 0              # orders that actually reached the venue book

    def __repr__(self) -> str:
        return f"MockBrokerAdapter(orders={len(self._orders)}, down={self.down})"

    # --- internals ----------------------------------------------------------------------------------
    def _up(self, method: str) -> None:
        self.calls.append(method)
        if self.down:
            raise BrokerUnavailable("mock venue unavailable")
        if not self.session_valid:
            raise BrokerUnavailable("mock venue: session not authenticated")

    def _report(self, coid: str) -> BrokerOrderReport:
        o = self._orders[coid]
        filled = sum((f.qty for f in o["fills"]), Decimal(0))
        return BrokerOrderReport(coid, o["status"], filled, tuple(o["fills"]), o["message"], o["broker_order_id"])

    def _fill(self, coid: str, qty: Decimal, price: Decimal | None = None) -> Fill:
        o = self._orders[coid]
        price = price if price is not None else self.prices[o["symbol"]]
        fee = price * qty * self.fee_rate
        if o["side"] is Side.BUY:
            self.cash -= price * qty + fee
            self.positions[o["symbol"]] = self.positions.get(o["symbol"], Decimal(0)) + qty
        else:
            self.cash += price * qty - fee
            self.positions[o["symbol"]] = self.positions.get(o["symbol"], Decimal(0)) - qty
        fill = Fill(f"{coid}-f{len(o['fills']) + 1}", coid, o["symbol"], o["side"], qty, price, fee, self.clock.now(),
                    OperatingMode.PAPER, self.data_status)
        o["fills"].append(fill)
        done = sum((f.qty for f in o["fills"]), Decimal(0))
        o["status"] = OrderStatus.FILLED if done >= o["qty"] else OrderStatus.PARTIALLY_FILLED
        return fill

    # --- Broker contract ------------------------------------------------------------------------------
    def health_check(self) -> BrokerHealth:
        self.calls.append("health_check")
        checks = (("venue_reachable", not self.down, "reachable" if not self.down else "unreachable"),
                  ("session_authenticated", self.session_valid, "session valid" if self.session_valid else "no session"),
                  ("order_entry", self.order_entry_enabled, "enabled" if self.order_entry_enabled else "disabled"),
                  ("account_readable", not self.down and self.session_valid, "account snapshot available"))
        return BrokerHealth(all(ok for _, ok, _ in checks), checks)

    def submit_order(self, client_order_id: str, symbol: str, side: Side, qty: Decimal) -> BrokerOrderReport:
        self._up("submit_order")
        if client_order_id in self._orders:
            return self._report(client_order_id)                  # venue-side idempotency on client order id
        if not self.order_entry_enabled:
            raise BrokerRejected("order entry disabled")
        step = self.script.pop(0) if self.script else "fill"
        if step == "timeout_before":
            raise BrokerTimeout("no response (not executed)")
        if step == "reject":
            self._orders[client_order_id] = {"symbol": symbol, "side": side, "qty": qty, "fills": [],
                                             "status": OrderStatus.REJECTED, "message": "rejected by venue",
                                             "broker_order_id": f"B{len(self._orders) + 1:06d}"}
            raise BrokerRejected("rejected by venue")
        if side is Side.SELL and qty > self.positions.get(symbol, Decimal(0)):
            raise BrokerRejected("insufficient position")
        self._orders[client_order_id] = {"symbol": symbol, "side": side, "qty": qty, "fills": [],
                                         "status": OrderStatus.ACKNOWLEDGED, "message": "",
                                         "broker_order_id": f"B{len(self._orders) + 1:06d}"}
        self.submissions += 1
        if step in ("fill", "timeout_after"):
            self._fill(client_order_id, qty)
        elif step.startswith("partial:"):
            self._fill(client_order_id, (qty * Decimal(step.split(":", 1)[1])).quantize(Decimal("0.00000001")))
        elif step != "ack":
            raise ValueError(f"unknown script step {step!r}")
        if step == "timeout_after":
            raise BrokerTimeout("response lost (executed)")
        return self._report(client_order_id)

    def get_order(self, client_order_id: str) -> BrokerOrderReport | None:
        self._up("get_order")
        return self._report(client_order_id) if client_order_id in self._orders else None

    def list_orders(self) -> tuple[BrokerOrderReport, ...]:
        self._up("list_orders")
        return tuple(self._report(k) for k in sorted(self._orders))

    def get_account(self) -> BrokerAccount:
        self._up("get_account")
        return BrokerAccount(self.cash, {s: q for s, q in self.positions.items() if q})

    def cancel_order(self, client_order_id: str) -> BrokerOrderReport:
        self._up("cancel_order")
        step = self.cancel_script.pop(0) if self.cancel_script else "ok"
        if client_order_id not in self._orders:
            return BrokerOrderReport(client_order_id, OrderStatus.NOT_FOUND, Decimal(0), (), "no such order")
        if step == "timeout":
            raise BrokerTimeout("cancel: no response")
        o = self._orders[client_order_id]
        if step == "ok" and o["status"] in OPEN:
            o["status"] = OrderStatus.PARTIAL_CANCELED if o["fills"] else OrderStatus.CANCELED
            o["message"] = "cancelled"
        return self._report(client_order_id)

    # --- scenario controls (the venue acting on its own) -------------------------------------------------
    def advance_fill(self, client_order_id: str, qty: Decimal | None = None) -> Fill:
        o = self._orders[client_order_id]
        if o["status"] not in OPEN:
            raise ValueError("order is not working")
        remaining = o["qty"] - sum((f.qty for f in o["fills"]), Decimal(0))
        return self._fill(client_order_id, min(qty or remaining, remaining))

    def expire(self, client_order_id: str) -> None:
        o = self._orders[client_order_id]
        if o["status"] in OPEN:
            o["status"], o["message"] = OrderStatus.EXPIRED, "expired"

    def inject_order(self, client_order_id: str, symbol: str, side: Side, qty: Decimal, fill: bool = True) -> None:
        """An order TradeTown never sent (e.g. placed by hand on the venue)."""
        self._orders[client_order_id] = {"symbol": symbol, "side": side, "qty": qty, "fills": [],
                                         "status": OrderStatus.ACKNOWLEDGED, "message": "external",
                                         "broker_order_id": f"X{len(self._orders) + 1:06d}"}
        if fill:
            self._fill(client_order_id, qty)

    def inject_fill(self, client_order_id: str, qty: Decimal, price: Decimal | None = None) -> Fill:
        """A fill the venue reports on an order TradeTown already considers final."""
        o = self._orders[client_order_id]
        o["qty"] = o["qty"] + qty
        status = o["status"]
        fill = self._fill(client_order_id, qty, price)
        o["status"] = status
        return fill

    def reprice_fill(self, client_order_id: str, index: int, price: Decimal) -> None:
        """The venue's record of a fill disagrees with what it reported earlier."""
        o = self._orders[client_order_id]
        o["fills"][index] = replace(o["fills"][index], price=price)

    def set_price(self, symbol: str, price: Decimal) -> None:
        self.prices[symbol] = price
