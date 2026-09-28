"""Execution engine: the only component that talks to a broker.

Guarantees:
- Accepts only a valid, unexpired, approved, signed ``RiskVerdict`` (no hidden bypass).
- Idempotent: the client order id is derived from the decision id; a decision can produce at most
  one order, and re-submitting returns the existing order without contacting the broker.
- Write-ahead: the order intent is durably journaled before the broker is called. If that write
  fails nothing is sent.
- Any uncertain outcome (timeout, lost response, contradictory broker answer, journal failure after
  submission, malformed report) marks the order UNKNOWN and HALTS all new submissions.
- ``reconcile`` queries the broker by client order id, applies missing fills (deduplicated by fill
  id), compares cash and positions, and only resumes when internal and external state agree.
- State is rebuilt from the journal on restart; unresolved orders keep the engine halted and
  reconciliation state starts UNKNOWN.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from ati.config import OperatingMode, assert_mode_permitted
from ati.core.errors import (ApprovalInvalid, BrokerError, BrokerRejected, ExecutionHalted, JournalWriteError,
                             ModeMismatch)
from ati.core.time import Clock
from ati.core.types import Side
from ati.execution.broker import TERMINAL, Broker, BrokerOrderReport, Order, OrderStatus
from ati.ledger.accounting import Account, Fill
from ati.ledger.journal import Journal, decode
from ati.market.models import DataStatus
from ati.risk.engine import ApprovalAuthority, PortfolioSnapshot, ReconState, RiskVerdict
from ati.security.secrets import SecretGuard

CASH_TOLERANCE = Decimal("0.00000001")


def client_order_id(decision_id: str) -> str:
    return "ati-" + hashlib.sha256(decision_id.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class ReconciliationResult:
    state: ReconState
    issues: tuple[str, ...]
    at: datetime


class ExecutionEngine:
    def __init__(self, broker: Broker, journal_path: Path | str, authority: ApprovalAuthority, clock: Clock, *,
                 mode: OperatingMode, data_status: DataStatus, initial_cash: Decimal, guard: SecretGuard | None = None):
        assert_mode_permitted(mode)
        if broker.mode is not mode:
            raise ModeMismatch(f"{broker.mode.value} broker wired into {mode.value} execution engine")
        self.broker = broker
        self.authority = authority
        self.clock = clock
        self.mode = mode
        self.journal = Journal(journal_path, kind="execution", clock=clock, guard=guard,
                               attrs={"mode": mode.value, "data_status": data_status.value, "initial_cash": str(initial_cash)})
        self.account = Account(mode, data_status, initial_cash)
        self.orders: dict[str, Order] = {}
        self.halted_reason: str | None = None
        self.recon_state = ReconState.UNKNOWN
        self._recover()

    # --- state -------------------------------------------------------------------------------
    @property
    def halted(self) -> bool:
        return self.halted_reason is not None

    def _record(self, type_: str, payload) -> None:
        try:
            self.journal.append(type_, payload)
        except JournalWriteError as exc:
            self.halted_reason = f"journal write failed: {exc}"
            self.recon_state = ReconState.UNKNOWN
            raise ExecutionHalted(self.halted_reason) from exc

    def _halt(self, reason: str) -> None:
        self.recon_state = ReconState.UNKNOWN if self.recon_state is ReconState.OK else self.recon_state
        if self.halted_reason != reason:
            self.halted_reason = reason
            self._record("halt", {"reason": reason})

    def _recover(self) -> None:
        for entry in self.journal.entries():
            p = decode(entry.payload)
            if entry.type == "order_intent":
                self.orders[p["client_order_id"]] = Order(
                    p["client_order_id"], p["decision_id"], p["strategy_key"], p["symbol"], Side(p["side"]),
                    p["intended_qty"], p["intended_price"], p["created_at"], OperatingMode(p["mode"]))
            elif entry.type == "order_status":
                order = self.orders[p["client_order_id"]]
                order.status, order.message = OrderStatus(p["status"]), p["message"]
            elif entry.type == "fill":
                f = p["fill"]
                fill = Fill(f["fill_id"], f["client_order_id"], f["symbol"], Side(f["side"]), f["qty"], f["price"],
                            f["fee"], f["at"], OperatingMode(f["mode"]), DataStatus(f["price_status"]))
                if self.account.apply_fill(fill):
                    self.orders[fill.client_order_id].fills.append(fill)
            elif entry.type == "external_adjustment":
                self.account.cash += p["cash_delta"]
            elif entry.type == "halt":
                self.halted_reason = p["reason"]
            elif entry.type == "resume":
                self.halted_reason = None
        unresolved = [o.client_order_id for o in self.orders.values() if o.status not in TERMINAL]
        if unresolved:
            self._halt(f"{len(unresolved)} unresolved order(s) after restart: reconcile required")

    # --- submission ----------------------------------------------------------------------------
    def submit(self, verdict: RiskVerdict) -> Order:
        if self.halted:
            raise ExecutionHalted(self.halted_reason)
        v = self.authority.verify(verdict)
        if not v.approved or v.qty <= 0:
            raise ApprovalInvalid("verdict is not an approval")
        if v.mode is not self.mode:
            raise ModeMismatch(f"{v.mode.value} approval presented to {self.mode.value} execution")
        if self.clock.now() > v.valid_until:
            raise ApprovalInvalid("risk approval expired; re-evaluate risk")
        coid = client_order_id(v.decision_id)
        if coid in self.orders:
            return self.orders[coid]  # idempotent: never submit twice for one decision
        order = Order(coid, v.decision_id, v.strategy_key, v.symbol, v.side, v.qty, v.entry_price, self.clock.now(), self.mode)
        self._record("order_intent", {"client_order_id": coid, "decision_id": v.decision_id, "strategy_key": v.strategy_key,
                                      "symbol": v.symbol, "side": v.side, "intended_qty": v.qty,
                                      "intended_price": v.entry_price, "created_at": order.created_at, "mode": self.mode,
                                      "risk_limits_hash": v.limits_hash})
        self.orders[coid] = order
        try:
            report = self.broker.submit_order(coid, v.symbol, v.side, v.qty)
        except BrokerRejected as exc:
            self._confirm_rejection(order, str(exc))
            return order
        except BrokerError as exc:
            self._set_status(order, OrderStatus.UNKNOWN, str(exc))
            self._halt(f"order {coid} outcome unknown: {exc}")
            return order
        except Exception as exc:  # any unexpected adapter failure is an unknown outcome
            self._set_status(order, OrderStatus.UNKNOWN, f"unexpected: {exc!r}")
            self._halt(f"order {coid} outcome unknown: unexpected adapter error")
            return order
        self._apply_report(order, report)
        return order

    def _confirm_rejection(self, order: Order, message: str) -> None:
        """A rejection is only believed after the venue confirms it has no fills for this id."""
        try:
            report = self.broker.get_order(order.client_order_id)
        except BrokerError as exc:
            self._set_status(order, OrderStatus.UNKNOWN, f"rejection unconfirmed: {exc}")
            self._halt(f"order {order.client_order_id}: rejection could not be confirmed")
            return
        if report is None or (not report.fills and report.status in (OrderStatus.REJECTED, OrderStatus.CANCELED)):
            self._set_status(order, OrderStatus.REJECTED, message)
            return
        self._apply_report(order, report)
        self._halt(f"order {order.client_order_id}: venue reported failure but order executed")

    def _set_status(self, order: Order, status: OrderStatus, message: str) -> None:
        self._record("order_status", {"client_order_id": order.client_order_id, "status": status, "message": message})
        order.status, order.message = status, message

    def _apply_report(self, order: Order, report: BrokerOrderReport) -> None:
        problems = []
        if not isinstance(report, BrokerOrderReport) or report.client_order_id != order.client_order_id:
            problems.append("report does not match order")
        else:
            for f in report.fills:
                if f.client_order_id != order.client_order_id or f.symbol != order.symbol or f.side is not order.side:
                    problems.append(f"fill {f.fill_id} does not belong to order")
            if sum((f.qty for f in report.fills), Decimal(0)) > order.intended_qty:
                problems.append("fills exceed intended quantity")
        if problems:
            self._set_status(order, OrderStatus.UNKNOWN, "; ".join(problems))
            self._halt(f"order {order.client_order_id}: malformed broker report")
            return
        for fill in report.fills:
            if fill.fill_id in {f.fill_id for f in order.fills}:
                continue  # duplicate fill report
            self._record("fill", {"fill": fill})
            self.account.apply_fill(fill)
            order.fills.append(fill)
        if report.status is not order.status:
            self._set_status(order, report.status, report.message)

    # --- reconciliation --------------------------------------------------------------------------
    def reconcile(self) -> ReconciliationResult:
        issues: list[str] = []
        unreachable = False
        for order in [o for o in self.orders.values() if o.status not in TERMINAL]:
            try:
                report = self.broker.get_order(order.client_order_id)
            except BrokerError as exc:
                issues.append(f"cannot query {order.client_order_id}: {exc}")
                unreachable = True
                continue
            if report is None:
                self._set_status(order, OrderStatus.NOT_FOUND, "venue has no record of this order")
            else:
                self._apply_report(order, report)
                if order.status not in TERMINAL:
                    issues.append(f"{order.client_order_id} still {order.status.value}")
        try:
            external = self.broker.get_account()
        except BrokerError as exc:
            issues.append(f"cannot read venue account: {exc}")
            unreachable = True
            external = None
        if external is not None:
            if abs(external.cash - self.account.cash) > CASH_TOLERANCE:
                issues.append(f"cash mismatch: venue {external.cash} vs internal {self.account.cash}")
            internal_pos = {s: p.qty for s, p in self.account.positions.items() if p.qty}
            for symbol in sorted(set(internal_pos) | set(external.positions)):
                a, b = internal_pos.get(symbol, Decimal(0)), external.positions.get(symbol, Decimal(0))
                if a != b:
                    issues.append(f"position mismatch {symbol}: venue {b} vs internal {a}")
        state = ReconState.OK if not issues else (ReconState.UNKNOWN if unreachable else ReconState.MISMATCH)
        result = ReconciliationResult(state, tuple(issues), self.clock.now())
        self._record("reconciliation", {"state": state, "issues": list(issues)})
        self.recon_state = state
        if state is ReconState.OK:
            if self.halted:
                self._record("resume", {"reason": "reconciled: internal state matches venue"})
                self.halted_reason = None
        else:
            self._halt(f"reconciliation {state.value}: " + "; ".join(issues))
        return result

    def record_external_adjustment(self, cash_delta: Decimal, reason: str, operator_ack: str) -> None:
        """Explicit, audited acceptance of a verified external cash change (e.g. a deposit)."""
        if operator_ack != "OPERATOR: external adjustment verified":
            raise PermissionError("external adjustments require operator acknowledgement")
        self._record("external_adjustment", {"cash_delta": cash_delta, "reason": reason})
        self.account.cash += cash_delta

    # --- views -------------------------------------------------------------------------------------
    def portfolio_snapshot(self, marks: dict[str, Decimal], day_start_equity: Decimal, peak_equity: Decimal) -> PortfolioSnapshot:
        known = not self.halted and self.recon_state is ReconState.OK
        try:
            equity = self.account.equity(marks)
        except KeyError:
            known, equity = False, self.account.cash
        positions = tuple((s, p.qty, marks.get(s, Decimal(0))) for s, p in sorted(self.account.positions.items()) if p.qty)
        return PortfolioSnapshot(self.clock.now(), self.mode, equity, self.account.cash, positions, day_start_equity,
                                 max(peak_equity, equity), self.recon_state, known)
