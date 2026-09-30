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
- Pre-submission gates (after the signed risk approval, before anything is written or sent), each refusal journaled
  as ``submission_refused`` with its rule, the decision id and the relevant state: execution mode (OBSERVE refuses
  everything), the LIVE_TRADING build constant, the kill switch (new risk), a fresh successful reconciliation, no
  other working/unresolved order for the symbol, operator approval (ASSISTED), autonomous limits
  (AUTONOMOUS_LIMITED), and the broker's own health check. Claude reaches none of these: it only ever supplies a
  proposal that the risk engine may turn into a signed approval.
- Venue acknowledgement and fills are recorded separately; ``cancel`` is write-ahead (CANCEL_REQUESTED) and an
  unconfirmed cancel is UNKNOWN; reconciliation also compares the venue's full order list (unexpected orders,
  missing/unexpected fills, quantity and price mismatches).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from ati.config import LIVE_TRADING, OperatingMode, assert_mode_permitted
from ati.core.errors import (ApprovalInvalid, BrokerError, BrokerRejected, ExecutionHalted, JournalWriteError,
                             ModeMismatch, SubmissionRefused)
from ati.core.time import Clock
from ati.core.types import Side
from ati.execution.broker import OPEN, TERMINAL, UNRESOLVED, Broker, BrokerOrderReport, Order, OrderStatus
from ati.execution.policy import APPROVAL_ACK, ExecutionMode, ExecutionPolicy
from ati.ledger.accounting import Account, Fill
from ati.ledger.journal import Journal, decode
from ati.market.models import DataStatus
from ati.risk.engine import ApprovalAuthority, PortfolioSnapshot, ReconState, RiskVerdict
from ati.risk.killswitch import KillSwitch
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
                 mode: OperatingMode, data_status: DataStatus, initial_cash: Decimal, kill_switch: KillSwitch,
                 policy: ExecutionPolicy = ExecutionPolicy(), guard: SecretGuard | None = None):
        assert_mode_permitted(mode)
        if not isinstance(kill_switch, KillSwitch):
            raise TypeError("the execution engine requires the system kill switch")
        if not isinstance(policy, ExecutionPolicy):
            raise TypeError("execution policy must be an ExecutionPolicy")
        if broker.mode is not mode:
            raise ModeMismatch(f"{broker.mode.value} broker wired into {mode.value} execution engine")
        self.broker = broker
        self.authority = authority
        self.clock = clock
        self.mode = mode
        self.journal = Journal(journal_path, kind="execution", clock=clock, guard=guard,
                               attrs={"mode": mode.value, "data_status": data_status.value, "initial_cash": str(initial_cash)})
        self.account = Account(mode, data_status, initial_cash)
        self.kill_switch = kill_switch
        self.policy = policy
        self.orders: dict[str, Order] = {}
        self.approvals: set[str] = set()            # decision ids an operator approved (ASSISTED)
        self.halted_reason: str | None = None
        self.recon_state = ReconState.UNKNOWN
        self.last_reconciled_at: datetime | None = None   # in memory only: a restarted process must reconcile first
        self.last_refusal: dict | None = None
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
                order.broker_order_id = p.get("broker_order_id") or order.broker_order_id
            elif entry.type == "operator_approval":
                self.approvals.add(p["decision_id"])
            elif entry.type == "submission_refused":
                self.last_refusal = p
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

    # --- operator ----------------------------------------------------------------------------------
    def approve(self, decision_id: str, acknowledgement: str) -> None:
        """ASSISTED mode: an operator approves one decision id. Journaled; survives restarts. Not reachable from any
        Claude action. The risk approval itself still expires after its TTL — an approval never revives it."""
        if acknowledgement != APPROVAL_ACK:
            raise PermissionError("operator approval requires the exact acknowledgement")
        if decision_id not in self.approvals:
            self._record("operator_approval", {"decision_id": decision_id})
            self.approvals.add(decision_id)

    # --- submission ----------------------------------------------------------------------------
    def _refuse(self, rule: str, reason: str, verdict: RiskVerdict | None, state: dict | None = None,
                exc: type[Exception] = SubmissionRefused):
        record = {"decision_id": getattr(verdict, "decision_id", None), "rule": rule, "reason": reason[:500],
                  "symbol": getattr(verdict, "symbol", None), "side": getattr(verdict, "side", None),
                  "authorized_qty": getattr(verdict, "qty", None), "execution_mode": self.policy.mode,
                  "venue_mode": self.mode, "at": self.clock.now(), "state": state or {}}
        self._record("submission_refused", record)
        self.last_refusal = record
        if exc is SubmissionRefused:
            raise SubmissionRefused(rule, reason)
        raise exc(reason)

    def _gates(self, v: RiskVerdict) -> None:
        now = self.clock.now()
        new_risk = v.side is Side.BUY
        if self.policy.mode is ExecutionMode.OBSERVE:
            self._refuse("execution_mode", "OBSERVE mode: proposals are analysed, never submitted", v)
        if self.mode is OperatingMode.LIVE and not LIVE_TRADING:     # unreachable by construction; checked anyway
            self._refuse("live_trading_disabled", "LIVE_TRADING is False in this build", v)
        if new_risk and self.kill_switch.engaged:
            ks = self.kill_switch.state()
            self._refuse("kill_switch", f"kill switch engaged: {ks.get('reason')}", v, {"kill_switch": ks})
        age = None if self.last_reconciled_at is None else now - self.last_reconciled_at
        if self.recon_state is not ReconState.OK or age is None or age > self.policy.max_account_age:
            self._refuse("account_state_fresh", f"account state not freshly reconciled (state {self.recon_state.value}, "
                                                f"age {age}, limit {self.policy.max_account_age})", v,
                         {"reconciliation": self.recon_state, "age_seconds": age.total_seconds() if age else None})
        working = [o.client_order_id for o in self.orders.values() if o.symbol == v.symbol and o.status not in TERMINAL]
        if working:
            self._refuse("open_order_exists", f"{len(working)} working/unresolved order(s) for {v.symbol}", v,
                         {"orders": working})
        if new_risk and self.policy.mode is ExecutionMode.ASSISTED and v.decision_id not in self.approvals:
            self._refuse("operator_approval_required", "ASSISTED mode: no operator approval for this decision", v)
        if new_risk and self.policy.mode is ExecutionMode.AUTONOMOUS_LIMITED:
            notional = v.qty * v.entry_price
            today = sum(1 for o in self.orders.values() if o.created_at.date() == now.date() and o.side is Side.BUY)
            problems = ([f"{v.symbol} not in autonomous symbols"] if v.symbol not in self.policy.autonomous_symbols else []) \
                + ([f"notional {notional:.2f} > {self.policy.autonomous_max_order_notional}"]
                   if notional > self.policy.autonomous_max_order_notional else []) \
                + ([f"{today} orders today >= {self.policy.autonomous_max_orders_per_day}"]
                   if today >= self.policy.autonomous_max_orders_per_day else [])
            if problems:
                self._refuse("autonomous_limit", "; ".join(problems), v)
        try:
            health = self.broker.health_check()
        except Exception as exc:                                    # a health check that cannot answer is unhealthy
            self._refuse("broker_health", f"health check failed: {type(exc).__name__}", v)
        if not health.ok:
            failed = [f"{n}: {d}" for n, ok, d in health.checks if not ok]
            self._refuse("broker_health", "broker unhealthy: " + "; ".join(failed), v, {"checks": failed})

    def submit(self, verdict: RiskVerdict) -> Order:
        if self.halted:
            self._refuse("halted", str(self.halted_reason), verdict if isinstance(verdict, RiskVerdict) else None,
                         exc=ExecutionHalted)
        try:
            v = self.authority.verify(verdict)
        except ApprovalInvalid as exc:
            self._refuse("risk_approval", str(exc), None, exc=ApprovalInvalid)
        if not v.approved or v.qty <= 0:
            self._refuse("risk_approval", "verdict is not an approval", v, exc=ApprovalInvalid)
        if v.mode is not self.mode:
            self._refuse("venue_mode", f"{v.mode.value} approval presented to {self.mode.value} execution", v,
                         exc=ModeMismatch)
        if self.clock.now() > v.valid_until:
            self._refuse("risk_approval", "risk approval expired; re-evaluate risk", v, exc=ApprovalInvalid)
        coid = client_order_id(v.decision_id)
        if coid in self.orders:
            return self.orders[coid]  # idempotent: never submit twice for one decision
        self._gates(v)
        order = Order(coid, v.decision_id, v.strategy_key, v.symbol, v.side, v.qty, v.entry_price, self.clock.now(), self.mode)
        self._record("order_intent", {"client_order_id": coid, "decision_id": v.decision_id, "strategy_key": v.strategy_key,
                                      "symbol": v.symbol, "side": v.side, "intended_qty": v.qty,
                                      "intended_price": v.entry_price, "created_at": order.created_at, "mode": self.mode,
                                      "execution_mode": self.policy.mode, "risk_limits_hash": v.limits_hash})
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

    def _set_status(self, order: Order, status: OrderStatus, message: str, broker_order_id: str = "") -> None:
        self._record("order_status", {"client_order_id": order.client_order_id, "status": status, "message": message,
                                      "broker_order_id": broker_order_id or order.broker_order_id})
        order.status, order.message = status, message
        order.broker_order_id = broker_order_id or order.broker_order_id

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
        if order.status is OrderStatus.CANCEL_REQUESTED and report.status in OPEN:
            return          # the venue has not processed the cancel yet: keep CANCEL_REQUESTED (unresolved)
        if report.status is not order.status or (report.broker_order_id and report.broker_order_id != order.broker_order_id):
            self._set_status(order, report.status, report.message, report.broker_order_id)

    # --- cancellation ------------------------------------------------------------------------------
    def cancel(self, client_order_id: str, reason: str) -> Order:
        """Cancel a working order. Write-ahead CANCEL_REQUESTED; an unconfirmed cancel is UNKNOWN and halts."""
        order = self.orders.get(client_order_id)
        if order is None:
            raise KeyError(client_order_id)
        if order.status not in OPEN:
            return order                                   # nothing to cancel (terminal, or already requested)
        self._set_status(order, OrderStatus.CANCEL_REQUESTED, reason[:200])
        try:
            report = self.broker.cancel_order(client_order_id)
        except BrokerError as exc:
            self._set_status(order, OrderStatus.UNKNOWN, f"cancel outcome unknown: {exc}")
            self._halt(f"order {client_order_id}: cancel could not be confirmed")
            return order
        except Exception as exc:
            self._set_status(order, OrderStatus.UNKNOWN, f"cancel outcome unknown: unexpected {exc!r}"[:300])
            self._halt(f"order {client_order_id}: cancel could not be confirmed")
            return order
        self._apply_report(order, report)
        return order

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
                if order.status is OrderStatus.CANCEL_REQUESTED or order.fills:
                    issues.append(f"{order.client_order_id}: venue lost an order it had acknowledged/filled")
                else:
                    self._set_status(order, OrderStatus.NOT_FOUND, "venue has no record of this order")
            else:
                self._apply_report(order, report)
                if order.status in UNRESOLVED:
                    issues.append(f"{order.client_order_id} still {order.status.value}")
        try:
            venue_orders = {r.client_order_id: r for r in self.broker.list_orders()}
        except BrokerError as exc:
            issues.append(f"cannot list venue orders: {exc}")
            unreachable = True
            venue_orders = None
        if venue_orders is not None:
            issues += self._compare_orders(venue_orders)
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
        self.last_reconciled_at = self.clock.now() if state is ReconState.OK else None
        if state is ReconState.OK:
            if self.halted:
                self._record("resume", {"reason": "reconciled: internal state matches venue"})
                self.halted_reason = None
        else:
            self._halt(f"reconciliation {state.value}: " + "; ".join(issues))
        return result

    def _compare_orders(self, venue: dict[str, BrokerOrderReport]) -> list[str]:
        """Venue order list vs internal orders: nothing TradeTown did not send, no fill it does not know, no fill it
        knows that the venue does not, same quantities and prices. Differences are reported, never auto-applied."""
        issues = []
        for coid in sorted(set(venue) - set(self.orders)):
            issues.append(f"unexpected venue order {coid} ({venue[coid].status.value}, filled {venue[coid].filled_qty})")
        for coid, order in sorted(self.orders.items()):
            report = venue.get(coid)
            if report is None:
                if order.fills:
                    issues.append(f"{coid}: {len(order.fills)} internal fill(s) the venue does not list")
                continue
            internal = {f.fill_id: f for f in order.fills}
            external = {f.fill_id: f for f in report.fills}
            for fid in sorted(set(external) - set(internal)):
                issues.append(f"unexpected fill {fid} on {coid} ({order.status.value})")
            for fid in sorted(set(internal) - set(external)):
                issues.append(f"missing fill {fid} on {coid}: recorded internally, not at the venue")
            for fid in sorted(set(internal) & set(external)):
                a, b = internal[fid], external[fid]
                if (a.qty, a.price) != (b.qty, b.price):
                    issues.append(f"fill {fid} differs: venue {b.qty}@{b.price} vs internal {a.qty}@{a.price}")
            if report.filled_qty != order.filled_qty:
                issues.append(f"{coid} quantity mismatch: venue {report.filled_qty} vs internal {order.filled_qty}")
            if order.status in TERMINAL and report.status not in TERMINAL:
                issues.append(f"{coid}: internal {order.status.value} but venue {report.status.value}")
        return issues

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
