"""Resumable autonomous loop.

One ``tick()`` runs one full cycle:

    WAKE → HEALTH → DATA → RECONCILE → RISK STATE → MONITOR (stops) → SCAN (champion signal)
         → [Claude proposal → adversarial review → risk engine → decision record → execution]
         → POST-TRADE REVIEW → MEMORY → END

Design for interruption: every tick is bracketed by journaled ``tick_start``/``tick_end`` entries.
If the previous tick has a start and no end, the process died mid-tick; the new tick records that
and — like every tick — reconciles against the venue before doing anything that could add risk.
Any stage failure ends the tick without new risk. UNKNOWN → STOP → RECONCILE → VERIFY → RESUME.

Scheduling is external (Routines / cron / a Claude session); the loop keeps no long-lived process
state that is not in a journal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal

from ati.agent.pipeline import Outcome
from ati.agent.reasoning import ReasoningBudgetExceeded, ReasoningPending
from ati.agent.roles import POST_TRADE, build_prompt
from ati.agent.schema import parse_post_trade_output
from ati.core.canonical import sha256_hex
from ati.core.errors import AtiError, DataIntegrityError, ExecutionHalted, ProviderError, SchemaViolation
from ati.core.time import to_iso
from ati.core.types import Side
from ati.data.dataset import Dataset
from ati.decision.records import FinalDecision
from ati.execution.broker import OrderStatus
from ati.ledger.journal import Journal, decode
from ati.market.validation import check_freshness
from ati.memory.store import MemoryEntry, MemoryKind, fingerprint
from ati.risk.engine import MarketSnapshot, ReconState
from ati.strategies.base import Target
from ati.system import DEFAULT_HISTORY_BARS, STALE_AFTER, System


@dataclass
class TickReport:
    tick: int
    resumed_after_interruption: bool = False
    stages: list[str] = field(default_factory=list)
    data_ok: bool = False
    reconciliation: str = "NOT_RUN"
    actions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stopped_at: str | None = None


class AutonomousLoop:
    def __init__(self, system: System, history_bars: int = DEFAULT_HISTORY_BARS):
        self.s = system
        self.history_bars = history_bars
        self.journal = Journal(system.state_dir / "loop.jsonl", kind="loop", clock=system.clock, guard=system.guard,
                               attrs={"mode": system.mode.value, "data_status": system.data_status.value})
        self.ticks = 0
        self.open_tick = False
        self.day = None
        self.day_start_equity: Decimal | None = None
        self.peak_equity = Decimal(0)
        self.stops: dict[str, tuple[Decimal, str, str]] = {}  # symbol → (stop, strategy_key, strategy_hash)
        self.entry_decisions: dict[str, str] = {}
        self.mistake_flags: dict[str, list[str]] = {}
        self.last_action = "none"
        self.current_stage = "IDLE"
        self._restore()

    # --- durable state --------------------------------------------------------------------------
    def _restore(self) -> None:
        for e in self.journal.entries():
            p = decode(e.payload)
            if e.type == "tick_start":
                self.ticks, self.open_tick = p["tick"], True
            elif e.type == "tick_end":
                self.open_tick = False
                self.last_action = p.get("last_action", self.last_action)
            elif e.type == "day_start":
                self.day, self.day_start_equity = p["day"], p["equity"]
            elif e.type == "peak_equity":
                self.peak_equity = p["equity"]
            elif e.type == "stop_set":
                self.stops[p["symbol"]] = (p["stop"], p["strategy_key"], p["strategy_hash"])
                self.entry_decisions[p["symbol"]] = p["decision_id"]
            elif e.type == "stop_cleared":
                self.stops.pop(p["symbol"], None)
            elif e.type == "mistake_flag":
                flags = self.mistake_flags.setdefault(p["fingerprint"], [])
                if p["review_id"] not in flags:
                    flags.append(p["review_id"])

    def _log(self, type_: str, payload: dict) -> None:
        self.journal.append(type_, payload)

    # --- one cycle ------------------------------------------------------------------------------
    def tick(self) -> TickReport:
        s = self.s
        resumed = self.open_tick
        self.ticks += 1
        report = TickReport(self.ticks, resumed_after_interruption=resumed)
        self._log("tick_start", {"tick": self.ticks, "resumed_after_interruption": resumed})
        self.open_tick = True
        if hasattr(s.reasoning, "budget"):
            s.reasoning.budget.reset()
        try:
            self._run(report)
        except (ExecutionHalted, AtiError) as exc:
            report.errors.append(f"{type(exc).__name__}: {exc}")
            report.stopped_at = self.current_stage
        self._log("tick_end", {"tick": self.ticks, "stages": report.stages, "actions": report.actions,
                               "errors": report.errors, "stopped_at": report.stopped_at, "last_action": self.last_action})
        self.open_tick = False
        self.current_stage = "WAIT"
        return report

    def _stage(self, report: TickReport, name: str) -> None:
        self.current_stage = name
        report.stages.append(name)

    def _run(self, report: TickReport) -> None:
        s = self.s
        now = s.clock.now()
        self._stage(report, "HEALTH")
        s.execution.journal.verify()
        s.decisions.journal.verify()
        datasets = self.refresh_data(report, now)
        if not self.reconcile(report):
            return
        if not report.data_ok:
            report.stopped_at = "DATA"
            return
        marks = self.update_risk_state(report, datasets, now)
        for symbol, ds in datasets.items():
            champion = s.strategies.champion(symbol, s.timeframe)   # this symbol's champion on this timeframe only
            candidate = self.manage_position(report, symbol, ds, marks, champion)
            if candidate is None:
                continue
            view, pf, mkt = candidate
            if s.kill_switch.engaged:
                report.actions.append(f"{symbol}: LONG signal ignored — kill switch engaged")
                continue
            self._stage(report, "DECIDE")
            self.record_entry(report, symbol, s.pipeline.entry(champion, view, pf, mkt), champion)

    # --- deterministic stages (shared with ati.company.control) --------------------------------------
    def refresh_data(self, report: TickReport, now) -> dict[str, Dataset]:
        """Fetch → archive → store → validated dataset per symbol. Any failure leaves ``data_ok`` False."""
        s = self.s
        self._stage(report, "DATA")
        datasets: dict[str, Dataset] = {}
        self.last_data_error: Exception | None = None
        try:
            for symbol in s.symbols:
                start = now - s.timeframe.delta * self.history_bars
                start = start - timedelta(microseconds=start.microsecond)
                start = start - timedelta(seconds=int(start.timestamp()) % s.timeframe.seconds)  # bar-aligned
                closed = [c for c in s.provider.fetch_candles(symbol, s.timeframe, start, now) if c.is_closed]
                if not closed:
                    raise DataIntegrityError("EMPTY", f"no closed candles for {symbol}")
                if closed[0].status is not s.data_status:
                    raise DataIntegrityError("STATUS_MIX", f"provider returned {closed[0].status.value}, system is {s.data_status.value}")
                s.archive.ingest(s.store, closed)
                series = s.store.series(s.provider.name, symbol, s.timeframe)[-self.history_bars:]
                check_freshness(series[-1], now, STALE_AFTER)
                datasets[symbol] = Dataset.build(series, data_version="live-window", realization="observed")
            report.data_ok = True
        except (ProviderError, DataIntegrityError, AtiError) as exc:
            self.last_data_error = exc
            report.errors.append(f"data: {type(exc).__name__}: {exc}")
            self._log("data_unhealthy", {"error": str(exc)[:500]})
        return datasets

    def reconcile(self, report: TickReport) -> bool:
        self._stage(report, "RECONCILE")
        recon = self.s.execution.reconcile()
        report.reconciliation = recon.state.value
        if recon.state is not ReconState.OK:
            report.stopped_at = "RECONCILE"
            return False
        return True

    def update_risk_state(self, report: TickReport, datasets: dict[str, Dataset], now) -> dict[str, Decimal]:
        self._stage(report, "RISK_STATE")
        marks = {sym: ds.candles[-1].close for sym, ds in datasets.items()}
        equity = self.s.execution.account.equity(marks)
        day = now.date().isoformat()
        if self.day != day:
            self.day, self.day_start_equity = day, equity
            self._log("day_start", {"day": day, "equity": equity})
        if equity > self.peak_equity:
            self.peak_equity = equity
            self._log("peak_equity", {"equity": equity})
        return marks

    def manage_position(self, report: TickReport, symbol: str, ds: Dataset, marks, champion):
        """Deterministic position management: stop exits, signal exits, stop ratchet. Returns
        ``(view, portfolio, market)`` only when the symbol is flat and the champion signals LONG."""
        s = self.s
        pf = s.execution.portfolio_snapshot(marks, self.day_start_equity, self.peak_equity)
        mkt = self._market(symbol, ds)
        held = pf.qty(symbol)
        self._stage(report, "MONITOR")
        if held > 0 and symbol in self.stops:
            stop, key, shash = self.stops[symbol]
            if ds.candles[-1].low <= stop:
                self._exit(report, symbol, "stop", key, shash, ds, pf, mkt)
                return None
        if held == 0 and symbol in self.stops:
            self._log("stop_cleared", {"symbol": symbol, "reason": "no position"})
            self.stops.pop(symbol)
        self._stage(report, "SCAN")
        if champion is None:
            report.actions.append(f"{symbol}: no champion — research only, no trading")
            return None
        # Cutoff = last closed bar, so the same information always maps to the same decision id.
        view = ds.view_at(ds.candles[-1].close_time, champion.lookback)
        signal = champion.signal(view, held > 0)
        if held > 0 and signal.target is Target.FLAT:
            self._exit(report, symbol, "signal", champion.key, champion.definition_hash, ds, pf, mkt)
            return None
        if held > 0 and signal.stop_price and symbol in self.stops and signal.stop_price > self.stops[symbol][0]:
            self._set_stop(symbol, signal.stop_price, champion.key, champion.definition_hash, self.entry_decisions.get(symbol, ""))
        if held == 0 and signal.target is Target.LONG:
            return view, pf, mkt
        return None

    def record_entry(self, report: TickReport, symbol: str, result, champion) -> None:
        if result.outcome is Outcome.PENDING_REASONING:
            report.actions.append(f"{symbol}: awaiting Claude reasoning (no risk taken)")
        elif result.outcome is Outcome.RECORDED:
            rec = result.decision
            report.actions.append(f"{symbol}: decision {rec.decision_id} → {rec.final_decision.value}")
            self.last_action = f"{rec.final_decision.value} {symbol}"
            order = result.order
            if order is not None and order.filled_qty > 0 and order.side is Side.BUY:
                self._set_stop(symbol, rec_stop(result), champion.key, champion.definition_hash, rec.decision_id)

    # --- helpers --------------------------------------------------------------------------------
    def _market(self, symbol: str, ds: Dataset) -> MarketSnapshot:
        last = ds.candles[-1]
        return MarketSnapshot(symbol, ds.identity.timeframe, last.close, last.close_time, last.status,
                              sum((c.volume for c in ds.candles[-24:]), Decimal(0)),
                              self.s.costs.half_spread_rate, self.s.costs.slippage_rate)

    def _set_stop(self, symbol, stop, key, shash, decision_id) -> None:
        self.stops[symbol] = (stop, key, shash)
        self.entry_decisions[symbol] = decision_id
        self._log("stop_set", {"symbol": symbol, "stop": stop, "strategy_key": key, "strategy_hash": shash,
                               "decision_id": decision_id})

    def _exit(self, report, symbol, why, key, shash, ds, pf, mkt) -> None:
        self._stage(report, "EXIT")
        result = self.s.pipeline.exit(key, shash, why, ds.candles[-1].close_time, pf, mkt)
        if result.outcome is not Outcome.RECORDED:
            return
        report.actions.append(f"{symbol}: exit ({why}) → {result.decision.final_decision.value}")
        self.last_action = f"EXIT {symbol} ({why})"
        order = result.order
        if order is not None and order.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_CANCELED):
            if self.s.execution.account.position_qty(symbol) == 0:
                self._log("stop_cleared", {"symbol": symbol, "reason": why})
                self.stops.pop(symbol, None)
            self._stage(report, "POST_TRADE_REVIEW")
            self._review(report, symbol, why, order, ds)

    def _review(self, report, symbol, why, exit_order, ds) -> None:
        """Deterministic facts first; Claude's process review second (LIGHT tier, optional)."""
        s = self.s
        entry_id = self.entry_decisions.get(symbol, "")
        entry_orders = [o for o in s.execution.orders.values() if o.decision_id == entry_id]
        entry_px = entry_orders[0].avg_fill_price if entry_orders else None
        facts = {"symbol": symbol, "entry_decision_id": entry_id, "exit_reason": why,
                 "entry_price": entry_px, "exit_price": exit_order.avg_fill_price, "qty": exit_order.filled_qty,
                 "fees": exit_order.fees + (entry_orders[0].fees if entry_orders else Decimal(0))}
        review_id = "rev_" + sha256_hex(facts)[:20]
        now = s.clock.now()
        s.evidence.register("trade_review", review_id, now, f"{symbol} exit {why}")
        self._log("trade_review", {"review_id": review_id, "facts": facts})
        try:
            raw = s.reasoning.complete(POST_TRADE.name, review_id,
                                       build_prompt(POST_TRADE, {"facts": {k: str(v) for k, v in facts.items()},
                                                                 "note": "judge process, not outcome"}, None, s.guard))
            review = parse_post_trade_output(raw)
        except (ReasoningPending, ReasoningBudgetExceeded, SchemaViolation) as exc:
            report.actions.append(f"{symbol}: post-trade reasoning unavailable ({type(exc).__name__})")
            return
        self._log("post_trade_review", {"review_id": review_id, "process_quality": review.process_quality,
                                        "notes": review.notes[:600], "possible_mistake": review.possible_mistake})
        if review.possible_mistake:
            self._stage(report, "MEMORY")
            fp = fingerprint(review.possible_mistake)
            self._log("mistake_flag", {"fingerprint": fp, "review_id": review_id})
            flags = self.mistake_flags.setdefault(fp, [])
            if review_id not in flags:
                flags.append(review_id)
            refs = tuple(s.evidence.resolve(r) for r in flags)
            if len(set(flags)) >= 2 and s.memory.accepts_doctrine:
                entry = MemoryEntry(MemoryKind.MISTAKE, review.possible_mistake, now, "claude:post_trade_reviewer",
                                    refs, 0.6)
            elif len(set(flags)) >= 2:
                # Non-market data (e.g. MOCK): keep the learning, never as doctrine.
                entry = MemoryEntry(MemoryKind.HYPOTHESIS,
                                    f"[{s.memory.data_status}, not doctrine] recurring possible mistake: "
                                    f"{review.possible_mistake}", now, "claude:post_trade_reviewer", refs, 0.3)
            else:
                entry = MemoryEntry(MemoryKind.HYPOTHESIS, f"possible mistake: {review.possible_mistake}", now,
                                    "claude:post_trade_reviewer", refs, 0.3)
            s.memory.add(entry)
            report.actions.append(f"memory: {entry.kind.value} recorded")


def rec_stop(result) -> Decimal:
    return result.verdict.stop_price
