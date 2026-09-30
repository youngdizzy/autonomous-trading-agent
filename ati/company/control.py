"""Company control plane — Company 1.0 Phase 1.

One ``run_cycle()`` call is one bounded company cycle:

    START → RECOVER → deterministic duties (data refresh, reconciliation, risk state, stop/signal exits)
          → HEALTH_CHECK → READ_STATE / PREPARE_CONTEXT → CLAUDE_ACTION (file exchange)
          → VALIDATE_ACTION (closed schema + health gate) → EXECUTE_ALLOWED_ACTION (existing systems only)
          → RECORD_RESULT → LEARN (deterministic outcome learning, ati.company.learning) → FINALIZE

Authority stays where it was: data truth in market/archive, research integrity in ResearchLog and the
research workflow, sizing and approval in the RiskEngine, orders in the ExecutionEngine, accounting in
the ledger, knowledge in MemoryStore. The control plane only decides *whether a validated Claude action
may be routed* to one of those systems, and records what happened.

Persistence is one hash-chained journal (``company.jsonl``, the existing Journal class): state
transitions, pause/resume, and each cycle's steps. Everything else is derived from existing state.

Idempotency: a cycle's id is derived from a fingerprint of the company's observable state, so the same
state maps to the same cycle, and a finished cycle is REPLAYed, never re-run. Within a cycle every step is
journaled once and recovery resumes after the last recorded step; downstream idempotency (decision ids →
client order ids, one run per hypothesis) remains authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from ati.agent.loop import AutonomousLoop, TickReport
from ati.agent.pipeline import Outcome
from ati.agent.reasoning import ReasoningBudgetExceeded, ReasoningPending
from ati.agent.roles import COMPANY, build_prompt
from ati.agent.schema import (COMPANY_ACTIONS, CompanyAction, CompanyContext, ValidationContext,
                              parse_company_response)
from ati.company import autonomy, budget, factory, objectives, scorecard
from ati.company.health import HealthReport, Status, assess
from ati.company.learning import LearningLedger
from ati.core.canonical import sha256_hex, sha256_text
from ati.core.errors import AtiError, CompanyStateError, HoldoutViolation, SchemaViolation
from ati.core.types import Side
from ati.execution.broker import TERMINAL
from ati.execution.policy import live_state
from ati.data.dataset import Dataset, Partition, sealed_ranges
from ati.decision.records import make_decision_id
from ati.ledger.journal import Journal, decode
from ati.market.conflict import ConflictRegister
from ati.market.health import contiguous_runs
from ati.memory.store import MemoryKind
from ati.research import conditions as conditions_mod
from ati.research import diagnostics
from ati.research import protocol as P
from ati.research import protocols as protocol_registry
from ati.research.adversarial import AdversarialPolicy
from ati.research.hypothesis import Criterion, PreRegistration, ResearchLog
from ati.research.walkforward import walk_forward
from ati.research.workflow import run_research_cycle
from ati.strategies.base import StrategyDefinition
from ati.strategies.registry import Lifecycle
from ati.validation.promotion import PromotionPolicy


class CompanyState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


_S = CompanyState
TRANSITIONS: dict[CompanyState, frozenset[CompanyState]] = {
    _S.IDLE: frozenset({_S.RUNNING, _S.PAUSED}),
    _S.COMPLETED: frozenset({_S.RUNNING, _S.PAUSED}),
    _S.BLOCKED: frozenset({_S.RUNNING, _S.PAUSED}),
    _S.FAILED: frozenset({_S.RUNNING, _S.PAUSED}),
    _S.RUNNING: frozenset({_S.COMPLETED, _S.BLOCKED, _S.FAILED, _S.PAUSED}),
    _S.PAUSED: frozenset({_S.RUNNING, _S.IDLE}),   # RUNNING = read-only cycle; IDLE = operator resume only
}
RESUME_ACK = "OPERATOR: company state reviewed; resume autonomous activity"
_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
# Routable research protocols (Phase 4B: every registered protocol resolves to its own timeframe deployment).
PROTOCOLS = {pid: p.strategy.key for pid, p in protocol_registry.REGISTRY.items() if p.executable}
PROTOCOL_PARAMS = {pid: dict(protocol_registry.REGISTRY[pid].base_params) for pid in PROTOCOLS}
CANDIDATE_EXPERIMENTS = frozenset({"SINGLE_VARIABLE", "INTERACTION", "STRUCTURAL"})  # full pipeline → challenger
DIAGNOSTIC_EXPERIMENTS = frozenset({"REGIME", "EXECUTION", "RISK"})                     # development-only, no candidate


@dataclass
class CycleRecord:
    cycle_id: str
    fingerprint: str
    steps: dict[str, dict] = field(default_factory=dict)
    status: str | None = None
    result: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CycleOutcome:
    cycle_id: str | None
    status: str          # COMPLETED | BLOCKED | FAILED | PAUSED | AWAITING_CLAUDE | REPLAY
    action: str | None
    detail: dict
    health: dict


class CompanyControlPlane:
    def __init__(self, system, *, loop: AutonomousLoop | None = None,
                 research_policies: tuple[AdversarialPolicy, PromotionPolicy] | None = None,
                 autonomy_level: autonomy.Autonomy = autonomy.Autonomy.PAPER_AUTONOMY):
        s = system
        self.autonomy = autonomy.require(autonomy_level)   # the ceiling is code, not config: live levels raise
        self.s = s
        self.loop = loop or AutonomousLoop(s)
        self.research_policies = research_policies or (AdversarialPolicy(), PromotionPolicy())
        self.journal = Journal(s.state_dir / "company.jsonl", kind="company", clock=s.clock, guard=s.guard,
                               attrs={"mode": s.mode.value, "data_status": s.data_status.value})
        attrs = {"mode": s.mode.value, "data_status": s.data_status.value}
        self.learning = LearningLedger(Journal(s.state_dir / "learning.jsonl", kind="learning", clock=s.clock,
                                               guard=s.guard, attrs=attrs))
        self.conflicts = ConflictRegister(Journal(s.state_dir / "conflicts.jsonl", kind="conflicts", clock=s.clock,
                                                  guard=s.guard, attrs=attrs))
        self.state = CompanyState.IDLE
        self.paused = False
        self.pause_epoch = 0
        self.cycles: dict[str, CycleRecord] = {}
        self.running: str | None = None
        self.last_finished: str | None = None
        self._replay()

    # --- durable state ------------------------------------------------------------------------------
    def _replay(self) -> None:
        for e in self.journal.entries():
            p = decode(e.payload)
            if e.type == "state":
                new = CompanyState(p["to"])
                if p["from"] != self.state.value or new not in TRANSITIONS[self.state]:
                    raise CompanyStateError(f"company journal holds illegal transition {p['from']} → {p['to']}")
                self.state = new
            elif e.type in ("pause", "resume"):
                self.paused = e.type == "pause"
                self.pause_epoch += 1
            elif e.type == "cycle_start":
                if self.running is not None:
                    raise CompanyStateError("company journal starts a cycle while another is running")
                self.cycles[p["cycle_id"]] = CycleRecord(p["cycle_id"], p["fingerprint"])
                self.running = p["cycle_id"]
            elif e.type == "cycle_step":
                cyc = self.cycles.get(p["cycle_id"])
                if cyc is None or cyc.status is not None or p["step"] in cyc.steps:
                    raise CompanyStateError(f"company journal step {p['step']} out of order")
                cyc.steps[p["step"]] = p["data"]
            elif e.type == "cycle_end":
                cyc = self.cycles.get(p["cycle_id"])
                if cyc is None or cyc.status is not None or self.running != p["cycle_id"]:
                    raise CompanyStateError("company journal ends a cycle that is not running")
                cyc.status, cyc.result = p["status"], p
                self.running, self.last_finished = None, p["cycle_id"]
        if (self.running is None) == (self.state is CompanyState.RUNNING):
            raise CompanyStateError(f"company state {self.state.value} contradicts running cycle {self.running}")

    def _transition(self, new: CompanyState, reason: str) -> None:
        if new not in TRANSITIONS[self.state]:
            raise CompanyStateError(f"illegal company transition {self.state.value} → {new.value}")
        self.journal.append("state", {"from": self.state.value, "to": new.value, "reason": reason[:300]})
        self.state = new

    def _step(self, cyc: CycleRecord, name: str, data: dict) -> None:
        self.journal.append("cycle_step", {"cycle_id": cyc.cycle_id, "step": name, "data": data})
        cyc.steps[name] = data

    # --- pause / resume -------------------------------------------------------------------------------
    def pause(self, source: str, reason: str) -> None:
        """Always permitted (stopping never needs permission). Persistent until an operator resumes."""
        if self.paused:
            return
        self.journal.append("pause", {"source": source, "reason": reason[:300]})
        self.paused, self.pause_epoch = True, self.pause_epoch + 1
        if self.state is not CompanyState.RUNNING:
            self._transition(CompanyState.PAUSED, f"paused by {source}")

    def resume(self, acknowledgement: str) -> None:
        """Operator only: no company action maps here, and Claude's vocabulary has no RESUME."""
        if acknowledgement != RESUME_ACK:
            raise PermissionError("resume requires the exact operator acknowledgement")
        if not self.paused or self.state is not CompanyState.PAUSED:
            raise CompanyStateError(f"cannot resume from {self.state.value} (paused={self.paused})")
        self.journal.append("resume", {"source": "operator"})
        self.paused, self.pause_epoch = False, self.pause_epoch + 1
        self._transition(CompanyState.IDLE, "operator resume")

    # --- one bounded cycle ------------------------------------------------------------------------------
    def run_cycle(self) -> CycleOutcome:
        s = self.s
        now = s.clock.now()
        report = TickReport(0)
        # Deterministic duties run every invocation, paused or not: risk-reducing exits never wait for Claude.
        datasets = self.loop.refresh_data(report, now)
        candidates: dict = {}
        marks: dict = {}
        if self.loop.reconcile(report) and report.data_ok:
            marks = self.loop.update_risk_state(report, datasets, now)
            for symbol, ds in datasets.items():
                cand = self.loop.manage_position(report, symbol, ds, marks, s.strategies.champion(symbol, s.timeframe))
                if cand is not None:
                    candidates[symbol] = cand
        health = assess(s, self.loop, datasets, self.journal, self.paused, self.conflicts,
                        extra_journals=(self.learning.journal, self.conflicts.journal))
        fingerprint = self._fingerprint(datasets, health)

        if self.running is not None:                                   # RECOVER
            cyc = self.cycles[self.running]
            if cyc.fingerprint != fingerprint and "action_started" not in cyc.steps:
                return self._finish(cyc, "BLOCKED", None, {
                    "reason": "stale: company state changed before any action executed; this cycle's response "
                              "will not be used"}, health)
        else:
            cycle_id = "cyc_" + fingerprint[:24]
            if cycle_id in self.cycles:
                done = self.cycles[cycle_id]
                return CycleOutcome(cycle_id, "REPLAY", done.result.get("action"),
                                    {"recorded_status": done.status}, health.as_dict())
            self._transition(CompanyState.RUNNING, f"cycle {cycle_id}")
            self.journal.append("cycle_start", {"cycle_id": cycle_id, "fingerprint": fingerprint,
                                                "paused": self.paused, "autonomy": self.autonomy.name})
            cyc = self.cycles[cycle_id] = CycleRecord(cycle_id, fingerprint)
            self.running = cycle_id

        if "health" not in cyc.steps:
            self._step(cyc, "health", health.as_dict())
        conditions = self._conditions(datasets)
        if "conditions" not in cyc.steps:
            self._step(cyc, "conditions", {"monitors": [{k: c[k] for k in ("assumption", "status", "statistic")}
                                                        for c in conditions]})
        allowed = [a for a in COMPANY_ACTIONS if not self._gate(a, health)]
        request_id = f"{cyc.cycle_id}:company"
        packet = self._context(cyc, request_id, health, datasets, candidates, marks, allowed, conditions)
        # Context identity: the hash of exactly what Claude is shown. Once issued it is fixed for this cycle, so a
        # response can only ever be bound to the context it was produced from.
        issued = cyc.steps.get("request_issued", {}).get("context_id")
        packet["context_id"] = issued or "ctx_" + sha256_hex(packet)[:24]
        prompt = build_prompt(COMPANY, packet, None, s.guard)

        # CLAUDE_ACTION — the response is read, hashed and bound to this request before anything else.
        try:
            raw = s.reasoning.complete(COMPANY.name, request_id, prompt)
        except ReasoningPending:
            if "request_issued" not in cyc.steps:
                self._issue(cyc, request_id, prompt, packet, allowed)
            return CycleOutcome(cyc.cycle_id, "AWAITING_CLAUDE", None, {"request_id": request_id}, health.as_dict())
        except (ReasoningBudgetExceeded, SchemaViolation) as exc:
            return self._finish(cyc, "FAILED", None, {"reason": f"CLAUDE_RESPONSE_REJECTED: {exc}"[:500]}, health)
        if "request_issued" not in cyc.steps:
            self._issue(cyc, request_id, prompt, packet, allowed)
        digest = sha256_text(raw)
        if "response" not in cyc.steps:
            self._step(cyc, "response", {"sha256": digest})
        elif cyc.steps["response"]["sha256"] != digest:
            return self._finish(cyc, "FAILED", None, {"reason": "CLAUDE_RESPONSE_REJECTED: response changed after it "
                                                                "was received (duplicate or tampered)"}, health)

        # VALIDATE_ACTION — closed schema, then the deterministic health gate.
        try:
            action = parse_company_response(raw, self._contract(cyc, request_id, health, datasets, packet["context_id"]))
        except SchemaViolation as exc:
            if "rejected" not in cyc.steps:
                self._step(cyc, "rejected", {"reason": str(exc)[:500]})
            return self._finish(cyc, "FAILED", None, {"reason": f"CLAUDE_RESPONSE_REJECTED: {exc}"[:500]}, health)
        if "action_result" in cyc.steps:   # crashed after recording the result: finish, never re-execute
            recorded = dict(cyc.steps["action_result"])
            return self._finish(cyc, recorded.pop("status"), action.action, recorded, health)
        reasons = self._gate(action.action, health)   # re-applied on recovery too: never looser
        if reasons:
            note = {"note": "blocked on recovery; downstream journals are authoritative for anything already done"} \
                if "action_started" in cyc.steps else {}
            return self._finish(cyc, "BLOCKED", action.action, {"reasons": reasons, **note}, health)
        if "validated" not in cyc.steps:
            self._step(cyc, "validated", {"action": action.action})

        # EXECUTE_ALLOWED_ACTION — only through existing systems.
        if "action_started" not in cyc.steps:
            started = {"action": action.action, "request_id": request_id, "context_id": packet["context_id"],
                       "strategy_fingerprints": self._champions(),
                       "datasets": {sym: ds.dataset_id for sym, ds in sorted(datasets.items())}}
            if action.trade is not None and action.symbol in candidates:
                started["cutoff"] = candidates[action.symbol][0].cutoff
            self._step(cyc, "action_started", started)
        try:
            status, detail = self._execute(cyc, action, health, datasets, candidates, marks, report)
        except ReasoningPending:
            return CycleOutcome(cyc.cycle_id, "AWAITING_CLAUDE", action.action,
                                {"reason": "awaiting adversarial review of the proposal"}, health.as_dict())
        except AtiError as exc:
            return self._finish(cyc, "FAILED", action.action, {"reason": f"{type(exc).__name__}: {exc}"[:500]}, health)
        self._step(cyc, "action_result", {"status": status, **detail})
        return self._finish(cyc, status, action.action, detail, health)

    # --- action routing ------------------------------------------------------------------------------------
    def _execute(self, cyc, action: CompanyAction, health: HealthReport, datasets, candidates, marks, report):
        s = self.s
        a = action.action
        if a == "NO_TRADE":
            return "COMPLETED", {"recorded": action.reason}
        if a == "PAUSE":
            self.pause("claude", action.reason)
            return "COMPLETED", {"paused": True}
        if a == "REQUEST_DATA":
            return "COMPLETED", {"need": action.need, "symbol": action.symbol, "data_state": health.data_state.value,
                                 "data_check": health.as_dict()["checks"]["data"]}
        if a == "REVIEW_POSITION":
            return "COMPLETED", {"positions": self._positions(marks, action.symbol)}
        if a == "REVIEW_RISK":
            return "COMPLETED", {"risk": self._risk_view(marks)}
        if a == "REVIEW_SYSTEM":
            return "COMPLETED", {"system": health.as_dict(), "company_state": self.state.value, "paused": self.paused}
        if a == "TRADE_PROPOSAL":
            return self._trade(cyc, action, candidates, report, datasets)
        if a == "RESEARCH_REQUEST":
            return self._research(cyc, action)
        raise CompanyStateError(f"no route for action {a!r}")  # unreachable: vocabulary is closed

    def _recover_trade(self, champion, proposal, cutoff, datasets) -> dict | None:
        """After a crash: if the decision for this proposal's cutoff is already journaled, report it (never
        re-route) and make sure a filled position is stop-protected. The stop is re-derived deterministically:
        the tighter of the proposal's stop and the champion's point-in-time stop at that same cutoff."""
        s = self.s
        decision_id = make_decision_id(champion.definition_hash, proposal.symbol, cutoff, "entry")
        if decision_id not in s.decisions:
            return None
        orders = [o for o in s.execution.orders.values() if o.decision_id == decision_id]
        final = next((decode(e.payload)["record"]["final_decision"] for e in s.decisions.journal.entries("decision")
                      if e.payload["decision_id"] == decision_id), None)
        detail = {"decision_id": decision_id, "final_decision": final,
                  "order": orders[0].status.value if orders else None, "recovered": True}
        filled = orders and orders[0].filled_qty > 0 and s.execution.account.position_qty(proposal.symbol) > 0
        if filled and proposal.symbol not in self.loop.stops and proposal.symbol in datasets:
            signal = champion.signal(datasets[proposal.symbol].view_at(cutoff, champion.lookback), False)
            stops = [x for x in (proposal.stop_price, signal.stop_price) if x is not None]
            if stops:
                self.loop._set_stop(proposal.symbol, max(stops), champion.key, champion.definition_hash, decision_id)
                detail["stop_restored"] = str(max(stops))
        if final == "EXECUTE" and not orders:
            detail["note"] = "decision recorded but no order intent exists (interrupted); the approval is not resubmitted"
        return detail

    def _trade(self, cyc, action: CompanyAction, candidates, report, datasets):
        s = self.s
        proposal = action.trade
        champion = s.strategies.champion(proposal.symbol, s.timeframe)       # this symbol's champion, this timeframe
        if champion is not None and proposal.strategy_key != champion.key:
            return "BLOCKED", {"reason": f"{proposal.strategy_key} is not the {proposal.symbol} {s.timeframe.value} champion"}
        if proposal.side is not Side.BUY:
            return "BLOCKED", {"reason": "position reduction by Claude is NOT IMPLEMENTED in Phase 1; exits are deterministic"}
        started = cyc.steps["action_started"]
        if champion is not None and started.get("cutoff") is not None:
            recovered = self._recover_trade(champion, proposal, started["cutoff"], datasets)
            if recovered is not None:
                return "COMPLETED", recovered
        cand = candidates.get(proposal.symbol)
        if champion is None or cand is None:
            return "BLOCKED", {"reason": f"the champion does not signal an entry for {proposal.symbol} at this cutoff"}
        view, pf, mkt = cand
        if started.get("cutoff") is not None and started["cutoff"] != view.cutoff:
            return "BLOCKED", {"reason": "stale proposal: market information changed before it could be routed"}
        result = s.pipeline.route_proposal(champion, view, pf, mkt, proposal)
        if result.outcome is Outcome.PENDING_REASONING:
            raise ReasoningPending("adversarial review pending")
        decision_id = make_decision_id(champion.definition_hash, proposal.symbol, view.cutoff, "entry")
        if result.outcome is Outcome.RECORDED:
            self.loop.record_entry(report, proposal.symbol, result, champion)
        orders = [o for o in s.execution.orders.values() if o.decision_id == decision_id]
        final = next((decode(e.payload)["record"]["final_decision"] for e in s.decisions.journal.entries("decision")
                      if e.payload["decision_id"] == decision_id), None)
        detail = {"decision_id": decision_id, "final_decision": final,
                  "order": orders[0].status.value if orders else None,
                  "recovered": result.outcome is Outcome.ALREADY_DECIDED}
        if final == "EXECUTE" and not orders:
            detail["note"] = "decision recorded but no order intent exists (interrupted); the approval is not resubmitted"
        return "COMPLETED", detail

    def _research(self, cyc, action: CompanyAction):
        s = self.s
        spec = action.research
        exp = spec.experiment
        log = self._research_log()
        if log is None:
            return "BLOCKED", {"reason": "research journal cannot be reconstructed (research FAIL)"}
        # Routing is resolved by the system from the protocol id: symbol, timeframe and strategy deployment come from
        # the registry, never from Claude's other fields (which the schema already required to match).
        proto = protocol_registry.REGISTRY[spec.protocol_id]
        base = proto.strategy
        if log.status(spec.hypothesis_id) != "UNKNOWN":
            if "research_invoked" in cyc.steps:  # this cycle already ran it before a crash: report, never rerun
                detail = {"hypothesis_id": spec.hypothesis_id, "research_status": log.status(spec.hypothesis_id),
                          "recovered": True, "experiment_id": cyc.steps["research_invoked"].get("experiment_id")}
                if exp is not None and exp.type in CANDIDATE_EXPERIMENTS:
                    detail["comparison"] = self._compare(spec, base, log)
                detail.update(self._recover_candidate(spec, log, cyc.steps["research_invoked"].get("experiment_id")))
                self._link(spec)
                return "COMPLETED", detail
            return "BLOCKED", {"reason": f"{spec.hypothesis_id} is already registered ({log.status(spec.hypothesis_id)}); "
                                         "criteria are locked — a new question needs a new hypothesis id"}
        lc = spec.learning_candidate_id
        if lc is not None:
            cand = self.learning.candidates.get(lc)
            if cand is None:
                return "BLOCKED", {"reason": f"unknown learning candidate {lc}"}
            if cand.holdout_derived:
                return "BLOCKED", {"reason": f"{lc} is derived from holdout/promotion outcomes: holdout results never "
                                             "motivate new hypotheses"}
            if not cand.research_eligible:
                return "BLOCKED", {"reason": f"{lc} is {cand.state.value} with {len(cand.evidence)} outcome(s): "
                                             "not eligible for a hypothesis"}
        # The candidate definition (if any) — only from already-registered strategy logic, never new code.
        candidate_def, grid = None, [dict(g) for g in proto.grid]
        if exp is not None and exp.candidate_params is not None:
            grid = [dict(exp.candidate_params)]
        elif exp is not None and exp.structure is not None:
            kind, params = exp.structure
            if kind == base.kind:
                return "BLOCKED", {"reason": "STRUCTURAL experiments use a different strategy logic than the baseline"}
            try:
                candidate_def = StrategyDefinition.create(kind, 1, kind, dict(params), proto.timeframe, s.clock.now(),
                                                          description=f"structural candidate for {spec.hypothesis_id}")
            except (KeyError, ValueError, TypeError) as exc:
                return "BLOCKED", {"reason": f"STRUCTURAL candidate invalid (only registered strategy logic may be "
                                             f"used): {exc}"[:300]}
            grid = [candidate_def.param_dict]
        diagnostic = exp is not None and exp.type in DIAGNOSTIC_EXPERIMENTS
        observation_refs = self._observation_refs(spec)
        registered = self._registered(base)
        series = s.store.series(s.provider.name, proto.symbol, proto.timeframe)
        window = self._window(series, diagnostic, proto)
        dev_bars = window[2] if isinstance(window, tuple) else 0
        compute = dev_bars * (len(grid) + (1 if exp is not None else 0)) if exp is not None else 0
        new_variant = candidate_def is None and not diagnostic and any(
            self._registered_with(base, g) is None for g in grid if g != dict(proto.base_params))
        if "research_invoked" not in cyc.steps:
            u = budget.usage(s, self.journal, s.clock.now(), registered.definition_hash if registered else None,
                             statement=spec.statement)
            over = budget.check(u, new_variant, compute)
            if over:
                return "BLOCKED", {"reason": "research budget: " + "; ".join(over), "budget_policy": u["policy_hash"]}
        criteria = [Criterion(*c) for c in proto.criteria]  # the protocol's floor always holds: requests can only add
        for metric, op, threshold in spec.success_criteria:
            c = Criterion(metric, op, threshold)
            if c not in criteria:
                criteria.append(c)
        design = self._design(spec, base, registered, criteria, candidate_def, compute) if exp is not None else None
        exp_id = factory.experiment_id(spec.hypothesis_id, sha256_hex(design) if design else "protocol-default")
        if "research_invoked" not in cyc.steps:
            self._step(cyc, "research_invoked", {"hypothesis_id": spec.hypothesis_id, "criteria": len(criteria),
                                                 "experiment_id": exp_id,
                                                 "deployment": protocol_registry.deployment(proto),
                                                 "experiment_type": exp.type if exp else "PROTOCOL_GRID"})
        if window is None or isinstance(window, str):
            reason = window or "no stored data for the protocol symbol"
            s.research_journal.append("research_not_run", {"hypothesis_id": spec.hypothesis_id, "reasons": [reason]})
            return "COMPLETED", {"hypothesis_id": spec.hypothesis_id, "research_status": "NOT_RUN", "reasons": [reason],
                                 "experiment_id": exp_id}
        full, boundary, _ = window
        protocol = protocol_registry.REGISTRY[spec.protocol_id]
        # the protocol id AND its material hash are recorded with the run: historical research always references the
        # protocol that actually existed when it ran
        s.research_journal.append("protocol_run", {
            "protocol_id": protocol.protocol_id, "protocol_hash": protocol.protocol_hash,
            **{k: v for k, v in protocol_registry.deployment(protocol).items() if k not in ("protocol_id", "protocol_hash")},
            "strategy_id": protocol.strategy_id, "strategy_version": protocol.strategy_version,
            "kind": "diagnostic" if diagnostic else "validation", "hypothesis_id": spec.hypothesis_id,
            "experiment_id": exp_id, "symbol": protocol.symbol, "timeframe": protocol.timeframe.value,
            "dataset_id": full.dataset_id})
        if exp is not None:
            # Baseline discipline: the baseline definition is registered under its own key *before* the run, so
            # a candidate is derived as a new version with the baseline as parent — never stored as the baseline.
            registered = registered or s.strategies.register(base)
            design["baseline"]["fingerprint"] = registered.definition_hash
            # Design recorded before any pre-registration or test.
            s.research_journal.append("experiment_design", design | {"experiment_id": exp_id,
                                                                     "dataset_id": full.dataset_id})
        if diagnostic:
            detail = self._diagnostic(spec, exp, base, full, boundary, criteria, observation_refs)
            self._link(spec)
            return "COMPLETED", detail | {"experiment_id": exp_id}
        adversarial, promotion = self.research_policies
        r = run_research_cycle(s, full, boundary, hypothesis_id=spec.hypothesis_id, statement=spec.statement,
                               base=candidate_def or base, grid=grid, criteria=tuple(criteria),
                               train_bars=proto.train_bars, test_bars=proto.test_bars, adversarial_policy=adversarial,
                               promotion_policy=promotion, min_candles=proto.min_candles, observation_refs=observation_refs)
        detail = {"hypothesis_id": spec.hypothesis_id, "experiment_id": exp_id, "research_status": r.status,
                  "reasons": list(r.reasons), "dev_verdict": r.dev_verdict and r.dev_verdict.value,
                  "holdout_verdict": r.holdout_verdict and r.holdout_verdict.value,
                  "promotion_approved": r.promotion.approved if r.promotion else None,
                  "research_window": {"dataset_id": full.dataset_id, "bars": len(full)}}
        if r.challenger_key:
            detail["candidate"] = self._persist_candidate(spec.hypothesis_id, r.challenger_key, exp_id, full.dataset_id,
                                                          registered.definition_hash if registered else None, proto)
        if exp is not None and exp.type in CANDIDATE_EXPERIMENTS and r.dev_verdict is not None:
            detail["comparison"] = self._compare(spec, base, ResearchLog(s.research_journal), full, boundary)
        self._link(spec)
        return "COMPLETED", detail

    def _window(self, series, diagnostic: bool, proto=None):
        """Deterministic research window: the most recent contiguous run of stored bars that overlaps no sealed
        holdout period. A holdout that has been used is never reused, overlapped, or folded into development data.
        Returns (full, boundary, dev_bars), or a NOT_RUN reason string, or None when nothing is stored."""
        if not series:
            return None
        first = series[0]
        # the protocol is the one declared for this series' own (symbol, timeframe) — derived from the data
        proto = proto or protocol_registry.for_series(first.symbol, first.timeframe)
        if proto is None or (proto.symbol, proto.timeframe) != (first.symbol, first.timeframe):
            return f"no research protocol declared for {first.symbol} {first.timeframe.value}"
        sealed = sealed_ranges(first.provider, first.symbol, first.timeframe,
                               getattr(self.s.provider, "realization", "observed"))
        segments, current = [], []
        for c in series:
            if any(c.open_time < end and c.close_time > start for start, end in sealed):
                if current:
                    segments.append(current)
                current = []
            else:
                current.append(c)
        if current:
            segments.append(current)
        # gaps split windows exactly like sealed periods do: a research window is one contiguous, unsealed run
        segments = [run for seg in segments for run in contiguous_runs(seg)]
        need = int(proto.min_candles * (1 - proto.holdout_fraction)) if diagnostic else proto.min_candles
        latest = segments[-1] if segments else []
        if len(latest) < need:
            return (f"INSUFFICIENT DATA outside sealed holdout periods: latest contiguous unsealed run has {len(latest)} bars "
                    f"< {need} required ({len(sealed)} holdout period(s) already used)")
        full = Dataset.build(latest, data_version="company-store",
                             realization=getattr(self.s.provider, "realization", "observed"))
        boundary = full.candles[int(len(full) * (1 - proto.holdout_fraction))].open_time
        dev_bars = sum(1 for c in full.candles if c.close_time <= boundary)
        return full, boundary, dev_bars

    def _design(self, spec, base, registered, criteria, candidate_def, compute) -> dict:
        exp = spec.experiment
        candidate = None
        if exp.candidate_params is not None:
            candidate = {"strategy_id": base.strategy_id, "kind": base.kind, "params": dict(exp.candidate_params)}
        elif candidate_def is not None:
            candidate = {"strategy_id": candidate_def.strategy_id, "kind": candidate_def.kind,
                         "params": candidate_def.param_dict, "code_hash": candidate_def.code_hash}
        changed = None
        if exp.candidate_params is not None:
            cp = dict(exp.candidate_params)
            base_params = dict(protocol_registry.REGISTRY[spec.protocol_id].base_params)
            changed = [{"variable": k, "baseline": base_params[k], "candidate": cp[k]}
                       for k in sorted(cp) if cp[k] != base_params[k]]
        return {"hypothesis_id": spec.hypothesis_id, "type": exp.type, "design_rationale": exp.design_rationale,
                "changed_variables": changed,
                "independent_variables": list(exp.independent_variables), "dependent_variable": exp.dependent_variable,
                "controls": exp.controls, "failure_criteria": exp.failure_criteria,
                "stopping_criteria": exp.stopping_criteria,
                "success_criteria": [[c.metric, c.op, c.threshold] for c in criteria],
                "baseline": {"strategy_key": base.registry_key, "params": base.param_dict,
                             "fingerprint": registered.definition_hash if registered else None},
                "protocol": {"protocol_id": spec.protocol_id,
                             "protocol_hash": protocol_registry.REGISTRY[spec.protocol_id].protocol_hash},
                "candidate": candidate, "condition": dict(exp.condition) if exp.condition else None,
                "motivation": spec.motivation, "expected_mechanism": spec.expected_mechanism,
                "evidence_refs": list(spec.evidence_refs), "learning_candidate_id": spec.learning_candidate_id,
                "compute_units": compute, "objective_contract": objectives.CONTRACT.contract_hash,
                "budget_policy": budget.POLICY.policy_hash}

    def _diagnostic(self, spec, exp, base, full, boundary, criteria, observation_refs=()) -> dict:
        """REGIME / EXECUTION / RISK: pre-registered, development partition only, recorded like any experiment,
        and never a candidate: no challenger, no holdout, no promotion."""
        s = self.s
        dev = Dataset.build([c for c in full.candles if c.close_time <= boundary], data_version=full.identity.data_version,
                            realization=full.identity.realization, partition=Partition.DEVELOPMENT)
        log = ResearchLog(s.research_journal)
        proto = protocol_registry.REGISTRY[spec.protocol_id]
        prereg = PreRegistration(spec.hypothesis_id, spec.statement, tuple(observation_refs), base.registry_key,
                                 base.definition_hash, dev.dataset_id, tuple(criteria),
                                 self.research_policies[0].min_oos_trades, s.clock.now())
        log.preregister(prereg)
        cond = dict(exp.condition)
        runner = {"REGIME": lambda: diagnostics.regime(base, dev, cond["regime"], train_bars=proto.train_bars,
                                                       test_bars=proto.test_bars),
                  "EXECUTION": lambda: diagnostics.execution(base, dev, cond, train_bars=proto.train_bars,
                                                             test_bars=proto.test_bars),
                  "RISK": lambda: diagnostics.risk(base, dev, cond, train_bars=proto.train_bars, test_bars=proto.test_bars)}
        try:
            metrics, evidence_hash, facts = runner[exp.type]()
        except (ValueError, HoldoutViolation) as exc:
            s.research_journal.append("research_note", {"hypothesis_id": spec.hypothesis_id,
                                                        "note": f"diagnostic not run: {exc}"[:300]})
            return {"hypothesis_id": spec.hypothesis_id, "research_status": "NOT_RUN", "reasons": [str(exc)[:300]]}
        s.evidence.register("experiment", evidence_hash + ":" + spec.hypothesis_id, s.clock.now(),
                            f"{exp.type} diagnostic on {dev.dataset_id}", dataset_id=dev.dataset_id,
                            strategy=base.definition_hash)
        verdict = log.record_experiment(spec.hypothesis_id, f"diagnostic:{exp.type.lower()}", evidence_hash, metrics,
                                        dev.dataset_id)
        s.research_journal.append("diagnostic_result", {"hypothesis_id": spec.hypothesis_id, "type": exp.type,
                                                        "facts": facts, "verdict": verdict.value,
                                                        "note": "development-only: never validation, never a candidate"})
        return {"hypothesis_id": spec.hypothesis_id, "research_status": "COMPLETED", "dev_verdict": verdict.value,
                "holdout_verdict": None, "promotion_approved": None, "diagnostic": facts,
                "n_trades": metrics.n_trades, "expectancy_r": metrics.expectancy_r}

    def _observation_refs(self, spec) -> tuple[str, ...]:
        """Provenance locked into the pre-registration: the learning candidate and the registered evidence behind
        it (added deterministically, not taken from Claude), plus Claude's own validated evidence refs."""
        refs: list[str] = []
        lc = spec.learning_candidate_id
        if lc is not None and lc in self.learning.candidates:
            refs.append(f"learning:{lc}")
            refs += [i for i in self.learning.candidates[lc].evidence if i in self.s.evidence][:12]
        refs += [r for r in spec.evidence_refs if r not in refs]
        return tuple(refs)

    def _persist_candidate(self, hypothesis_id, challenger_key, exp_id, full_dataset_id, baseline_hash, proto) -> dict:
        s = self.s
        challenger = s.strategies.get(challenger_key)
        dev_id = next((e["dataset_id"] for e in ResearchLog(s.research_journal).experiments
                       if e["hypothesis_id"] == hypothesis_id and e["stage"] == "walk_forward_oos"), None)
        factory.record_lineage(s, challenger, hypothesis_id=hypothesis_id, experiment_id=exp_id, dev_dataset_id=dev_id,
                               full_dataset_id=full_dataset_id, baseline_hash=baseline_hash, protocol=proto)
        return {"key": challenger.registry_key, "fingerprint": challenger.definition_hash, "parent_fingerprint": challenger.parent_hash,
                "candidate_id": factory.candidate_id(challenger.definition_hash, hypothesis_id)}

    def _recover_candidate(self, spec, log: ResearchLog, exp_id) -> dict:
        """After a crash inside a research run: persist missing lineage (idempotent) and report — never re-run,
        never re-evaluate the holdout, never apply a promotion. A challenger left mid-review is reported for a new
        promotion review rather than being promoted or rejected by recovery code."""
        s = self.s
        hold_id = spec.hypothesis_id + ":holdout"
        if log.status(hold_id) == "UNKNOWN":
            return {}
        fingerprint = log.get(hold_id).strategy_hash
        key = next((k for k in s.strategies.definitions() if s.strategies.get(k).definition_hash == fingerprint), None)
        if key is None:
            return {}
        proto = protocol_registry.REGISTRY[spec.protocol_id]
        out = {"candidate": self._persist_candidate(spec.hypothesis_id, key, exp_id, None, None, proto)}
        promos = [decode(e.payload)["record"] for e in s.research_journal.entries("promotion_decision")
                  if e.payload["record"]["challenger_hash"] == fingerprint]
        state = s.strategies.state(key, proto.symbol).value
        if state == "CHALLENGER" and promos:
            out["requires_review"] = (f"{key} is still CHALLENGER after a recorded promotion decision "
                                      f"({'approved' if promos[-1]['approved'] else 'denied'}) — the run was interrupted; "
                                      "recovery does not apply or reverse promotions")
        return out

    def _link(self, spec) -> None:
        if spec.learning_candidate_id is not None:
            self.learning.link_hypothesis(spec.learning_candidate_id, spec.hypothesis_id)

    def _registered(self, base):
        try:
            return self.s.strategies.get(base.registry_key)
        except KeyError:
            return None

    def _registered_with(self, base, params: dict):
        target = tuple(sorted(params.items()))
        for key in self.s.strategies.definitions(base.timeframe):
            d = self.s.strategies.get(key)
            if d.strategy_id == base.strategy_id and d.kind == base.kind and tuple(sorted(d.params)) == target:
                return d
        return None

    def _compare(self, spec, base, log: ResearchLog, full: Dataset | None = None, boundary=None) -> dict:
        """BASELINE_* vs CANDIDATE_* on the *recorded* development partition only (never the holdout). Written once
        per hypothesis; the objective contract decides the conclusion, not Claude. The candidate's walk-forward is
        recomputed and must reproduce the recorded evidence hash."""
        s = self.s
        existing = next((decode(e.payload) for e in s.research_journal.entries("experiment_comparison")
                         if e.payload["hypothesis_id"] == spec.hypothesis_id), None)
        if existing is not None:
            return {"conclusion": existing["conclusion"], "recorded": True}
        design = next((decode(e.payload) for e in s.research_journal.entries("experiment_design")
                       if e.payload["hypothesis_id"] == spec.hypothesis_id), None)
        rows = [e for e in log.experiments if e["hypothesis_id"] == spec.hypothesis_id and e["stage"] == "walk_forward_oos"]
        if not rows or design is None or design.get("candidate") is None:
            return {"conclusion": "NOT_AVAILABLE", "reason": "no development experiment or design recorded"}
        row = rows[-1]
        if full is None:
            proto = protocol_registry.REGISTRY[spec.protocol_id]
            window = self._window(s.store.series(s.provider.name, proto.symbol, proto.timeframe), False, proto)
            candidates = [w for w in [window] if isinstance(w, tuple) and w[0].dataset_id == design.get("dataset_id")]
            if not candidates:
                return {"conclusion": "NOT_AVAILABLE", "reason": "research window changed; comparison not reconstructable"}
            full, boundary, _ = candidates[0]
        dev = Dataset.build([c for c in full.candles if c.close_time <= boundary], data_version=full.identity.data_version,
                            realization=full.identity.realization, partition=Partition.DEVELOPMENT)
        if dev.dataset_id != row["dataset_id"]:
            return {"conclusion": "NOT_AVAILABLE", "reason": "development partition does not match the recorded experiment"}
        cd = design["candidate"]
        cand_base = base if cd["strategy_id"] == base.strategy_id else StrategyDefinition.create(
            cd["strategy_id"], 1, cd["kind"], cd["params"], base.timeframe, s.clock.now())
        proto = protocol_registry.REGISTRY[spec.protocol_id]
        cand_wf = walk_forward(cand_base, dev, [cd["params"]], train_bars=proto.train_bars, test_bars=proto.test_bars)
        registered_cand = self._registered_with(cand_base, cd["params"])
        cand_def_hash = registered_cand.definition_hash if registered_cand else None
        if cand_wf.evidence_hash != row["evidence_hash"]:
            return {"conclusion": "NOT_AVAILABLE", "reason": "candidate walk-forward does not reproduce the recorded evidence"}
        baseline = walk_forward(base, dev, [base.param_dict], train_bars=proto.train_bars, test_bars=proto.test_bars)
        stages = {e["stage"]: e["verdict"] for e in log.experiments
                  if e["hypothesis_id"] in (spec.hypothesis_id, spec.hypothesis_id + ":holdout")}
        adv = [decode(e.payload)["report"] for e in s.research_journal.entries("adversarial_report")]
        verdicts = {"walk_forward_oos": stages.get("walk_forward_oos"), "holdout": stages.get("holdout")}
        hold_id = spec.hypothesis_id + ":holdout"
        challenger_hash = log.get(hold_id).strategy_hash if log.status(hold_id) != "UNKNOWN" else None
        mine = [a for a in adv if a["dataset_id"] == dev.dataset_id and a["strategy_hash"] == challenger_hash]
        if mine and stages.get("holdout") is not None:
            blocking = any(o["verdict"] in ("FAIL", "INSUFFICIENT_EVIDENCE") for o in mine[-1]["objections"])
            verdicts["adversarial"] = "BLOCKING" if blocking else "NON_BLOCKING"
        extra = {"wfo_stability": (baseline.positive_fold_fraction, cand_wf.positive_fold_fraction)}
        result = objectives.compare(baseline.oos_metrics, cand_wf.oos_metrics, verdicts, extra=extra)
        s.research_journal.append("experiment_comparison", {
            "hypothesis_id": spec.hypothesis_id, "experiment_id": design.get("experiment_id"),
            "baseline_strategy_id": base.registry_key, "baseline_fingerprint": design["baseline"]["fingerprint"],
            "baseline_dataset_identity": dev.dataset_id, "baseline_result": baseline.oos_metrics,
            "baseline_evidence": baseline.evidence_hash,
            "candidate_strategy_id": f"{cd['strategy_id']}:{sorted(cd['params'].items())}",
            "candidate_fingerprint": challenger_hash or cand_def_hash, "candidate_dataset_identity": dev.dataset_id,
            "candidate_result": cand_wf.oos_metrics, "candidate_evidence": row["evidence_hash"],
            "attempts": factory.attempts(s, spec.hypothesis_id),
            "dev_dataset_id": dev.dataset_id, "verdicts": verdicts, **result,
            "baseline_replaced": False})
        return {"conclusion": result["conclusion"], "dimensions": result["dimensions"],
                "constraint_violations": result["constraint_violations"][:4], "recorded": False}

    # --- finishing ---------------------------------------------------------------------------------------
    def _issue(self, cyc, request_id, prompt, packet, allowed) -> None:
        # What was provided is recorded by category and hash — not a second copy of the context.
        self._step(cyc, "request_issued", {"request_id": request_id, "request_sha256": sha256_text(prompt),
                                           "context_id": packet["context_id"], "issued_at": self.s.clock.now(),
                                           "context_categories": sorted(packet), "allowed_actions": allowed})

    def _finish(self, cyc: CycleRecord, status: str, action: str | None, detail: dict,
                health: HealthReport) -> CycleOutcome:
        if "learn" not in cyc.steps:
            # LEARN: deterministic, idempotent, from journals only. A learning failure is recorded, never hidden,
            # and never changes the cycle's outcome (learning has no authority over what already happened).
            try:
                learned = self.learning.learn(self.s, self.loop.journal, self.journal)
            except AtiError as exc:
                learned = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            self._step(cyc, "learn", learned)
        self.journal.append("cycle_end", {"cycle_id": cyc.cycle_id, "status": status, "action": action,
                                          "detail": detail})
        cyc.status, cyc.result = status, {"status": status, "action": action, "detail": detail}
        self.running, self.last_finished = None, cyc.cycle_id
        end = CompanyState.PAUSED if self.paused else CompanyState(status)
        self._transition(end, f"cycle {cyc.cycle_id} {status}")
        return CycleOutcome(cyc.cycle_id, "PAUSED" if self.paused and status == "COMPLETED" else status,
                            action, detail, health.as_dict())

    # --- deterministic views -----------------------------------------------------------------------------
    def _research_log(self) -> ResearchLog | None:
        """None when the research journal cannot be reconstructed; the health gate reports that as FAIL."""
        try:
            return ResearchLog(self.s.research_journal)
        except AtiError:
            return None

    def _fingerprint(self, datasets: dict, health: HealthReport) -> str:
        s = self.s
        return sha256_hex({
            "data": {sym: [ds.candles[-1].close_time, len(ds), ds.dataset_id] for sym, ds in sorted(datasets.items())},
            "health": {c.component: c.status.value for c in health.checks}, "data_state": health.data_state.value,
            "cash": s.execution.account.cash,
            "positions": {k: v.qty for k, v in sorted(s.execution.account.positions.items())},
            "orders": len(s.execution.orders), "decisions": sum(1 for _ in s.decisions.journal.entries("decision")),
            "champions": self._champions(),
            "research_entries": len(s.research_journal), "memory": len(s.memory),
            "pause_epoch": self.pause_epoch, "stops": sorted(self.loop.stops),
        })

    def _champions(self) -> dict:
        """Champion fingerprint per research dimension the system trades (symbol × its timeframe)."""
        s = self.s
        return {f"{sym} {s.timeframe.value}": (c.definition_hash if (c := s.strategies.champion(sym, s.timeframe)) else None)
                for sym in s.symbols}

    def _gate(self, action: str, health: HealthReport) -> list[str]:
        reasons = health.gate(action, self.paused)
        if action == "TRADE_PROPOSAL" and self.autonomy < autonomy.Autonomy.PAPER_AUTONOMY:
            reasons = reasons + [f"autonomy {self.autonomy.name} does not include paper trading"]
        return reasons

    def _conditions(self, datasets) -> list[dict]:
        """Assumption monitors on the point-in-time data refreshed this cycle, plus realized paper entry costs."""
        s = self.s
        fills = []
        ctx = {e.payload["decision_id"]: dict(decode(e.payload)["record"]["market_context"])
               for e in s.decisions.journal.entries("decision")}
        for o in s.execution.orders.values():
            ref = ctx.get(o.decision_id, {}).get("last_price")
            if o.side is Side.BUY and o.filled_qty > 0 and ref is not None and o.avg_fill_price is not None:
                fills.append((Decimal(ref), o.avg_fill_price, o.filled_qty))
        return conditions_mod.monitor({sym: ds.candles for sym, ds in datasets.items()}, fills,
                                      s.costs.half_spread_rate + s.costs.slippage_rate)

    def _contract(self, cyc, request_id, health: HealthReport, datasets, context_id: str = "") -> CompanyContext:
        s = self.s
        now = s.clock.now()
        keys = {c.key for sym in s.symbols if (c := s.strategies.champion(sym, s.timeframe)) is not None}
        trade = ValidationContext(
            universe=s.universe, allowed_strategy_keys=frozenset(keys),
            last_prices={sym: ds.candles[-1].close for sym, ds in datasets.items()},
            evidence_exists=lambda r: r in s.evidence and s.evidence.resolve(r).available_at <= now,
            account_state_known=health.status("execution") is Status.PASS)
        return CompanyContext(request_id, cyc.cycle_id, trade, PROTOCOLS, PROTOCOL_PARAMS, context_id,
                              lambda r: s.evidence.resolve(r).kind if r in s.evidence else None)

    def _positions(self, marks, symbol=None) -> list[dict]:
        out = []
        for sym, pos in sorted(self.s.execution.account.positions.items()):
            if pos.qty and (symbol is None or sym == symbol):
                mark = marks.get(sym)
                out.append({"symbol": sym, "qty": str(pos.qty), "avg_price": str(pos.avg_price),
                            "mark": str(mark) if mark is not None else None,
                            "unrealized_pnl": str((mark - pos.avg_price) * pos.qty) if mark is not None else None,
                            "stop": str(self.loop.stops[sym][0]) if sym in self.loop.stops else None,
                            "strategy": self.loop.stops[sym][1] if sym in self.loop.stops else None})
        return out

    def _risk_view(self, marks) -> dict:
        s = self.s
        view = {"kill_switch": s.kill_switch.state(), "reconciliation": s.execution.recon_state.value,
                "limits": {k: str(v) for k, v in s.limits.__dict__.items() if k != "allowed_data"},
                "limits_hash": s.limits.limits_hash}
        if marks and self.loop.day_start_equity is not None:
            pf = s.execution.portfolio_snapshot(marks, self.loop.day_start_equity, self.loop.peak_equity)
            view["portfolio"] = {"equity": str(pf.equity), "cash": str(pf.cash), "exposure": str(pf.exposure()),
                                 "day_start_equity": str(pf.day_start_equity), "peak_equity": str(pf.peak_equity),
                                 "account_state_known": pf.account_state_known}
        else:
            view["portfolio"] = "marks unavailable (no fresh data): equity not computed"
        return view

    def _context(self, cyc, request_id, health, datasets, candidates, marks, allowed, conditions=()) -> dict:
        """Bounded, deterministic, point-in-time context in named sections. Summaries and ids only — no journals,
        datasets, holdout contents or internals. Every learning/memory item carries its evidence level explicitly
        (VALIDATED_FINDING / SUPPORTED_PATTERN / HYPOTHESIS / OBSERVATION / REJECTED_HYPOTHESIS): memory informs,
        evidence authorizes."""
        s = self.s
        now = s.clock.now()
        log = self._research_log()
        prev = self.cycles.get(self.last_finished) if self.last_finished else None
        memory = s.memory.query(now)
        level = {"VALIDATED_FINDING": "VALIDATED_FINDING", "FINDING": "SUPPORTED_PATTERN", "HYPOTHESIS": "HYPOTHESIS",
                 "REJECTED_HYPOTHESIS": "REJECTED_HYPOTHESIS", "MISTAKE": "SUPPORTED_PATTERN", "LESSON": "SUPPORTED_PATTERN"}
        learning_level = {"VALIDATED_EFFECT": "VALIDATED_FINDING", "SUPPORTED_PATTERN": "SUPPORTED_PATTERN"}
        learnings = [dict(r, evidence_level=learning_level.get(r["class"], "OBSERVATION"))
                     for r in self.learning.summary(s)]
        return {
            "request_id": request_id, "cycle_id": cyc.cycle_id, "as_of": now.isoformat(), "allowed_actions": allowed,
            "COMPANY_STATE": {
                "state": self.state.value, "paused": self.paused,
                "previous_cycle": {"cycle_id": prev.cycle_id, "status": prev.status,
                                   "action": prev.result.get("action")} if prev else None,
                "autonomy": {"level": self.autonomy.name, "maximum_permitted": autonomy.maximum_permitted().name},
                "objective": {"primary": objectives.CONTRACT.primary_objective,
                              "contract_hash": objectives.CONTRACT.contract_hash,
                              "constraints": {"max_drawdown": objectives.CONTRACT.max_drawdown,
                                              "min_trades": objectives.CONTRACT.min_trades,
                                              "max_top5_profit_share": objectives.CONTRACT.max_top5_profit_share,
                                              "max_cost_share_of_gross": objectives.CONTRACT.max_cost_share_of_gross},
                              "required_evidence": list(objectives.CONTRACT.required_verdicts)}},
            "DATA_HEALTH": {
                "data_state": health.data_state.value, "data_check": health.as_dict()["checks"]["data"],
                "market": {sym: {"data_status": ds.identity.status.value, "dataset_id": ds.dataset_id,
                                 "latest_close": str(ds.candles[-1].close_time), "last_price": str(ds.candles[-1].close),
                                 "bars": len(ds)} for sym, ds in sorted(datasets.items())},
                "open_conflicts": sorted(self.conflicts.open)[:5],
                "conditions": [{k: c[k] for k in ("assumption", "status", "statistic", "invalidates")} for c in conditions]},
            "STRATEGY": {"timeframe": s.timeframe.value,
                         "champions": {sym: ({"key": c.key, "registry_key": c.registry_key, "fingerprint": c.definition_hash,
                                              "version": c.version, "params": c.param_dict}
                                             if (c := s.strategies.champion(sym, s.timeframe)) else None)
                                       for sym in s.symbols},
                         "entry_signals": {sym: {"cutoff": str(c[0].cutoff), "last_close": str(c[0].latest.close)}
                                           for sym, c in sorted(candidates.items())}},
            "RISK_STATE": {"kill_switch_engaged": s.kill_switch.state()["engaged"],
                           "reconciliation": s.execution.recon_state.value,
                           "max_risk_per_trade_fraction": str(s.limits.max_risk_per_trade_fraction),
                           "limits_hash": s.limits.limits_hash, "owner": "code; not changeable by any action",
                           "portfolio": {"cash": str(s.execution.account.cash), "positions": self._positions(marks),
                                         "realized_pnl": str(s.execution.account.realized_pnl)},
                           # read-only execution facts; Claude cannot change the mode, the live state or any order
                           "execution": {"mode": s.execution.policy.mode.value, "live": live_state(),
                                         "halted": s.execution.halted,
                                         "working_orders": [{"symbol": o.symbol, "side": o.side.value,
                                                             "status": o.status.value}
                                                            for o in s.execution.orders.values()
                                                            if o.status not in TERMINAL],
                                         "last_refusal_rule": (s.execution.last_refusal or {}).get("rule")}},
            "RECENT_OUTCOMES": [{"outcome_id": o.outcome_id, "source": o.source, "at": o.at, "data_category": o.data_category,
                                 "realized": o.realized, "deviation": o.deviation}
                                for o in list(self.learning.outcomes.values())[-6:]],
            "RELEVANT_LEARNINGS": {
                "learning_candidates": learnings,
                "memory": [{"entry_id": m.entry_id, "evidence_level": level[m.kind.value], "statement": m.statement[:300],
                            "confidence": m.confidence} for m in memory
                           if m.kind.value not in ("VALIDATED_FINDING", "REJECTED_HYPOTHESIS")][-6:]},
            "ACTIVE_HYPOTHESES": self._active_hypotheses(log),
            "RECENT_EXPERIMENTS": [{"hypothesis_id": e["hypothesis_id"], "stage": e["stage"], "verdict": e["verdict"],
                                    # holdout identities never enter Claude's context: evaluation-only, access-limited
                                    "dataset_id": "SEALED_HOLDOUT" if e["stage"] == "holdout" else e["dataset_id"]}
                                   for e in (log.experiments[-5:] if log else [])],
            "FAILED_EXPERIMENTS": [{"hypothesis_id": e["hypothesis_id"], "stage": e["stage"], "verdict": e["verdict"]}
                                   for e in (log.experiments if log else []) if e["verdict"] != "PASS"][-8:],
            "REJECTED_HYPOTHESES": [{"entry_id": m.entry_id, "evidence_level": "REJECTED_HYPOTHESIS",
                                     "statement": m.statement[:300]} for m in memory
                                    if m.kind is MemoryKind.REJECTED_HYPOTHESIS][-6:],
            "VALIDATED_FINDINGS": [{"entry_id": m.entry_id, "evidence_level": "VALIDATED_FINDING", "statement": m.statement,
                                    "confidence": m.confidence} for m in memory
                                   if m.kind is MemoryKind.VALIDATED_FINDING][-5:],
            "CHALLENGERS": self._validation_status(),
            "SYSTEM_HEALTH": {"health": health.as_dict(), "scorecard": scorecard.build(s, health, self.journal, self.learning)},
            "RESEARCH_BUDGET": self._budget_view(),
            "RESEARCH_PROTOCOL": {
                "protocols": [{"protocol_id": pid, "symbol": p.symbol, "timeframe": p.timeframe.value,
                               "strategy_key": p.strategy.key, "statement": p.statement, "min_bars": p.min_candles,
                               "locked_minimum_criteria": [list(c) for c in p.criteria]}
                              for pid, p in protocol_registry.REGISTRY.items() if pid in PROTOCOLS],
                "candidate_types": sorted(CANDIDATE_EXPERIMENTS), "diagnostic_types": sorted(DIAGNOSTIC_EXPERIMENTS),
                "regime_labels": list(diagnostics.REGIME_LABELS), "execution_bounds": diagnostics.EXECUTION_BOUNDS,
                "risk_bounds": diagnostics.RISK_BOUNDS, "baseline_params": PROTOCOL_PARAMS},
        }

    def _active_hypotheses(self, log) -> list[dict]:
        if log is None:
            return []
        tested = {e["hypothesis_id"] for e in log.experiments}
        linked = {c.hypothesis_id: c.candidate_id for c in self.learning.candidates.values() if c.hypothesis_id}
        out = []
        for e in self.s.research_journal.entries("preregistration"):
            h = e.payload["prereg"]["hypothesis_id"]
            if ":" not in h:
                out.append({"hypothesis_id": h, "status": log.status(h), "learning_candidate": linked.get(h),
                            "tested": h in tested})
        return out[-8:]

    def _validation_status(self) -> dict:
        s = self.s
        promos = [decode(e.payload)["record"] for e in s.research_journal.entries("promotion_decision")]
        latest = promos[-1] if promos else None
        lineage = [e.payload for e in s.research_journal.entries("candidate_lineage")][-3:]
        return {"registered_challengers": {sym: s.strategies.keys(Lifecycle.CHALLENGER, sym) for sym in s.universe.symbols},
                "latest_promotion": {"promotion_id": factory.promotion_id(latest["record_hash"]),
                                     "challenger": latest["challenger_key"], "approved": latest["approved"],
                                     "reasons": list(latest["reasons"])[:3]} if latest else "NOT_AVAILABLE",
                "recent_candidates": [{"candidate_id": l_["candidate_id"], "key": l_["key"], "hypothesis_id": l_["hypothesis_id"],
                                       "furthest_stage": factory.stages(s, l_["fingerprint"])["furthest_stage"],
                                       "attempts_before": l_.get("attempts")} for l_ in lineage]}

    def _budget_view(self) -> dict:
        s = self.s
        base = self._registered(protocol_registry.REGISTRY[P.PROTOCOL_ID].strategy)
        try:
            u = budget.usage(s, self.journal, s.clock.now(), base.definition_hash if base else None)
        except AtiError:
            return {"state": "UNAVAILABLE (research FAIL): no research may run"}
        return {k: v for k, v in u.items() if k != "policy_hash"}
