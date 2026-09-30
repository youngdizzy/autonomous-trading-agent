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
from enum import Enum

from ati.agent.loop import AutonomousLoop, TickReport
from ati.agent.pipeline import Outcome
from ati.agent.reasoning import ReasoningBudgetExceeded, ReasoningPending
from ati.agent.roles import COMPANY, build_prompt
from ati.agent.schema import (COMPANY_ACTIONS, CompanyAction, CompanyContext, ValidationContext,
                              parse_company_response)
from ati.company import autonomy, budget, objectives, scorecard
from ati.company.health import HealthReport, Status, assess
from ati.company.learning import LearningLedger
from ati.core.canonical import sha256_hex, sha256_text
from ati.core.errors import AtiError, CompanyStateError, SchemaViolation
from ati.core.types import Side
from ati.data.dataset import Dataset, Partition
from ati.decision.records import make_decision_id
from ati.ledger.journal import Journal, decode
from ati.market.conflict import ConflictRegister
from ati.memory.store import MemoryKind
from ati.research import protocol as P
from ati.research.adversarial import AdversarialPolicy
from ati.research.hypothesis import Criterion, ResearchLog
from ati.research.metrics import Metrics
from ati.research.walkforward import walk_forward
from ati.research.workflow import run_research_cycle
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
PROTOCOLS = {P.PROTOCOL_ID: P.base_definition(_EPOCH).key}
PROTOCOL_PARAMS = {P.PROTOCOL_ID: dict(P.BASE_PARAMS)}
EXECUTABLE_EXPERIMENTS = frozenset({"SINGLE_VARIABLE", "INTERACTION"})   # others: executor NOT IMPLEMENTED


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
                 research_policies: tuple[AdversarialPolicy, PromotionPolicy] | None = None):
        s = system
        self.autonomy = autonomy.require(autonomy.Autonomy.PAPER_AUTONOMY)   # the ceiling is code, not config
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
            champion = s.strategies.champion()
            for symbol, ds in datasets.items():
                cand = self.loop.manage_position(report, symbol, ds, marks, champion)
                if cand is not None:
                    candidates[symbol] = cand
        health = assess(s, self.loop, datasets, self.journal, self.paused, self.conflicts)
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
                                                "paused": self.paused})
            cyc = self.cycles[cycle_id] = CycleRecord(cycle_id, fingerprint)
            self.running = cycle_id

        if "health" not in cyc.steps:
            self._step(cyc, "health", health.as_dict())
        allowed = [a for a in COMPANY_ACTIONS if not health.gate(a, self.paused)]
        request_id = f"{cyc.cycle_id}:company"
        packet = self._context(cyc, request_id, health, datasets, candidates, marks, allowed)
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
            action = parse_company_response(raw, self._contract(cyc, request_id, health, datasets))
        except SchemaViolation as exc:
            if "rejected" not in cyc.steps:
                self._step(cyc, "rejected", {"reason": str(exc)[:500]})
            return self._finish(cyc, "FAILED", None, {"reason": f"CLAUDE_RESPONSE_REJECTED: {exc}"[:500]}, health)
        if "action_result" in cyc.steps:   # crashed after recording the result: finish, never re-execute
            recorded = dict(cyc.steps["action_result"])
            return self._finish(cyc, recorded.pop("status"), action.action, recorded, health)
        reasons = health.gate(action.action, self.paused)   # re-applied on recovery too: never looser
        if reasons:
            note = {"note": "blocked on recovery; downstream journals are authoritative for anything already done"} \
                if "action_started" in cyc.steps else {}
            return self._finish(cyc, "BLOCKED", action.action, {"reasons": reasons, **note}, health)
        if "validated" not in cyc.steps:
            self._step(cyc, "validated", {"action": action.action})

        # EXECUTE_ALLOWED_ACTION — only through existing systems.
        if "action_started" not in cyc.steps:
            started = {"action": action.action}
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
        champion = s.strategies.champion()
        proposal = action.trade
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
        base = P.base_definition(s.clock.now())
        if log.status(spec.hypothesis_id) != "UNKNOWN":
            if "research_invoked" in cyc.steps:  # this cycle already ran it before a crash: report, never rerun
                detail = {"hypothesis_id": spec.hypothesis_id, "research_status": log.status(spec.hypothesis_id),
                          "recovered": True}
                if exp is not None and exp.candidate_params is not None:
                    detail["comparison"] = self._compare(spec, base, log)
                self._link(spec)
                return "COMPLETED", detail
            return "BLOCKED", {"reason": f"{spec.hypothesis_id} is already registered ({log.status(spec.hypothesis_id)}); "
                                         "criteria are locked — a new question needs a new hypothesis id"}
        if exp is not None and exp.type not in EXECUTABLE_EXPERIMENTS:
            return "BLOCKED", {"reason": f"{exp.type} experiment executor is NOT IMPLEMENTED; nothing was run or "
                                         "approximated"}
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
        registered = self._registered(base)
        grid = [dict(exp.candidate_params)] if exp is not None and exp.candidate_params is not None else P.GRID
        new_variant = any(self._registered_with(base, g) is None for g in grid if g != dict(P.BASE_PARAMS))
        if "research_invoked" not in cyc.steps:
            u = budget.usage(s, self.journal, s.clock.now(), registered.definition_hash if registered else None)
            over = budget.check(u, new_variant)
            if over:
                return "BLOCKED", {"reason": "research budget: " + "; ".join(over), "budget_policy": u["policy_hash"]}
        criteria = list(P.CRITERIA)  # the protocol's floor is always included: requests can only add criteria
        for metric, op, threshold in spec.success_criteria:
            c = Criterion(metric, op, threshold)
            if c not in criteria:
                criteria.append(c)
        series = s.store.series(s.provider.name, P.SYMBOL, P.TIMEFRAME)
        if "research_invoked" not in cyc.steps:
            self._step(cyc, "research_invoked", {"hypothesis_id": spec.hypothesis_id, "criteria": len(criteria)})
        if not series:
            return "COMPLETED", {"hypothesis_id": spec.hypothesis_id, "research_status": "NOT_RUN",
                                 "reasons": ["no stored data for the protocol symbol"]}
        if exp is not None:
            # Baseline discipline: the baseline definition is registered under its own key *before* the run, so
            # the candidate is derived as a new version with the baseline as parent — never stored as the baseline.
            registered = registered or s.strategies.register(base)
            # Design recorded before the run: the baseline is named explicitly and never silently replaced.
            s.research_journal.append("experiment_design", {
                "hypothesis_id": spec.hypothesis_id, "type": exp.type,
                "independent_variables": list(exp.independent_variables), "dependent_variable": exp.dependent_variable,
                "controls": exp.controls, "failure_criteria": exp.failure_criteria,
                "stopping_criteria": exp.stopping_criteria, "success_criteria": [[c.metric, c.op, c.threshold]
                                                                                 for c in criteria],
                "baseline": {"strategy_key": base.key, "params": dict(P.BASE_PARAMS),
                             "fingerprint": registered.definition_hash if registered else None},
                "candidate_params": dict(exp.candidate_params) if exp.candidate_params else None,
                "motivation": spec.motivation, "expected_mechanism": spec.expected_mechanism,
                "evidence_refs": list(spec.evidence_refs), "learning_candidate_id": lc,
                "objective_contract": objectives.CONTRACT.contract_hash, "budget_policy": budget.POLICY.policy_hash})
        full = Dataset.build(series, data_version="company-store",
                             realization=getattr(s.provider, "realization", "observed"))
        boundary = full.candles[int(len(full) * (1 - P.HOLDOUT_FRACTION))].open_time
        adversarial, promotion = self.research_policies
        r = run_research_cycle(s, full, boundary, hypothesis_id=spec.hypothesis_id, statement=spec.statement,
                               base=base, grid=grid, criteria=tuple(criteria),
                               train_bars=P.TRAIN_BARS, test_bars=P.TEST_BARS, adversarial_policy=adversarial,
                               promotion_policy=promotion, min_candles=P.MIN_CANDLES)
        detail = {"hypothesis_id": spec.hypothesis_id, "research_status": r.status,
                  "reasons": list(r.reasons), "dev_verdict": r.dev_verdict and r.dev_verdict.value,
                  "holdout_verdict": r.holdout_verdict and r.holdout_verdict.value,
                  "promotion_approved": r.promotion.approved if r.promotion else None}
        if exp is not None and exp.candidate_params is not None and r.dev_verdict is not None:
            detail["comparison"] = self._compare(spec, base, ResearchLog(s.research_journal), full, boundary)
        self._link(spec)
        return "COMPLETED", detail

    def _link(self, spec) -> None:
        if spec.learning_candidate_id is not None:
            self.learning.link_hypothesis(spec.learning_candidate_id, spec.hypothesis_id)

    def _registered(self, base):
        try:
            return self.s.strategies.get(base.key)
        except KeyError:
            return None

    def _registered_with(self, base, params: dict):
        target = tuple(sorted(params.items()))
        for key in self.s.strategies.keys():
            d = self.s.strategies.get(key)
            if d.strategy_id == base.strategy_id and d.kind == base.kind and tuple(sorted(d.params)) == target:
                return d
        return None

    def _compare(self, spec, base, log: ResearchLog, full: Dataset | None = None, boundary=None) -> dict:
        """Baseline vs candidate on the *recorded* development partition only (never the holdout). Written once
        per hypothesis to the research journal; the objective contract decides the conclusion, not Claude."""
        s = self.s
        existing = next((decode(e.payload) for e in s.research_journal.entries("experiment_comparison")
                         if e.payload["hypothesis_id"] == spec.hypothesis_id), None)
        if existing is not None:
            return {"conclusion": existing["conclusion"], "recorded": True}
        rows = [e for e in log.experiments if e["hypothesis_id"] == spec.hypothesis_id and e["stage"] == "walk_forward_oos"]
        if not rows:
            return {"conclusion": "NOT_AVAILABLE", "reason": "no development experiment recorded"}
        row = rows[-1]
        if full is None:
            series = s.store.series(s.provider.name, P.SYMBOL, P.TIMEFRAME)
            full = Dataset.build(series, data_version="company-store",
                                 realization=getattr(s.provider, "realization", "observed"))
            boundary = full.candles[int(len(full) * (1 - P.HOLDOUT_FRACTION))].open_time
        dev = Dataset.build([c for c in full.candles if c.close_time <= boundary], data_version=full.identity.data_version,
                            realization=full.identity.realization, partition=Partition.DEVELOPMENT)
        if dev.dataset_id != row["dataset_id"]:
            return {"conclusion": "NOT_AVAILABLE", "reason": "development partition does not match the recorded experiment"}
        baseline = walk_forward(base, dev, [dict(P.BASE_PARAMS)], train_bars=P.TRAIN_BARS, test_bars=P.TEST_BARS)
        candidate = Metrics(**row["metrics"])
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
        result = objectives.compare(baseline.oos_metrics, candidate, verdicts)
        s.research_journal.append("experiment_comparison", {
            "hypothesis_id": spec.hypothesis_id, "dev_dataset_id": dev.dataset_id,
            "baseline": {"strategy_key": base.key, "params": dict(P.BASE_PARAMS), "evidence_hash": baseline.evidence_hash},
            "candidate": {"params": dict(spec.experiment.candidate_params), "evidence_hash": row["evidence_hash"]},
            "verdicts": verdicts, **result})
        return {"conclusion": result["conclusion"], "dimensions": result["dimensions"],
                "constraint_violations": result["constraint_violations"][:4], "recorded": False}

    # --- finishing ---------------------------------------------------------------------------------------
    def _issue(self, cyc, request_id, prompt, packet, allowed) -> None:
        # What was provided is recorded by category and hash — not a second copy of the context.
        self._step(cyc, "request_issued", {"request_id": request_id, "request_sha256": sha256_text(prompt),
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
        champion = s.strategies.champion()
        return sha256_hex({
            "data": {sym: [ds.candles[-1].close_time, len(ds), ds.dataset_id] for sym, ds in sorted(datasets.items())},
            "health": {c.component: c.status.value for c in health.checks}, "data_state": health.data_state.value,
            "cash": s.execution.account.cash,
            "positions": {k: v.qty for k, v in sorted(s.execution.account.positions.items())},
            "orders": len(s.execution.orders), "decisions": sum(1 for _ in s.decisions.journal.entries("decision")),
            "champion": champion.definition_hash if champion else None,
            "research_entries": len(s.research_journal), "memory": len(s.memory),
            "pause_epoch": self.pause_epoch, "stops": sorted(self.loop.stops),
        })

    def _contract(self, cyc, request_id, health: HealthReport, datasets) -> CompanyContext:
        s = self.s
        champion = s.strategies.champion()
        now = s.clock.now()
        trade = ValidationContext(
            universe=s.universe, allowed_strategy_keys=frozenset({champion.key}) if champion else frozenset(),
            last_prices={sym: ds.candles[-1].close for sym, ds in datasets.items()},
            evidence_exists=lambda r: r in s.evidence and s.evidence.resolve(r).available_at <= now,
            account_state_known=health.status("execution") is Status.PASS)
        return CompanyContext(request_id, cyc.cycle_id, trade, PROTOCOLS, PROTOCOL_PARAMS,
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

    def _context(self, cyc, request_id, health, datasets, candidates, marks, allowed) -> dict:
        """Bounded, point-in-time context: summaries only — no journals, datasets, holdout or internals."""
        s = self.s
        champion = s.strategies.champion()
        log = self._research_log()
        prev = self.cycles.get(self.last_finished) if self.last_finished else None
        return {
            "request_id": request_id, "cycle_id": cyc.cycle_id, "allowed_actions": allowed,
            "company": {"state": self.state.value, "paused": self.paused,
                        "previous_cycle": {"cycle_id": prev.cycle_id, "status": prev.status,
                                           "action": prev.result.get("action")} if prev else None},
            "health": health.as_dict(),
            "market": {sym: {"data_status": ds.identity.status.value, "latest_close": str(ds.candles[-1].close_time),
                             "last_price": str(ds.candles[-1].close), "bars": len(ds)}
                       for sym, ds in sorted(datasets.items())},
            "portfolio": {"cash": str(s.execution.account.cash), "positions": self._positions(marks),
                          "realized_pnl": str(s.execution.account.realized_pnl)},
            "risk": {"kill_switch_engaged": s.kill_switch.state()["engaged"],
                     "reconciliation": s.execution.recon_state.value,
                     "max_risk_per_trade_fraction": str(s.limits.max_risk_per_trade_fraction)},
            "strategy": {"champion": champion.key if champion else None,
                         "entry_signals": {sym: {"cutoff": str(c[0].cutoff), "last_close": str(c[0].latest.close)}
                                           for sym, c in sorted(candidates.items())}},
            "research": {"hypotheses_tested": log.hypotheses_tested if log else "UNAVAILABLE (research FAIL)",
                         "recent": [{"hypothesis_id": e["hypothesis_id"], "stage": e["stage"], "verdict": e["verdict"]}
                                    for e in (log.experiments[-5:] if log else [])],
                         "protocols": [{"protocol_id": P.PROTOCOL_ID, "strategy_key": PROTOCOLS[P.PROTOCOL_ID],
                                        "statement": P.STATEMENT, "min_bars": P.MIN_CANDLES,
                                        "locked_minimum_criteria": [[c.metric, c.op, c.threshold] for c in P.CRITERIA]}]},
            "memory": [{"kind": m.kind.value, "statement": m.statement, "confidence": m.confidence}
                       for m in s.memory.query(s.clock.now())[-10:]],
            # Self-improvement context: informs Claude; none of it authorizes anything.
            "autonomy": {"level": self.autonomy.name, "maximum_permitted": autonomy.maximum_permitted().name},
            "objective": {"primary": objectives.CONTRACT.primary_objective,
                          "contract_hash": objectives.CONTRACT.contract_hash,
                          "constraints": {"max_drawdown": objectives.CONTRACT.max_drawdown,
                                          "min_trades": objectives.CONTRACT.min_trades,
                                          "max_top5_profit_share": objectives.CONTRACT.max_top5_profit_share,
                                          "max_cost_share_of_gross": objectives.CONTRACT.max_cost_share_of_gross},
                          "required_evidence": list(objectives.CONTRACT.required_verdicts)},
            "research_budget": self._budget_view(),
            "scorecard": scorecard.build(s, health, self.journal, self.learning),
            "learning_candidates": self.learning.summary(s),
            "failed_experiments": [{"hypothesis_id": e["hypothesis_id"], "stage": e["stage"], "verdict": e["verdict"]}
                                   for e in (log.experiments if log else []) if e["verdict"] != "PASS"][-8:],
            "validated_findings": [{"statement": m.statement, "confidence": m.confidence}
                                   for m in s.memory.query(s.clock.now()) if m.kind is MemoryKind.VALIDATED_FINDING][-5:],
            "experiment_protocol": {"types_executable": sorted(EXECUTABLE_EXPERIMENTS),
                                    "baseline_params": PROTOCOL_PARAMS[P.PROTOCOL_ID]},
        }

    def _budget_view(self) -> dict:
        s = self.s
        base = self._registered(P.base_definition(s.clock.now()))
        try:
            u = budget.usage(s, self.journal, s.clock.now(), base.definition_hash if base else None)
        except AtiError:
            return {"state": "UNAVAILABLE (research FAIL): no research may run"}
        return {k: v for k, v in u.items() if k != "policy_hash"}
