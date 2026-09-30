"""Outcome learning — the LEARN stage of the company cycle.

    OUTCOME (journals) → OutcomeRecord (facts) → deterministic pattern detection → LearningCandidate
      → [Claude may cite it in a RESEARCH_REQUEST] → PREREGISTERED → TESTING → SUPPORTED/REJECTED/INCONCLUSIVE

Everything here is deterministic and derived from the existing journals:
  - trade outcomes      loop journal ``trade_review`` (+ the decision's expected risk)
  - research outcomes   research journal ``experiment`` rows and ``promotion_decision`` records
  - data failures       loop journal ``data_unhealthy``
  - contract failures   company journal ``rejected`` steps

A learning candidate is a *record*, never a rule. This module holds no reference to the strategy registry,
risk engine or execution engine, so no learning path can change a strategy, a limit, or an order. The
only way forward from a candidate is a pre-registered hypothesis through the research workflow.

Evidence quality (deterministic, never Claude's opinion):
  OBSERVATION        one supporting outcome
  WEAK_EVIDENCE      two independent outcomes
  REPEATED_EVIDENCE  three or more independent outcomes
  SUPPORTED_EVIDENCE a pre-registered test of the linked hypothesis passed development and holdout
  VALIDATED_EVIDENCE as SUPPORTED, on market-category data, with the promotion gate approving
Outcomes count as independent only if they are distinct source events; stages of one experiment, and
repeated observations of one event, count once.

Holdout isolation: outcomes of a holdout evaluation are recorded (failures stay visible) but are marked
``holdout_derived`` and can never seed a new hypothesis — holdout is for evaluation, not learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from ati.core.canonical import sha256_hex
from ati.core.errors import LifecycleError
from ati.ledger.journal import Journal, decode
from ati.market.models import MARKET_EVIDENCE_STATUSES


class Quality(str, Enum):
    OBSERVATION = "OBSERVATION"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    REPEATED_EVIDENCE = "REPEATED_EVIDENCE"
    SUPPORTED_EVIDENCE = "SUPPORTED_EVIDENCE"
    VALIDATED_EVIDENCE = "VALIDATED_EVIDENCE"


class PatternClass(str, Enum):
    ONE_OFF = "ONE_OFF"
    RECURRING = "RECURRING_PATTERN"
    STATISTICALLY_SUPPORTED = "STATISTICALLY_SUPPORTED_EFFECT"
    VALIDATED_DOCTRINE = "VALIDATED_DOCTRINE"


class CandidateState(str, Enum):
    OBSERVED = "OBSERVED"
    ANALYZED = "ANALYZED"
    HYPOTHESIS_CANDIDATE = "HYPOTHESIS_CANDIDATE"
    PREREGISTERED = "PREREGISTERED"
    TESTING = "TESTING"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


_C = CandidateState
TRANSITIONS: dict[CandidateState, frozenset[CandidateState]] = {
    _C.OBSERVED: frozenset({_C.ANALYZED, _C.HYPOTHESIS_CANDIDATE, _C.PREREGISTERED}),
    _C.ANALYZED: frozenset({_C.HYPOTHESIS_CANDIDATE, _C.PREREGISTERED}),
    _C.HYPOTHESIS_CANDIDATE: frozenset({_C.PREREGISTERED}),
    _C.PREREGISTERED: frozenset({_C.TESTING, _C.INCONCLUSIVE}),
    _C.TESTING: frozenset({_C.SUPPORTED, _C.REJECTED, _C.INCONCLUSIVE}),
    _C.SUPPORTED: frozenset(), _C.REJECTED: frozenset(), _C.INCONCLUSIVE: frozenset(),  # terminal, visible forever
}


@dataclass(frozen=True)
class OutcomeRecord:
    outcome_id: str
    source: str                    # trade | experiment | promotion | data_failure | response_rejected
    at: str
    data_category: str
    symbol: str | None = None
    timeframe: str | None = None
    strategy: str | None = None    # fingerprint (definition hash) or strategy key where that is all there is
    dataset_id: str | None = None
    expected: dict = field(default_factory=dict)
    realized: dict = field(default_factory=dict)
    deviation: dict = field(default_factory=dict)
    holdout_derived: bool = False
    facts: dict = field(default_factory=dict)


@dataclass
class LearningCandidate:
    candidate_id: str
    pattern: str                   # detector name
    key: dict                      # what the pattern is conditioned on
    statement: str                 # deterministic template — never free text from Claude
    evidence: list[str]            # distinct outcome ids
    state: CandidateState
    holdout_derived: bool
    hypothesis_id: str | None = None
    history: list[tuple[str, str, str]] = field(default_factory=list)

    def quality(self, supported: bool = False, validated: bool = False) -> Quality:
        if validated:
            return Quality.VALIDATED_EVIDENCE
        if supported:
            return Quality.SUPPORTED_EVIDENCE
        n = len(self.evidence)
        return Quality.OBSERVATION if n <= 1 else Quality.WEAK_EVIDENCE if n == 2 else Quality.REPEATED_EVIDENCE

    @property
    def research_eligible(self) -> bool:
        return not self.holdout_derived and self.state in (_C.HYPOTHESIS_CANDIDATE, _C.ANALYZED)


def _v(value) -> str:
    return str(getattr(value, "value", value))


# --- outcome extraction ------------------------------------------------------------------------------------
def extract_outcomes(system, loop_journal: Journal, company_journal: Journal) -> list[OutcomeRecord]:
    s = system
    cat = s.data_status.value
    out: list[OutcomeRecord] = []
    decisions = {e.payload["decision_id"]: decode(e.payload)["record"] for e in s.decisions.journal.entries("decision")}
    for e in loop_journal.entries("trade_review"):
        p = decode(e.payload)
        f = p["facts"]
        rec = decisions.get(f.get("entry_decision_id"), {})
        entry, exit_, qty, fees = f.get("entry_price"), f.get("exit_price"), f.get("qty"), f.get("fees") or Decimal(0)
        pnl = (exit_ - entry) * qty - fees if None not in (entry, exit_, qty) else None
        expected_loss = rec.get("expected_risk")
        deviation = {}
        if pnl is not None and expected_loss is not None and pnl < 0 and -pnl > expected_loss * Decimal("1.1"):
            deviation["loss_beyond_expected_risk"] = str(-pnl - expected_loss)
        out.append(OutcomeRecord(p["review_id"], "trade", e.at, cat, f.get("symbol"), s.timeframe.value,
                                 rec.get("strategy_hash"), None,
                                 {"max_loss": str(expected_loss) if expected_loss is not None else None},
                                 {"pnl": str(pnl) if pnl is not None else None, "exit_reason": f.get("exit_reason"),
                                  "fees": str(fees)}, deviation, False, {"entry_decision_id": f.get("entry_decision_id")}))
    for e in s.research_journal.entries("experiment"):
        p = decode(e.payload)
        out.append(OutcomeRecord(f"exp:{e.hash[:24]}", "experiment", e.at, cat, None, None, p["prereg_hash"][:24],
                                 p["dataset_id"], {"criteria": p["prereg_hash"][:12]},
                                 {"verdict": _v(p["verdict"]), "stage": p["stage"]}, {}, p["stage"] == "holdout",
                                 {"hypothesis_id": p["hypothesis_id"]}))
    for e in s.research_journal.entries("promotion_decision"):
        r = decode(e.payload)["record"]
        out.append(OutcomeRecord(f"promo:{r['record_hash'][:24]}", "promotion", e.at, cat, None, None,
                                 r["challenger_hash"], None, {}, {"approved": r["approved"]}, {}, True,
                                 {"reasons": list(r["reasons"])[:3], "challenger_key": r["challenger_key"]}))
    for e in loop_journal.entries("data_unhealthy"):
        err = str(e.payload.get("error", ""))
        out.append(OutcomeRecord(f"data:{e.hash[:24]}", "data_failure", e.at, cat, facts={"error": err[:200]},
                                 realized={"kind": err.split(":")[0][:60] or "unknown"}))
    for e in company_journal.entries("cycle_step"):
        if e.payload["step"] == "rejected":
            out.append(OutcomeRecord(f"resp:{e.hash[:24]}", "response_rejected", e.at, cat,
                                     facts={"reason": str(e.payload["data"].get("reason", ""))[:200]},
                                     realized={"kind": str(e.payload["data"].get("reason", "")).split(":")[0][:60]}))
    return out


# --- deterministic detectors -------------------------------------------------------------------------------
def _groups(outcomes: list[OutcomeRecord]) -> dict[tuple, list[OutcomeRecord]]:
    g: dict[tuple, list[OutcomeRecord]] = {}

    def add(pattern, key: dict, o: OutcomeRecord):
        g.setdefault((pattern, tuple(sorted(key.items()))), []).append(o)

    for o in outcomes:
        if o.source == "trade" and o.realized.get("pnl") is not None and Decimal(o.realized["pnl"]) < 0:
            add("REPEATED_LOSS", {"strategy": o.strategy, "symbol": o.symbol, "exit_reason": o.realized.get("exit_reason")}, o)
        if o.source == "trade" and o.deviation.get("loss_beyond_expected_risk"):
            add("EXECUTION_DEVIATION", {"strategy": o.strategy, "symbol": o.symbol}, o)
        if o.source == "experiment" and o.realized["verdict"] in ("FAIL", "INSUFFICIENT_EVIDENCE"):
            kind = "HOLDOUT_FAILURE" if o.holdout_derived else "RESEARCH_FAILURE"
            add(kind, {"stage": o.realized["stage"], "verdict": o.realized["verdict"]}, o)
        if o.source == "promotion" and not o.realized["approved"]:
            add("PROMOTION_DENIED", {"challenger": o.facts.get("challenger_key")}, o)
        if o.source == "data_failure":
            add("DATA_FAILURE", {"kind": o.realized["kind"]}, o)
        if o.source == "response_rejected":
            add("CONTRACT_REJECTION", {"kind": o.realized["kind"]}, o)
    return g


_TEMPLATES = {
    "REPEATED_LOSS": "Losses recur for strategy {strategy} on {symbol} with exit reason {exit_reason}",
    "EXECUTION_DEVIATION": "Realized losses exceeded the risk engine's expected maximum for {strategy} on {symbol}",
    "RESEARCH_FAILURE": "Experiments fail at stage {stage} with verdict {verdict}",
    "HOLDOUT_FAILURE": "Holdout evaluations end {verdict} (evaluation outcome; not usable as training feedback)",
    "PROMOTION_DENIED": "Promotion of {challenger} was denied",
    "DATA_FAILURE": "Data failures recur: {kind}",
    "CONTRACT_REJECTION": "Claude responses are rejected by the contract: {kind}",
}


class LearningLedger:
    """Append-only learning ledger (existing Journal class, ``learning.jsonl``). Candidates and their state
    changes are replayed from it; a restart never loses or re-creates learning."""

    def __init__(self, journal: Journal):
        self.journal = journal
        self.outcomes: dict[str, OutcomeRecord] = {}
        self.candidates: dict[str, LearningCandidate] = {}
        for e in journal.entries():
            p = decode(e.payload)
            if e.type == "outcome":
                self.outcomes[p["outcome_id"]] = OutcomeRecord(**p["record"])
            elif e.type == "candidate":
                self.candidates[p["candidate_id"]] = LearningCandidate(
                    p["candidate_id"], p["pattern"], p["key"], p["statement"], [], CandidateState.OBSERVED,
                    p["holdout_derived"])
            elif e.type == "candidate_evidence":
                self.candidates[p["candidate_id"]].evidence.append(p["outcome_id"])
            elif e.type == "candidate_state":
                c = self.candidates[p["candidate_id"]]
                new = CandidateState(p["to"])
                if p["from"] != c.state.value or new not in TRANSITIONS[c.state]:
                    raise LifecycleError(f"learning journal holds illegal transition {p['from']} → {p['to']}")
                c.state = new
                c.history.append((p["from"], p["to"], p["reason"]))
            elif e.type == "candidate_hypothesis":
                self.candidates[p["candidate_id"]].hypothesis_id = p["hypothesis_id"]

    def _transition(self, c: LearningCandidate, new: CandidateState, reason: str) -> None:
        if new not in TRANSITIONS[c.state]:
            raise LifecycleError(f"learning candidate {c.candidate_id}: {c.state.value} → {new.value} not allowed")
        self.journal.append("candidate_state", {"candidate_id": c.candidate_id, "from": c.state.value,
                                                "to": new.value, "reason": reason[:300]})
        c.history.append((c.state.value, new.value, reason))
        c.state = new

    # --- the LEARN stage ------------------------------------------------------------------------------
    def learn(self, system, loop_journal: Journal, company_journal: Journal) -> dict:
        """Idempotent: re-running over unchanged journals writes nothing."""
        new_outcomes = 0
        for o in extract_outcomes(system, loop_journal, company_journal):
            if o.outcome_id not in self.outcomes:
                self.journal.append("outcome", {"outcome_id": o.outcome_id, "record": o})
                self.outcomes[o.outcome_id] = o
                new_outcomes += 1
        for (pattern, key), members in _groups(list(self.outcomes.values())).items():
            cid = "lc_" + sha256_hex({"pattern": pattern, "key": list(key)})[:20]
            c = self.candidates.get(cid)
            if c is None:
                keyd = dict(key)
                holdout = pattern in ("HOLDOUT_FAILURE", "PROMOTION_DENIED")
                statement = _TEMPLATES[pattern].format(**{k: keyd.get(k) for k in keyd})
                self.journal.append("candidate", {"candidate_id": cid, "pattern": pattern, "key": keyd,
                                                  "statement": statement, "holdout_derived": holdout})
                c = self.candidates[cid] = LearningCandidate(cid, pattern, keyd, statement, [], CandidateState.OBSERVED,
                                                             holdout)
            for o in members:
                if o.outcome_id not in c.evidence:          # a source event is evidence once
                    self.journal.append("candidate_evidence", {"candidate_id": cid, "outcome_id": o.outcome_id})
                    c.evidence.append(o.outcome_id)
            n = len(c.evidence)
            if c.state is _C.OBSERVED and n >= 2:
                self._transition(c, _C.ANALYZED, f"{n} independent outcomes: recurring pattern")
            if c.state is _C.ANALYZED and n >= 3 and not c.holdout_derived:
                self._transition(c, _C.HYPOTHESIS_CANDIDATE, f"{n} independent outcomes: eligible for a hypothesis")
        self._sync_research(system)
        return {"new_outcomes": new_outcomes, "candidates": len(self.candidates)}

    def link_hypothesis(self, candidate_id: str, hypothesis_id: str) -> None:
        c = self.candidates[candidate_id]
        if c.hypothesis_id not in (None, hypothesis_id):
            raise LifecycleError(f"{candidate_id} is already linked to {c.hypothesis_id}")
        if c.hypothesis_id is None:
            self.journal.append("candidate_hypothesis", {"candidate_id": candidate_id, "hypothesis_id": hypothesis_id})
            c.hypothesis_id = hypothesis_id

    def _sync_research(self, system) -> None:
        """Advance linked candidates from the research journal's facts — never from Claude's claims."""
        from ati.research.hypothesis import ResearchLog

        log = ResearchLog(system.research_journal)
        market = system.data_status in MARKET_EVIDENCE_STATUSES
        for c in self.candidates.values():
            h = c.hypothesis_id
            if h is None or c.state in (_C.SUPPORTED, _C.REJECTED, _C.INCONCLUSIVE):
                continue
            dev, hold = log.status(h), log.status(f"{h}:holdout")
            if dev != "UNKNOWN" and c.state in (_C.OBSERVED, _C.ANALYZED, _C.HYPOTHESIS_CANDIDATE):
                self._transition(c, _C.PREREGISTERED, f"hypothesis {h} pre-registered")
            if dev.startswith("TESTED:") and c.state is _C.PREREGISTERED:
                self._transition(c, _C.TESTING, f"{h} tested in development")
            if c.state is not _C.TESTING:
                continue
            if dev == "TESTED:FAIL" or hold == "TESTED:FAIL":
                self._transition(c, _C.REJECTED, f"{h}: development {dev}, holdout {hold}")
            elif dev == "TESTED:PASS" and hold == "TESTED:PASS":
                self._transition(c, _C.SUPPORTED, f"{h}: passed development and holdout "
                                 f"({'market' if market else 'non-market'} data)")
            elif hold.startswith("TESTED:") or dev == "TESTED:INSUFFICIENT_EVIDENCE":
                self._transition(c, _C.INCONCLUSIVE, f"{h}: development {dev}, holdout {hold}")

    # --- views -----------------------------------------------------------------------------------------------
    def quality(self, c: LearningCandidate, system) -> Quality:
        supported = c.state is _C.SUPPORTED
        validated = supported and system.data_status in MARKET_EVIDENCE_STATUSES and c.hypothesis_id is not None and any(
            decode(e.payload)["record"]["approved"] for e in system.research_journal.entries("promotion_decision"))
        return c.quality(supported, validated)

    def pattern_class(self, c: LearningCandidate, system) -> PatternClass:
        q = self.quality(c, system)
        if q is Quality.VALIDATED_EVIDENCE:
            return PatternClass.VALIDATED_DOCTRINE
        if q is Quality.SUPPORTED_EVIDENCE:
            return PatternClass.STATISTICALLY_SUPPORTED
        return PatternClass.ONE_OFF if len(c.evidence) <= 1 else PatternClass.RECURRING

    def summary(self, system, limit: int = 8) -> list[dict]:
        ordered = sorted(self.candidates.values(), key=lambda c: (-len(c.evidence), c.candidate_id))[:limit]
        return [{"candidate_id": c.candidate_id, "pattern": c.pattern, "statement": c.statement, "state": c.state.value,
                 "evidence_count": len(c.evidence), "quality": self.quality(c, system).value,
                 "class": self.pattern_class(c, system).value, "research_eligible": c.research_eligible,
                 "holdout_derived": c.holdout_derived, "hypothesis_id": c.hypothesis_id} for c in ordered]
