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
    ONE_OFF = "ONE_OFF"                        # 1 outcome: never doctrine, never a rule
    POSSIBLE_PATTERN = "POSSIBLE_PATTERN"      # 2 independent outcomes
    RECURRING_PATTERN = "RECURRING_PATTERN"    # 3+ independent outcomes
    SUPPORTED_EFFECT = "SUPPORTED_EFFECT"      # linked hypothesis passed development AND holdout
    VALIDATED_EFFECT = "VALIDATED_EFFECT"      # as supported, on market data, with an approved promotion


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
def _regime_at(system, symbol: str, cutoff) -> str | None:
    """Point-in-time regime label (volatility × trend) at a decision cutoff, from stored bars up to the cutoff."""
    from ati.data.dataset import Dataset
    from ati.research.robustness import regime_labels

    bars = [c for c in system.store.series(system.provider.name, symbol, system.timeframe) if c.close_time <= cutoff]
    if len(bars) < 160:
        return None
    ds = Dataset.build(bars[-300:], data_version="learning-view", realization=getattr(system.provider, "realization", "observed"))
    label = regime_labels(ds).get(bars[-1].close_time)
    return "/".join(label) if label else None


def extract_outcomes(system, loop_journal: Journal, company_journal: Journal,
                     known: frozenset[str] | set[str] = frozenset()) -> list[OutcomeRecord]:
    """``known``: outcome ids already recorded — skipped before any expensive derivation (regime labels)."""
    s = system
    cat = s.data_status.value
    out: list[OutcomeRecord] = []
    decisions = {e.payload["decision_id"]: decode(e.payload)["record"] for e in s.decisions.journal.entries("decision")}
    for e in loop_journal.entries("trade_review"):
        if e.payload["review_id"] in known:
            continue
        p = decode(e.payload)
        f = p["facts"]
        rec = decisions.get(f.get("entry_decision_id"), {})
        entry, exit_, qty, fees = f.get("entry_price"), f.get("exit_price"), f.get("qty"), f.get("fees") or Decimal(0)
        gross = (exit_ - entry) * qty if None not in (entry, exit_, qty) else None
        pnl = gross - fees if gross is not None else None
        expected_loss = rec.get("expected_risk")
        mctx = dict(rec.get("market_context") or ())
        ref = Decimal(mctx["last_price"]) if "last_price" in mctx else None
        slippage = (entry - ref) / ref if entry is not None and ref else None
        modelled = (Decimal(mctx.get("est_spread_rate", "0")) + Decimal(mctx.get("est_slippage_rate", "0"))) if mctx else None
        deviation = {}
        if pnl is not None and expected_loss is not None and pnl < 0 and -pnl > expected_loss * Decimal("1.1"):
            deviation["loss_beyond_expected_risk"] = str(-pnl - expected_loss)
        if slippage is not None and modelled and slippage > 2 * modelled:
            deviation["entry_slippage_beyond_model"] = str(slippage)
        if gross is not None and gross != 0 and fees / abs(gross) > Decimal("0.5"):
            deviation["costs_over_half_of_gross"] = str(fees / abs(gross))
        cutoff = rec.get("available_information_cutoff")
        out.append(OutcomeRecord(
            p["review_id"], "trade", e.at, cat, f.get("symbol"), s.timeframe.value, rec.get("strategy_hash"), None,
            {"max_loss": str(expected_loss) if expected_loss is not None else None,
             "expected_reward": str(rec["expected_reward"]) if rec.get("expected_reward") is not None else None,
             "modelled_cost_rate": str(modelled) if modelled is not None else None},
            {"pnl": str(pnl) if pnl is not None else None, "gross": str(gross) if gross is not None else None,
             "exit_reason": f.get("exit_reason"), "fees": str(fees), "entry": str(entry), "exit": str(exit_),
             "entry_slippage": str(slippage) if slippage is not None else None},
            deviation, False,
            {"trade_id": f.get("entry_decision_id"), "thesis": (rec.get("thesis") or "")[:200],
             "risk_state": {"approved_size": str(rec.get("approved_size")), "limits_hash": rec.get("risk_limits_hash")},
             "regime": _regime_at(s, f["symbol"], cutoff) if cutoff and f.get("symbol") else None,
             "execution_quality": "WITHIN_MODEL" if "entry_slippage_beyond_model" not in deviation else "DEVIATED"}))
    for e in s.research_journal.entries("experiment"):
        p = decode(e.payload)
        out.append(OutcomeRecord(f"exp:{e.hash[:24]}", "experiment", e.at, cat, None, None, p["prereg_hash"][:24],
                                 p["dataset_id"], {"criteria": p["prereg_hash"][:12]},
                                 {"verdict": _v(p["verdict"]), "stage": p["stage"]}, {}, p["stage"] == "holdout",
                                 {"hypothesis_id": p["hypothesis_id"]}))
    for e in s.research_journal.entries("adversarial_report"):
        r = decode(e.payload)["report"]
        failed = sorted(o["question"] for o in r["objections"] if _v(o["verdict"]) == "FAIL")
        if failed:   # development-partition robustness evidence (the adversarial review never sees the holdout)
            out.append(OutcomeRecord(f"adv:{e.hash[:24]}", "robustness", e.at, cat, None, None, r["strategy_hash"],
                                     r["dataset_id"], {}, {"failed": failed}, {}, False, {"strategy_key": r["strategy_key"]}))
    for e in s.research_journal.entries("promotion_decision"):
        r = decode(e.payload)["record"]
        out.append(OutcomeRecord(f"promo:{r['record_hash'][:24]}", "promotion", e.at, cat, None, None,
                                 r["challenger_hash"], None, {}, {"approved": r["approved"]}, {}, True,
                                 {"reasons": list(r["reasons"])[:3], "challenger_key": r["challenger_key"]}))
    for e in loop_journal.entries("data_unhealthy"):
        err = str(e.payload.get("error", ""))
        out.append(OutcomeRecord(f"data:{e.hash[:24]}", "data_failure", e.at, cat, facts={"error": err[:200]},
                                 realized={"kind": err.split(":")[0][:60] or "unknown"}))
    seen_shift_days: set[tuple[str, str]] = set()
    for e in company_journal.entries("cycle_step"):
        if e.payload["step"] == "rejected":
            out.append(OutcomeRecord(f"resp:{e.hash[:24]}", "response_rejected", e.at, cat,
                                     facts={"reason": str(e.payload["data"].get("reason", ""))[:200]},
                                     realized={"kind": str(e.payload["data"].get("reason", "")).split(":")[0][:60]}))
        elif e.payload["step"] == "conditions":
            for m in e.payload["data"]["monitors"]:
                day = (m["assumption"], e.at[:10])   # a persisting shift is one observation per day, not one per cycle
                if m["status"] == "SHIFT_DETECTED" and day not in seen_shift_days:
                    seen_shift_days.add(day)
                    out.append(OutcomeRecord(f"cond:{m['assumption']}:{e.at[:10]}", "condition_shift", e.at, cat,
                                             realized={"assumption": m["assumption"], "statistic": m["statistic"]}))
    return out


# --- deterministic detectors -------------------------------------------------------------------------------
def _pnl(o: OutcomeRecord) -> Decimal | None:
    return Decimal(o.realized["pnl"]) if o.realized.get("pnl") not in (None, "None") else None


def _groups(outcomes: list[OutcomeRecord]) -> dict[tuple, list[OutcomeRecord]]:
    g: dict[tuple, list[OutcomeRecord]] = {}

    def add(pattern, key: dict, o: OutcomeRecord):
        g.setdefault((pattern, tuple(sorted(key.items()))), []).append(o)

    trades_by_strategy: dict[str, list[OutcomeRecord]] = {}
    for o in outcomes:
        pnl = _pnl(o) if o.source == "trade" else None
        if o.source == "trade" and pnl is not None:
            trades_by_strategy.setdefault(o.strategy, []).append(o)
        if pnl is not None and pnl < 0:
            add("REPEATED_LOSS", {"strategy": o.strategy, "symbol": o.symbol, "exit_reason": o.realized.get("exit_reason")}, o)
            if o.facts.get("regime"):
                add("REGIME_FAILURE", {"strategy": o.strategy, "regime": o.facts["regime"]}, o)
        if o.source == "trade" and (o.deviation.get("loss_beyond_expected_risk") or
                                    o.deviation.get("entry_slippage_beyond_model")):
            add("EXECUTION_DEVIATION", {"strategy": o.strategy, "symbol": o.symbol}, o)
        if o.source == "trade" and o.deviation.get("costs_over_half_of_gross"):
            add("UNEXPECTED_COSTS", {"strategy": o.strategy, "symbol": o.symbol}, o)
        if o.source == "experiment" and o.realized["verdict"] in ("FAIL", "INSUFFICIENT_EVIDENCE"):
            kind = "HOLDOUT_FAILURE" if o.holdout_derived else "RESEARCH_FAILURE"
            add(kind, {"stage": o.realized["stage"], "verdict": o.realized["verdict"]}, o)
        if o.source == "robustness":
            for q in o.realized["failed"]:
                if "parameter" in q:
                    add("PARAMETER_INSTABILITY", {"question": q}, o)
                elif any(w in q for w in ("costs", "delay", "regimes", "time periods", "liquidity")):
                    add("ROBUSTNESS_FAILURE", {"question": q}, o)
        if o.source == "promotion" and not o.realized["approved"]:
            add("PROMOTION_DENIED", {"challenger": o.facts.get("challenger_key")}, o)
        if o.source == "data_failure":
            add("DATA_FAILURE", {"kind": o.realized["kind"]}, o)
        if o.source == "response_rejected":
            add("CONTRACT_REJECTION", {"kind": o.realized["kind"]}, o)
        if o.source == "condition_shift":
            add("CONDITION_SHIFT", {"assumption": o.realized["assumption"]}, o)
    for strategy, trades in trades_by_strategy.items():
        trades.sort(key=lambda o: o.at)
        streak: list[OutcomeRecord] = []
        for o in trades:                       # drawdown cluster: 3+ consecutive losing trades
            streak = streak + [o] if _pnl(o) < 0 else []
            if len(streak) >= 3:
                for m in streak:
                    add("DRAWDOWN_CLUSTER", {"strategy": strategy}, m)
        if len(trades) >= 10:                  # degradation: later half worse than earlier half, and negative
            half = len(trades) // 2
            early = sum(_pnl(o) for o in trades[:half]) / half
            late_trades = trades[half:]
            late = sum(_pnl(o) for o in late_trades) / len(late_trades)
            if late < early and late < 0:
                for m in late_trades:
                    add("STRATEGY_DEGRADATION", {"strategy": strategy}, m)
    for k in list(g):                          # an outcome is evidence once per pattern
        seen, uniq = set(), []
        for o in g[k]:
            if o.outcome_id not in seen:
                seen.add(o.outcome_id)
                uniq.append(o)
        g[k] = uniq
    return g


_TEMPLATES = {
    "REPEATED_LOSS": "Losses recur for strategy {strategy} on {symbol} with exit reason {exit_reason}",
    "REGIME_FAILURE": "Losses of strategy {strategy} concentrate in regime {regime}",
    "EXECUTION_DEVIATION": "Paper execution deviated from the risk engine's expectation for {strategy} on {symbol}",
    "UNEXPECTED_COSTS": "Costs consumed more than half of gross P&L for {strategy} on {symbol}",
    "DRAWDOWN_CLUSTER": "Losing trades of strategy {strategy} cluster (3+ consecutive losses)",
    "STRATEGY_DEGRADATION": "Recent trades of strategy {strategy} perform worse than earlier ones and are net negative",
    "RESEARCH_FAILURE": "Experiments fail at stage {stage} with verdict {verdict}",
    "HOLDOUT_FAILURE": "Holdout evaluations end {verdict} (evaluation outcome; not usable as training feedback)",
    "ROBUSTNESS_FAILURE": "Adversarial robustness check fails repeatedly: {question}",
    "PARAMETER_INSTABILITY": "Results are unstable under parameter perturbation: {question}",
    "PROMOTION_DENIED": "Promotion of {challenger} was denied",
    "DATA_FAILURE": "Data failures recur: {kind}",
    "CONTRACT_REJECTION": "Claude responses are rejected by the contract: {kind}",
    "CONDITION_SHIFT": "Assumption monitor reports a shift: {assumption}",
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
        for o in extract_outcomes(system, loop_journal, company_journal, set(self.outcomes)):
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
            diagnostic = any(e["hypothesis_id"] == h and str(e["stage"]).startswith("diagnostic:") for e in log.experiments)
            if diagnostic and dev != "TESTED:FAIL":
                self._transition(c, _C.INCONCLUSIVE, f"{h}: development-only diagnostic {dev}; development evidence "
                                                     "alone cannot support an effect")
            elif dev == "TESTED:FAIL" or hold == "TESTED:FAIL":
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
            return PatternClass.VALIDATED_EFFECT
        if q is Quality.SUPPORTED_EVIDENCE:
            return PatternClass.SUPPORTED_EFFECT
        n = len(c.evidence)
        return PatternClass.ONE_OFF if n <= 1 else PatternClass.POSSIBLE_PATTERN if n == 2 else PatternClass.RECURRING_PATTERN

    def summary(self, system, limit: int = 8) -> list[dict]:
        ordered = sorted(self.candidates.values(), key=lambda c: (-len(c.evidence), c.candidate_id))[:limit]
        return [{"candidate_id": c.candidate_id, "pattern": c.pattern, "statement": c.statement, "state": c.state.value,
                 "evidence_count": len(c.evidence), "quality": self.quality(c, system).value,
                 "class": self.pattern_class(c, system).value, "research_eligible": c.research_eligible,
                 "holdout_derived": c.holdout_derived, "hypothesis_id": c.hypothesis_id} for c in ordered]
