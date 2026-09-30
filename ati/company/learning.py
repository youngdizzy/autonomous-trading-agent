"""Outcome learning — the LEARN stage of the company cycle.

    OUTCOME (journals) → OutcomeRecord (facts + provenance) → deterministic pattern detection → LEARNING CANDIDATE
      → [Claude cites it in a designed RESEARCH_REQUEST] → PROMOTED_TO_HYPOTHESIS → existing pre-registration,
        experiment, validation and promotion machinery (research status is read from the ResearchLog)

Sources, all existing journals: paper trades (loop ``trade_review`` + the decision record), experiments and
promotion decisions (research journal), adversarial robustness findings, data failures, rejected Claude
responses, and assumption-monitor shifts (company journal).

A learning candidate keeps FACT (what happened, computed from recorded outcomes), INTERPRETATION (what *may*
explain it, a hedged template) and PROPOSED_QUESTION (what to test) apart. It is a record, never a rule: this
module holds no reference to the strategy registry, risk engine or execution engine. Its only memory write is a
non-doctrinal HYPOTHESIS entry through MemoryStore's evidence gates, once per recurring trade pattern.

Status: OBSERVED → ANALYZED (2 outcomes) → HYPOTHESIS_CANDIDATE (3+) → PROMOTED_TO_HYPOTHESIS (linked hypothesis
pre-registered); REJECTED when later matching outcomes contradict the pattern. Terminal states stay visible.
Classification: ONE_OFF · POSSIBLE_PATTERN · RECURRING_PATTERN · SUPPORTED_PATTERN (linked hypothesis passed
development and holdout) · VALIDATED_EFFECT (as supported, on market data, with the promotion gate approving that
hypothesis's own challenger) — only the last is doctrine-eligible, and only through the research workflow.

Holdout isolation: holdout and promotion outcomes are recorded (failures stay visible) but are marked
``holdout_derived`` and can never seed a hypothesis; their dataset identities never appear in Claude-facing views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from ati.core.canonical import sha256_hex
from ati.core.errors import LifecycleError
from ati.core.time import parse_utc
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
    SUPPORTED_PATTERN = "SUPPORTED_PATTERN"    # the linked hypothesis passed development AND holdout
    VALIDATED_EFFECT = "VALIDATED_EFFECT"      # as supported, on market data, with the promotion gate approving
                                               # *that hypothesis's* challenger — the only doctrine-eligible class


class CandidateState(str, Enum):
    """Status of the *learning candidate* itself. What happened to a hypothesis it produced is research status,
    derived from the ResearchLog (``research_outcome``), never stored a second time here."""
    OBSERVED = "OBSERVED"
    ANALYZED = "ANALYZED"
    HYPOTHESIS_CANDIDATE = "HYPOTHESIS_CANDIDATE"
    PROMOTED_TO_HYPOTHESIS = "PROMOTED_TO_HYPOTHESIS"
    REJECTED = "REJECTED"                      # contradicted by later outcomes of the same kind


_C = CandidateState
TRANSITIONS: dict[CandidateState, frozenset[CandidateState]] = {
    _C.OBSERVED: frozenset({_C.ANALYZED, _C.PROMOTED_TO_HYPOTHESIS, _C.REJECTED}),
    _C.ANALYZED: frozenset({_C.HYPOTHESIS_CANDIDATE, _C.PROMOTED_TO_HYPOTHESIS, _C.REJECTED}),
    _C.HYPOTHESIS_CANDIDATE: frozenset({_C.PROMOTED_TO_HYPOTHESIS, _C.REJECTED}),
    _C.PROMOTED_TO_HYPOTHESIS: frozenset(), _C.REJECTED: frozenset(),   # terminal, visible forever
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
    candidate_id: str              # learning_id
    pattern: str                   # detector name
    key: dict                      # what the pattern is conditioned on
    statement: str                 # deterministic pattern label — never free text from Claude
    evidence: list[str]            # distinct outcome ids
    state: CandidateState
    holdout_derived: bool
    hypothesis_id: str | None = None
    history: list[tuple[str, str, str]] = field(default_factory=list)
    interpretation: str = ""       # what *may* explain it (template, hedged) — never a fact
    question: str = ""             # what should be tested
    created_at: str = ""

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


_INTERPRETATIONS = {
    "REPEATED_LOSS": "The strategy's entries may be unreliable in the conditions that preceded these exits.",
    "REGIME_FAILURE": "The strategy's edge may depend on market regime; regime {regime} may reduce its reliability.",
    "EXECUTION_DEVIATION": "Realized execution may differ from what the risk model assumes for this strategy.",
    "UNEXPECTED_COSTS": "Trading costs may be large relative to this strategy's typical gross move.",
    "DRAWDOWN_CLUSTER": "Losses may be serially dependent (clustered), not independent draws.",
    "STRATEGY_DEGRADATION": "The strategy's behaviour may have changed relative to its earlier trades.",
    "RESEARCH_FAILURE": "Research at this stage may be asking questions the available data cannot answer.",
    "HOLDOUT_FAILURE": "Development results may be overfitted; this is an evaluation observation only.",
    "ROBUSTNESS_FAILURE": "Results may be fragile to this robustness perturbation.",
    "PARAMETER_INSTABILITY": "Results may depend on narrowly chosen parameters.",
    "PROMOTION_DENIED": "Candidates may not be meeting the promotion evidence standard (evaluation observation only).",
    "DATA_FAILURE": "The data path may be unreliable for this failure kind.",
    "CONTRACT_REJECTION": "Claude responses may be misunderstanding the action contract.",
    "CONDITION_SHIFT": "A research assumption may no longer hold ({assumption}).",
}
_QUESTIONS = {
    "REPEATED_LOSS": "Does a pre-registered filter on the entry condition improve out-of-sample expectancy versus the baseline?",
    "REGIME_FAILURE": "Is the baseline's out-of-sample expectancy lower in regime {regime} than in other regimes (REGIME experiment)?",
    "EXECUTION_DEVIATION": "Does the baseline's edge survive stressed execution costs or delay (EXECUTION experiment)?",
    "UNEXPECTED_COSTS": "Does the baseline's edge survive a cost multiplier of 2 (EXECUTION experiment)?",
    "DRAWDOWN_CLUSTER": "Does a different per-trade risk fraction change drawdown clustering in backtest (RISK experiment)?",
    "STRATEGY_DEGRADATION": "Does the strategy's out-of-sample edge persist on the most recent unsealed data?",
    "RESEARCH_FAILURE": "Is more data, or a different design, needed before this question is testable?",
    "HOLDOUT_FAILURE": "Is the research process overfitting? (answerable only with new, unused holdout periods)",
    "ROBUSTNESS_FAILURE": "Does a design that addresses this perturbation survive walk-forward and adversarial review?",
    "PARAMETER_INSTABILITY": "Is there a parameter region that is stable under perturbation (SINGLE_VARIABLE experiment)?",
    "PROMOTION_DENIED": "Which evidence requirement is most often missing? (process question; no candidate change)",
    "DATA_FAILURE": "Can this data failure be prevented or detected earlier? (operational question)",
    "CONTRACT_REJECTION": "Is the action contract documented clearly enough in the context? (operational question)",
    "CONDITION_SHIFT": "Does the baseline's edge hold under the shifted condition ({assumption})?",
}
_TRADE_PATTERNS = frozenset({"REPEATED_LOSS", "REGIME_FAILURE", "EXECUTION_DEVIATION", "UNEXPECTED_COSTS",
                             "DRAWDOWN_CLUSTER", "STRATEGY_DEGRADATION"})


def _dataset_view(o: OutcomeRecord) -> str | None:
    """Holdout identities stay in the learning journal (audit) but never appear in views that reach Claude."""
    return "SEALED_HOLDOUT" if o.holdout_derived and o.source == "experiment" else o.dataset_id


class LearningLedger:
    """Append-only learning ledger (existing Journal class, ``learning.jsonl``). Candidates and their state
    changes are replayed from it; a restart never loses or re-creates learning."""

    def __init__(self, journal: Journal):
        self.journal = journal
        self.outcomes: dict[str, OutcomeRecord] = {}
        self.candidates: dict[str, LearningCandidate] = {}
        self.memory_written: dict[str, str] = {}
        for e in journal.entries():
            p = decode(e.payload)
            if e.type == "outcome":
                self.outcomes[p["outcome_id"]] = OutcomeRecord(**p["record"])
            elif e.type == "candidate":
                self.candidates[p["candidate_id"]] = LearningCandidate(
                    p["candidate_id"], p["pattern"], p["key"], p["statement"], [], CandidateState.OBSERVED,
                    p["holdout_derived"], interpretation=p.get("interpretation", ""), question=p.get("question", ""),
                    created_at=e.at)
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
            elif e.type == "candidate_memory":
                self.memory_written[p["candidate_id"]] = p["entry_id"]

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
                fmt = {k: keyd.get(k) for k in keyd}
                statement = _TEMPLATES[pattern].format(**fmt)
                interpretation = _INTERPRETATIONS[pattern].format(**fmt)
                question = _QUESTIONS[pattern].format(**fmt)
                entry = self.journal.append("candidate", {"candidate_id": cid, "pattern": pattern, "key": keyd,
                                                          "statement": statement, "holdout_derived": holdout,
                                                          "interpretation": interpretation, "question": question})
                c = self.candidates[cid] = LearningCandidate(cid, pattern, keyd, statement, [], CandidateState.OBSERVED,
                                                             holdout, interpretation=interpretation, question=question,
                                                             created_at=entry.at)
            for o in members:
                if o.outcome_id not in c.evidence:          # a source event is evidence once
                    self.journal.append("candidate_evidence", {"candidate_id": cid, "outcome_id": o.outcome_id})
                    c.evidence.append(o.outcome_id)
            n = len(c.evidence)
            contradiction = self._contradiction(c)
            if contradiction and c.state in (_C.OBSERVED, _C.ANALYZED, _C.HYPOTHESIS_CANDIDATE):
                self._transition(c, _C.REJECTED, contradiction)
                continue
            if c.state is _C.OBSERVED and n >= 2:
                self._transition(c, _C.ANALYZED, f"{n} independent outcomes: possible pattern")
            if c.state is _C.ANALYZED and n >= 3 and not c.holdout_derived:
                self._transition(c, _C.HYPOTHESIS_CANDIDATE, f"{n} independent outcomes: recurring pattern, eligible "
                                                              "for a hypothesis")
        self._sync_research(system)
        memory = self._remember(system)
        return {"new_outcomes": new_outcomes, "candidates": len(self.candidates), "memory_entries": memory}

    def _matching_trades(self, c: LearningCandidate) -> list[OutcomeRecord]:
        k = c.key
        out = []
        for o in self.outcomes.values():
            if o.source != "trade" or _pnl(o) is None or o.strategy != k.get("strategy"):
                continue
            if c.pattern == "REPEATED_LOSS" and (o.symbol, o.realized.get("exit_reason")) != (k.get("symbol"), k.get("exit_reason")):
                continue
            if c.pattern == "REGIME_FAILURE" and o.facts.get("regime") != k.get("regime"):
                continue
            out.append(o)
        return out

    def _contradiction(self, c: LearningCandidate) -> str | None:
        """Contradictory evidence: once enough trades share the pattern's conditions, a loss pattern whose
        matching trades are mostly profitable is REJECTED (kept visible), not silently dropped or kept."""
        if c.pattern not in ("REPEATED_LOSS", "REGIME_FAILURE"):
            return None
        matching = self._matching_trades(c)
        wins = sum(1 for o in matching if _pnl(o) > 0)
        if len(matching) >= 4 and wins * 2 > len(matching):
            return f"contradicted by later outcomes: {wins} of {len(matching)} matching trades were profitable"
        return None

    def _remember(self, system) -> int:
        """Learning enters memory only through MemoryStore's own evidence gates, and only as HYPOTHESIS (never
        doctrine): once per recurring (3+) trade pattern, citing the registered trade_review evidence. The entry
        is deterministic (timestamp = the third outcome's), so a replay returns the same entry id."""
        from ati.memory.store import MemoryEntry, MemoryKind

        written = 0
        for c in self.candidates.values():
            if c.candidate_id in self.memory_written or c.pattern not in _TRADE_PATTERNS or len(c.evidence) < 3:
                continue
            outcomes = [self.outcomes[i] for i in c.evidence[:3]]
            refs = [system.evidence.resolve(o.outcome_id) for o in outcomes if o.outcome_id in system.evidence]
            if len(refs) < 3:
                continue
            created = max(max(r.available_at for r in refs), parse_utc(outcomes[-1].at))
            tag = "" if system.data_status in MARKET_EVIDENCE_STATUSES else f"[{system.data_status.value}] "
            statement = (f"{tag}RECURRING_PATTERN (not validated): {self.fact(c)} Interpretation (untested): "
                         f"{c.interpretation} Question: {c.question}")[:1900]
            entry = MemoryEntry(MemoryKind.HYPOTHESIS, statement, created, f"learning:{c.candidate_id}", tuple(refs),
                                0.2, (), tuple(sorted({o.strategy for o in outcomes if o.strategy})), "")
            entry_id = system.memory.add(entry)
            self.journal.append("candidate_memory", {"candidate_id": c.candidate_id, "entry_id": entry_id})
            self.memory_written[c.candidate_id] = entry_id
            written += 1
        return written

    def link_hypothesis(self, candidate_id: str, hypothesis_id: str) -> None:
        c = self.candidates[candidate_id]
        if c.hypothesis_id not in (None, hypothesis_id):
            raise LifecycleError(f"{candidate_id} is already linked to {c.hypothesis_id}")
        if c.hypothesis_id is None:
            self.journal.append("candidate_hypothesis", {"candidate_id": candidate_id, "hypothesis_id": hypothesis_id})
            c.hypothesis_id = hypothesis_id

    def _sync_research(self, system) -> None:
        """A linked candidate becomes PROMOTED_TO_HYPOTHESIS once the ResearchLog holds the pre-registration —
        a fact, never Claude's claim. What the hypothesis then yields is research status (see research_outcome)."""
        from ati.research.hypothesis import ResearchLog

        log = ResearchLog(system.research_journal)
        for c in self.candidates.values():
            h = c.hypothesis_id
            if h is None or c.state in (_C.PROMOTED_TO_HYPOTHESIS, _C.REJECTED):
                continue
            if log.status(h) != "UNKNOWN":
                self._transition(c, _C.PROMOTED_TO_HYPOTHESIS, f"hypothesis {h} pre-registered")

    def research_outcome(self, c: LearningCandidate, system) -> str:
        """NONE | PREREGISTERED | TESTING | SUPPORTED | REJECTED | INCONCLUSIVE — derived from the ResearchLog."""
        from ati.research.hypothesis import ResearchLog

        if c.hypothesis_id is None:
            return "NONE"
        log = ResearchLog(system.research_journal)
        h = c.hypothesis_id
        dev, hold = log.status(h), log.status(f"{h}:holdout")
        if dev == "UNKNOWN":
            return "NONE"
        if dev == "PREREGISTERED":
            return "PREREGISTERED"
        diagnostic = any(e["hypothesis_id"] == h and str(e["stage"]).startswith("diagnostic:") for e in log.experiments)
        if dev == "TESTED:FAIL" or hold == "TESTED:FAIL":
            return "REJECTED"
        if diagnostic:
            return "INCONCLUSIVE"    # development-only evidence cannot support an effect
        if dev == "TESTED:PASS" and hold == "TESTED:PASS":
            return "SUPPORTED"
        if hold.startswith("TESTED:") or dev == "TESTED:INSUFFICIENT_EVIDENCE":
            return "INCONCLUSIVE"
        return "TESTING"

    def _validated(self, c: LearningCandidate, system) -> bool:
        """VALIDATED requires: supported research outcome, market-category data, and an approved promotion of
        the challenger *this hypothesis's holdout pre-registration locked* — not any promotion in the journal."""
        from ati.research.hypothesis import ResearchLog

        if system.data_status not in MARKET_EVIDENCE_STATUSES or self.research_outcome(c, system) != "SUPPORTED":
            return False
        log = ResearchLog(system.research_journal)
        challenger = log.get(f"{c.hypothesis_id}:holdout").strategy_hash
        return any(decode(e.payload)["record"]["approved"] and e.payload["record"]["challenger_hash"] == challenger
                   for e in system.research_journal.entries("promotion_decision"))

    # --- views -----------------------------------------------------------------------------------------------
    def quality(self, c: LearningCandidate, system) -> Quality:
        supported = self.research_outcome(c, system) == "SUPPORTED"
        return c.quality(supported, supported and self._validated(c, system))

    def pattern_class(self, c: LearningCandidate, system) -> PatternClass:
        q = self.quality(c, system)
        if q is Quality.VALIDATED_EVIDENCE:
            return PatternClass.VALIDATED_EFFECT
        if q is Quality.SUPPORTED_EVIDENCE:
            return PatternClass.SUPPORTED_PATTERN
        n = len(c.evidence)
        return PatternClass.ONE_OFF if n <= 1 else PatternClass.POSSIBLE_PATTERN if n == 2 else PatternClass.RECURRING_PATTERN

    def fact(self, c: LearningCandidate) -> str:
        """FACT: what objectively happened, computed from the recorded outcomes (no interpretation)."""
        outs = [self.outcomes[i] for i in c.evidence if i in self.outcomes]
        n = len(outs)
        if c.pattern in _TRADE_PATTERNS:
            pnls = [_pnl(o) for o in outs if _pnl(o) is not None]
            lost = sum(1 for p in pnls if p < 0)
            cat = outs[0].data_category if outs else "UNKNOWN"
            key = ", ".join(f"{k}={v}" for k, v in sorted(c.key.items()))
            return (f"{n} {cat} paper trade outcome(s) matched [{key}]: {lost} lost, {len(pnls) - lost} did not; "
                    f"net P&L {sum(pnls, Decimal(0)):.2f}.")
        if c.pattern in ("RESEARCH_FAILURE", "HOLDOUT_FAILURE"):
            return f"{n} experiment(s) at stage {c.key.get('stage')} ended {c.key.get('verdict')}."
        if c.pattern in ("ROBUSTNESS_FAILURE", "PARAMETER_INSTABILITY"):
            return f"{n} adversarial review(s) failed the check: {c.key.get('question')}"
        return f"{n} recorded {c.pattern.lower()} event(s): " + ", ".join(f"{k}={v}" for k, v in sorted(c.key.items()))

    def evidence_references(self, c: LearningCandidate, system) -> list[dict]:
        refs = []
        for i in c.evidence:
            o = self.outcomes.get(i)
            if o is None:
                continue
            refs.append({"outcome_id": i, "source_type": o.source, "registered_evidence": i if i in system.evidence else None,
                         "decision_id": o.facts.get("trade_id"), "hypothesis_id": o.facts.get("hypothesis_id"),
                         "dataset_id": _dataset_view(o), "at": o.at})
        return refs

    def record(self, c: LearningCandidate, system) -> dict:
        """The learning-candidate contract: identity, provenance, FACT / INTERPRETATION / QUESTION kept apart."""
        first = self.outcomes.get(c.evidence[0]) if c.evidence else None
        return {
            "learning_id": c.candidate_id,
            "source_event": {"source_type": first.source if first else None, "source_id": first.outcome_id if first else None},
            "provenance": {"source_type": first.source if first else None, "source_id": first.outcome_id if first else None,
                           "strategy_fingerprint": first.strategy if first else None,
                           "dataset_identity": _dataset_view(first) if first else None,
                           "data_category": first.data_category if first else system.data_status.value,
                           "created_at": c.created_at},
            "pattern": c.pattern,
            "FACT": self.fact(c),
            "INTERPRETATION": c.interpretation,
            "PROPOSED_QUESTION": c.question,
            "evidence_references": self.evidence_references(c, system),
            "confidence_class": self.quality(c, system).value,
            "classification": self.pattern_class(c, system).value,
            "status": c.state.value,
            "research_eligible": c.research_eligible,
            "holdout_derived": c.holdout_derived,
            "hypothesis": {"hypothesis_id": c.hypothesis_id, "research_outcome": self.research_outcome(c, system)},
            "memory_entry": self.memory_written.get(c.candidate_id),
        }

    def summary(self, system, limit: int = 8) -> list[dict]:
        ordered = sorted(self.candidates.values(), key=lambda c: (-len(c.evidence), c.candidate_id))[:limit]
        out = []
        for c in ordered:
            r = self.record(c, system)
            r["evidence_references"] = [e["outcome_id"] for e in r["evidence_references"]][:6]
            out.append(r | {"candidate_id": c.candidate_id, "state": c.state.value, "evidence_count": len(c.evidence),
                            "quality": r["confidence_class"], "class": r["classification"],
                            "hypothesis_id": c.hypothesis_id, "statement": c.statement})
        return out
