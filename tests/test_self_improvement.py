"""Company 1.0 Phase 1 — self-improvement engine tests.

Outcome learning, learning-candidate lifecycle, experiment design and baseline discipline, the objective
contract, the research budget, autonomy ceiling, multi-source DATA_CONFLICT, text-evidence layering and the
scorecard. All data is MOCK; nothing here is evidence about real markets.
"""

import dataclasses
import inspect
import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ati.company import autonomy, budget, learning, objectives, scorecard
from ati.company.control import CompanyState
from ati.company.learning import CandidateState, LearningCandidate, LearningLedger, OutcomeRecord, Quality
from ati.core.errors import DataConflictError, LifecycleError, LiveTradingDisabled, TextEvidenceError
from ati.core.time import FixedClock
from ati.ledger.journal import Journal
from ati.market.conflict import RESOLVE_ACK, ConflictRegister, compare
from ati.market.models import Timeframe
from ati.memory.textual import ExtractedFact, Interpretation, RawSource, TextEvidence
from ati.research.hypothesis import ResearchLog
from ati.research.metrics import Metrics
from tests.helpers import T0
from tests.test_company_control import MECHANICS, counts, plane, reply, research_payload, with_history

ROOT = Path(__file__).resolve().parents[1]


def experiment(kind="SINGLE_VARIABLE", params=None, iv=None, **over):
    params = params if params is not None else {"fast": 20, "slow": 50, "atr_period": 14, "stop_atr": 3.0}
    body = {"type": kind, "independent_variables": iv or ["fast"], "dependent_variable": "expectancy_r",
            "controls": "same data, costs, walk-forward windows and criteria as the baseline",
            "failure_criteria": "expectancy not above baseline on development data",
            "stopping_criteria": "one pre-registered run; no retries",
            "design_rationale": "one variable isolates the effect of entry responsiveness", "candidate_params": params}
    return body | over


def condition_experiment(kind, condition, **over):
    body = {k: v for k, v in experiment(kind, iv=list(condition)).items() if k != "candidate_params"}
    return body | {"condition": condition, "design_rationale": f"{kind.lower()} question about the fixed baseline"} | over


def designed(hid="H-co-1", exp=None, **over):
    """A RESEARCH_REQUEST carrying an experiment design (motivation and mechanism are then mandatory)."""
    return research_payload(hid, experiment=exp if exp is not None else experiment(),
                            motivation="recurring observation worth testing",
                            expected_mechanism="the stated variable changes when and how often trends are captured") | over


def metrics(**over) -> Metrics:
    base = dict(n_trades=60, net_pnl=500.0, net_return=0.05, max_drawdown=0.08, expectancy_r=0.2, win_rate=0.45,
                profit_factor=1.3, sharpe_annualized=0.9, mean_trade_return=0.001, t_stat_trade_return=1.5,
                exposure=0.3, total_fees=50.0, total_spread_slippage=20.0, cost_share_of_gross=0.2,
                top5_profit_share=0.4, partial_fills=0)
    return Metrics(**(base | over))


PASSING = {"walk_forward_oos": "PASS", "adversarial": "NON_BLOCKING", "holdout": "PASS"}


# --- objective contract (no single metric) -----------------------------------------------------------------
class TestObjectiveContract:
    def test_constraint_violation_beats_any_improvement(self):
        better_but_risky = metrics(net_return=0.5, sharpe_annualized=3.0, max_drawdown=0.35)
        out = objectives.compare(metrics(), better_but_risky, PASSING)
        assert out["conclusion"] == "NOT_AN_IMPROVEMENT" and any("drawdown" in v for v in out["constraint_violations"])
        assert out["dimensions"]["return"] == "BETTER"          # reported, but it cannot compensate

    def test_missing_evidence_is_inconclusive_and_never_validated(self):
        out = objectives.compare(metrics(), metrics(net_return=0.08), {"walk_forward_oos": "PASS"})
        assert out["conclusion"] == "INCONCLUSIVE" and "holdout:PASS" in out["missing_evidence"]
        full = objectives.compare(metrics(), metrics(net_return=0.08, sharpe_annualized=1.2), PASSING)
        assert full["conclusion"] == "IMPROVED_ON_DEVELOPMENT"
        assert {c.value for c in objectives.Conclusion} == {"NOT_AN_IMPROVEMENT", "INCONCLUSIVE", "MIXED",
                                                           "IMPROVED_ON_DEVELOPMENT"}   # no VALIDATED / PROMOTE

    def test_tradeoff_is_mixed_not_averaged(self):
        out = objectives.compare(metrics(), metrics(net_return=0.09, max_drawdown=0.15), PASSING)
        assert out["conclusion"] == "MIXED" and "score" not in json.dumps(out)

    def test_too_few_trades_is_insufficient_evidence(self):
        assert any(v.startswith("INSUFFICIENT_EVIDENCE") for v in objectives.constraint_violations(metrics(n_trades=5)))

    def test_goal_never_overrides_constraint(self):
        g = objectives.goal_progress(metrics(max_drawdown=0.5))
        assert g["gap"] != "no constraint violated on development data" and "new pre-registered" in g["next_test"]
        assert objectives.goal_progress(None)["current_state"] == "NOT_AVAILABLE"

    def test_contract_is_frozen_code(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            objectives.CONTRACT.max_drawdown = 0.9


# --- autonomy ------------------------------------------------------------------------------------------------
class TestAutonomy:
    def test_live_levels_impossible(self):
        assert autonomy.maximum_permitted() is autonomy.Autonomy.PAPER_AUTONOMY
        for level in (autonomy.Autonomy.SUPERVISED_LIVE, autonomy.Autonomy.FULL_LIVE):
            with pytest.raises(LiveTradingDisabled):
                autonomy.require(level)
        with pytest.raises(TypeError):
            autonomy.require(4)

    def test_control_plane_runs_at_paper_autonomy(self, tmp_path):
        cp, _, _ = plane(tmp_path / "st")
        assert cp.autonomy is autonomy.Autonomy.PAPER_AUTONOMY


# --- research budget -------------------------------------------------------------------------------------------
class TestBudget:
    def test_check_reasons(self):
        u = {"root_hypotheses_tested": 12, "holdout_evaluations": 6, "variants_of_baseline": 4, "research_runs_today": 2}
        reasons = budget.check(u, new_variant=True)
        assert len(reasons) == 4
        assert budget.check(u | {"variants_of_baseline": 0, "root_hypotheses_tested": 0, "holdout_evaluations": 0,
                                 "research_runs_today": 0}, True) == []

    def test_exhausted_budget_blocks_research_before_anything_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(budget, "POLICY", budget.BudgetPolicy(max_research_runs_per_day=0))
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST", research_payload())})
        before = counts(s)
        out = cp.run_cycle()
        assert out.status == "BLOCKED" and "research budget" in out.detail["reason"]
        assert counts(s) == before and not any(True for _ in s.research_journal.entries("preregistration"))

    def test_usage_is_derived_from_journals(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                             script={"company": reply("RESEARCH_REQUEST", research_payload())})
        with_history(s, 3200)
        cp.run_cycle()
        u = budget.usage(s, cp.journal, clock.now())
        cp2, s2, _ = plane(tmp_path / "st", hours=3200, clock=clock)       # restart: nothing resets
        assert budget.usage(s2, cp2.journal, clock.now()) == u
        assert u["research_runs_today"] == 1 and u["root_hypotheses_tested"] == 1


# --- experiment design contract --------------------------------------------------------------------------------
class TestExperimentSchema:
    @pytest.mark.parametrize("exp,needle", [
        (experiment(params={"fast": 20, "slow": 100, "atr_period": 14, "stop_atr": 3.0}), "exactly one parameter"),
        (experiment("INTERACTION", iv=["fast"]), "at least two"),
        (experiment(iv=["slow"]), "must name exactly the changed"),
        (experiment("REGIME"), "do not take ['candidate_params']"),
        (condition_experiment("EXECUTION", {"cost_multiplier": 0.5}), "never reduced below the model"),
        (condition_experiment("RISK", {"risk_fraction": 0.5}), "risk_fraction must be within"),
        (condition_experiment("REGIME", {"regime": "MOON/UP"}), "regime must be one of"),
        (condition_experiment("REGIME", {"regime": "HIGH_VOL/UP"}, independent_variables=["fast"]), "exactly ['regime']"),
        ({k: v for k, v in experiment().items() if k != "design_rationale"}, "design_rationale"),
        ({k: v for k, v in experiment("STRUCTURAL", iv=["structure"]).items() if k != "candidate_params"}
         | {"structure": {"kind": "not a kind", "params": {}}}, "structure must be"),
        (experiment("MAGIC"), "experiment type"),
        (experiment(params={"fast": 20, "slow": 50, "atr_period": 14, "stop_atr": 3.0, "leverage": 5}), "every baseline"),
        (experiment(params={"fast": -5, "slow": 50, "atr_period": 14, "stop_atr": 3.0}), "positive"),
        (experiment(dependent_variable="vibes"), "not a recorded metric"),
        (experiment(risk_limit=0.5), "unrecognized"),
        ({k: v for k, v in experiment().items() if k != "candidate_params"}, "require candidate_params"),
    ])
    def test_malformed_experiments_rejected(self, tmp_path, exp, needle):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST", designed(exp=exp))})
        out = cp.run_cycle()
        assert out.status == "FAILED" and needle in out.detail["reason"], out.detail
        assert counts(s)["experiments"] == 0

    def test_holdout_evidence_cannot_motivate_a_hypothesis(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st")
        s.evidence.register("holdout", "hold_ref_1", clock.now() - timedelta(hours=1), "holdout evaluation")
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", research_payload(evidence_refs=["hold_ref_1"]))
        out = cp.run_cycle()
        assert out.status == "FAILED" and "holdout/promotion outcomes cannot motivate" in out.detail["reason"]

    def test_unresolvable_evidence_ref_rejected(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST",
                                                                   research_payload(evidence_refs=["made_up"]))})
        out = cp.run_cycle()
        assert out.status == "FAILED" and "does not resolve" in out.detail["reason"]

    def test_designed_experiment_requires_motivation_and_mechanism(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST",
                                                                   research_payload(experiment=experiment()))})
        out = cp.run_cycle()
        assert out.status == "FAILED" and "motivation" in out.detail["reason"]

    def test_learning_candidate_hypothesis_requires_a_design(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply(
            "RESEARCH_REQUEST", research_payload(learning_candidate_id="lc_" + "0" * 20))})
        out = cp.run_cycle()
        assert out.status == "FAILED" and "explicit experiment design" in out.detail["reason"]

    def test_unknown_learning_candidate_blocks(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply(
            "RESEARCH_REQUEST", designed(learning_candidate_id="lc_" + "0" * 20))})
        out = cp.run_cycle()
        assert out.status == "BLOCKED" and "unknown learning candidate" in out.detail["reason"]


class TestSingleVariableExperiment:
    def test_design_recorded_before_run_baseline_kept_and_compared(self, tmp_path):
        payload = designed("H-sv-1", motivation="shorter fast MA reacts earlier",
                           expected_mechanism="earlier entries capture more of each trend")
        cp, s, _ = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                         script={"company": reply("RESEARCH_REQUEST", payload)})
        with_history(s, 3200)
        out = cp.run_cycle()
        assert out.status == "COMPLETED", out.detail
        entries = list(s.research_journal.entries())
        seq = {e.type: e.seq for e in reversed(entries)}          # first occurrence of each type
        assert seq["experiment_design"] < seq["preregistration"]  # design locked before any test
        design = next(e.payload for e in entries if e.type == "experiment_design")
        assert design["baseline"]["params"] == {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0}
        assert design["candidate"]["params"]["fast"] == 20 and design["experiment_id"].startswith("exp_") and design["objective_contract"] == objectives.CONTRACT.contract_hash
        # the walk-forward tested exactly the candidate; the baseline definition is unchanged in the registry
        wf_rows = [e for e in ResearchLog(s.research_journal).experiments if e["stage"] == "walk_forward_oos"]
        assert len(wf_rows) == 1
        comp = [e.payload for e in s.research_journal.entries("experiment_comparison")]
        assert len(comp) == 1 and comp[0]["conclusion"] in {c.value for c in objectives.Conclusion}
        assert comp[0]["dev_dataset_id"] == wf_rows[0]["dataset_id"]            # development partition only
        assert "promot" in comp[0]["note"] and comp[0]["baseline_replaced"] is False
        for side in ("baseline", "candidate"):
            assert {f"{side}_strategy", f"{side}_fingerprint", f"{side}_dataset", f"{side}_results"} <= set(comp[0])
        assert comp[0]["baseline_strategy"] == "trend@v1" and comp[0]["candidate_evidence"] == wf_rows[0]["evidence_hash"]
        assert comp[0]["baseline_dataset"] == comp[0]["candidate_dataset"] == wf_rows[0]["dataset_id"]
        assert "wfo_stability" in comp[0]["dimensions"] and comp[0]["not_measured"]
        assert out.detail["comparison"]["conclusion"] == comp[0]["conclusion"]
        # baseline never silently replaced: trend@v1 keeps the baseline params; the candidate is a child version
        baseline = s.strategies.get("trend@v1")
        assert baseline.param_dict["fast"] == 10 and design["baseline"]["fingerprint"] == baseline.definition_hash
        children = [s.strategies.get(k) for k in s.strategies.keys() if s.strategies.get(k).parent_hash == baseline.definition_hash]
        assert len(children) == 1 and children[0].param_dict["fast"] == 20 and children[0].key != "trend@v1"
        # candidate factory: lineage record binds fingerprint, parent, hypothesis, experiment, datasets, code identity
        lineage = [e.payload for e in s.research_journal.entries("candidate_lineage")]
        assert len(lineage) == 1 and lineage[0]["fingerprint"] == children[0].definition_hash
        assert lineage[0]["parent_fingerprint"] == baseline.definition_hash == lineage[0]["baseline_fingerprint"]
        assert lineage[0]["hypothesis_id"] == "H-sv-1" and lineage[0]["experiment_id"] == design["experiment_id"]
        assert lineage[0]["datasets"]["development"] == wf_rows[0]["dataset_id"] and lineage[0]["code_hash"]
        assert lineage[0]["provenance"]["data_status"] == "MOCK"
        from ati.company import factory
        trail = factory.stages(s, children[0].definition_hash)
        assert trail["stages"]["CANDIDATE_GENERATED"] == children[0].key and trail["stages"]["WFO"] == "PASS"
        assert trail["stages"]["ADVERSARIAL"] != "NOT_REACHED" and trail["stages"]["PROMOTION_REVIEW"] != "NOT_REACHED"
        assert trail["stages"]["VALIDATION"] == ("APPROVED" if out.detail["promotion_approved"] else "DENIED")
        # the LEARN stage ran for this cycle and learned from the experiment outcome(s)
        learn = next(e.payload for e in cp.journal.entries("cycle_step") if e.payload["step"] == "learn")
        assert learn["data"]["new_outcomes"] >= 1


# --- learning -------------------------------------------------------------------------------------------------------
def rejected_cycles(tmp_path, n):
    cp, s, clock = plane(tmp_path / "st", script={"company": reply("RESUME")})
    for _ in range(n):
        out = cp.run_cycle()
        assert out.status == "FAILED"
        clock.advance(timedelta(hours=1))
    return cp, s, clock


class TestLearning:
    def test_repeated_outcomes_become_a_candidate_not_a_rule(self, tmp_path):
        cp, s, clock = rejected_cycles(tmp_path, 3)
        cands = [c for c in cp.learning.candidates.values() if c.pattern == "CONTRACT_REJECTION"]
        assert len(cands) == 1
        c = cands[0]
        assert len(c.evidence) == 3 and c.state is CandidateState.HYPOTHESIS_CANDIDATE and c.research_eligible
        assert cp.learning.quality(c, s) is Quality.REPEATED_EVIDENCE
        assert cp.learning.pattern_class(c, s).value == "RECURRING_PATTERN"
        # a learning candidate is a record: no strategy, memory doctrine or limit changed
        assert s.strategies.keys() == [] and len(s.memory) == 0

    def test_evidence_quality_ladder(self):
        c = LearningCandidate("lc_x", "P", {}, "s", ["a"], CandidateState.OBSERVED, False)
        assert c.quality() is Quality.OBSERVATION
        c.evidence.append("b")
        assert c.quality() is Quality.WEAK_EVIDENCE
        c.evidence.append("c")
        assert c.quality() is Quality.REPEATED_EVIDENCE
        assert c.quality(supported=True) is Quality.SUPPORTED_EVIDENCE
        assert c.quality(supported=True, validated=True) is Quality.VALIDATED_EVIDENCE

    def test_learn_is_idempotent_and_survives_restart(self, tmp_path):
        cp, s, clock = rejected_cycles(tmp_path, 2)
        n = len(cp.learning.journal)
        assert cp.learning.learn(s, cp.loop.journal, cp.journal)["new_outcomes"] == 0
        assert len(cp.learning.journal) == n
        cp2, s2, _ = plane(tmp_path / "st", clock=clock)
        assert {k: (c.state, c.evidence) for k, c in cp2.learning.candidates.items()} == \
               {k: (c.state, c.evidence) for k, c in cp.learning.candidates.items()}
        cp2.learning.journal.verify()

    def test_same_event_counts_once(self, tmp_path):
        cp, s, clock = rejected_cycles(tmp_path, 1)
        for _ in range(3):
            cp.learning.learn(s, cp.loop.journal, cp.journal)
        c = next(c for c in cp.learning.candidates.values() if c.pattern == "CONTRACT_REJECTION")
        assert len(c.evidence) == 1 and c.state is CandidateState.OBSERVED

    def test_holdout_derived_learning_never_becomes_research_eligible(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st")
        for i in range(5):
            o = OutcomeRecord(f"exp:h{i}", "experiment", clock.now().isoformat(), "MOCK", realized={
                "verdict": "FAIL", "stage": "holdout"}, holdout_derived=True, facts={"hypothesis_id": f"H{i}:holdout"})
            cp.learning.outcomes[o.outcome_id] = o
        cp.learning.learn(s, cp.loop.journal, cp.journal)
        c = next(c for c in cp.learning.candidates.values() if c.pattern == "HOLDOUT_FAILURE")
        assert c.holdout_derived and len(c.evidence) == 5
        assert c.state is CandidateState.ANALYZED and not c.research_eligible   # stops short of HYPOTHESIS_CANDIDATE
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed(learning_candidate_id=c.candidate_id))
        clock.advance(timedelta(hours=1))
        out = cp.run_cycle()
        assert out.status == "BLOCKED" and "holdout" in out.detail["reason"]

    def test_illegal_transitions_refused_live_and_on_replay(self, tmp_path):
        cp, s, clock = rejected_cycles(tmp_path, 1)
        c = next(iter(cp.learning.candidates.values()))
        with pytest.raises(LifecycleError):
            cp.learning._transition(c, CandidateState.SUPPORTED, "skip the evidence")
        j = cp.learning.journal
        j.append("candidate_state", {"candidate_id": c.candidate_id, "from": "OBSERVED", "to": "SUPPORTED", "reason": "x"})
        with pytest.raises(LifecycleError):
            LearningLedger(Journal(j.path, kind="learning", clock=s.clock, guard=s.guard, attrs=j.attrs))

    def test_terminal_states_are_permanent_and_visible(self):
        for terminal in (CandidateState.SUPPORTED, CandidateState.REJECTED, CandidateState.INCONCLUSIVE):
            assert learning.TRANSITIONS[terminal] == frozenset()

    def test_learning_has_no_path_to_authority(self):
        for mod in (learning, objectives, budget, scorecard, autonomy):
            src = inspect.getsource(mod)
            for forbidden in ("ati.risk", "ati.execution", "strategies.transition", "strategies.register",
                              "apply_promotion", ".derive(", "memory.add", "LIVE_TRADING =", "limits."):
                assert forbidden not in src or (mod is scorecard and forbidden == "limits."), (mod.__name__, forbidden)


# --- data conflict --------------------------------------------------------------------------------------------------
def two_sources(tmp_path, delta=Decimal("0")):
    cp, s, clock = plane(tmp_path / "st")
    a = [c for c in s.provider.fetch_candles("BTC/USD", Timeframe.H1, T0, T0 + timedelta(hours=10)) if c.is_closed]
    b = [dataclasses.replace(c, provider="mock-b", close=c.close + delta, high=max(c.high, c.close + delta)) for c in a]
    return cp, s, clock, a, b


class TestDataConflict:
    def test_agreeing_sources_have_no_conflict(self, tmp_path):
        _, _, _, a, b = two_sources(tmp_path)
        assert compare(a, b) == []

    def test_a_source_cannot_corroborate_itself(self, tmp_path):
        _, _, _, a, _ = two_sources(tmp_path)
        with pytest.raises(DataConflictError):
            compare(a, a)

    def test_conflict_blocks_research_and_trading_until_operator_resolves(self, tmp_path):
        cp, s, clock, a, b = two_sources(tmp_path, Decimal("500"))
        conflicts = compare(a, b)
        assert conflicts and cp.conflicts.record(conflicts) == len(conflicts)
        assert cp.conflicts.record(conflicts) == 0                    # idempotent
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", research_payload())
        out = cp.run_cycle()
        assert out.status == "BLOCKED" and out.health["data_state"] == "INVALID_DATA"
        assert "DATA_CONFLICT" in out.health["checks"]["data"]["detail"]
        with pytest.raises(PermissionError):
            cp.conflicts.resolve(conflicts[0].conflict_id, "mock-b", "just fix it")
        with pytest.raises(DataConflictError):
            cp.conflicts.resolve(conflicts[0].conflict_id, "some-third-source", RESOLVE_ACK)
        cp2, _, _ = plane(tmp_path / "st", clock=clock)                # persists across restart
        assert set(cp2.conflicts.open) == {c.conflict_id for c in conflicts}
        for c in conflicts:
            cp2.conflicts.resolve(c.conflict_id, "mock", RESOLVE_ACK)
        assert not ConflictRegister(Journal(cp2.conflicts.journal.path, kind="conflicts", clock=s.clock, guard=s.guard,
                                            attrs=cp2.conflicts.journal.attrs)).open


# --- text evidence -------------------------------------------------------------------------------------------------
class TestTextEvidence:
    def src(self, sid="news-1", text="Exchange X halted withdrawals on Tuesday.", hours=0):
        return RawSource(sid, "wire", T0 + timedelta(hours=hours), T0 + timedelta(hours=hours, minutes=5), text)

    def test_facts_must_be_verbatim(self):
        te = TextEvidence()
        te.add_source(self.src())
        te.add_fact(ExtractedFact("news-1", "halted withdrawals"))
        with pytest.raises(TextEvidenceError):
            te.add_fact(ExtractedFact("news-1", "Exchange X is insolvent"))   # an interpretation posing as a fact

    def test_interpretations_of_one_source_are_one_source(self):
        te = TextEvidence()
        te.add_source(self.src())
        for model in ("claude-a", "claude-b", "claude-c"):
            te.add_interpretation(Interpretation("news-1", model, "bearish", T0 + timedelta(hours=1)))
        assert te.independent_sources(["news-1", "news-1", "news-1"], T0 + timedelta(hours=2)) == 1
        te.add_source(self.src("news-2", "Regulator opened an inquiry.", hours=1))
        assert te.independent_sources(["news-1", "news-2"], T0 + timedelta(hours=2)) == 2
        assert te.independent_sources(["news-1", "news-2"], T0 + timedelta(minutes=30)) == 1   # point-in-time

    def test_look_ahead_and_rewrites_refused(self):
        te = TextEvidence()
        te.add_source(self.src())
        with pytest.raises(TextEvidenceError):
            te.add_interpretation(Interpretation("news-1", "m", "x", T0))       # before retrieval
        with pytest.raises(TextEvidenceError):
            te.add_source(self.src(text="Edited after the fact."))
        ctx = te.context(T0 + timedelta(hours=1))
        assert set(ctx[0]) >= {"RAW_SOURCE", "EXTRACTED_FACTS", "MODEL_INTERPRETATION"}
        assert "untrusted_text" in ctx[0]["RAW_SOURCE"]


# --- scorecard and context -------------------------------------------------------------------------------------------
class TestScorecardAndContext:
    def test_scorecard_has_independent_dimensions_and_no_aggregate(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("NO_TRADE")})
        cp.run_cycle()
        from ati.company.health import assess
        card = scorecard.build(s, assess(s, cp.loop, {}, cp.journal, False), cp.journal, cp.learning)
        assert tuple(card) == scorecard.DIMENSIONS and len(card) == 10
        assert card["EVIDENCE_QUALITY"]["state"] == "NOT_AVAILABLE"            # MOCK is never market evidence
        assert all(set(v) == {"state", "detail"} for v in card.values())
        assert not any(k in json.dumps(card).lower() for k in ('"score"', '"total"', '"overall"'))

    def test_context_carries_self_improvement_categories(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("NO_TRADE")})
        cp.run_cycle()
        issued = next(e.payload["data"] for e in cp.journal.entries("cycle_step") if e.payload["step"] == "request_issued")
        assert set(issued["context_categories"]) >= {"research_budget", "scorecard", "learning_candidates", "objective",
                                                     "autonomy", "failed_experiments", "validated_findings"}
        assert cp.state is CompanyState.COMPLETED

    def test_every_cycle_learns_once(self, tmp_path):
        cp, s, clock = rejected_cycles(tmp_path, 2)
        steps = [e.payload for e in cp.journal.entries("cycle_step") if e.payload["step"] == "learn"]
        assert len(steps) == 2 and len({p["cycle_id"] for p in steps}) == 2


class TestLearningToResearch:
    def test_eligible_candidate_links_and_state_follows_research_facts(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS, script={"company": reply("RESUME")})
        with_history(s, 3200)
        for _ in range(3):
            assert cp.run_cycle().status == "FAILED"
            clock.advance(timedelta(hours=1))
        c = next(c for c in cp.learning.candidates.values() if c.pattern == "CONTRACT_REJECTION")
        assert c.research_eligible
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed(
            "H-lc-1", learning_candidate_id=c.candidate_id))
        out = cp.run_cycle()
        assert out.status == "COMPLETED", out.detail
        c = cp.learning.candidates[c.candidate_id]
        assert c.hypothesis_id == "H-lc-1"
        # state advanced only from ResearchLog facts, through every intermediate state, to a terminal state
        path = [h[1] for h in c.history]
        assert path[:3] == ["ANALYZED", "HYPOTHESIS_CANDIDATE", "PREREGISTERED"] and "TESTING" in path
        log = ResearchLog(s.research_journal)
        expected = {"TESTED:FAIL": "REJECTED"}.get(log.status("H-lc-1"), None) or \
            {"TESTED:FAIL": "REJECTED"}.get(log.status("H-lc-1:holdout"))
        if expected:
            assert c.state.value == expected
        else:
            assert c.state.value in ("SUPPORTED", "INCONCLUSIVE")
        # MOCK data can never reach VALIDATED_EVIDENCE, whatever the outcome
        assert cp.learning.quality(c, s) is not Quality.VALIDATED_EVIDENCE
        # a second request citing a now-terminal candidate is refused
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed(
            "H-lc-2", learning_candidate_id=c.candidate_id))
        clock.advance(timedelta(hours=1))
        out2 = cp.run_cycle()
        assert out2.status == "BLOCKED" and "not eligible" in out2.detail["reason"]
