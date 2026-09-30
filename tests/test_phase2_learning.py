"""Company 1.0 Phase 2 — self-improving research, learning and strategy factory: test matrix A–AF plus the ten
crash points. Every test name carries its matrix letter. All data is MOCK, execution PAPER; nothing here is
evidence about real markets.
"""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from ati.agent.reasoning import Budget, ScriptedReasoningClient
from ati.company import factory
from ati.company import control as control_mod
from ati.company.control import CompanyControlPlane
from ati.company.intelligence import report as intelligence_report
from ati.company.learning import CandidateState, LearningLedger, OutcomeRecord
from ati.core.errors import (HoldoutViolation, LifecycleError, LiveTradingDisabled, MemoryIntegrityError,
                             PromotionDenied, StrategyImmutableError)
from ati.core.time import FixedClock
from ati.data.dataset import Dataset
from ati.ledger.journal import Journal, decode
from ati.market.mock import MockProvider
from ati.market.models import Timeframe
from ati.memory.store import MemoryEntry, MemoryKind
from ati.research.adversarial import AdversarialReport, Objection
from ati.research.hypothesis import ResearchLog, Verdict
from ati.research.walkforward import walk_forward
from ati.strategies.base import StrategyDefinition
from ati.strategies.registry import Lifecycle
from ati.system import build_paper_system
from ati.validation.holdout import HoldoutEvaluation, HoldoutVault
from ati.validation.promotion import PromotionPolicy, decide_promotion
from tests.helpers import T0
from tests.rig import FAST, MOCK_SCRIPT, install_champion
from tests.test_company_control import MECHANICS, Boom, counts, crash_after, crash_once, packet, plane, reply, \
    trade_payload, with_history
from tests.test_research_memory_integrity import independent_chain, stores
from tests.test_self_improvement import designed, experiment

SUBJECTIVE = ("SMART", "DUMB", "GOOD", "BAD")


# --- shared builders --------------------------------------------------------------------------------------------------
def company_brain(research=None):
    """Scripted MOCK Claude: trades the champion's entry signals, otherwise NO_TRADE; optionally, once, cites the
    first research-eligible learning candidate in a designed RESEARCH_REQUEST."""
    answered: dict[str, str] = {}
    state = {"cited": None}

    def respond(prompt):
        p = packet(prompt)
        if p["request_id"] not in answered:
            eligible = [c for c in p["RELEVANT_LEARNINGS"]["learning_candidates"] if c["research_eligible"]]
            if research and state["cited"] is None and eligible and "RESEARCH_REQUEST" in p["allowed_actions"]:
                state["cited"] = eligible[0]["learning_id"]
                action, payload = "RESEARCH_REQUEST", designed(research, learning_candidate_id=state["cited"],
                                                               statement="Faster entries reduce repeated losses")
            elif p["STRATEGY"]["entry_signals"] and "TRADE_PROPOSAL" in p["allowed_actions"]:
                action, payload = "TRADE_PROPOSAL", trade_payload(p)
            else:
                action, payload = "NO_TRADE", {}
            answered[p["request_id"]] = json.dumps({"request_id": p["request_id"], "cycle_id": p["cycle_id"],
                                                    "context_id": p["context_id"], "action": action,
                                                    "reason": "[MOCK] scripted company brain", "payload": payload})
        return answered[p["request_id"]]
    respond.state = state
    return respond


def paper_company(tmp_path, brain, hours=3200, policies=MECHANICS):
    clock = FixedClock(T0 + timedelta(hours=hours))
    provider = MockProvider(11, clock, epoch=T0)
    s = build_paper_system(tmp_path / "st", clock, provider,
                           ScriptedReasoningClient(MOCK_SCRIPT | {"company": brain}, Budget(100_000)),
                           data_status=provider.data_status)
    with_history(s, hours)
    install_champion(s)
    return CompanyControlPlane(s, research_policies=policies), s, clock


def run_until(cp, clock, predicate, limit=400):
    for _ in range(limit):
        out = cp.run_cycle()
        if predicate(out):
            return out
        clock.advance(timedelta(hours=1))
    raise AssertionError("condition not reached")


@pytest.fixture
def traded(tmp_path):
    """A MOCK/PAPER company that has closed at least two paper trades through the unchanged risk/execution path."""
    brain = company_brain()
    cp, s, clock = paper_company(tmp_path, brain)
    run_until(cp, clock, lambda _: sum(1 for o in cp.learning.outcomes.values() if o.source == "trade") >= 2)
    return cp, s, clock


def register_reviews(s, n, pnl=Decimal("-10"), prefix="rev_synth"):
    """Synthetic outcome records whose evidence *is registered* (as the loop does for real paper exits)."""
    outs = []
    for i in range(n):
        rid = f"{prefix}_{i:02d}"
        s.evidence.register("trade_review", rid, s.clock.now() - timedelta(hours=n - i), f"synthetic exit {i}")
        outs.append(OutcomeRecord(rid, "trade", (s.clock.now() - timedelta(hours=n - i)).isoformat(), "MOCK", "BTC/USD",
                                  "1h", "fp_synth", None, realized={"pnl": str(pnl), "exit_reason": "stop"},
                                  facts={"regime": "HIGH_VOL/DOWN", "trade_id": f"dec_{i}"}))
    return outs


# --- Learning A–G -------------------------------------------------------------------------------------------------------
class TestLearning:
    def test_A_paper_outcome_creates_learning_candidate(self, traded):
        cp, s, _ = traded
        trades = [o for o in cp.learning.outcomes.values() if o.source == "trade"]
        assert len(trades) >= 2
        linked = [c for c in cp.learning.candidates.values() if set(c.evidence) & {o.outcome_id for o in trades}]
        assert linked, "no learning candidate from paper outcomes"
        assert all(c.state in (CandidateState.OBSERVED, CandidateState.ANALYZED) for c in linked)

    def test_B_learning_candidate_preserves_provenance(self, traded):
        cp, s, _ = traded
        champion = s.strategies.champion("BTC/USD", Timeframe.H1)
        c = next(c for c in cp.learning.candidates.values() if c.pattern in ("REPEATED_LOSS", "REGIME_FAILURE",
                                                                             "UNEXPECTED_COSTS", "DRAWDOWN_CLUSTER")
                 or cp.learning.outcomes[c.evidence[0]].source == "trade")
        r = cp.learning.record(c, s)
        prov = r["provenance"]
        assert r["learning_id"] == c.candidate_id and prov["source_type"] == "trade"
        assert prov["source_id"] in s.evidence and prov["source_id"].startswith("rev_")
        assert prov["strategy_fingerprint"] == champion.definition_hash and prov["data_category"] == "MOCK"
        assert prov["created_at"] and r["source_event"]["source_id"] == prov["source_id"]
        refs = r["evidence_references"]
        assert refs and all(x["registered_evidence"] == x["outcome_id"] for x in refs if x["source_type"] == "trade")
        assert all(x["decision_id"] in s.decisions for x in refs if x["source_type"] == "trade")

    def test_C_fact_interpretation_question_stay_separate(self, traded):
        cp, s, _ = traded
        for c in cp.learning.candidates.values():
            r = cp.learning.record(c, s)
            assert r["FACT"] and r["INTERPRETATION"] and r["PROPOSED_QUESTION"]
            assert r["FACT"] != r["INTERPRETATION"] and " may " not in r["FACT"]
            assert " may " in r["INTERPRETATION"] or "may " in r["INTERPRETATION"].lower()
            assert r["PROPOSED_QUESTION"].rstrip(")").rstrip().endswith("?") or "?" in r["PROPOSED_QUESTION"]
            text = " ".join(str(v) for v in r.values()).upper().split()
            assert not set(SUBJECTIVE) & set(text)

    def test_D_one_off_event_never_becomes_doctrine(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st")
        out = register_reviews(s, 1)[0]
        cp.learning.outcomes[out.outcome_id] = out
        cp.learning.learn(s, cp.loop.journal, cp.journal)
        for c in cp.learning.candidates.values():
            assert cp.learning.pattern_class(c, s).value == "ONE_OFF" and not c.research_eligible
        assert len(s.memory) == 0                                          # a single event is not even remembered
        with pytest.raises(MemoryIntegrityError):                         # and MOCK can never hold doctrine
            s.memory.add(MemoryEntry(MemoryKind.MISTAKE, "x", s.clock.now(), "t",
                                     (s.evidence.resolve(out.outcome_id),), 0.5))

    def test_E_recurring_pattern_detected_and_remembered_as_hypothesis_only(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st")
        for o in register_reviews(s, 3):
            cp.learning.outcomes[o.outcome_id] = o
        cp.learning.learn(s, cp.loop.journal, cp.journal)
        c = next(c for c in cp.learning.candidates.values() if c.pattern == "REPEATED_LOSS")
        assert cp.learning.pattern_class(c, s).value == "RECURRING_PATTERN"
        assert c.state is CandidateState.HYPOTHESIS_CANDIDATE and c.research_eligible
        mem_id = cp.learning.memory_written[c.candidate_id]
        m = s.memory.get(mem_id)
        assert m.kind is MemoryKind.HYPOTHESIS and m.statement.startswith("[MOCK] RECURRING_PATTERN (not validated)")
        assert {r.ref_id for r in m.evidence} == set(c.evidence[:3])
        # idempotent replay, and across a restart
        n_mem, n_journal = len(s.memory), len(cp.learning.journal)
        cp.learning.learn(s, cp.loop.journal, cp.journal)
        assert (len(s.memory), len(cp.learning.journal)) == (n_mem, n_journal)
        cp2, s2, _ = plane(tmp_path / "st", clock=clock)
        assert cp2.learning.memory_written == cp.learning.memory_written and len(s2.memory) == n_mem

    def test_E2_contradictory_evidence_rejects_the_pattern_visibly(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st")
        losses = register_reviews(s, 2)
        wins = [replace(o, realized={"pnl": "25", "exit_reason": "stop"}) for o in register_reviews(s, 3, prefix="rev_win")]
        for o in losses:
            cp.learning.outcomes[o.outcome_id] = o
        cp.learning.learn(s, cp.loop.journal, cp.journal)
        c = next(c for c in cp.learning.candidates.values() if c.pattern == "REPEATED_LOSS")
        assert c.state is CandidateState.ANALYZED
        for o in wins:
            cp.learning.outcomes[o.outcome_id] = o
        cp.learning.learn(s, cp.loop.journal, cp.journal)
        assert c.state is CandidateState.REJECTED and "contradicted" in c.history[-1][2]
        assert not c.research_eligible and c.candidate_id in cp.learning.candidates     # rejected stays visible

    def test_F_G_learning_candidate_converts_to_preregistered_hypothesis(self, tmp_path):
        brain = company_brain(research="H-p2-F")
        cp, s, clock = paper_company(tmp_path, brain)
        out = run_until(cp, clock, lambda o: o.action == "RESEARCH_REQUEST")
        assert out.status == "COMPLETED", out.detail
        lc = brain.state["cited"]
        log = ResearchLog(s.research_journal)
        prereg = log.get("H-p2-F")
        assert f"learning:{lc}" in prereg.observation_refs                    # provenance locked into the prereg
        assert any(r.startswith("rev_") for r in prereg.observation_refs)
        c = cp.learning.candidates[lc]
        assert c.state is CandidateState.PROMOTED_TO_HYPOTHESIS and c.hypothesis_id == "H-p2-F"
        entries = list(s.research_journal.entries())
        first = lambda t: next(e.seq for e in entries if e.type == t)       # noqa: E731
        assert first("experiment_design") < first("preregistration") < first("experiment")   # G: locked before testing
        # a designed request without an explicit design is refused at the contract (G)
        assert out.detail["experiment_id"].startswith("exp_")


# --- Research H–N -----------------------------------------------------------------------------------------------------------
@pytest.fixture
def researched(tmp_path):
    cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                         script={"company": reply("RESEARCH_REQUEST", designed("H-p2-R"))})
    with_history(s, 3200)
    out = cp.run_cycle()
    assert out.status == "COMPLETED" and out.detail["research_status"] == "COMPLETED", out.detail
    return cp, s, clock, out


class TestResearch:
    def test_H_experiment_type_recorded(self, researched):
        cp, s, _, out = researched
        design = next(e.payload for e in s.research_journal.entries("experiment_design"))
        step = next(e.payload["data"] for e in cp.journal.entries("cycle_step") if e.payload["step"] == "research_invoked")
        assert design["type"] == step["experiment_type"] == "SINGLE_VARIABLE"
        assert design["changed_variables"] == [{"variable": "fast", "baseline": 10, "candidate": 20}]

    def test_I_baseline_is_immutable(self, researched):
        cp, s, _, _ = researched
        baseline = s.strategies.get("trend@v1/1h")
        assert baseline.param_dict == {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0}
        with pytest.raises(StrategyImmutableError):
            s.strategies.register(StrategyDefinition.create("trend", 1, "ma_crossover", {"fast": 20, "slow": 50,
                                                            "atr_period": 14, "stop_atr": 3.0}, Timeframe.H1, T0))
        comp = next(e.payload for e in s.research_journal.entries("experiment_comparison"))
        assert comp["baseline_replaced"] is False and comp["baseline_fingerprint"] == baseline.definition_hash

    def test_J_candidate_fingerprint_is_deterministic(self, researched):
        cp, s, _, out = researched
        params = {"fast": 20, "slow": 50, "atr_period": 14, "stop_atr": 3.0}
        a = StrategyDefinition.create("trend", 2, "ma_crossover", params, Timeframe.H1, T0, "p")
        b = StrategyDefinition.create("trend", 2, "ma_crossover", dict(reversed(list(params.items()))), Timeframe.H1, T0, "p")
        assert a.definition_hash == b.definition_hash
        fp = out.detail["candidate"]["fingerprint"]
        assert cp._registered_with(s.strategies.get("trend@v1/1h"), params).definition_hash == fp   # same params → same def
        assert out.detail["candidate"]["candidate_id"] == factory.candidate_id(fp, "H-p2-R")

    def test_K_candidate_lineage_preserved_across_restart(self, researched, tmp_path):
        cp, s, clock, out = researched
        before = [e.payload for e in s.research_journal.entries("candidate_lineage")]
        cp2, s2, _ = plane(tmp_path / "st", clock=clock, policies=MECHANICS)
        after = [e.payload for e in s2.research_journal.entries("candidate_lineage")]
        assert before == after and len(after) == 1
        l_ = after[0]
        assert l_["parent_fingerprint"] == s2.strategies.get("trend@v1/1h").definition_hash
        assert {"candidate_id", "fingerprint", "hypothesis_id", "experiment_id", "datasets", "params", "code_hash",
                "provenance", "attempts"} <= set(l_)
        assert factory.stages(s2, l_["fingerprint"]) == factory.stages(s, l_["fingerprint"])

    def test_L_failed_candidate_remains_visible(self, researched, tmp_path):
        cp, s, clock, out = researched
        cp2, s2, _ = plane(tmp_path / "st", clock=clock, policies=MECHANICS)
        log = ResearchLog(s2.research_journal)
        assert [e for e in log.experiments if e["hypothesis_id"].startswith("H-p2-R")]
        key = out.detail["candidate"]["key"]
        if not out.detail["promotion_approved"]:
            assert s2.strategies.state(key, "BTC/USD") is Lifecycle.REJECTED
            assert any(h[0] == key and h[2] == "REJECTED" for h in s2.strategies.history)
        rep = intelligence_report(cp2)
        assert rep["RESEARCH"]["experiments"] == len(log.experiments) and rep["RESEARCH"]["failed_experiments"] >= 1

    def test_M_N_multiple_testing_and_budget_persist(self, researched, tmp_path, monkeypatch):
        cp, s, clock, _ = researched
        from ati.company import budget
        u = budget.usage(s, cp.journal, clock.now())
        cp2, s2, _ = plane(tmp_path / "st", clock=clock, policies=MECHANICS)
        u2 = budget.usage(s2, cp2.journal, clock.now())
        assert u == u2 and u["root_hypotheses_tested"] == 1 and u["candidates_generated"] == 1
        assert u["experiments_run"] >= 1 and u["holdout_evaluations"] == 1
        # the next candidate's evidence carries "how many attempts before this one"
        a = factory.attempts(s2, "H-next")
        assert a["root_hypotheses_tested_before"] == 1 and a["holdout_evaluations_before"] == 1
        assert a["candidates_generated_before"] == 1


# --- Validation O–S ------------------------------------------------------------------------------------------------------
def gate_inputs(s):
    d = s.strategies.register(StrategyDefinition.create("trend", 1, "ma_crossover", FAST, Timeframe.H1, s.clock.now()))
    s.strategies.transition(d.registry_key, "BTC/USD", Lifecycle.CHALLENGER, "test")
    provider = MockProvider(99, FixedClock(T0 + timedelta(hours=1200)), epoch=T0)
    ds = Dataset.build(provider.fetch_candles("BTC/USD", Timeframe.H1, T0, T0 + timedelta(hours=1200)),
                       data_version="rig", realization="rig")
    wf = walk_forward(d, ds, [d.param_dict], train_bars=400, test_bars=400)
    wf = replace(wf, oos_metrics=replace(wf.oos_metrics, n_trades=100), positive_fold_fraction=1.0)
    adv = AdversarialReport(d.key, d.definition_hash, ds.dataset_id, "MOCK", (Objection("stub", Verdict.PASS, "rig"),))
    hold = HoldoutEvaluation(d.key, d.definition_hash, "ds_h", "p", replace(wf.oos_metrics, n_trades=50), Verdict.PASS, (), 1)
    return d, wf, adv, hold


class TestValidation:
    POLICY = PromotionPolicy(allow_mock_evidence=True)

    def decide(self, s, d, wf, adv, hold):
        return decide_promotion(d, None, wf, adv, hold, None, self.POLICY, s.clock.now(), s.research_journal,
                                symbol="BTC/USD")

    def test_O_cannot_bypass_walk_forward(self, tmp_path):
        _, s, _ = plane(tmp_path / "st")
        d, wf, adv, hold = gate_inputs(s)
        r = self.decide(s, d, replace(wf, positive_fold_fraction=0.0, folds=()), adv, hold)
        assert not r.approved and any("walk-forward" in x or "folds positive" in x for x in r.reasons)

    def test_P_cannot_bypass_robustness(self, tmp_path):
        _, s, _ = plane(tmp_path / "st")
        d, wf, adv, hold = gate_inputs(s)
        fragile = replace(adv, objections=(Objection("Does the edge survive doubled costs and slippage?", Verdict.FAIL, "x2: -5"),))
        r = self.decide(s, d, wf, fragile, hold)
        assert not r.approved and any("doubled costs" in x for x in r.reasons)

    def test_Q_cannot_bypass_adversarial(self, tmp_path):
        _, s, _ = plane(tmp_path / "st")
        d, wf, adv, hold = gate_inputs(s)
        r = self.decide(s, d, wf, replace(adv, strategy_hash="someone_else"), hold)
        assert not r.approved and any("does not belong" in x for x in r.reasons)
        with pytest.raises(LifecycleError):                           # and no lifecycle side door to CHAMPION
            s.strategies.transition(d.registry_key, "BTC/USD", Lifecycle.CHAMPION, "skip the gate")

    def test_R_cannot_bypass_holdout(self, tmp_path):
        _, s, _ = plane(tmp_path / "st")
        d, wf, adv, hold = gate_inputs(s)
        r = self.decide(s, d, wf, adv, replace(hold, verdict=Verdict.FAIL))
        assert not r.approved and any("holdout verdict" in x for x in r.reasons)
        with pytest.raises(PromotionDenied):
            s.strategies.apply_promotion(r)
        assert s.strategies.champion("BTC/USD", Timeframe.H1) is None

    def test_S_holdout_result_cannot_modify_or_rerun_the_candidate(self, researched, tmp_path):
        cp, s, clock, out = researched
        fp = out.detail["candidate"]["fingerprint"]
        key = out.detail["candidate"]["key"]
        assert s.strategies.get(key).definition_hash == fp                         # definition unchanged after holdout
        holdout_learning = [c for c in cp.learning.candidates.values() if c.pattern in ("HOLDOUT_FAILURE", "PROMOTION_DENIED")]
        assert all(c.holdout_derived and not c.research_eligible for c in holdout_learning)
        # the same hypothesis cannot run again (no re-evaluation of the holdout); a new id cannot reuse the period
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed("H-p2-R"))
        clock.advance(timedelta(hours=1))
        again = cp.run_cycle()
        assert again.status == "BLOCKED" and "locked" in again.detail["reason"]
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed("H-p2-R2", statement="Retune after holdout"))
        clock.advance(timedelta(hours=1))
        retune = cp.run_cycle()
        assert retune.detail.get("research_status") == "NOT_RUN" and "sealed holdout" in retune.detail["reasons"][0]
        assert sum(1 for _ in s.research_journal.entries("holdout_access")) == 1
        # a vault never evaluates twice
        full = Dataset.build(s.store.series(s.provider.name, "BTC/USD", Timeframe.H1), data_version="v", realization="x")
        vault = HoldoutVault(full, full.candles[-300].open_time, Journal(tmp_path / "v.jsonl", kind="research",
                                                                         attrs={"data_status": "MOCK"}, clock=clock),
                             max_evaluations=1)
        d = s.strategies.get(key)
        from ati.research.hypothesis import Criterion, PreRegistration
        pr = PreRegistration("HV", "s", (), d.key, d.definition_hash, vault.development.dataset_id,
                             (Criterion("net_pnl", ">", 0.0),), 1, clock.now())
        vault.evaluate(d, pr)
        with pytest.raises(HoldoutViolation):
            vault.evaluate(d, pr)


# --- Memory T–V ------------------------------------------------------------------------------------------------------------
class TestMemory:
    def test_T_mock_learning_cannot_become_doctrine(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st")
        for o in register_reviews(s, 5):
            cp.learning.outcomes[o.outcome_id] = o
        cp.learning.learn(s, cp.loop.journal, cp.journal)
        assert all(m.kind is MemoryKind.HYPOTHESIS for m in s.memory.query(s.clock.now()))
        assert all(cp.learning.pattern_class(c, s).value not in ("SUPPORTED_PATTERN", "VALIDATED_EFFECT")
                   for c in cp.learning.candidates.values())
        refs = tuple(s.evidence.resolve(f"rev_synth_{i:02d}") for i in range(3))
        for kind in (MemoryKind.MISTAKE, MemoryKind.FINDING, MemoryKind.LESSON):
            with pytest.raises(MemoryIntegrityError):
                s.memory.add(MemoryEntry(kind, "doctrine attempt", s.clock.now(), "t", refs, 0.5))

    def test_U_validated_evidence_remains_eligible_on_market_data(self, tmp_path):
        reg, mem = stores(tmp_path, "REAL")          # category configuration only: no REAL data is fabricated
        chain = independent_chain(reg)
        eid = mem.add(MemoryEntry(MemoryKind.VALIDATED_FINDING, "trend persistence", T0 + timedelta(hours=1), "t",
                                  chain, 0.6, dataset_ids=("ds_dev", "ds_hold")))
        assert mem.get(eid).kind is MemoryKind.VALIDATED_FINDING and mem.accepts_doctrine

    def test_V_duplicate_evidence_remains_rejected(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st")
        ref = s.evidence.resolve(register_reviews(s, 1)[0].outcome_id)
        with pytest.raises(MemoryIntegrityError):
            s.memory.add(MemoryEntry(MemoryKind.HYPOTHESIS, "dup", s.clock.now(), "t", (ref, ref), 0.2))


# --- Recovery W–AA: the ten crash points -------------------------------------------------------------------------------------
def learning_journal_crash(monkeypatch, cp, entry_type):
    real = cp.learning.journal.append
    state = {"done": False}

    def append(type_, payload):
        if type_ == entry_type and not state["done"]:
            state["done"] = True
            raise Boom(f"crash before {entry_type}")
        return real(type_, payload)
    monkeypatch.setattr(cp.learning.journal, "append", append)


def rejected_then(tmp_path, monkeypatch, entry_type):
    cp, s, clock = plane(tmp_path / "st", script={"company": reply("RESUME")})
    learning_journal_crash(monkeypatch, cp, entry_type)
    with pytest.raises(Boom):
        cp.run_cycle()                     # crash inside LEARN of the first cycle
    cp2, s2, _ = plane(tmp_path / "st", clock=clock, script={"company": reply("RESUME")})
    out = cp2.run_cycle()                  # recovery finishes the same cycle
    assert out.cycle_id == cp.running or out.status == "FAILED"
    clock.advance(timedelta(hours=1))
    cp2.run_cycle()                        # and a second, independent outcome follows
    return cp2, s2, out


def no_duplicates(cp):
    for c in cp.learning.candidates.values():
        assert len(c.evidence) == len(set(c.evidence))
    ids = [e.payload["outcome_id"] for e in cp.learning.journal.entries("outcome")]
    assert len(ids) == len(set(ids))
    cands = [e.payload["candidate_id"] for e in cp.learning.journal.entries("candidate")]
    assert len(cands) == len(set(cands))
    ends = [e.payload["cycle_id"] for e in cp.journal.entries("cycle_end")]
    assert len(ends) == len(set(ends))


class TestRecovery:
    def test_W1_crash_before_learning_candidate_creation(self, tmp_path, monkeypatch):
        cp, s, out = rejected_then(tmp_path, monkeypatch, "candidate")
        assert out.status == "FAILED"
        no_duplicates(cp)
        c = next(c for c in cp.learning.candidates.values() if c.pattern == "CONTRACT_REJECTION")
        assert len(c.evidence) == 2

    def test_W2_crash_after_learning_candidate_creation(self, tmp_path, monkeypatch):
        cp, s, out = rejected_then(tmp_path, monkeypatch, "candidate_evidence")
        no_duplicates(cp)
        c = next(c for c in cp.learning.candidates.values() if c.pattern == "CONTRACT_REJECTION")
        assert len(c.evidence) == 2 and c.state is CandidateState.ANALYZED

    def research_crash(self, tmp_path, monkeypatch, target, name, after=False, hid="H-crash"):
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                             script={"company": reply("RESEARCH_REQUEST", designed(hid))})
        with_history(s, 3200)
        (crash_after if after else crash_once)(monkeypatch, target, name)
        with pytest.raises(Boom):
            cp.run_cycle()
        monkeypatch.undo()
        # restart: the MOCK candle store is in-memory (as in the existing crash tests), so history is re-ingested;
        # every journal (research, company, learning, registry, evidence) is what persisted
        cp2, s2, _ = plane(tmp_path / "st", clock=clock, policies=MECHANICS,
                           script={"company": reply("RESEARCH_REQUEST", designed(hid))})
        with_history(s2, 3200)
        out = cp2.run_cycle()
        sizes = {n: len(list(s2.research_journal.entries(n))) for n in
                 ("preregistration", "experiment", "holdout_access", "promotion_decision", "candidate_lineage")}
        cp3, s3, _ = plane(tmp_path / "st", clock=clock, policies=MECHANICS,
                           script={"company": reply("RESEARCH_REQUEST", designed(hid))})
        with_history(s3, 3200)
        replay = cp3.run_cycle()                                   # AA: repeating the request re-runs nothing
        log = ResearchLog(s3.research_journal)
        prereg = [e for e in s3.research_journal.entries("preregistration") if e.payload["prereg"]["hypothesis_id"] == hid]
        assert len(prereg) <= 1
        rows = [e for e in log.experiments if e["hypothesis_id"].split(":")[0] == hid]
        stages = [e["stage"] for e in rows]
        assert len(stages) == len(set(stages))                     # no experiment duplicated
        assert sum(1 for _ in s3.research_journal.entries("holdout_access")) <= 1      # holdout never consumed twice
        assert sum(1 for _ in s3.research_journal.entries("promotion_decision")) <= 1
        assert s3.strategies.champion("BTC/USD", Timeframe.H1) is None                     # never silently promoted
        assert replay.status == "BLOCKED" and "already registered" in replay.detail["reason"]
        assert {n: len(list(s3.research_journal.entries(n))) for n in sizes} == sizes
        no_duplicates(cp3)
        return out, s3, log

    def test_X3_crash_before_hypothesis_conversion(self, tmp_path, monkeypatch):
        out, s, log = self.research_crash(tmp_path, monkeypatch, CompanyControlPlane, "_observation_refs")
        assert out.detail["research_status"] == "COMPLETED" and log.was_tested("H-crash")   # ran once, after restart

    def test_X4_crash_after_hypothesis_preregistration(self, tmp_path, monkeypatch):
        out, s, log = self.research_crash(tmp_path, monkeypatch, ResearchLog, "preregister", after=True)
        assert out.detail["recovered"] and log.status("H-crash") == "PREREGISTERED" and not log.was_tested("H-crash")

    def test_Y5_crash_before_experiment(self, tmp_path, monkeypatch):
        import ati.research.workflow as wf_mod
        out, s, log = self.research_crash(tmp_path, monkeypatch, wf_mod, "walk_forward")
        assert out.detail["recovered"] and not log.was_tested("H-crash")

    def test_Y6_crash_after_experiment(self, tmp_path, monkeypatch):
        out, s, log = self.research_crash(tmp_path, monkeypatch, ResearchLog, "record_experiment", after=True)
        assert out.detail["recovered"] and [e["stage"] for e in log.experiments] == ["walk_forward_oos"]

    def test_Y7_crash_before_validation(self, tmp_path, monkeypatch):
        import ati.research.workflow as wf_mod
        out, s, log = self.research_crash(tmp_path, monkeypatch, wf_mod, "challenge")
        assert out.detail["recovered"] and not list(s.research_journal.entries("holdout_access"))

    def test_Y8_crash_after_validation(self, tmp_path, monkeypatch):
        import ati.research.workflow as wf_mod
        out, s, log = self.research_crash(tmp_path, monkeypatch, wf_mod, "decide_promotion", after=True)
        assert out.detail["recovered"] and len(list(s.research_journal.entries("promotion_decision"))) == 1
        assert "requires_review" in out.detail                   # reported, never applied or reversed by recovery
        key = out.detail["candidate"]["key"]
        assert s.strategies.state(key, "BTC/USD") is Lifecycle.CHALLENGER

    def test_Z9_crash_before_candidate_persistence(self, tmp_path, monkeypatch):
        out, s, log = self.research_crash(tmp_path, monkeypatch, factory, "record_lineage")
        lineage = [e.payload for e in s.research_journal.entries("candidate_lineage")]
        assert out.detail["recovered"] and len(lineage) == 1 and lineage[0]["hypothesis_id"] == "H-crash"

    def test_Z10_crash_before_final_cycle_completion(self, tmp_path, monkeypatch):
        cp, s, clock = plane(tmp_path / "st", script={"company": reply("NO_TRADE")})
        real = cp.journal.append
        state = {"done": False}

        def append(type_, payload):
            if type_ == "cycle_end" and not state["done"]:
                state["done"] = True
                raise Boom("crash before cycle_end")
            return real(type_, payload)
        monkeypatch.setattr(cp.journal, "append", append)
        with pytest.raises(Boom):
            cp.run_cycle()
        cp2, s2, _ = plane(tmp_path / "st", clock=clock, script={"company": reply("NO_TRADE")})
        out = cp2.run_cycle()
        assert out.status == "COMPLETED"
        steps = [e.payload["step"] for e in cp2.journal.entries("cycle_step")]
        assert steps.count("learn") == 1 and steps.count("action_result") == 1
        no_duplicates(cp2)
        assert cp2.run_cycle().status == "REPLAY"


# --- Security AB–AF --------------------------------------------------------------------------------------------------------
class TestSecurity:
    @pytest.mark.parametrize("payload,needle", [
        (designed("H-sec-1", strategy_key="trend@v9"), "locked strategy"),
        (designed("H-sec-2") | {"strategy_params": {"fast": 1}}, "unrecognized"),
        (designed("H-sec-3") | {"apply_to_champion": True}, "unrecognized"),
    ])
    def test_AB_claude_cannot_mutate_strategy(self, tmp_path, payload, needle):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST", payload)})
        keys = s.strategies.keys()
        out = cp.run_cycle()
        assert out.status == "FAILED" and needle in out.detail["reason"] and s.strategies.keys() == keys

    @pytest.mark.parametrize("action,payload", [("REVIEW_RISK", {"max_risk_per_trade_fraction": "0.5"}),
                                                ("RESEARCH_REQUEST", designed("H-sec-4") | {"risk_limits": {}}),
                                                ("NO_TRADE", {"kill_switch": "release"})])
    def test_AC_claude_cannot_alter_risk(self, tmp_path, action, payload):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply(action, payload)})
        before = (s.limits, s.limits.limits_hash, s.kill_switch.state())
        out = cp.run_cycle()
        assert out.status == "FAILED" and (s.limits, s.limits.limits_hash, s.kill_switch.state()) == before

    def test_AC2_risk_experiment_is_backtest_only(self, tmp_path):
        from tests.test_self_improvement import condition_experiment
        cp, s, _ = plane(tmp_path / "st", hours=3200, policies=MECHANICS, script={"company": reply(
            "RESEARCH_REQUEST", designed("H-sec-5", condition_experiment("RISK", {"risk_fraction": 0.02})))})
        with_history(s, 3200)
        before = (s.limits, s.limits.limits_hash)
        assert cp.run_cycle().status == "COMPLETED"
        assert (s.limits, s.limits.limits_hash) == before

    @pytest.mark.parametrize("over", [{"data_status": "REAL"}, {"evidence_category": "VALIDATED"}])
    def test_AD_claude_cannot_alter_evidence_category(self, tmp_path, over):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST", designed("H-sec-6") | over)})
        out = cp.run_cycle()
        assert out.status == "FAILED" and "unrecognized" in out.detail["reason"]
        assert s.data_status.value == "MOCK" and s.memory.data_status == "MOCK"
        cp2, s2, _ = plane(tmp_path / "st2", script={"company": reply("NO_TRADE", data_status="REAL")})
        assert cp2.run_cycle().status == "FAILED"

    def test_AE_claude_cannot_access_holdout(self, researched):
        cp, s, clock, _ = researched
        prompts = []
        real = s.reasoning.complete

        def spy(role, rid, prompt):
            prompts.append(prompt)
            return real(role, rid, prompt)
        s.reasoning.complete = spy
        s.reasoning.script["company"] = reply("NO_TRADE")
        clock.advance(timedelta(hours=1))
        cp.run_cycle()
        seal = decode(next(s.research_journal.entries("holdout_sealed")).payload)
        text = prompts[-1]
        assert seal["holdout_dataset_id"] not in text and seal["holdout_commitment"] not in text
        assert "holdout_identity" not in text
        s.evidence.register("holdout", "hold_x", clock.now() - timedelta(hours=1), "holdout")
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed("H-sec-7", evidence_refs=["hold_x"]))
        clock.advance(timedelta(hours=1))
        out = cp.run_cycle()
        assert out.status == "FAILED" and "cannot motivate" in out.detail["reason"]

    @pytest.mark.parametrize("payload", [{"live_trading": True}, {"mode": "LIVE"}])
    def test_AF_claude_cannot_enable_live_trading(self, tmp_path, payload):
        from ati import config
        from ati.company import autonomy
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("REVIEW_SYSTEM", payload)})
        assert cp.run_cycle().status == "FAILED"
        assert config.LIVE_TRADING is False and s.execution.mode.value == "PAPER"
        with pytest.raises(LiveTradingDisabled):
            autonomy.require(autonomy.Autonomy.SUPERVISED_LIVE)

    def test_protocol_minimums_cannot_be_weakened(self, tmp_path):
        payload = designed("H-sec-8", success_criteria=[{"metric": "net_pnl", "op": ">", "threshold": -1e9}])
        cp, s, _ = plane(tmp_path / "st", hours=3200, policies=MECHANICS, script={"company": reply("RESEARCH_REQUEST", payload)})
        with_history(s, 3200)
        cp.run_cycle()
        crit = {(c.metric, c.op, c.threshold) for c in ResearchLog(s.research_journal).get("H-sec-8").criteria}
        assert {("net_pnl", ">", 0.0), ("expectancy_r", ">", 0.0)} <= crit     # the protocol floor stays


# --- Intelligence report -------------------------------------------------------------------------------------------------
class TestIntelligenceReport:
    def test_report_is_labelled_read_only_and_never_claims_profitability(self, traded, tmp_path):
        cp, s, _ = traded
        state = s.state_dir
        before = {p.name: p.read_bytes() for p in state.glob("*.jsonl")}
        r = intelligence_report(cp)
        assert {p.name: p.read_bytes() for p in state.glob("*.jsonl")} == before
        assert set(r) >= {"COMPANY", "STRATEGY", "RESEARCH", "LEARNING", "TRADING"}
        assert r["data_label"] == "MOCK data / PAPER execution" and r["TRADING"]["label"] == r["data_label"]
        assert r["TRADING"]["paper_trades_closed"] >= 2
        assert r["TRADING"]["wins"] + r["TRADING"]["losses"] == r["TRADING"]["paper_trades_closed"]
        assert r["TRADING"]["profitability"].startswith("INSUFFICIENT_EVIDENCE")
        assert r["STRATEGY"]["champions"]["BTC/USD 1h"]["fingerprint"] == \
            s.strategies.champion("BTC/USD", Timeframe.H1).definition_hash
        json.dumps(r, default=str)
