"""Autonomous Trading Company 1.0 — full intelligence build.

Research windows that never reuse a holdout, development-only diagnostic experiments (REGIME / EXECUTION /
RISK), STRUCTURAL candidates from registered logic only, candidate lineage and evaluation trail, idea
saturation and compute budgets, research-only autonomy, context identity, persistence health, assumption
monitors, the extended mistake detectors, the readiness report, and one end-to-end MOCK/PAPER company cycle.

Everything is MOCK/PAPER. Nothing here is evidence about real markets.
"""

import json
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from ati.agent.reasoning import Budget, ScriptedReasoningClient
from ati.company import autonomy, budget, factory
from ati.company.control import CompanyControlPlane
from ati.company.learning import CandidateState, OutcomeRecord, _groups
from ati.company.readiness import report
from ati.core.errors import LiveTradingDisabled, TextEvidenceError
from ati.core.time import FixedClock
from ati.data.dataset import sealed_ranges
from ati.ledger.journal import decode
from ati.market.mock import MockProvider
from ati.market.models import Timeframe
from ati.memory.textual import Interpretation, RawSource, TextEvidence
from ati.research import conditions
from ati.research.hypothesis import ResearchLog
from ati.system import build_paper_system
from tests.helpers import T0
from tests.rig import MOCK_SCRIPT, install_champion
from tests.test_company_control import MECHANICS, counts, packet, plane, reply, trade_payload, with_history
from tests.test_self_improvement import condition_experiment, designed, experiment


def research_plane(tmp_path, script, hours=3200):
    cp, s, clock = plane(tmp_path / "st", hours=hours, policies=MECHANICS, script={"company": script})
    with_history(s, hours)
    return cp, s, clock


def extend_history(s, clock, hours_from, hours_to):
    clock.advance(timedelta(hours=hours_to - hours_from))
    s.store.ingest(c for c in s.provider.fetch_candles("BTC/USD", Timeframe.H1, T0 + timedelta(hours=hours_from),
                                                       T0 + timedelta(hours=hours_to)) if c.is_closed)


# --- holdout: research windows never reuse or overlap a used holdout ------------------------------------------------
class TestResearchWindow:
    def test_second_run_uses_only_data_outside_sealed_holdout(self, tmp_path, monkeypatch):
        cp, s, clock = research_plane(tmp_path, reply("RESEARCH_REQUEST", designed("H-w-1")))
        first = cp.run_cycle()
        assert first.status == "COMPLETED" and first.detail["research_status"] == "COMPLETED", first.detail
        sealed = sealed_ranges("mock", "BTC/USD", Timeframe.H1, s.provider.realization)
        assert len(sealed) == 1
        # same data: a new hypothesis cannot run — the only unsealed run is too short — and says why
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed("H-w-2", statement="A different idea: "
                                                                                             "shorter trends persist"))
        clock.advance(timedelta(hours=1))
        second = cp.run_cycle()
        assert second.detail["research_status"] == "NOT_RUN"
        assert "outside sealed holdout periods" in second.detail["reasons"][0]
        # the same idea tested again is recognised as saturated before anything runs
        monkeypatch.setattr(budget, "POLICY", budget.BudgetPolicy(max_tests_per_idea=1))
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed("H-w-3"))
        clock.advance(timedelta(hours=1))
        third = cp.run_cycle()
        assert third.status == "BLOCKED" and "IDEA SATURATED" in third.detail["reason"]
        monkeypatch.setattr(budget, "POLICY", budget.BudgetPolicy(max_research_runs_per_day=10))
        # fresh data after the sealed period: research runs again, entirely after the used holdout
        extend_history(s, clock, 3200, 6300)
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed("H-w-4", statement="Trend persistence "
                                                                                             "on fresh data"))
        fourth = cp.run_cycle()
        assert fourth.detail["research_status"] == "COMPLETED", fourth.detail
        rows = [e for e in ResearchLog(s.research_journal).experiments if e["hypothesis_id"] == "H-w-4"]
        seals = [decode(e.payload) for e in s.research_journal.entries("holdout_sealed")]
        assert rows and len(seals) == 2
        first_holdout, second_dev = seals[0]["holdout_identity"], seals[1]["development"]
        assert second_dev["start"] >= first_holdout["end"]               # new development data starts after the used holdout
        assert seals[1]["holdout_identity"]["start"] > first_holdout["end"]   # and the new holdout is a fresh period


# --- diagnostic experiments -------------------------------------------------------------------------------------------
class TestDiagnostics:
    @pytest.mark.parametrize("kind,condition", [("REGIME", {"regime": "HIGH_VOL/UP"}),
                                                ("EXECUTION", {"cost_multiplier": 2.0}),
                                                ("EXECUTION", {"entry_delay_bars": 1}),
                                                ("RISK", {"risk_fraction": 0.005})])
    def test_diagnostics_are_preregistered_dev_only_and_never_candidates(self, tmp_path, kind, condition):
        cp, s, _ = research_plane(tmp_path, reply("RESEARCH_REQUEST", designed("H-d-1", condition_experiment(kind, condition))))
        before_keys = set(s.strategies.keys())
        limits_before = (s.limits, s.limits.limits_hash)
        out = cp.run_cycle()
        assert out.status == "COMPLETED" and out.detail["research_status"] == "COMPLETED", out.detail
        log = ResearchLog(s.research_journal)
        rows = [e for e in log.experiments if e["hypothesis_id"] == "H-d-1"]
        assert len(rows) == 1 and rows[0]["stage"] == f"diagnostic:{kind.lower()}"
        assert log.hypotheses_tested == 1                                          # counts for multiple testing
        # never a holdout, never a candidate, never a promotion
        assert not list(s.research_journal.entries("holdout_access"))
        assert not list(s.research_journal.entries("holdout_sealed"))
        assert not list(s.research_journal.entries("promotion_decision"))
        assert set(s.strategies.keys()) - before_keys <= {"trend@v1"}              # only the baseline registered
        design = next(e.payload for e in s.research_journal.entries("experiment_design"))
        assert design["condition"] == condition and design["candidate"] is None
        assert next(e.seq for e in s.research_journal.entries("experiment_design")) < \
            next(e.seq for e in s.research_journal.entries("preregistration"))
        assert (s.limits, s.limits.limits_hash) == limits_before       # backtest sizing only: live limits untouched
        assert s.risk.limits is s.limits


class TestStructural:
    def test_structural_candidate_from_registered_logic_goes_through_full_pipeline(self, tmp_path):
        exp = {k: v for k, v in experiment("STRUCTURAL", iv=["structure"]).items() if k != "candidate_params"} | {
            "structure": {"kind": "buy_and_hold", "params": {"stop_fraction": 0.2}},
            "design_rationale": "compares trend-following with a structurally different exposure rule"}
        cp, s, _ = research_plane(tmp_path, reply("RESEARCH_REQUEST", designed("H-s-1", exp)))
        out = cp.run_cycle()
        assert out.status == "COMPLETED" and out.detail["research_status"] == "COMPLETED", out.detail
        assert s.strategies.get("trend@v1").param_dict["fast"] == 10                 # baseline untouched
        # buy-and-hold trades once per window, so walk-forward cannot select it (>= 5 trades per training window):
        # development evidence is INSUFFICIENT and no candidate is generated — reported, not forced.
        assert out.detail["dev_verdict"] == "INSUFFICIENT_EVIDENCE" and "candidate" not in out.detail
        assert not list(s.research_journal.entries("candidate_lineage"))
        assert not list(s.research_journal.entries("holdout_access"))
        comp = [e.payload for e in s.research_journal.entries("experiment_comparison")]
        assert len(comp) == 1 and comp[0]["candidate_strategy"].startswith("buy_and_hold")
        assert comp[0]["baseline_strategy"] == "trend@v1"

    @pytest.mark.parametrize("structure,needle", [({"kind": "ma_crossover", "params": {"fast": 3}}, "different strategy logic"),
                                                  ({"kind": "martingale", "params": {"x": 2}}, "only registered strategy logic")])
    def test_structural_refusals(self, tmp_path, structure, needle):
        exp = {k: v for k, v in experiment("STRUCTURAL", iv=["structure"]).items() if k != "candidate_params"} | {
            "structure": structure}
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST", designed("H-s-2", exp))})
        out = cp.run_cycle()
        assert out.status == "BLOCKED" and needle in out.detail["reason"]
        assert counts(s)["experiments"] == 0


class TestComputeBudget:
    def test_compute_budget_blocks_before_anything_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(budget, "POLICY", budget.BudgetPolicy(max_compute_units_per_day=10))
        cp, s, _ = research_plane(tmp_path, reply("RESEARCH_REQUEST", designed("H-c-1")))
        out = cp.run_cycle()
        assert out.status == "BLOCKED" and "compute budget" in out.detail["reason"]
        assert counts(s)["experiments"] == 0 and not list(s.research_journal.entries("experiment_design"))


# --- autonomy, context identity, persistence --------------------------------------------------------------------------
class TestAutonomyAndIdentity:
    def test_research_autonomy_cannot_trade(self, tmp_path):
        s_cp, s, clock = plane(tmp_path / "st")
        cp = CompanyControlPlane(s, loop=s_cp.loop, autonomy_level=autonomy.Autonomy.RESEARCH_AUTONOMY)
        install_champion(s)
        s.reasoning.script["company"] = reply("TRADE_PROPOSAL", trade_payload)
        out = cp.run_cycle()
        issued = next(e.payload["data"] for e in cp.journal.entries("cycle_step") if e.payload["step"] == "request_issued")
        assert "TRADE_PROPOSAL" not in issued["allowed_actions"]
        assert out.status in ("BLOCKED", "FAILED") and counts(s)["orders"] == 0

    def test_live_autonomy_cannot_be_constructed(self, tmp_path):
        _, s, _ = plane(tmp_path / "st")
        for level in (autonomy.Autonomy.SUPERVISED_LIVE, autonomy.Autonomy.FULL_LIVE):
            with pytest.raises(LiveTradingDisabled):
                CompanyControlPlane(s, autonomy_level=level)

    def test_context_identity_is_recorded_and_bound(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("NO_TRADE")})
        out = cp.run_cycle()
        steps = {e.payload["step"]: e.payload["data"] for e in cp.journal.entries("cycle_step")}
        assert steps["request_issued"]["context_id"].startswith("ctx_")
        assert steps["action_started"]["context_id"] == steps["request_issued"]["context_id"]
        assert steps["action_started"]["request_id"] == steps["request_issued"]["request_id"]
        assert "datasets" in steps["action_started"] and "strategy_fingerprint" in steps["action_started"]
        assert out.status == "COMPLETED"

    def test_tampered_learning_journal_fails_persistence_and_blocks_research(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", script={"company": reply("NO_TRADE")})
        cp.run_cycle()
        path = cp.learning.journal.path
        path.write_text(path.read_text() + '{"seq": 999, "garbage": true}\n')
        clock.advance(timedelta(hours=1))
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed("H-p-1"))
        out = cp.run_cycle()
        assert out.health["checks"]["persistence"]["status"] == "FAIL"
        assert out.status == "BLOCKED" and any("persistence FAIL" in r for r in out.detail["reasons"])


# --- future-condition monitors -------------------------------------------------------------------------------------------
def bars(n, vol=0.01, volume=100.0, drift=0.0, start=100.0, seed=1):
    import random
    rng = random.Random(seed)
    out, px = [], start
    for _ in range(n):
        px *= 1 + drift + rng.gauss(0, vol)
        out.append(SimpleNamespace(close=Decimal(str(round(px, 6))), volume=Decimal(str(volume))))
    return out


class TestConditions:
    def by_name(self, rows):
        return {r["assumption"]: r for r in rows}

    def test_stable_series_is_stable_and_missing_inputs_are_explicit(self):
        rows = self.by_name(conditions.monitor({"X": bars(1000)}))
        assert rows["volatility"]["status"] == "STABLE" and rows["distribution"]["status"] == "STABLE"
        assert rows["correlation"]["status"] == "NOT_AVAILABLE" and rows["execution_cost"]["status"] == "NOT_AVAILABLE"
        assert all(r["invalidates"] for r in rows.values())

    def test_volatility_and_liquidity_shifts_detected(self):
        series = bars(832, vol=0.005, volume=100.0) + bars(168, vol=0.03, volume=20.0, start=100.0, seed=2)
        rows = self.by_name(conditions.monitor({"X": series}))
        assert rows["volatility"]["status"] == "SHIFT_DETECTED" and rows["liquidity"]["status"] == "SHIFT_DETECTED"

    def test_insufficient_history(self):
        rows = self.by_name(conditions.monitor({"X": bars(100)}))
        assert rows["volatility"]["status"] == "INSUFFICIENT_EVIDENCE"

    def test_execution_cost_uses_realized_fills(self):
        fills = [(Decimal("100"), Decimal("100.5"), Decimal("1"))] * 6
        rows = self.by_name(conditions.monitor({"X": bars(1000)}, fills, Decimal("0.0005")))
        assert rows["execution_cost"]["status"] == "SHIFT_DETECTED"

    def test_cycle_records_conditions_and_shift_is_one_outcome_per_day(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", script={"company": reply("NO_TRADE")})
        cp.run_cycle()
        step = next(e.payload["data"] for e in cp.journal.entries("cycle_step") if e.payload["step"] == "conditions")
        assert {m["assumption"] for m in step["monitors"]} == set(conditions.ASSUMPTIONS)
        # two cycles the same day reporting a shift yield one learning outcome, not two
        for _ in range(2):
            cp.journal.append("cycle_step", {"cycle_id": "x", "step": "conditions", "data": {"monitors": [
                {"assumption": "volatility", "status": "SHIFT_DETECTED", "statistic": 2.0}]}})
        from ati.company.learning import extract_outcomes
        shifts = [o for o in extract_outcomes(s, cp.loop.journal, cp.journal) if o.source == "condition_shift"]
        assert len(shifts) == 1


# --- mistake detection ------------------------------------------------------------------------------------------------------
def trade(i, pnl, strategy="S", regime="HIGH_VOL/DOWN", fees="1", gross=None, at=None):
    return OutcomeRecord(f"rev_{i}", "trade", at or f"2024-01-{i + 1:02d}T00:00:00+00:00", "MOCK", "BTC/USD", "1h", strategy,
                         realized={"pnl": str(pnl), "gross": str(gross if gross is not None else pnl), "exit_reason": "stop"},
                         deviation={"costs_over_half_of_gross": "0.9"} if gross is not None else {},
                         facts={"regime": regime})


class TestMistakeDetection:
    def patterns(self, outcomes):
        return {k[0]: v for k, v in _groups(outcomes).items()}

    def test_drawdown_cluster_needs_three_consecutive_losses(self):
        assert "DRAWDOWN_CLUSTER" not in self.patterns([trade(0, -1), trade(1, -1), trade(2, 5), trade(3, -1)])
        g = self.patterns([trade(0, -1), trade(1, -1), trade(2, -1)])
        assert len(g["DRAWDOWN_CLUSTER"]) == 3 and len(g["REGIME_FAILURE"]) == 3

    def test_degradation_requires_worse_and_negative_recent_half(self):
        good_then_bad = [trade(i, 10) for i in range(5)] + [trade(i, -3) for i in range(5, 10)]
        assert len(self.patterns(good_then_bad)["STRATEGY_DEGRADATION"]) == 5
        steady = [trade(i, 10) for i in range(10)]
        assert "STRATEGY_DEGRADATION" not in self.patterns(steady)

    def test_unexpected_costs_and_robustness_detectors(self):
        g = self.patterns([trade(0, -1, gross=1)])
        assert "UNEXPECTED_COSTS" in g
        rob = OutcomeRecord("adv:1", "robustness", "t", "MOCK", strategy="h", dataset_id="d", realized={"failed": [
            "Does it survive parameter changes?", "Does the edge survive doubled costs and slippage?"]})
        g = self.patterns([rob])
        assert "PARAMETER_INSTABILITY" in g and "ROBUSTNESS_FAILURE" in g

    def test_one_off_never_becomes_doctrine(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st")
        cp.learning.outcomes["rev_x"] = trade(0, -1)
        cp.learning.learn(s, cp.loop.journal, cp.journal)
        for c in cp.learning.candidates.values():
            assert cp.learning.pattern_class(c, s).value == "ONE_OFF" and c.state is CandidateState.OBSERVED
            assert not c.research_eligible
        assert len(s.memory) == 0


# --- text evidence: hypothesis layer --------------------------------------------------------------------------------------
class TestTextHypothesis:
    def test_hypothesis_counts_independent_sources_only(self):
        te = TextEvidence()
        te.add_source(RawSource("n1", "wire", T0, T0, "Exchange X paused withdrawals."))
        for m in ("a", "b"):
            te.add_interpretation(Interpretation("n1", m, "bearish", T0 + timedelta(hours=1)))
        row = te.add_hypothesis("H-news-1", ["n1", "n1"], T0 + timedelta(hours=2))
        assert row["independent_sources"] == 1 and row["layer"] == "HYPOTHESIS"
        with pytest.raises(TextEvidenceError):
            te.add_hypothesis("H-news-2", ["n1"], T0 - timedelta(hours=1))     # look-ahead
        with pytest.raises(TextEvidenceError):
            te.add_hypothesis("H-news-3", ["unknown"], T0 + timedelta(hours=2))


# --- readiness ----------------------------------------------------------------------------------------------------------------
class TestReadiness:
    def test_report_is_complete_explicit_and_read_only(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("NO_TRADE")})
        cp.run_cycle()
        state = tmp_path / "st"
        before = {p.name: p.read_bytes() for p in state.glob("*.jsonl")}
        r = report(cp)
        assert {p.name: p.read_bytes() for p in state.glob("*.jsonl")} == before       # read-only
        assert set(r) >= {"DATA", "STRATEGY", "RESEARCH", "LEARNING", "VALIDATION", "RISK", "EXECUTION"}
        assert r["DATA"]["category"] == "MOCK" and r["DATA"]["market_evidence"] is False
        assert r["EXECUTION"]["live_trading"] is False and r["EXECUTION"]["mode"] == "PAPER"
        assert r["EXECUTION"]["maximum_autonomy"] == "PAPER_AUTONOMY"
        assert r["VALIDATION"]["holdout"] == "NOT_RUN" and r["STRATEGY"]["champion"] == "NOT_AVAILABLE"
        assert r["LEARNING"]["validated_findings"].startswith("NONE")
        json.dumps(r, default=str)


# --- end-to-end: one MOCK/PAPER company traversal ----------------------------------------------------------------------------
class TestEndToEndPaperCycle:
    def test_outcome_to_learning_to_hypothesis_to_experiment(self, tmp_path):
        """Company starts → health → bounded context → Claude (MOCK) acts → validated → paper outcomes → LEARN →
        learning candidate → Claude cites it in a designed hypothesis → pre-registered → experiment → candidate
        evaluated → memory updated → cycle finalized. All MOCK/PAPER."""
        mode = {"research": None}
        answered: dict[str, str] = {}

        def claude(prompt):
            p = packet(prompt)
            if p["request_id"] not in answered:
                eligible = [c for c in p["learning_candidates"] if c["research_eligible"]]
                if mode["research"] is None and eligible and "RESEARCH_REQUEST" in p["allowed_actions"]:
                    mode["research"] = eligible[0]["candidate_id"]
                    action, payload = "RESEARCH_REQUEST", designed("H-e2e-1", learning_candidate_id=mode["research"],
                                                                   statement="Faster entries reduce repeated losses")
                elif p["strategy"]["entry_signals"] and "TRADE_PROPOSAL" in p["allowed_actions"]:
                    action, payload = "TRADE_PROPOSAL", trade_payload(p)
                else:
                    action, payload = "NO_TRADE", {}
                answered[p["request_id"]] = json.dumps({
                    "request_id": p["request_id"], "cycle_id": p["cycle_id"], "context_id": p["context_id"],
                    "action": action, "reason": "[MOCK] scripted company brain", "payload": payload})
            return answered[p["request_id"]]

        clock = FixedClock(T0 + timedelta(hours=3200))
        provider = MockProvider(11, clock, epoch=T0)
        s = build_paper_system(tmp_path / "st", clock, provider,
                               ScriptedReasoningClient(MOCK_SCRIPT | {"company": claude}, Budget(100_000)),
                               data_status=provider.data_status)
        with_history(s, 3200)
        install_champion(s)
        cp = CompanyControlPlane(s, research_policies=MECHANICS)
        research_out = None
        for _ in range(400):
            out = cp.run_cycle()
            assert out.status in ("COMPLETED", "REPLAY", "BLOCKED"), out.detail
            if out.action == "RESEARCH_REQUEST":
                research_out = out
                break
            clock.advance(timedelta(hours=1))
        assert research_out is not None, "no learning candidate became research-eligible"
        trades = [o for o in cp.learning.outcomes.values() if o.source == "trade"]
        assert trades and all(o.data_category == "MOCK" for o in trades)       # paper outcomes, labelled MOCK
        assert all(o.facts["trade_id"] and o.strategy for o in trades)          # lineage to decision + fingerprint
        assert research_out.status == "COMPLETED", research_out.detail
        assert research_out.detail["research_status"] == "COMPLETED", research_out.detail
        log = ResearchLog(s.research_journal)
        assert log.status("H-e2e-1") != "UNKNOWN"                                # pre-registered
        design = next(e.payload for e in s.research_journal.entries("experiment_design"))
        assert design["learning_candidate_id"] == mode["research"] and design["experiment_id"].startswith("exp_")
        cand = cp.learning.candidates[mode["research"]]
        assert cand.hypothesis_id == "H-e2e-1" and cand.state is not CandidateState.OBSERVED
        assert log.was_tested("H-e2e-1") and "comparison" in research_out.detail   # candidate evaluated vs baseline
        assert len(s.memory) >= 1                                                  # workflow memory (MOCK-tagged)
        assert all(m.kind.value in ("HYPOTHESIS", "REJECTED_HYPOTHESIS") for m in s.memory.query(clock.now()))
        assert all(m.statement.startswith("[MOCK]") for m in s.memory.query(clock.now()))
        steps = [e.payload["step"] for e in cp.journal.entries("cycle_step") if e.payload["cycle_id"] == research_out.cycle_id]
        assert steps[:3] == ["health", "conditions", "request_issued"] and steps[-1] == "learn"
        assert cp.state.value == "COMPLETED"
        for j in (s.research_journal, cp.journal, cp.learning.journal, s.decisions.journal):
            j.verify()
