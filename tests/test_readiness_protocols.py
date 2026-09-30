"""Phase 4A — readiness states and the research protocol registry.

No REAL candles exist or are created. REAL-state semantics are tested through the pure readiness contract with
explicit inputs (``classify`` / ``validation_outcome``); integration paths use MOCK systems (labelled MOCK) and an
empty REAL-bound state directory. The only REAL-labelled candle below is a forged in-memory one that must be
rejected.
"""

import json
from dataclasses import replace
from datetime import timedelta

import pytest

from ati.agent.reasoning import Budget, ScriptedReasoningClient
from ati.company import factory
from ati.core.canonical import sha256_text
from ati.core.errors import DataIntegrityError
from ati.core.time import FixedClock
from ati.data.dataset import Dataset
from ati.ledger.journal import decode
from ati.market.accumulate import SERIES
from ati.market.health import (classify, protocol_validation, readiness_table, research_readiness, series_health,
                               validation_outcome)
from ati.market.kraken import KrakenPublicOHLC
from ati.market.models import Candle, DataStatus, Provenance, Timeframe
from ati.market.provider import UrllibTransport
from ati.market.universe import default_universe
from ati.research import protocol as P
from ati.research.hypothesis import ResearchLog
from ati.research.protocols import REGISTRY, compatibility, for_series
from ati.strategies.base import StrategyDefinition
from ati.system import build_paper_system
from tests.helpers import T0, make_candle, mock_dataset
from tests.rig import MOCK_SCRIPT
from tests.test_company_control import MECHANICS, plane, reply, with_history
from tests.test_self_improvement import designed

H1, H4 = Timeframe.H1, Timeframe.H4
STATES = {"REAL_DATA_UNAVAILABLE", "BLOCKED", "INSUFFICIENT_REAL_CANDLES", "VALIDATION_PENDING",
          "INSUFFICIENT_EVIDENCE", "VALIDATION_FAILED", "VALIDATION_PASSED", "NOT_APPLICABLE_MOCK"}


# --- readiness contract A–H -------------------------------------------------------------------------------------------
class TestReadinessContract:
    def test_A_zero_real_candles_is_real_data_unavailable(self, tmp_path):
        r = classify("REAL", 0, 0, 3000, "NOT_READY", "NOT_RUN")
        assert r["state"] == "REAL_DATA_UNAVAILABLE" and r["real_data"] == "REAL_DATA_UNAVAILABLE"
        clock = FixedClock(T0 + timedelta(hours=10))
        s = build_paper_system(tmp_path / "real", clock, KrakenPublicOHLC(UrllibTransport(), clock),   # empty, REAL-bound
                               ScriptedReasoningClient(MOCK_SCRIPT, Budget(1)), data_status=DataStatus.REAL)
        for row in readiness_table(s, SERIES):
            assert row["overall"] == "REAL_DATA_UNAVAILABLE" and "FAIL" not in json.dumps(row)

    def test_B_some_real_candles_below_floor_is_distinct_from_no_data(self):
        r = classify("REAL", 1200, 1200, 3000, "PASS", "NOT_RUN")
        assert r["state"] == "INSUFFICIENT_REAL_CANDLES" and r["real_data"].startswith("REAL_DATA_PRESENT")
        assert r["state"] != classify("REAL", 0, 0, 3000, "NOT_READY", "NOT_RUN")["state"]
        # enough candles in total but not contiguous/unsealed: still insufficient
        assert classify("REAL", 5000, 2999, 3000, "PASS", "NOT_RUN")["state"] == "INSUFFICIENT_REAL_CANDLES"

    def test_C_enough_candles_below_evidence_floor_is_insufficient_evidence(self):
        assert validation_outcome("INSUFFICIENT_EVIDENCE", None, None) == "INSUFFICIENT_EVIDENCE"
        assert validation_outcome("PASS", "INSUFFICIENT_EVIDENCE", ["PASS"]) == "INSUFFICIENT_EVIDENCE"
        assert validation_outcome("PASS", "PASS", ["PASS", "INSUFFICIENT_EVIDENCE"]) == "INSUFFICIENT_EVIDENCE"
        r = classify("REAL", 3200, 3200, 3000, "PASS", "INSUFFICIENT_EVIDENCE")
        assert r["state"] == "INSUFFICIENT_EVIDENCE" and r["evidence_readiness"] == "INSUFFICIENT_EVIDENCE"
        assert r["candle_readiness"] == "SUFFICIENT" and "FAILED" not in r["state"]

    def test_D_sufficient_evidence_awaiting_validation_is_pending(self):
        assert classify("REAL", 3000, 3000, 3000, "PASS", "NOT_RUN")["state"] == "VALIDATION_PENDING"
        assert validation_outcome("PASS", None, None) == "VALIDATION_PENDING"      # run not finished
        assert validation_outcome(None, None, None) == "NOT_RUN"

    @pytest.mark.parametrize("dev,hold,adv", [("FAIL", None, None), ("PASS", "FAIL", ["PASS"]), ("PASS", "PASS", ["FAIL"])])
    def test_E_actually_failed_validation(self, dev, hold, adv):
        assert validation_outcome(dev, hold, adv) == "VALIDATION_FAILED"
        assert classify("REAL", 3200, 3200, 3000, "PASS", "VALIDATION_FAILED")["state"] == "VALIDATION_FAILED"

    def test_F_actually_passed_validation(self):
        assert validation_outcome("PASS", "PASS", ["PASS", "NOT_AUTOMATED"]) == "VALIDATION_PASSED"
        r = classify("REAL", 3200, 3200, 3000, "PASS", "VALIDATION_PASSED")
        assert r["state"] == r["validation"] == "VALIDATION_PASSED" and r["evidence_readiness"] == "SUFFICIENT"
        # a stage that never ran can never read as passed
        assert validation_outcome("PASS", None, ["PASS"]) != "VALIDATION_PASSED"
        assert validation_outcome("PASS", "PASS", None) != "VALIDATION_PASSED"

    def test_G_mock_evidence_never_produces_real_readiness(self, tmp_path):
        r = classify("MOCK", 5000, 5000, 3000, "PASS", "VALIDATION_PASSED")
        assert r["state"] == "NOT_APPLICABLE_MOCK" and r["validation"] == "NOT_APPLICABLE"
        cp, s, _ = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                         script={"company": reply("RESEARCH_REQUEST", designed("H-4a-g"))})
        with_history(s, 3200)
        assert cp.run_cycle().detail["research_status"] == "COMPLETED"
        rr = research_readiness(s, "BTC/USD", H1, series_health(s, "BTC/USD", H1))
        assert rr["state"] == "NOT_APPLICABLE_MOCK" and rr["validation"] == "NOT_APPLICABLE"
        assert rr["non_market_mechanics_outcome"].startswith("[MOCK]")          # visible, labelled, never REAL

    def test_H_mixed_provenance_never_valid(self):
        r = classify("MIXED", 3200, 3200, 3000, "PASS", "VALIDATION_PASSED")
        assert r["state"] == "BLOCKED" and r["real_data"] == "MIXED_INVALID"
        with pytest.raises(DataIntegrityError):
            Dataset.build([make_candle(0), make_candle(1, status=DataStatus.SYNTHETIC)], data_version="v", realization="r")

    def test_every_state_is_from_the_contract_vocabulary(self):
        for cat in ("REAL", "MOCK", "MIXED"):
            for n in (0, 10, 3000):
                for health in ("PASS", "FAIL", "NOT_READY"):
                    for v in ("NOT_RUN", "VALIDATION_PENDING", "INSUFFICIENT_EVIDENCE", "VALIDATION_FAILED",
                              "VALIDATION_PASSED"):
                        assert classify(cat, n, n, 3000, health, v)["state"] in STATES


# --- readiness from recorded runs, restart, non-mutation I–J ------------------------------------------------------------
class TestRecordedReadiness:
    def test_I_J_readiness_is_derived_reconstructs_and_mutates_nothing(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                             script={"company": reply("RESEARCH_REQUEST", designed("H-4a-i"))})
        with_history(s, 3200)
        out = cp.run_cycle()
        run = decode(next(s.research_journal.entries("protocol_run")).payload)
        assert run["protocol_id"] == P.PROTOCOL_ID and run["protocol_hash"] == REGISTRY[P.PROTOCOL_ID].protocol_hash
        assert run["kind"] == "validation" and run["hypothesis_id"] == "H-4a-i"
        v = protocol_validation(s, P.PROTOCOL_ID)
        log = ResearchLog(s.research_journal)
        assert v["hypothesis_id"] == "H-4a-i" and v["stages"]["walk_forward_oos"] == log.status("H-4a-i").split(":")[1]
        state = s.state_dir
        before = {p.name: p.read_bytes() for p in state.glob("*.jsonl")}
        lineage = [e.payload["fingerprint"] for e in s.research_journal.entries("candidate_lineage")]
        trails = [factory.stages(s, f) for f in lineage]
        table = readiness_table(s, SERIES)
        assert {p.name: p.read_bytes() for p in state.glob("*.jsonl")} == before           # J: read-only
        assert [factory.stages(s, f) for f in lineage] == trails
        cp2, s2, _ = plane(tmp_path / "st", hours=3200, clock=clock, policies=MECHANICS)
        with_history(s2, 3200)
        assert readiness_table(s2, SERIES) == table and protocol_validation(s2, P.PROTOCOL_ID) == v   # I: restart

    def test_research_status_is_identical_before_and_after_restart(self, tmp_path):
        """Found by this milestone's audit: in-process status read 'TESTED:Verdict.X', reloaded read 'TESTED:X'."""
        from ati.company import budget
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                             script={"company": reply("RESEARCH_REQUEST", designed("H-4a-rs"))})
        with_history(s, 3200)
        cp.run_cycle()
        live = ResearchLog(s.research_journal)
        cp2, s2, _ = plane(tmp_path / "st", hours=3200, clock=clock, policies=MECHANICS)
        reloaded = ResearchLog(s2.research_journal)
        for h in ("H-4a-rs", "H-4a-rs:holdout"):
            assert live.status(h) == reloaded.status(h) and "Verdict." not in live.status(h)
        assert budget.usage(s, cp.journal, clock.now())["rejected_hypotheses"] == \
            budget.usage(s2, cp2.journal, clock.now())["rejected_hypotheses"]

    def test_diagnostics_are_not_validation(self, tmp_path):
        from tests.test_self_improvement import condition_experiment
        cp, s, _ = plane(tmp_path / "st", hours=3200, policies=MECHANICS, script={"company": reply(
            "RESEARCH_REQUEST", designed("H-4a-diag", condition_experiment("REGIME", {"regime": "HIGH_VOL/UP"})))})
        with_history(s, 3200)
        cp.run_cycle()
        assert decode(next(s.research_journal.entries("protocol_run")).payload)["kind"] == "diagnostic"
        assert protocol_validation(s, P.PROTOCOL_ID)["outcome"] == "NOT_RUN"      # a diagnostic is not validation


# --- protocol registry and identity -----------------------------------------------------------------------------------
class TestRegistry:
    def test_exactly_the_four_approved_datasets(self):
        assert {(p.symbol, p.timeframe) for p in REGISTRY.values()} == set(SERIES)
        assert len(REGISTRY) == 4 and all(p.symbol in default_universe() for p in REGISTRY.values())

    def test_reference_protocol_is_reproduced_exactly(self):
        p = REGISTRY[P.PROTOCOL_ID]
        assert (p.symbol, p.timeframe, p.hypothesis_id, p.statement) == (P.SYMBOL, P.TIMEFRAME, P.HYPOTHESIS_ID, P.STATEMENT)
        assert dict(p.base_params) == P.BASE_PARAMS and [dict(g) for g in p.grid] == P.GRID
        assert p.criteria == tuple((c.metric, c.op, c.threshold) for c in P.CRITERIA)
        assert (p.min_candles, p.holdout_fraction, p.train_bars, p.test_bars) == \
               (P.MIN_CANDLES, P.HOLDOUT_FRACTION, P.TRAIN_BARS, P.TEST_BARS)
        assert p.strategy_fingerprint == P.base_definition(T0).definition_hash and p.executable

    def test_equivalent_requirements_no_invented_thresholds(self):
        ref = REGISTRY[P.PROTOCOL_ID]
        same = ("statement", "strategy_id", "strategy_version", "strategy_kind", "base_params", "grid", "criteria",
                "min_candles", "holdout_fraction", "train_bars", "test_bars", "min_oos_trades", "min_holdout_trades",
                "min_positive_fold_fraction", "validation_stages", "permitted_provenance")
        for p in REGISTRY.values():
            assert all(getattr(p, f) == getattr(ref, f) for f in same), p.protocol_id
            assert p.min_candles == 3000 and p.min_holdout_trades == 20 and p.min_oos_trades == 30
            assert p.permitted_provenance == ("DELAYED", "HISTORICAL", "REAL")
            assert p.validation_stages == ("walk_forward_oos:PASS", "adversarial:NON_BLOCKING", "holdout:PASS",
                                           "promotion_gate:APPROVED")
        # Phase 4B: the registry is timeframe-aware, so every protocol resolves to its own deployment
        for pid in ("REAL-PROTOCOL-002", "REAL-PROTOCOL-003", "REAL-PROTOCOL-004"):
            assert REGISTRY[pid].executable and not REGISTRY[pid].not_executable_reason

    def test_identity_is_deterministic_and_material_changes_change_it(self):
        p = REGISTRY["REAL-PROTOCOL-002"]
        assert replace(p).protocol_hash == p.protocol_hash                       # same definition, same identity
        assert replace(p, executable=True).protocol_hash == p.protocol_hash      # state is not identity
        for change in ({"min_candles": 2999}, {"timeframe": H1}, {"symbol": "ETH/USD"}, {"min_holdout_trades": 19},
                       {"base_params": tuple(sorted((P.BASE_PARAMS | {"fast": 11}).items()))}, {"holdout_fraction": 0.1}):
            assert replace(p, **change).protocol_hash != p.protocol_hash, change
        assert len({q.protocol_hash for q in REGISTRY.values()}) == 4
        assert REGISTRY["REAL-PROTOCOL-001"].strategy_fingerprint == REGISTRY["REAL-PROTOCOL-003"].strategy_fingerprint
        assert REGISTRY["REAL-PROTOCOL-001"].strategy_fingerprint != REGISTRY["REAL-PROTOCOL-002"].strategy_fingerprint

    def test_unknown_protocols_cannot_be_requested(self, tmp_path):
        payload = designed("H-4a-x") | {"protocol_id": "REAL-PROTOCOL-999"}
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST", payload)})
        out = cp.run_cycle()
        assert out.status == "FAILED" and "unknown research protocol" in out.detail["reason"]


# --- compatibility --------------------------------------------------------------------------------------------------------
class TestCompatibility:
    @pytest.fixture
    def system(self, tmp_path):
        return plane(tmp_path / "st")[1]

    def strategy(self, tf=H1, **params):
        return StrategyDefinition.create("trend", 1, "ma_crossover", P.BASE_PARAMS | params, tf, T0)

    @pytest.mark.parametrize("symbol,tf,pid,needle", [
        ("BTC/USD", H1, "REAL-PROTOCOL-002", "timeframe 1h ≠ protocol 4h"),       # BTC 1h → BTC 4h
        ("BTC/USD", H4, "REAL-PROTOCOL-004", "symbol BTC/USD ≠ protocol ETH/USD"),  # BTC 4h → ETH 4h
        ("ETH/USD", H1, "REAL-PROTOCOL-004", "timeframe 1h ≠ protocol 4h"),       # ETH 1h → ETH 4h
    ])
    def test_series_mismatches_rejected(self, system, symbol, tf, pid, needle):
        ds = mock_dataset(120, tf=tf, symbol=symbol)
        reasons = compatibility(REGISTRY[pid], ds, REGISTRY[pid].strategy, system)
        assert any(needle in r for r in reasons), reasons

    def test_mock_rejected_by_real_protocol(self, system):
        reasons = compatibility(REGISTRY[P.PROTOCOL_ID], mock_dataset(120), REGISTRY[P.PROTOCOL_ID].strategy, system)
        assert any("provenance MOCK is not permitted" in r for r in reasons)

    def test_forged_real_label_is_not_proof(self, system):
        """A candle *labelled* REAL with a plausible payload hash, never received from a provider."""
        forged = [Candle(provider="kraken", symbol="BTC/USD", timeframe=H1, open_time=c.open_time, open=c.open,
                         high=c.high, low=c.low, close=c.close, volume=c.volume, is_closed=True, status=DataStatus.REAL,
                         received_at=c.received_at, provenance=Provenance("kraken", "forged", c.received_at,
                                                                          raw_sha256=sha256_text("not a response")))
                  for c in mock_dataset(120).candles]
        ds = Dataset.build(forged, data_version="v", realization="observed")
        reasons = compatibility(REGISTRY[P.PROTOCOL_ID], ds, REGISTRY[P.PROTOCOL_ID].strategy, system)
        assert any("not proven by the payload archive" in r for r in reasons)

    def test_strategy_identity_and_fingerprint_enforced(self, system):
        ds = mock_dataset(120)
        p = REGISTRY[P.PROTOCOL_ID]
        assert any("strategy deployment" in r for r in compatibility(p, ds, self.strategy(fast=11), system))
        assert any("strategy deployment" in r for r in compatibility(p, ds, self.strategy(tf=H4), system))
        child = StrategyDefinition.create("trend", 1, "ma_crossover", P.BASE_PARAMS, H1, T0, parent_hash="x" * 64)
        assert any("fingerprint" in r for r in compatibility(p, ds, child, system))

    def test_candle_floor_and_gaps_enforced(self, system):
        ds = mock_dataset(120)
        assert any("< 3000" in r for r in compatibility(REGISTRY[P.PROTOCOL_ID], ds, REGISTRY[P.PROTOCOL_ID].strategy,
                                                         system))
        gapped = Dataset.build([c for i, c in enumerate(mock_dataset(120).candles) if i != 50], data_version="v",
                               realization="r")
        assert any("missing interval" in r for r in compatibility(REGISTRY[P.PROTOCOL_ID], gapped,
                                                                  REGISTRY[P.PROTOCOL_ID].strategy, system))


# --- holdout safety ------------------------------------------------------------------------------------------------------
class TestHoldoutSafety:
    def test_protocol_metadata_and_claude_context_carry_no_holdout_evidence(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                             script={"company": reply("RESEARCH_REQUEST", designed("H-4a-h"))})
        with_history(s, 3200)
        cp.run_cycle()
        seal = decode(next(s.research_journal.entries("holdout_sealed")).payload)
        described = json.dumps([p.describe() for p in REGISTRY.values()], default=str)
        assert seal["holdout_dataset_id"] not in described and "ds_" not in described
        assert '"sealed_holdout_required": true' in described
        seen = []
        real = s.reasoning.complete
        s.reasoning.complete = lambda role, rid, prompt: (seen.append(prompt), real(role, rid, prompt))[1]
        s.reasoning.script["company"] = reply("NO_TRADE")
        clock.advance(timedelta(hours=1))
        cp.run_cycle()
        prompt = seen[-1]
        assert seal["holdout_dataset_id"] not in prompt and seal["holdout_commitment"] not in prompt
        assert "protocol_run" not in prompt and "validation_detail" not in prompt
        assert "holdout_identity" not in prompt and "SEALED_HOLDOUT" in prompt
