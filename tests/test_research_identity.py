"""Phase 4B — timeframe-aware strategy and research identity.

Identity model under test:
  strategy fingerprint   ``definition_hash``: id, version, kind, params, timeframe, code, lineage (unchanged contract)
  behaviour fingerprint  ``behavior_fingerprint``: the same without timeframe — 1h and 4h deployments share it
  registry key           ``strategy_id@vN/<timeframe>`` — unique; ``strategy_id@vN`` is only a behavioural name
  research dimension     (symbol, timeframe) — lifecycle/champion state is scoped to it; there is no global champion
  dataset identity       content hash of candles (independent of strategy identity)
  protocol identity      hash of the protocol's material requirements

All data is MOCK; nothing here is market evidence.
"""

import hashlib
import json
from dataclasses import replace
from datetime import timedelta

import pytest

from ati.company import factory
from ati.company.control import PROTOCOLS
from ati.core.errors import LifecycleError, PromotionDenied, StrategyImmutableError
from ati.ledger.journal import Journal, decode
from ati.market.health import classify
from ati.market.models import DataStatus, Timeframe
from ati.research import protocol as P
from ati.research.backtest import run_backtest
from ati.research.protocols import REGISTRY, deployment, identity_mismatches
from ati.research.workflow import run_research_cycle
from ati.risk.engine import RiskLimits
from ati.strategies.base import StrategyDefinition
from ati.strategies.registry import LEGACY_SCOPE, Lifecycle, StrategyRegistry
from ati.validation.promotion import PromotionPolicy
from tests.helpers import T0, mock_dataset
from tests.rig import install_champion
from tests.test_company_control import MECHANICS, plane, reply, with_history
from tests.test_self_improvement import designed

H1, H4 = Timeframe.H1, Timeframe.H4
EXPECTED = {"REAL-PROTOCOL-001": ("BTC/USD", H1), "REAL-PROTOCOL-002": ("BTC/USD", H4),
            "REAL-PROTOCOL-003": ("ETH/USD", H1), "REAL-PROTOCOL-004": ("ETH/USD", H4)}
# Protocol identities recorded when Phase 4A declared them; Phase 4B only changed the executable *state*.
PHASE_4A_HASHES = {"REAL-PROTOCOL-001": "ec07c9e015d6d12d", "REAL-PROTOCOL-002": "5a7d0199afda836a",
                   "REAL-PROTOCOL-003": "4530136be2b327ec", "REAL-PROTOCOL-004": "5b680b7aceb8b7d5"}
# Signal + backtest digests computed with the pre-change code (7ab9bb2) and with this code: identical.
GOLDEN = {"1h/base": "9b9a87a8db3b6a083ba0fafb", "1h/fast": "5a2fdbbee6e72335238ab1ab",
          "4h/base": "3bca5919126e69873c6907a7", "4h/fast": "5089f6340e1165c53d17bd21"}
FAST = {"fast": 5, "slow": 20, "atr_period": 14, "stop_atr": 2.0}


def trend(tf, params=None, **kw):
    return StrategyDefinition.create("trend", 1, "ma_crossover", params or P.BASE_PARAMS, tf, T0, **kw)


# --- identity A–D, I–J -------------------------------------------------------------------------------------------------
class TestIdentity:
    def test_A_B_C_one_strategy_two_timeframe_deployments_without_collision(self):
        reg = StrategyRegistry()
        a, b = reg.register(trend(H1)), reg.register(trend(H4))
        assert (a.registry_key, b.registry_key) == ("trend@v1/1h", "trend@v1/4h")
        assert a.key == b.key == "trend@v1"                                      # the behavioural name is shared
        assert a.definition_hash != b.definition_hash                            # deployment-specific fingerprint
        assert reg.get("trend@v1/1h") is a and reg.get("trend@v1/4h") is b
        assert sorted(reg.definitions()) == ["trend@v1/1h", "trend@v1/4h"]
        assert reg.register(trend(H1)) is a                                      # idempotent, per deployment
        with pytest.raises(StrategyImmutableError):
            reg.register(trend(H1, {**P.BASE_PARAMS, "fast": 11}))              # same key, different content

    def test_D_fingerprint_semantics(self):
        h1, h4 = trend(H1), trend(H4)
        assert h1.behavior_fingerprint == h4.behavior_fingerprint                # same rule, same code, same params
        assert h1.definition_hash != h4.definition_hash                          # timeframe is part of the deployment
        assert trend(H1, FAST).behavior_fingerprint != h1.behavior_fingerprint   # params change the behaviour
        assert h1.definition_hash == REGISTRY["REAL-PROTOCOL-001"].strategy_fingerprint
        assert h4.definition_hash == REGISTRY["REAL-PROTOCOL-002"].strategy_fingerprint

    def test_versions_are_counted_per_timeframe(self):
        reg = StrategyRegistry()
        reg.register(trend(H1))
        reg.register(trend(H4))
        child = reg.derive("trend@v1/4h", FAST, T0, "variant")
        assert child.registry_key == "trend@v2/4h" and child.parent_hash == reg.get("trend@v1/4h").definition_hash
        assert "trend@v2/1h" not in reg.definitions()

    @pytest.mark.parametrize("tf", [Timeframe.M5, Timeframe.D1])
    def test_I_unsupported_timeframe_rejected(self, tf):
        reg = StrategyRegistry()
        with pytest.raises(LifecycleError, match="not an approved research timeframe"):
            reg.register(trend(tf))
        with pytest.raises(LifecycleError):
            reg.champion("BTC/USD", tf)

    def test_J_missing_or_malformed_timeframe_rejected(self):
        with pytest.raises(ValueError, match="timeframe must be a Timeframe"):
            StrategyDefinition("trend", 1, "ma_crossover", tuple(P.BASE_PARAMS.items()), None, T0)
        with pytest.raises(ValueError):
            StrategyDefinition("trend", 1, "ma_crossover", tuple(P.BASE_PARAMS.items()), "1h", T0)
        reg = StrategyRegistry()
        reg.register(trend(H1))
        with pytest.raises(KeyError, match="has no timeframe"):
            reg.get("trend@v1")                                                  # no implicit fallback
        for bad in (None, "1h"):
            with pytest.raises(LifecycleError):
                reg.champion("BTC/USD", bad)
        for symbol in (None, "", LEGACY_SCOPE, "*"):
            with pytest.raises(LifecycleError):
                reg.state("trend@v1/1h", symbol)


# --- routing E–H, protocols K–O ---------------------------------------------------------------------------------------
class TestRouting:
    @pytest.mark.parametrize("pid", sorted(EXPECTED))
    def test_E_F_G_H_each_protocol_resolves_its_own_deployment(self, tmp_path, pid):
        symbol, tf = EXPECTED[pid]
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST", designed("H-4b-r") | {"protocol_id": pid})})
        cp.run_cycle()
        step = next(e.payload["data"] for e in cp.journal.entries("cycle_step") if e.payload["step"] == "research_invoked")
        dep = step["deployment"]
        assert (dep["symbol"], dep["timeframe"], dep["protocol_id"]) == (symbol, tf.value, pid)
        assert dep["registry_key"] == f"trend@v1/{tf.value}" and dep["strategy_fingerprint"] == trend(tf).definition_hash
        assert dep["protocol_hash"] == REGISTRY[pid].protocol_hash

    def test_K_all_four_protocols_are_routable(self):
        assert set(PROTOCOLS) == set(EXPECTED) and all(p.executable for p in REGISTRY.values())
        assert {(p.symbol, p.timeframe) for p in REGISTRY.values()} == set(EXPECTED.values())

    @pytest.mark.parametrize("pid", sorted(EXPECTED))
    def test_identity_layer_accepts_the_matching_deployment(self, pid):
        symbol, tf = EXPECTED[pid]
        assert identity_mismatches(REGISTRY[pid], symbol, tf, trend(tf)) == []

    @pytest.mark.parametrize("symbol,tf,strategy_tf,pid,needle", [
        ("BTC/USD", H1, H4, "REAL-PROTOCOL-001", "strategy deployment"),   # BTC 1h data + 4h deployment
        ("BTC/USD", H4, H4, "REAL-PROTOCOL-004", "symbol"),               # BTC 4h → ETH 4h protocol
        ("ETH/USD", H1, H1, "REAL-PROTOCOL-004", "timeframe"),            # ETH 1h → ETH 4h protocol
        ("BTC/USD", H1, H1, "REAL-PROTOCOL-003", "symbol"),               # wrong protocol
        ("BTC/USD", Timeframe.M5, H1, "REAL-PROTOCOL-001", "timeframe"),  # unsupported timeframe
    ])
    def test_L_M_N_mismatches_rejected(self, symbol, tf, strategy_tf, pid, needle):
        assert any(needle in r for r in identity_mismatches(REGISTRY[pid], symbol, tf, trend(strategy_tf)))

    def test_O_wrong_fingerprint_rejected(self):
        p = REGISTRY["REAL-PROTOCOL-002"]
        assert any("deployment" in r for r in identity_mismatches(p, "BTC/USD", H4, trend(H4, FAST)))
        lineage = trend(H4, parent_hash="f" * 64)
        assert any("fingerprint" in r for r in identity_mismatches(p, "BTC/USD", H4, lineage))

    def test_T_protocol_identity_is_unchanged_by_making_it_executable(self):
        for pid, prefix in PHASE_4A_HASHES.items():
            assert REGISTRY[pid].protocol_hash.startswith(prefix), pid
            assert replace(REGISTRY[pid], executable=False).protocol_hash == REGISTRY[pid].protocol_hash


# --- champion isolation P–R ------------------------------------------------------------------------------------------------
class TestChampions:
    def test_P_Q_champions_are_isolated_by_dimension(self, tmp_path):
        _, s, _ = plane(tmp_path / "st")
        d = install_champion(s)                                                  # BTC/USD 1h only
        assert s.strategies.champion("BTC/USD", H1) is d
        for sym, tf in (("BTC/USD", H4), ("ETH/USD", H1), ("ETH/USD", H4)):
            assert s.strategies.champion(sym, tf) is None, (sym, tf)
        assert s.strategies.state(d.registry_key, "ETH/USD") is Lifecycle.CANDIDATE
        e = install_champion(s, symbol="ETH/USD", timeframe=H4)
        assert s.strategies.champion("ETH/USD", H4) is e and s.strategies.champion("BTC/USD", H1) is d
        assert e.registry_key == "trend@v1/4h"

    def test_Q_a_record_for_one_dimension_cannot_promote_in_another(self, tmp_path):
        from tests.test_phase2_learning import gate_inputs
        _, s, _ = plane(tmp_path / "st")
        d, wf, adv, hold = gate_inputs(s)                                         # CHALLENGER for BTC/USD
        from ati.validation.promotion import decide_promotion
        rec = decide_promotion(d, None, wf, adv, hold, None, PromotionPolicy(allow_mock_evidence=True), s.clock.now(),
                               s.research_journal, symbol="ETH/USD")
        with pytest.raises(PromotionDenied, match="not CHALLENGER"):
            s.strategies.apply_promotion(rec)                                     # ETH scope has no such challenger
        unscoped = decide_promotion(d, None, wf, adv, hold, None, PromotionPolicy(allow_mock_evidence=True),
                                    s.clock.now(), s.research_journal)
        with pytest.raises(PromotionDenied, match="no research dimension"):
            s.strategies.apply_promotion(unscoped)
        tampered = replace(rec, symbol="BTC/USD")                                 # the dimension is integrity-hashed
        with pytest.raises(PromotionDenied, match="intact"):
            s.strategies.apply_promotion(tampered)
        assert s.strategies.champion("BTC/USD", H1) is None and s.strategies.champion("ETH/USD", H1) is None

    def test_R_cross_timeframe_champion_comparison_is_refused(self, tmp_path):
        from tests.test_phase2_learning import gate_inputs
        from ati.validation.promotion import decide_promotion
        _, s, _ = plane(tmp_path / "st")
        d, wf, adv, hold = gate_inputs(s)
        other = trend(H4)
        rec = decide_promotion(d, other, wf, adv, hold, None, PromotionPolicy(allow_mock_evidence=True), s.clock.now(),
                               s.research_journal, symbol="BTC/USD")
        assert not rec.approved and any("different timeframes" in r for r in rec.reasons)


# --- end-to-end routed research: champion scope and provenance (S, AE) ------------------------------------------------------
class TestRoutedResearch:
    def test_ETH_1h_research_is_isolated_from_the_BTC_1h_champion(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS, script={"company": reply(
            "RESEARCH_REQUEST", designed("H-4b-eth") | {"protocol_id": "REAL-PROTOCOL-003"})})
        btc_champion = install_champion(s)
        s.store.ingest(c for c in s.provider.fetch_candles("ETH/USD", H1, T0, T0 + timedelta(hours=3200)) if c.is_closed)
        out = cp.run_cycle()
        assert out.detail["research_status"] == "COMPLETED", out.detail
        run = decode(next(s.research_journal.entries("protocol_run")).payload)
        assert (run["symbol"], run["timeframe"], run["registry_key"]) == ("ETH/USD", "1h", "trend@v1/1h")   # S
        assert run["strategy_fingerprint"] == trend(H1).definition_hash and run["protocol_hash"] == \
            REGISTRY["REAL-PROTOCOL-003"].protocol_hash
        seal = decode(next(s.research_journal.entries("holdout_sealed")).payload)
        assert (seal["development"]["symbol"], seal["development"]["timeframe"]) == ("ETH/USD", "1h")
        assert run["dataset_id"].startswith("ds_")
        promos = [decode(e.payload)["record"] for e in s.research_journal.entries("promotion_decision")]
        eth = [p for p in promos if p["symbol"] == "ETH/USD"]
        assert eth and all(p["champion_key"] is None and p["timeframe"] == "1h" for p in eth)   # BTC champion never compared
        assert s.strategies.champion("BTC/USD", H1) is btc_champion                           # AE: BTC trading unaffected
        lineage = [e.payload for e in s.research_journal.entries("candidate_lineage")]
        if lineage:
            assert lineage[0]["symbol"] == "ETH/USD" and lineage[0]["timeframe"] == "1h"
            assert lineage[0]["protocol_id"] == "REAL-PROTOCOL-003"

    def test_BTC_4h_research_runs_on_the_4h_deployment(self, tmp_path):
        hours = 3200 * 4
        cp, s, clock = plane(tmp_path / "st", hours=hours, policies=MECHANICS, script={"company": reply(
            "RESEARCH_REQUEST", designed("H-4b-4h") | {"protocol_id": "REAL-PROTOCOL-002"})})
        s.store.ingest(c for c in s.provider.fetch_candles("BTC/USD", H4, T0, T0 + timedelta(hours=hours)) if c.is_closed)
        install_champion(s)                                                     # a BTC/USD 1h champion exists
        out = cp.run_cycle()
        assert out.detail["research_status"] == "COMPLETED", out.detail
        assert sorted(k for k in s.strategies.definitions() if k.startswith("trend@v1")) == ["trend@v1/1h", "trend@v1/4h"]
        design = next(e.payload for e in s.research_journal.entries("experiment_design"))
        assert design["baseline"]["strategy_key"] == "trend@v1/4h" and design["protocol"]["protocol_id"] == "REAL-PROTOCOL-002"
        seal = decode(next(s.research_journal.entries("holdout_sealed")).payload)
        assert seal["development"]["timeframe"] == seal["holdout_identity"]["timeframe"] == "4h"
        assert s.strategies.champion("BTC/USD", H1).registry_key == "trend@v1/1h"          # 1h champion untouched
        promos = [decode(e.payload)["record"] for e in s.research_journal.entries("promotion_decision")]
        four_hour = [p for p in promos if p["challenger_key"].endswith("/4h")]
        assert four_hour and all(p["timeframe"] == "4h" and p["champion_key"] is None for p in four_hour)
        assert not any(p["challenger_key"].endswith("/4h") and (p["champion_key"] or "").endswith("/1h") for p in promos)


# --- historical compatibility U–W ---------------------------------------------------------------------------------------------
def legacy_journal(tmp_path, clock):
    """A research journal written in the pre-4B format: keys without timeframe, lifecycle entries without symbol."""
    j = Journal(tmp_path / "legacy.jsonl", kind="research", attrs={"data_status": "MOCK"}, clock=clock)
    d = trend(H1)
    j.append("strategy_registered", {"key": "trend@v1", "definition_hash": d.definition_hash, "definition": d})
    j.append("strategy_lifecycle", {"key": "trend@v1", "from": "-", "to": "CANDIDATE", "reason": "registered"})
    j.append("strategy_lifecycle", {"key": "trend@v1", "from": "CANDIDATE", "to": "CHALLENGER", "reason": "legacy"})
    j.append("strategy_lifecycle", {"key": "trend@v1", "from": "CHALLENGER", "to": "CHAMPION", "reason": "legacy"})
    return j, d


class TestHistorical:
    def test_U_V_W_legacy_records_are_replayed_classified_and_never_guessed(self, tmp_path, clock):
        j, d = legacy_journal(tmp_path, clock)
        before = j.path.read_bytes()
        reg = StrategyRegistry(Journal(j.path, kind="research", attrs={"data_status": "MOCK"}, clock=clock))
        assert reg.get("trend@v1/1h").definition_hash == d.definition_hash   # timeframe from the recorded definition
        legacy = reg.legacy_records
        assert legacy["legacy_keys"] == {"trend@v1": "trend@v1/1h"}
        assert legacy["legacy_unscoped_states"] == {"trend@v1/1h": "CHAMPION"}
        for sym in ("BTC/USD", "ETH/USD"):                                   # the legacy champion is NOT assigned
            assert reg.champion(sym, H1) is None and reg.state("trend@v1/1h", sym) is Lifecycle.CANDIDATE
        assert j.path.read_bytes() == before                                 # U: replay rewrote nothing

    def test_U_new_runs_do_not_mutate_old_research_records(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                             script={"company": reply("RESEARCH_REQUEST", designed("H-4b-u"))})
        with_history(s, 3200)
        cp.run_cycle()
        old = s.research_journal.path.read_bytes()
        cp2, s2, _ = plane(tmp_path / "st", hours=3200, clock=clock, policies=MECHANICS)
        with_history(s2, 3200)
        for e in s2.research_journal.entries("candidate_lineage"):              # read paths never write
            factory.stages(s2, e.payload["fingerprint"])
        s2.strategies.legacy_records
        assert s2.research_journal.path.read_bytes() == old                     # history untouched by the new model


# --- behaviour X–Z --------------------------------------------------------------------------------------------------------
class TestBehaviour:
    def test_X_signals_and_backtests_identical_to_pre_4B_code(self):
        out = {}
        for tf in (H1, H4):
            ds = mock_dataset(600, tf=tf)
            for name, params in (("base", P.BASE_PARAMS), ("fast", FAST)):
                d = trend(tf, params)
                sigs = [[str(x.target), str(x.stop_price)] for x in
                        (d.signal(ds.view_at(c.close_time, d.lookback), False) for c in ds.candles[60:])]
                trades = [[t.decided_at.isoformat(), str(t.net_pnl)] for t in run_backtest(d, ds).trades]
                out[f"{tf.value}/{name}"] = hashlib.sha256(json.dumps([sigs, trades]).encode()).hexdigest()[:24]
        assert out == GOLDEN

    def test_Y_parameters_and_logic_unchanged(self):
        assert P.BASE_PARAMS == {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0}
        assert all(dict(p.base_params) == P.BASE_PARAMS for p in REGISTRY.values())
        assert trend(H1).code_hash == "85e5a1f9245058795615db5a925a51a76652de3515152da9a953e9a814f36179"

    def test_Z_risk_limits_unchanged(self, tmp_path):
        _, s, _ = plane(tmp_path / "st")
        assert s.limits == RiskLimits() and s.execution.mode.value == "PAPER"


# --- safety AA–AE ---------------------------------------------------------------------------------------------------------
class TestSafety:
    def test_AA_AB_provenance_still_authoritative(self):
        assert classify("MOCK", 5000, 5000, 3000, "PASS", "VALIDATION_PASSED")["state"] == "NOT_APPLICABLE_MOCK"
        assert classify("MIXED", 5000, 5000, 3000, "PASS", "VALIDATION_PASSED")["state"] == "BLOCKED"
        assert all(p.permitted_provenance == ("DELAYED", "HISTORICAL", "REAL") for p in REGISTRY.values())

    @pytest.mark.parametrize("over,needle", [
        ({"protocol_id": "REAL-PROTOCOL-004", "strategy_key": "trend@v2"}, "locked strategy"),
        ({"protocol_id": "REAL-PROTOCOL-004", "timeframe": "1h"}, "unrecognized"),
        ({"protocol_id": "REAL-PROTOCOL-004", "symbol": "BTC/USD"}, "unrecognized"),
        ({"protocol_id": "BTC-1h"}, "unknown research protocol"),
    ])
    def test_AD_claude_cannot_bypass_routing(self, tmp_path, over, needle):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST", designed("H-4b-ad") | over)})
        out = cp.run_cycle()
        assert out.status == "FAILED" and needle in out.detail["reason"]

    def test_AD_trade_must_cite_its_own_symbols_champion(self, tmp_path):
        from tests.test_company_control import trade_payload, until_signal
        cp, s, clock = plane(tmp_path / "st")
        install_champion(s)
        until_signal(cp, s, clock)
        s.reasoning.script["company"] = reply("TRADE_PROPOSAL", lambda p: trade_payload(p) | {"strategy_key": "trend@v9"})
        out = cp.run_cycle()
        assert out.status == "FAILED" and "not active for trading" in out.detail["reason"]

    def test_AC_context_has_no_holdout_after_routed_4h_research(self, tmp_path):
        hours = 3200 * 4
        cp, s, clock = plane(tmp_path / "st", hours=hours, policies=MECHANICS, script={"company": reply(
            "RESEARCH_REQUEST", designed("H-4b-ac") | {"protocol_id": "REAL-PROTOCOL-002"})})
        s.store.ingest(c for c in s.provider.fetch_candles("BTC/USD", H4, T0, T0 + timedelta(hours=hours)) if c.is_closed)
        cp.run_cycle()
        seal = decode(next(s.research_journal.entries("holdout_sealed")).payload)
        seen = []
        real = s.reasoning.complete
        s.reasoning.complete = lambda role, rid, prompt: (seen.append(prompt), real(role, rid, prompt))[1]
        s.reasoning.script["company"] = reply("NO_TRADE")
        clock.advance(timedelta(hours=1))
        cp.run_cycle()
        assert seal["holdout_dataset_id"] not in seen[-1] and seal["holdout_commitment"] not in seen[-1]
