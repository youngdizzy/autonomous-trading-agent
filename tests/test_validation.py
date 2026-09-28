from dataclasses import replace
from datetime import timedelta

import pytest

from ati.core.errors import HoldoutViolation, JournalWriteError, PromotionDenied
from ati.data.dataset import Partition
from ati.market.models import Timeframe
from ati.research.adversarial import AdversarialReport, Objection
from ati.research.backtest import run_backtest
from ati.research.hypothesis import Criterion, PreRegistration, Verdict
from ati.research.walkforward import walk_forward
from ati.strategies import library  # noqa: F401
from ati.strategies.base import StrategyDefinition
from ati.strategies.registry import Lifecycle, StrategyRegistry
from ati.validation.holdout import HoldoutEvaluation, HoldoutVault
from ati.validation.promotion import PromotionPolicy, decide_promotion
from tests.helpers import T0, mock_dataset

P = {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0}


def trend(sid="trend", version=1, **over):
    return StrategyDefinition.create(sid, version, "ma_crossover", P | over, Timeframe.H1, T0)


@pytest.fixture
def full():
    return mock_dataset(2000)


@pytest.fixture
def vault(full, journal):
    return HoldoutVault(full, full.candles[1500].open_time, journal, max_evaluations=2)


def prereg(vault, strategy, hid="H1"):
    return PreRegistration(hid, "s", (), strategy.key, strategy.definition_hash, vault.development.dataset_id,
                           (Criterion("net_pnl", ">", 0.0),), 5, T0)


class TestHoldoutStructure:
    def test_development_contains_no_holdout_bars(self, vault, full):
        boundary = full.candles[1500].open_time
        assert vault.development.partition is Partition.DEVELOPMENT
        assert all(c.close_time <= boundary for c in vault.development.candles)
        assert vault.holdout_identity.start >= boundary

    def test_public_surface_exposes_no_holdout_candles(self, vault):
        public = [a for a in dir(vault) if not a.startswith("_")]
        assert set(public) == {"development", "holdout_identity", "burned", "evaluations_used", "evaluate"}
        assert not hasattr(vault.holdout_identity, "candles")

    def test_seal_is_journaled_with_commitment(self, vault, journal):
        seal = next(journal.entries("holdout_sealed"))
        assert seal.payload["holdout_commitment"] == vault.holdout_identity.content_sha256

    def test_requires_locked_preregistration_for_exact_strategy(self, vault):
        s = trend()
        with pytest.raises(HoldoutViolation):
            vault.evaluate(s, prereg(vault, trend(fast=11)))

    def test_one_evaluation_per_lineage(self, vault):
        s = trend()
        ev = vault.evaluate(s, prereg(vault, s))
        assert isinstance(ev, HoldoutEvaluation)
        tweaked = trend(version=2, fast=12)
        with pytest.raises(HoldoutViolation):
            vault.evaluate(tweaked, prereg(vault, tweaked, "H2"))

    def test_budget_burns_holdout(self, vault):
        for sid in ("a", "b"):
            s = trend(sid)
            vault.evaluate(s, prereg(vault, s, sid))
        assert vault.burned
        s = trend("c")
        with pytest.raises(HoldoutViolation):
            vault.evaluate(s, prereg(vault, s, "c"))

    def test_access_logged_before_computation(self, vault, journal, monkeypatch):
        s = trend()
        def fail(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(journal, "_write_line", fail)
        with pytest.raises(JournalWriteError):
            vault.evaluate(s, prereg(vault, s))
        assert vault.evaluations_used == 0

    def test_holdout_only_trades_are_counted(self, vault, full):
        s = trend()
        ev = vault.evaluate(s, prereg(vault, s))
        assert ev.metrics.n_trades >= 0 and ev.evaluation_number == 1

    def test_optimizers_refuse_holdout_partition(self, full):
        _, hold = full.split(full.candles[1500].open_time)
        with pytest.raises(HoldoutViolation):
            walk_forward(trend(), hold, [P], train_bars=200, test_bars=100)
        with pytest.raises(HoldoutViolation):
            run_backtest(trend(), hold)


def _evidence(challenger, full, verdict=Verdict.PASS, status="REAL", holdout_verdict=Verdict.PASS):
    wf = walk_forward(challenger, full, [challenger.param_dict], train_bars=600, test_bars=300)
    adv = AdversarialReport(challenger.key, challenger.definition_hash, full.dataset_id, status,
                            (Objection("q", verdict, "d"),))
    m = replace(wf.oos_metrics, n_trades=100)
    wf = replace(wf, oos_metrics=m, positive_fold_fraction=1.0)
    hold = HoldoutEvaluation(challenger.key, challenger.definition_hash, "ds_h", "p", replace(m, n_trades=50),
                             holdout_verdict, (), 1)
    return wf, adv, hold


class TestPromotion:
    def test_approved_path_and_registry_application(self, full, journal):
        reg = StrategyRegistry(journal)
        c = reg.register(trend())
        reg.transition(c.key, Lifecycle.CHALLENGER, "ready")
        wf, adv, hold = _evidence(c, full)
        rec = decide_promotion(c, None, wf, adv, hold, None, PromotionPolicy(), T0, journal)
        assert rec.approved, rec.reasons
        reg.apply_promotion(rec)
        assert reg.champion().key == c.key

    def test_denials_are_recorded_not_hidden(self, full, journal):
        c = trend()
        wf, adv, hold = _evidence(c, full, verdict=Verdict.FAIL)
        rec = decide_promotion(c, None, wf, adv, hold, None, PromotionPolicy(), T0, journal)
        assert not rec.approved
        recorded = [e.payload["record"] for e in journal.entries("promotion_decision")]
        assert recorded and recorded[-1]["approved"] is False

    @pytest.mark.parametrize("kwargs,needle", [
        (dict(verdict=Verdict.INSUFFICIENT_EVIDENCE), "adversarial"),
        (dict(holdout_verdict=Verdict.FAIL), "holdout verdict"),
        (dict(status="MOCK"), "not market data"),
    ])
    def test_denial_reasons(self, full, journal, kwargs, needle):
        c = trend()
        wf, adv, hold = _evidence(c, full, **kwargs)
        rec = decide_promotion(c, None, wf, adv, hold, None, PromotionPolicy(), T0, journal)
        assert not rec.approved and any(needle in r for r in rec.reasons)

    def test_evidence_for_other_strategy_rejected(self, full, journal):
        c, other = trend(), trend(fast=11)
        wf, adv, hold = _evidence(other, full)
        rec = decide_promotion(c, None, wf, adv, hold, None, PromotionPolicy(), T0, journal)
        assert not rec.approved

    def test_denied_record_cannot_be_applied(self, full, journal):
        reg = StrategyRegistry()
        c = reg.register(trend())
        reg.transition(c.key, Lifecycle.CHALLENGER, "x")
        wf, adv, hold = _evidence(c, full, verdict=Verdict.FAIL)
        rec = decide_promotion(c, None, wf, adv, hold, None, PromotionPolicy(), T0, journal)
        with pytest.raises(PromotionDenied):
            reg.apply_promotion(rec)

    def test_tampered_record_cannot_be_applied(self, full, journal):
        reg = StrategyRegistry()
        c = reg.register(trend())
        reg.transition(c.key, Lifecycle.CHALLENGER, "x")
        wf, adv, hold = _evidence(c, full, verdict=Verdict.FAIL)
        rec = decide_promotion(c, None, wf, adv, hold, None, PromotionPolicy(), T0, journal)
        forged = replace(rec, approved=True)
        with pytest.raises(PromotionDenied):
            reg.apply_promotion(forged)

    def test_challenger_must_beat_champion_like_for_like(self, full, journal):
        reg = StrategyRegistry()
        champ = reg.register(trend("champ"))
        reg.transition(champ.key, Lifecycle.CHALLENGER, "x")
        wf0, adv0, h0 = _evidence(champ, full)
        reg.apply_promotion(decide_promotion(champ, None, wf0, adv0, h0, None, PromotionPolicy(), T0, journal))
        chal = reg.register(trend("chal"))
        reg.transition(chal.key, Lifecycle.CHALLENGER, "x")
        wf, adv, hold = _evidence(chal, full)
        champ_wf = replace(wf0, oos_metrics=replace(wf0.oos_metrics, net_pnl=wf.oos_metrics.net_pnl + 1))
        rec = decide_promotion(chal, champ, wf, adv, hold, champ_wf, PromotionPolicy(), T0, journal)
        assert not rec.approved and any("beat champion" in r for r in rec.reasons)
        assert reg.champion().key == champ.key
