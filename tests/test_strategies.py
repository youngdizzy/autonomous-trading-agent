from datetime import timedelta

import pytest

from ati.core.errors import LifecycleError, PromotionDenied, StrategyImmutableError
from ati.market.models import Timeframe
from ati.strategies import library  # noqa: F401  (registers logic)
from ati.strategies.base import StrategyDefinition, Target
from ati.strategies.registry import Lifecycle, StrategyRegistry
from tests.helpers import T0, mock_dataset

P = {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0}


def defn(version=1, **over):
    return StrategyDefinition.create("trend", version, "ma_crossover", P | over, Timeframe.H1, T0)


def test_identity_is_content_derived():
    assert defn().definition_hash == defn().definition_hash
    assert defn().definition_hash != defn(fast=11).definition_hash
    assert defn().definition_hash != defn(version=2).definition_hash
    later = StrategyDefinition.create("trend", 1, "ma_crossover", P, Timeframe.H1, T0 + timedelta(days=5), description="x")
    assert later.definition_hash == defn().definition_hash  # creation time / prose are not identity


def test_identity_covers_source_code():
    d = defn()
    assert len(d.code_hash) == 64
    with pytest.raises(ValueError):
        StrategyDefinition("trend", 1, "ma_crossover", tuple(P.items()), Timeframe.H1, T0, code_hash="0" * 64)


def test_definition_is_immutable():
    d = defn()
    with pytest.raises(Exception):
        d.params = ()


@pytest.mark.parametrize("bad", [{"fast": 60}, {"fast": 0}, {"stop_atr": 3}, {"extra": 1}])
def test_invalid_params_rejected(bad):
    with pytest.raises(ValueError):
        StrategyDefinition.create("trend", 1, "ma_crossover", P | bad, Timeframe.H1, T0)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        StrategyDefinition.create("x", 1, "does_not_exist", {}, Timeframe.H1, T0)


def test_registry_refuses_in_place_modification():
    reg = StrategyRegistry()
    reg.register(defn())
    reg.register(defn())  # identical: idempotent
    with pytest.raises(StrategyImmutableError):
        reg.register(defn(fast=12))


def test_derive_creates_new_candidate_version_with_lineage():
    reg = StrategyRegistry()
    parent = reg.register(defn())
    child = reg.derive(parent.registry_key, P | {"fast": 12}, T0, "tweak")
    assert child.version == 2 and child.parent_hash == parent.definition_hash
    assert reg.state(child.registry_key, "BTC/USD") is Lifecycle.CANDIDATE
    assert reg.get(parent.registry_key).param_dict["fast"] == 10


def test_champion_cannot_be_set_without_promotion_record():
    reg = StrategyRegistry()
    d = reg.register(defn())
    reg.transition(d.registry_key, "BTC/USD", Lifecycle.CHALLENGER, "evaluated")
    with pytest.raises(LifecycleError):
        reg.transition(d.registry_key, "BTC/USD", Lifecycle.CHAMPION, "trust me")
    with pytest.raises(PromotionDenied):
        reg.apply_promotion({"approved": True, "challenger_key": d.key})
    assert reg.champion("BTC/USD", Timeframe.H1) is None


def test_candidates_are_isolated_from_champion():
    reg = StrategyRegistry()
    d = reg.register(defn())
    reg.derive(d.registry_key, P | {"slow": 60}, T0, "c")
    assert reg.champion("BTC/USD", Timeframe.H1) is None
    assert set(reg.keys(Lifecycle.CANDIDATE, "BTC/USD")) == {"trend@v1/1h", "trend@v2/1h"}


def test_signal_is_stamped_with_cutoff_and_uses_only_view():
    ds = mock_dataset(200)
    d = defn()
    view = ds.view_at(T0 + timedelta(hours=120), d.lookback)
    sig = d.signal(view, in_position=False)
    assert sig.as_of == view.cutoff
    assert sig.target in (Target.LONG, Target.FLAT)
    if sig.target is Target.LONG:
        assert sig.stop_price < view.latest.close


def test_insufficient_history_is_flat():
    ds = mock_dataset(200)
    view = ds.view_at(T0 + timedelta(hours=10))
    assert defn().signal(view, False).target is Target.FLAT
