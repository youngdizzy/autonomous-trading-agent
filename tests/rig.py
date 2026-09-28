"""Full-system test rig. Everything here is MOCK: generated prices and scripted reasoning."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta

from ati.agent.loop import AutonomousLoop
from ati.agent.reasoning import Budget, ScriptedReasoningClient
from ati.cli import MOCK_SCRIPT
from ati.core.time import FixedClock
from ati.data.dataset import Dataset
from ati.market.mock import MockProvider
from ati.market.models import DataStatus, Timeframe
from ati.research.adversarial import AdversarialReport, Objection
from ati.research.hypothesis import Verdict
from ati.research.walkforward import walk_forward
from ati.strategies.base import StrategyDefinition
from ati.strategies.registry import Lifecycle
from ati.system import build_paper_system
from ati.validation.holdout import HoldoutEvaluation
from ati.validation.promotion import PromotionPolicy, decide_promotion
from tests.helpers import T0

FAST = {"fast": 5, "slow": 20, "atr_period": 14, "stop_atr": 2.0}


def make_system(state_dir, *, script=None, seed=11, start_hours=400, provider=None, clock=None):
    clock = clock or FixedClock(T0 + timedelta(hours=start_hours))
    provider = provider or MockProvider(seed, clock, epoch=T0)
    reasoning = ScriptedReasoningClient(script or MOCK_SCRIPT, Budget(10))
    return build_paper_system(state_dir, clock, provider, reasoning, data_status=DataStatus.MOCK), clock


def install_champion(system, params=FAST):
    """TEST-ONLY: promotes through the real gate using stub evidence objects so the paper path can
    be exercised. The gate itself is unchanged; its decision is journaled like any other."""
    s = system
    d = s.strategies.register(StrategyDefinition.create("trend", 1, "ma_crossover", params, Timeframe.H1, s.clock.now()))
    s.strategies.transition(d.key, Lifecycle.CHALLENGER, "test rig")
    provider = MockProvider(99, FixedClock(T0 + timedelta(hours=1200)), epoch=T0)
    ds = Dataset.build(provider.fetch_candles("BTC/USD", Timeframe.H1, T0, T0 + timedelta(hours=1200)),
                       data_version="rig", realization="rig")
    wf = walk_forward(d, ds, [d.param_dict], train_bars=400, test_bars=400)
    wf = replace(wf, oos_metrics=replace(wf.oos_metrics, n_trades=100), positive_fold_fraction=1.0)
    adv = AdversarialReport(d.key, d.definition_hash, ds.dataset_id, "MOCK", (Objection("stub", Verdict.PASS, "rig"),))
    hold = HoldoutEvaluation(d.key, d.definition_hash, "ds_rig", "p", replace(wf.oos_metrics, n_trades=50), Verdict.PASS, (), 1)
    record = decide_promotion(d, None, wf, adv, hold, None, PromotionPolicy(allow_mock_evidence=True), s.clock.now(),
                              s.research_journal)
    assert record.approved, record.reasons
    s.strategies.apply_promotion(record)
    return d


def run_until(loop, clock, predicate, max_ticks=300):
    for _ in range(max_ticks):
        clock.advance(timedelta(hours=1))
        report = loop.tick()
        if predicate(report):
            return report
    raise AssertionError("condition not reached")


def entered(report):
    return any("EXECUTE" in a for a in report.actions)


def script_with(primary):
    return MOCK_SCRIPT | {"primary_decision": primary}


def packet(prompt):
    return json.loads(prompt.split("\n\nPACKET:\n\n", 1)[1].split("\n\n<<<UNTRUSTED_DATA", 1)[0])
