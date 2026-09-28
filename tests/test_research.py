from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from ati.core.errors import HoldoutViolation, LifecycleError
from ati.data.dataset import Dataset
from ati.market.models import Timeframe
from ati.research.adversarial import AdversarialPolicy, challenge, required_t
from ati.research.attribution import attribute
from ati.research.backtest import BacktestConfig, run_backtest
from ati.research.bootstrap import block_bootstrap_mean, shuffle_drawdown
from ati.research.costs import CostModel
from ati.research.hypothesis import Criterion, PreRegistration, ResearchLog, Verdict
from ati.research.metrics import compute_metrics
from ati.research.robustness import cost_stress, parameter_perturbation, regime_labels, timing_stress
from ati.research.walkforward import walk_forward
from ati.strategies import library  # noqa: F401
from ati.strategies.base import StrategyDefinition
from tests.helpers import T0, make_candle, mock_dataset

P = {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0}
BPY = Timeframe.H1.bars_per_year


def trend(**over):
    return StrategyDefinition.create("trend", 1, "ma_crossover", P | over, Timeframe.H1, T0)


@pytest.fixture(scope="module")
def ds():
    return mock_dataset(2000)


class TestBacktest:
    def test_deterministic(self, ds):
        a, b = run_backtest(trend(), ds), run_backtest(trend(), ds)
        assert [t.net_pnl for t in a.trades] == [t.net_pnl for t in b.trades]

    def test_fills_never_precede_decisions(self, ds):
        r = run_backtest(trend(), ds)
        assert r.trades
        for t in r.trades:
            assert t.entry_time >= t.decided_at
            if t.exit_decided_at:
                assert t.exit_time >= t.exit_decided_at

    def test_future_data_cannot_change_past_decisions(self, ds):
        """Replace the second half of history with a crash: every trade decided and closed
        before the change point must be identical."""
        cut = 1000
        crash = [make_candle(i, o="100", h="100", l="1", c="1", v="10") for i in range(cut, 2000)]
        altered = Dataset.build(list(ds.candles[:cut]) + [replace(c, provider="mock") for c in crash],
                                data_version="x", realization="x")
        boundary = ds.candles[cut].open_time
        a = [t for t in run_backtest(trend(), ds).trades if t.exit_time < boundary]
        b = [t for t in run_backtest(trend(), altered).trades if t.exit_time < boundary]
        assert [(t.decided_at, t.entry_price, t.exit_price) for t in a] == [(t.decided_at, t.entry_price, t.exit_price) for t in b]

    def test_costs_accounted(self, ds):
        free = run_backtest(trend(), ds, BacktestConfig(costs=CostModel.frictionless()))
        real = run_backtest(trend(), ds)
        assert all(t.fees == 0 and t.spread_slip_cost == 0 for t in free.trades)
        assert all(t.fees > 0 and t.spread_slip_cost > 0 for t in real.trades)
        for t in real.trades:
            assert t.net_pnl == t.gross_pnl - t.spread_slip_cost - t.fees

    def test_buy_fills_above_mid_sell_below(self, ds):
        for t in run_backtest(trend(), ds).trades:
            assert t.entry_price > t.entry_mid and t.exit_price < t.exit_mid

    def test_stop_exit_is_gap_aware(self):
        # bar 60 gaps down through the stop: exit must be at the open, not the stop
        candles = [make_candle(i, o=str(100 + i * 0.1), h=str(100.5 + i * 0.1), l=str(99.5 + i * 0.1), c=str(100.1 + i * 0.1))
                   for i in range(60)]
        candles.append(make_candle(60, o="50", h="51", l="49", c="50"))
        candles += [make_candle(i, o="50", h="51", l="49", c="50") for i in range(61, 70)]
        ds = Dataset.build(candles, data_version="v", realization="r")
        r = run_backtest(StrategyDefinition.create("bh", 1, "buy_and_hold", {"stop_fraction": 0.05}, Timeframe.H1, T0), ds)
        stop_trade = next(t for t in r.trades if t.exit_reason == "stop")
        assert stop_trade.exit_mid == Decimal("50")

    def test_partial_fills_respect_participation(self, ds):
        cfg = BacktestConfig(costs=replace(CostModel(), max_participation=Decimal("0.001")))
        r = run_backtest(trend(), ds, cfg)
        assert r.partial_fills > 0
        for t in r.trades:
            assert t.qty <= t.requested_qty

    def test_risk_sizing_bounds_loss_at_stop(self, ds):
        cfg = BacktestConfig()
        for t in run_backtest(trend(), ds, cfg).trades:
            assert t.initial_risk <= cfg.initial_equity * cfg.risk_fraction * Decimal("1.5")

    def test_refuses_holdout(self, ds):
        _, hold = ds.split(ds.candles[1500].open_time)
        with pytest.raises(HoldoutViolation):
            run_backtest(trend(), hold)

    def test_metrics_report_insufficient_evidence(self):
        small = mock_dataset(120)
        m = compute_metrics(run_backtest(trend(), small), BPY)
        assert m.n_trades < 20 and m.expectancy_r is None and m.t_stat_trade_return is None and not m.sufficient


class TestWalkForwardAndRobustness:
    def test_walk_forward_only_counts_oos(self, ds):
        grid = [P | {"fast": f} for f in (5, 10, 20)]
        wf = walk_forward(trend(), ds, grid, train_bars=600, test_bars=300)
        assert wf.configs_tried == 3 * len(wf.folds)
        for fold in wf.folds:
            assert fold.train_end <= fold.test_end
        for t in wf.oos_trades:
            assert any(f.train_end <= t.decided_at <= f.test_end for f in wf.folds)
        assert len(wf.evidence_hash) == 64

    def test_walk_forward_refuses_holdout(self, ds):
        _, hold = ds.split(ds.candles[1000].open_time)
        with pytest.raises(HoldoutViolation):
            walk_forward(trend(), hold, [P], train_bars=300, test_bars=200)

    def test_probes(self, ds):
        costs = cost_stress(trend(), ds, BacktestConfig())
        assert [p.variant for p in costs] == ["x1", "x2", "x3"]
        assert costs[0].net_pnl > costs[1].net_pnl > costs[2].net_pnl
        assert len(timing_stress(trend(), ds, BacktestConfig())) == 3
        assert len(parameter_perturbation(trend(), ds, BacktestConfig())) >= 6

    def test_regime_labels_are_point_in_time(self, ds):
        a = regime_labels(ds)
        truncated = Dataset.build(ds.candles[:1200], data_version="t", realization="t")
        b = regime_labels(truncated)
        for k, v in b.items():
            assert a[k] == v

    def test_attribution_reports_residual(self, ds):
        att = attribute(trend(), ds, BacktestConfig())
        total = att["signal"] + att["sizing"] + att["execution"] + att["fees"] + att["interaction"]
        assert abs(total - att["net"]) < 1e-6
        assert att["fees"] < 0 and att["execution"] < 0


class TestStatistics:
    def test_bootstrap_is_seeded_and_requires_sample(self):
        vals = [0.01, -0.005, 0.02, 0.0, -0.01, 0.015, 0.005, -0.002, 0.01, 0.003, 0.004]
        assert block_bootstrap_mean(vals, seed=1) == block_bootstrap_mean(vals, seed=1)
        assert block_bootstrap_mean(vals[:5]) is None
        ci = block_bootstrap_mean(vals, seed=1)
        assert ci.lo <= ci.mean <= ci.hi

    def test_shuffle_drawdown(self):
        pn = [100, -50, 30, -80, 20, 60, -40, 10, -10, 25]
        out = shuffle_drawdown(pn, 1000.0, runs=500, seed=3)
        assert 0 <= out["median"] <= out["p95"] <= out["max"]

    def test_bonferroni_threshold_grows_with_trials(self):
        assert required_t(100, 0.05) > required_t(1, 0.05) > 1.9


class TestHypothesisDiscipline:
    def prereg(self, ds, **over):
        base = dict(hypothesis_id="H1", statement="trend persistence in mock data", observation_refs=("obs-1",),
                    strategy_key="trend@v1", strategy_hash=trend().definition_hash, dev_dataset_id=ds.dataset_id,
                    criteria=(Criterion("net_pnl", ">", 0.0), Criterion("expectancy_r", ">", 0.05)), min_trades=20,
                    locked_at=T0)
        return PreRegistration(**(base | over))

    def test_cannot_record_result_without_preregistration(self, journal, ds):
        log = ResearchLog(journal)
        m = compute_metrics(run_backtest(trend(), ds), BPY)
        with pytest.raises(LifecycleError):
            log.record_experiment("H1", "dev", "x", m, ds.dataset_id)

    def test_criteria_cannot_be_redefined_after_results(self, journal, ds):
        log = ResearchLog(journal)
        log.preregister(self.prereg(ds))
        with pytest.raises(LifecycleError):
            log.preregister(self.prereg(ds, criteria=(Criterion("net_pnl", ">", -1e9),)))

    def test_failed_experiments_are_retained(self, journal, ds):
        log = ResearchLog(journal)
        log.preregister(self.prereg(ds, criteria=(Criterion("net_pnl", ">", 1e12),)))
        m = compute_metrics(run_backtest(trend(), ds), BPY)
        assert log.record_experiment("H1", "dev", "e1", m, ds.dataset_id) is Verdict.FAIL
        assert [e.payload["verdict"] for e in journal.entries("experiment")] == ["FAIL"]
        assert log.hypotheses_tested == 1

    def test_insufficient_sample_is_not_a_pass(self, journal):
        small = mock_dataset(150)
        log = ResearchLog(journal)
        log.preregister(self.prereg(small, criteria=(Criterion("net_pnl", ">", -1e12),)))
        m = compute_metrics(run_backtest(trend(), small), BPY)
        assert log.record_experiment("H1", "dev", "e", m, small.dataset_id) is Verdict.INSUFFICIENT_EVIDENCE

    def test_criteria_need_real_metric(self):
        with pytest.raises(ValueError):
            Criterion("vibes", ">", 0)


class TestAdversarial:
    def test_mock_data_can_never_be_market_evidence(self, ds):
        wf = walk_forward(trend(), ds, [P], train_bars=600, test_bars=300)
        report = challenge(trend(), ds, BacktestConfig(), wf, hypotheses_tested=1)
        data_q = report.objections[0]
        assert data_q.verdict is Verdict.INSUFFICIENT_EVIDENCE and "MOCK" in data_q.detail
        assert report.blocking

    def test_losing_strategy_fails_edge_question(self):
        ds = mock_dataset(1500, seed=3)
        wf = walk_forward(trend(), ds, [P], train_bars=500, test_bars=250,
                          config=BacktestConfig(costs=CostModel().scaled(Decimal(20))))
        report = challenge(trend(), ds, BacktestConfig(costs=CostModel().scaled(Decimal(20))), wf, 1,
                           AdversarialPolicy(allow_non_market_data=True))
        edge = next(o for o in report.objections if o.question.startswith("Is there any out-of-sample edge"))
        assert edge.verdict is Verdict.FAIL and report.blocking

    def test_judgement_questions_never_auto_pass(self, ds):
        wf = walk_forward(trend(), ds, [P], train_bars=600, test_bars=300)
        report = challenge(trend(), ds, BacktestConfig(), wf, 1)
        manual = [o for o in report.objections if o.verdict is Verdict.NOT_AUTOMATED]
        assert len(manual) >= 3
