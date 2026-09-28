"""Research Workflow Integrity 1.2 — legitimate retests of previously rejected strategy definitions.

All data here is MOCK (independent generated realizations per hypothesis); nothing is REAL evidence.
Each ``cycle`` builds fresh system objects over the same state directory, i.e. a process restart.
"""

from datetime import timedelta
from pathlib import Path

import pytest

from ati.core.errors import LifecycleError, MemoryIntegrityError, ResearchIntegrityError
from ati.core.time import FixedClock
from ati.data.dataset import Dataset
from ati.ledger.journal import Journal, decode
from ati.market.mock import MockProvider
from ati.market.models import Timeframe
from ati.memory.store import MemoryEntry, MemoryKind, _check_independence
from ati.research import workflow
from ati.research.adversarial import AdversarialPolicy
from ati.research.hypothesis import Criterion, PreRegistration, ResearchLog
from ati.research.workflow import run_research_cycle
from ati.strategies.base import StrategyDefinition
from ati.strategies.registry import Lifecycle
from ati.validation.promotion import PromotionPolicy
from tests.helpers import T0
from tests.rig import FAST, install_champion, make_system

P = {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0}
HOURS = 1000
LENIENT = (Criterion("net_pnl", ">", -1e12),)


def system(state):
    return make_system(state, clock=FixedClock(T0 + timedelta(hours=HOURS)))[0]


def dataset(seed, clock):
    prov = MockProvider(seed, clock, epoch=T0)
    return Dataset.build(prov.fetch_candles("BTC/USD", Timeframe.H1, T0, T0 + timedelta(hours=HOURS)),
                         data_version="v", realization=prov.realization)


def cycle(state, seed, hid, grid=(P,), params=P, s=None, criteria=LENIENT):
    s = s or system(state)
    full = dataset(seed, s.clock)
    base = StrategyDefinition.create("trend", 1, "ma_crossover", params, Timeframe.H1, T0)
    result = run_research_cycle(
        s, full, full.candles[800].open_time, hypothesis_id=hid, statement="trend persists", base=base,
        grid=list(grid), criteria=criteria, train_bars=300, test_bars=200,
        adversarial_policy=AdversarialPolicy(allow_non_market_data=True, min_oos_trades=1),
        promotion_policy=PromotionPolicy(allow_mock_evidence=True))
    return s, result


def entries(s, type_):
    return [decode(e.payload) for e in s.research_journal.entries(type_)]


def history(s, key):
    return [(f, t) for k, f, t, _ in s.strategies.history if k == key]


def trials(s):
    report = entries(s, "adversarial_report")[-1]["report"]
    q = next(o["question"] for o in report["objections"] if o["question"].startswith("How many"))
    return int(q.split("(")[1].split(")")[0])


@pytest.fixture
def rejected(tmp_path):
    """H1 tests strategy S and gets it REJECTED (promotion denied on MOCK evidence)."""
    state = tmp_path / "st"
    s, r1 = cycle(state, 1, "H1")
    assert r1.status == "COMPLETED" and not r1.promotion.approved
    assert s.strategies.state(r1.challenger_key) is Lifecycle.REJECTED
    lines = (state / "research.jsonl").read_bytes().splitlines(keepends=True)
    return state, r1, s.strategies.get(r1.challenger_key).definition_hash, lines


class TestRetest:
    def test_rejected_definition_is_retested_cleanly_after_restart(self, rejected):           # 1, 2, 11
        state, r1, h1_hash, _ = rejected
        s2, r2 = cycle(state, 2, "H2")
        assert r2.status == "COMPLETED" and r2.challenger_key == r1.challenger_key
        assert r2.promotion is not None and r2.memory_entry_id

    def test_fingerprint_identical_when_truly_retesting(self, rejected):                       # 5
        state, r1, h1_hash, _ = rejected
        s2, r2 = cycle(state, 2, "H2")
        assert s2.strategies.get(r2.challenger_key).definition_hash == h1_hash
        assert r2.promotion.challenger_hash == r1.promotion.challenger_hash == h1_hash
        assert len([k for k in s2.strategies.keys() if k.startswith("trend@")]) == 1

    def test_changed_fingerprint_is_a_different_definition(self, rejected):                     # 6, Phase 4
        state, r1, h1_hash, _ = rejected
        s2, r2 = cycle(state, 2, "H2", grid=(P | {"fast": 5},))
        new = s2.strategies.get(r2.challenger_key)
        assert r2.challenger_key == "trend@v2" and new.definition_hash != h1_hash
        assert new.parent_hash == h1_hash and new.param_dict["fast"] == 5
        assert s2.strategies.state(r1.challenger_key) is Lifecycle.REJECTED        # v1 untouched
        assert history(s2, r1.challenger_key) == [("-", "CANDIDATE"), ("CANDIDATE", "CHALLENGER"), ("CHALLENGER", "REJECTED")]

    def test_prior_rejection_stays_and_history_shows_readmission(self, rejected):             # 3, 12
        state, r1, _, _ = rejected
        s2, r2 = cycle(state, 2, "H2")
        key = r1.challenger_key
        assert history(s2, key) == [("-", "CANDIDATE"), ("CANDIDATE", "CHALLENGER"), ("CHALLENGER", "REJECTED"),
                                    ("REJECTED", "CHALLENGER"), ("CHALLENGER", "REJECTED")]
        reasons = [r for k, f, t, r in s2.strategies.history if k == key and f == "REJECTED"]
        assert reasons and "re-admitted" in reasons[0] and "H2" in reasons[0]
        decisions = entries(s2, "promotion_decision")
        assert [d["record"]["record_hash"] for d in decisions][0] == r1.promotion.record_hash
        assert len(decisions) == 2

    def test_old_journal_bytes_unchanged_and_new_records_are_independent(self, rejected):    # 4, 19, 20, 21
        state, r1, _, lines_after_h1 = rejected
        s2, r2 = cycle(state, 2, "H2")
        lines = (state / "research.jsonl").read_bytes().splitlines(keepends=True)
        assert lines[:len(lines_after_h1)] == lines_after_h1          # append-only: H1 bytes identical
        exps = entries(s2, "experiment")
        h1 = [e for e in exps if e["hypothesis_id"].startswith("H1")]
        h2 = [e for e in exps if e["hypothesis_id"].startswith("H2")]
        assert len(h1) == len(h2) == 2
        assert {e["prereg_hash"] for e in h1}.isdisjoint({e["prereg_hash"] for e in h2})
        assert {e["evidence_hash"] for e in h1}.isdisjoint({e["evidence_hash"] for e in h2})
        assert {e["dataset_id"] for e in h1}.isdisjoint({e["dataset_id"] for e in h2})
        log = ResearchLog(s2.research_journal)
        assert log.status("H1") == "TESTED:PASS" and log.get("H1").criteria == LENIENT


class TestMultipleTesting:
    def test_retest_is_not_free(self, rejected):                                               # 7, 9, 10
        state, _, _, _ = rejected
        first = trials(system(state))
        s2, _ = cycle(state, 2, "H2")
        log = ResearchLog(s2.research_journal)
        assert log.hypotheses_tested == 2                  # H1(+H1:holdout) and H2(+H2:holdout)
        assert trials(s2) == 2 * first
        assert ResearchLog(system(state).research_journal).hypotheses_tested == 2

    def test_stages_count_once(self, rejected):                                                  # 8
        state, _, _, _ = rejected
        log = ResearchLog(system(state).research_journal)
        assert {e["hypothesis_id"] for e in log.experiments} == {"H1", "H1:holdout"}
        assert log.hypotheses_tested == 1


class TestLocksAndExistingProtections:
    def test_new_hypothesis_criteria_locked_after_restart(self, rejected):                      # 13, 14
        state, _, _, _ = rejected
        cycle(state, 2, "H2")
        log = ResearchLog(system(state).research_journal)
        locked = log.get("H2")
        with pytest.raises(LifecycleError):
            log.preregister(PreRegistration("H2", locked.statement, (), locked.strategy_key, locked.strategy_hash,
                                            locked.dev_dataset_id, (Criterion("net_pnl", ">", -1e15),),
                                            locked.min_trades, locked.locked_at))

    def test_same_hypothesis_cannot_run_twice(self, rejected):                                    # 26
        state, _, _, _ = rejected
        s, again = cycle(state, 3, "H1")
        assert again.status == "NOT_RUN" and any("exactly once" in r for r in again.reasons)
        assert len([e for e in entries(s, "experiment") if e["hypothesis_id"].startswith("H1")]) == 2

    def test_same_holdout_period_cannot_be_reused_by_a_retest(self, rejected):
        state, _, _, _ = rejected
        _, r = cycle(state, 1, "H2")                        # same realization → overlaps H1's sealed holdout
        assert r.status == "NOT_RUN" and any("sealed holdout" in x for x in r.reasons)

    def test_conflicting_and_tampered_preregistrations_still_fail_closed(self, rejected):      # 27, 28
        state, _, _, _ = rejected
        j = Journal(state / "research.jsonl", kind="research", attrs={"mode": "PAPER", "data_status": "MOCK"},
                    clock=FixedClock(T0))
        p = PreRegistration("H1", "other", (), "k", "h", "ds", LENIENT, 1, T0)
        j.append("preregistration", {"prereg": p, "prereg_hash": p.prereg_hash})
        with pytest.raises(ResearchIntegrityError):
            ResearchLog(j)


class TestCrashRecovery:
    def test_crash_after_preregistration_keeps_lock_and_fails_closed_on_rerun(self, rejected, monkeypatch):  # 15
        state, _, _, _ = rejected
        def boom(*a, **k):
            raise RuntimeError("crash after preregistration")
        monkeypatch.setattr(workflow, "walk_forward", boom)
        with pytest.raises(RuntimeError):
            cycle(state, 2, "H2")
        monkeypatch.undo()
        log = ResearchLog(system(state).research_journal)
        assert log.status("H2") == "PREREGISTERED" and not log.was_tested("H2")
        _, again = cycle(state, 2, "H2")                     # its holdout period was already sealed
        assert again.status == "NOT_RUN"
        assert len([e for e in entries(system(state), "preregistration") if e["prereg"]["hypothesis_id"] == "H2"]) == 1
        _, fresh = cycle(state, 3, "H3")                     # research continues on independent data
        assert fresh.status == "COMPLETED"

    def test_crash_after_experiment_before_finalization(self, rejected, monkeypatch):              # 16, 17, 18
        state, r1, _, _ = rejected
        def boom(*a, **k):
            raise RuntimeError("crash before lifecycle finalization")
        monkeypatch.setattr(workflow, "decide_promotion", boom)
        with pytest.raises(RuntimeError):
            cycle(state, 2, "H2")
        monkeypatch.undo()
        s = system(state)
        assert s.strategies.state(r1.challenger_key) is Lifecycle.CHALLENGER   # re-admitted, never finalized
        log = ResearchLog(s.research_journal)
        assert log.status("H2") == "TESTED:PASS" and log.status("H2:holdout").startswith("TESTED:")
        n = len(log.experiments)
        _, rerun = cycle(state, 4, "H2")
        assert rerun.status == "NOT_RUN" and len(ResearchLog(system(state).research_journal).experiments) == n
        s3, r3 = cycle(state, 3, "H3")                       # a new hypothesis finalizes from CHALLENGER
        assert r3.status == "COMPLETED" and s3.strategies.state(r1.challenger_key) is Lifecycle.REJECTED
        assert ResearchLog(s3.research_journal).hypotheses_tested == 3


class TestIneligibleStates:
    def test_champion_retest_stops_before_holdout_without_lifecycle_change(self, tmp_path):
        state = tmp_path / "st"
        s = system(state)
        champ = install_champion(s, FAST)        # TEST-ONLY promotion through the real gate (see rig)
        holdout_before = len(entries(s, "holdout_access"))
        _, r = cycle(state, 2, "H-champ", grid=(FAST,), params=FAST, s=s)
        assert r.status == "COMPLETED" and r.holdout_verdict is None and r.promotion is None
        assert "CHAMPION" in r.reasons[0] and "holdout not used" in r.reasons[0]
        assert len(entries(s, "holdout_access")) == holdout_before
        assert s.strategies.champion().key == champ.key


class TestMemoryUnaffected:
    def test_retest_memory_is_new_non_doctrinal_and_leaves_history(self, rejected):             # 22, 24, 25
        state, r1, _, _ = rejected
        before = system(state).memory.get(r1.memory_entry_id)
        s2, r2 = cycle(state, 2, "H2")
        assert s2.memory.get(r1.memory_entry_id) == before
        new = s2.memory.get(r2.memory_entry_id)
        assert new.kind is MemoryKind.REJECTED_HYPOTHESIS and new.supersedes is None
        assert r2.memory_entry_id != r1.memory_entry_id
        assert not s2.memory.query(s2.clock.now(), MemoryKind.VALIDATED_FINDING)

    def test_retest_evidence_cannot_become_mock_doctrine_or_skip_independence(self, rejected):  # 22, 23, 29, 30
        from dataclasses import replace
        state, _, _, _ = rejected
        s2, _ = cycle(state, 2, "H2")
        wf, hold = (s2.evidence.resolve(e["evidence_hash"]) for e in entries(s2, "experiment")
                    if e["hypothesis_id"].startswith("H2"))
        adv = s2.evidence.resolve(entries(s2, "adversarial_report")[-1]["evidence_hash"])
        assert (wf.kind, hold.kind, adv.kind) == ("walk_forward", "holdout", "adversarial")
        with pytest.raises(MemoryIntegrityError, match="doctrine"):          # MOCK category gate
            s2.memory.add(MemoryEntry(MemoryKind.VALIDATED_FINDING, "trend edge", s2.clock.now(), "x",
                                      (wf, adv, hold), 0.9))
        _check_independence([wf, adv, hold])                                  # retest's own chain is independent
        with pytest.raises(MemoryIntegrityError, match="not independent"):
            _check_independence([wf, adv, replace(hold, dataset_id=wf.dataset_id)])
        with pytest.raises(MemoryIntegrityError, match="canonical"):          # forged metadata
            s2.memory.add(MemoryEntry(MemoryKind.REJECTED_HYPOTHESIS, "forged", s2.clock.now(), "x",
                                      (replace(wf, data_status="REAL"),), 0.5))
        with pytest.raises(MemoryIntegrityError, match="duplicate"):
            s2.memory.add(MemoryEntry(MemoryKind.REJECTED_HYPOTHESIS, "dup", s2.clock.now(), "x", (wf, wf), 0.5))
