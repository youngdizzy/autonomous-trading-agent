"""Research & Memory Integrity 1.1 — regression tests for every reproduced defect.

No REAL evidence is created anywhere in this file. Market-category behaviour is tested through the
pure rule functions on canonical references; stores bound to MOCK / SYNTHETIC / UNKNOWN are tested
end to end.
"""

import json
import subprocess
import sys
import textwrap
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from ati.core.errors import LifecycleError, MemoryIntegrityError, ProvenanceError, ResearchIntegrityError
from ati.core.time import FixedClock
from ati.data.dataset import Dataset
from ati.ledger.journal import Journal
from ati.market.mock import MockProvider
from ati.market.models import Timeframe
from ati.memory.evidence import EvidenceRef, EvidenceRegistry
from ati.memory.store import (MemoryEntry, MemoryKind, MemoryStore, _check_independence,
                              _check_validated_supersession, _evidence_ok)
from ati.research.adversarial import AdversarialPolicy, required_t
from ati.research.hypothesis import Criterion, PreRegistration, ResearchLog, Verdict
from ati.research.metrics import Metrics
from ati.research.workflow import run_research_cycle
from ati.strategies.base import StrategyDefinition
from ati.validation.promotion import PromotionPolicy
from tests.helpers import T0

ROOT = Path(__file__).resolve().parents[1]
M = Metrics(30, 1.0, 0.0, 0.1, 0.1, 0.5, 1.2, 1.0, 0.01, 1.0, 0.5, 1.0, 1.0, 0.1, 0.5, 0)
LOSING = replace(M, net_pnl=-5.0)


def journal(path):
    return Journal(path, kind="research", attrs={"data_status": "MOCK"}, clock=FixedClock(T0))


def prereg(hid, threshold=0.0, statement="s"):
    return PreRegistration(hid, statement, (), "k", "h", "ds", (Criterion("net_pnl", ">", threshold),), 1, T0)


def in_new_process(code: str, path: Path) -> str:
    """Run code in a separate interpreter: a genuine process boundary."""
    header = textwrap.dedent(f"""
        from pathlib import Path
        from tests.test_research_memory_integrity import journal, prereg, M
        from ati.research.hypothesis import ResearchLog
        J = lambda: journal(Path({str(path)!r}))
    """)
    r = subprocess.run([sys.executable, "-c", header + textwrap.dedent(code)], capture_output=True, text=True, cwd=ROOT)
    return (r.stdout + r.stderr).strip()


# --- research ---------------------------------------------------------------------------------------
class TestResearchPersistence:
    def test_preregistration_survives_restart_and_stays_locked(self, tmp_path):             # 1, 2, 9
        path = tmp_path / "r.jsonl"
        ResearchLog(journal(path)).preregister(prereg("H1", 0.0))
        log = ResearchLog(journal(path))
        assert log.get("H1").criteria == (Criterion("net_pnl", ">", 0.0),)
        assert log.status("H1") == "PREREGISTERED"
        log.preregister(prereg("H1", 0.0))  # identical: idempotent, no new entry
        assert len(list(journal(path).entries("preregistration"))) == 1

    @pytest.mark.parametrize("threshold", [-1e9, 1e9])                                       # 3 (weaker and stronger)
    def test_changed_criteria_rejected_across_real_process_boundary(self, tmp_path, threshold):
        path = tmp_path / "r.jsonl"
        assert in_new_process('ResearchLog(J()).preregister(prereg("H1", 0.0)); print("locked")', path) == "locked"
        out = in_new_process(f"""
            try:
                ResearchLog(J()).preregister(prereg("H1", {threshold!r})); print("ACCEPTED")
            except Exception as e:
                print(type(e).__name__)""", path)
        assert out == "LifecycleError"
        assert len(list(journal(path).entries("preregistration"))) == 1

    def test_hypothesis_count_survives_restart(self, tmp_path):                             # 4, 7
        path = tmp_path / "r.jsonl"
        in_new_process("""
            log = ResearchLog(J())
            for h in ("H1", "H2", "H3"):
                log.preregister(prereg(h)); log.record_experiment(h, "dev", "e-" + h, M, "ds")""", path)
        out = in_new_process("""
            log = ResearchLog(J()); before = log.hypotheses_tested
            log.preregister(prereg("H4")); log.record_experiment("H4", "dev", "e4", M, "ds")
            print(before, log.hypotheses_tested, len(log.experiments))""", path)
        assert out == "3 4 4"

    def test_counting_rule_no_double_count_no_untested(self, tmp_path):                      # 5
        log = ResearchLog(journal(tmp_path / "r.jsonl"))
        log.preregister(prereg("H1")); log.record_experiment("H1", "dev", "e1", M, "ds")
        log.record_experiment("H1", "dev", "e1b", M, "ds")                  # same hypothesis again
        log.preregister(prereg("H1:holdout")); log.record_experiment("H1:holdout", "holdout", "e1h", M, "dsh")
        log.preregister(prereg("H9"))                                       # never tested
        assert log.hypotheses_tested == 1
        assert ResearchLog(journal(tmp_path / "r.jsonl")).hypotheses_tested == 1

    def test_failed_experiment_survives_restart(self, tmp_path):                              # 6
        path = tmp_path / "r.jsonl"
        log = ResearchLog(journal(path))
        log.preregister(prereg("H1"))
        assert log.record_experiment("H1", "dev", "e1", LOSING, "ds") is Verdict.FAIL
        again = ResearchLog(journal(path))
        assert again.status("H1") == "TESTED:FAIL" and again.was_tested("H1") and again.hypotheses_tested == 1
        assert [e["verdict"] for e in again.experiments] == ["FAIL"]

    def test_crash_between_preregistration_and_experiment_preserves_lock(self, tmp_path):      # 10
        path = tmp_path / "r.jsonl"
        in_new_process('ResearchLog(J()).preregister(prereg("H1", 0.0))', path)   # process dies here
        out = in_new_process("""
            log = ResearchLog(J())
            try: log.preregister(prereg("H1", -1.0)); print("ACCEPTED")
            except Exception as e: print(type(e).__name__, end=" ")
            print(log.record_experiment("H1", "dev", "e1", M, "ds").value)""", path)
        assert out == "LifecycleError PASS"

    def test_stage_requires_root_with_same_statement(self, tmp_path):
        log = ResearchLog(journal(tmp_path / "r.jsonl"))
        with pytest.raises(LifecycleError):
            log.preregister(prereg("H1:holdout"))
        log.preregister(prereg("H1", statement="a"))
        with pytest.raises(LifecycleError):
            log.preregister(prereg("H1:holdout", statement="b"))

    def test_journal_with_conflicting_preregistrations_fails_closed(self, tmp_path):
        """A journal written by the defective pre-1.1 code (two locks for H1) must not load."""
        j = journal(tmp_path / "r.jsonl")
        for p in (prereg("H1", 0.0), prereg("H1", -1e9)):
            j.append("preregistration", {"prereg": p, "prereg_hash": p.prereg_hash})
        with pytest.raises(ResearchIntegrityError):
            ResearchLog(journal(tmp_path / "r.jsonl"))

    def test_tampered_prereg_hash_fails_closed(self, tmp_path):
        j = journal(tmp_path / "r.jsonl")
        j.append("preregistration", {"prereg": prereg("H1"), "prereg_hash": "0" * 64})
        with pytest.raises(ResearchIntegrityError):
            ResearchLog(journal(tmp_path / "r.jsonl"))


def test_multiple_testing_penalty_uses_persistent_history(tmp_path):                        # 8, Phase 14
    """Two hypotheses on independent MOCK realizations, separated by a restart: the second
    adversarial review must be penalised for both."""
    from tests.rig import make_system

    def cycle(seed, hid):
        s, clock = make_system(tmp_path / "st", clock=FixedClock(T0 + timedelta(hours=1000)))
        provider = MockProvider(seed, clock, epoch=T0)
        full = Dataset.build(provider.fetch_candles("BTC/USD", Timeframe.H1, T0, T0 + timedelta(hours=1000)),
                             data_version="v", realization=provider.realization)
        base = StrategyDefinition.create(f"trend_{hid[-1]}", 1, "ma_crossover",
                                         {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0}, Timeframe.H1, T0)
        result = run_research_cycle(
            s, full, full.candles[800].open_time, hypothesis_id=hid, statement=f"statement {hid}", base=base,
            grid=[base.param_dict], criteria=(Criterion("net_pnl", ">", -1e12),), train_bars=300, test_bars=200,
            adversarial_policy=AdversarialPolicy(allow_non_market_data=True, min_oos_trades=1),
            promotion_policy=PromotionPolicy(allow_mock_evidence=True))
        assert result.status == "COMPLETED", result.reasons
        reports = [e.payload["report"] for e in s.research_journal.entries("adversarial_report")]
        question = next(o["question"] for o in reports[-1]["objections"] if o["question"].startswith("How many"))
        return int(question.split("(")[1].split(")")[0]), s

    first_trials, _ = cycle(1, "H-a")
    second_trials, s = cycle(2, "H-b")          # new System objects = restart
    assert second_trials == 2 * first_trials
    assert ResearchLog(s.research_journal).hypotheses_tested == 2
    assert required_t(second_trials, 0.05) > required_t(first_trials, 0.05)


# --- memory ------------------------------------------------------------------------------------------
def stores(tmp_path, status):
    attrs = {} if status is None else {"mode": "PAPER", "data_status": status}
    clock = FixedClock(T0 + timedelta(days=1))
    reg = EvidenceRegistry(Journal(tmp_path / "ev.jsonl", kind="evidence", attrs=attrs, clock=clock))
    return reg, MemoryStore(Journal(tmp_path / "mem.jsonl", kind="memory", attrs=attrs, clock=clock), reg)


def independent_chain(reg):
    return (reg.register("walk_forward", "wf", T0, dataset_id="ds_dev", strategy="trend"),
            reg.register("adversarial", "adv", T0, dataset_id="ds_dev", strategy="hash_c"),
            reg.register("holdout", "hold", T0, dataset_id="ds_hold", strategy="hash_c"))


def entry(kind, refs=(), conf=0.3, statement="trend persistence", **kw):
    return MemoryEntry(kind, statement, T0 + timedelta(hours=1), "test", tuple(refs), conf, **kw)


class TestMemoryCategoryGate:
    @pytest.mark.parametrize("status", ["MOCK", "SYNTHETIC", "UNKNOWN", None])                  # 11, 12, 13
    def test_non_market_store_refuses_every_doctrinal_kind(self, tmp_path, status):
        reg, mem = stores(tmp_path, status)
        chain = independent_chain(reg)             # otherwise perfect, independent evidence
        reviews = (reg.register("trade_review", "r1", T0), reg.register("trade_review", "r2", T0))
        for kind, refs in ((MemoryKind.VALIDATED_FINDING, chain), (MemoryKind.MISTAKE, reviews),
                           (MemoryKind.FINDING, chain[:1]), (MemoryKind.LESSON, chain[:1])):
            with pytest.raises(MemoryIntegrityError):
                mem.add(entry(kind, refs, conf=0.5))
        assert len(mem) == 0 and not mem.accepts_doctrine
        mem.add(entry(MemoryKind.HYPOTHESIS, conf=0.2))                       # non-doctrinal records remain
        mem.add(entry(MemoryKind.REJECTED_HYPOTHESIS, chain[:1], conf=0.8))
        assert len(mem) == 2

    def test_market_bound_store_accepts_doctrine_category_only_structurally(self, tmp_path):     # 23
        # No evidence is registered: only the category configuration is exercised.
        _, mem = stores(tmp_path, "REAL")
        assert mem.accepts_doctrine

    def test_memory_and_evidence_categories_must_match(self, tmp_path):
        clock = FixedClock(T0)
        reg = EvidenceRegistry(Journal(tmp_path / "ev.jsonl", kind="evidence", attrs={"data_status": "MOCK"}, clock=clock))
        with pytest.raises(ProvenanceError):
            MemoryStore(Journal(tmp_path / "m.jsonl", kind="memory", attrs={"data_status": "REAL"}, clock=clock), reg)


class TestEvidenceRules:
    def test_duplicate_reference_rejected_not_deduplicated(self, tmp_path):                  # 14
        reg, mem = stores(tmp_path, "MOCK")
        r = reg.register("experiment", "trade_review_123", T0)
        with pytest.raises(MemoryIntegrityError, match="duplicate"):
            mem.add(entry(MemoryKind.REJECTED_HYPOTHESIS, (r, r), conf=0.5))
        assert len(mem) == 0

    def test_same_dataset_is_not_independent(self):                                          # 15, 17
        refs = [EvidenceRef("walk_forward", "w", T0, dataset_id="ds_A"),
                EvidenceRef("adversarial", "a", T0, dataset_id="ds_A", strategy="c"),
                EvidenceRef("holdout", "h", T0, dataset_id="ds_A", strategy="c")]
        with pytest.raises(MemoryIntegrityError, match="not independent"):
            _check_independence(refs)

    @pytest.mark.parametrize("missing", ["walk_forward", "adversarial", "holdout"])             # 16
    def test_missing_dataset_identity_fails_closed(self, missing):
        refs = [EvidenceRef("walk_forward", "w", T0, dataset_id="ds_dev"),
                EvidenceRef("adversarial", "a", T0, dataset_id="ds_dev", strategy="c"),
                EvidenceRef("holdout", "h", T0, dataset_id="ds_hold", strategy="c")]
        refs = [replace(r, dataset_id=None) if r.kind == missing else r for r in refs]
        with pytest.raises(MemoryIntegrityError, match="dataset identity"):
            _check_independence(refs)

    def test_holdout_must_exist_and_concern_the_same_strategy(self):                          # 18
        dev = [EvidenceRef("walk_forward", "w", T0, dataset_id="ds_dev"),
               EvidenceRef("adversarial", "a", T0, dataset_id="ds_dev", strategy="c")]
        assert _evidence_ok(MemoryKind.VALIDATED_FINDING, tuple(dev)) is not None
        with pytest.raises(MemoryIntegrityError):
            _check_independence(dev)
        with pytest.raises(MemoryIntegrityError, match="same strategy"):
            _check_independence(dev + [EvidenceRef("holdout", "h", T0, dataset_id="ds_hold", strategy="other")])
        _check_independence(dev + [EvidenceRef("holdout", "h", T0, dataset_id="ds_hold", strategy="c")])

    def test_caller_cannot_forge_category_or_metadata(self, tmp_path):                         # 19
        reg, mem = stores(tmp_path, "MOCK")
        r = reg.register("experiment", "e1", T0, dataset_id="ds_1")
        assert r.data_status == "MOCK"
        for forged in (replace(r, data_status="REAL"), replace(r, dataset_id="ds_other"), replace(r, strategy="x")):
            with pytest.raises(MemoryIntegrityError, match="canonical"):
                mem.add(entry(MemoryKind.REJECTED_HYPOTHESIS, (forged,), conf=0.5))

    def test_registry_journal_claiming_real_inside_mock_fails_closed(self, tmp_path):
        j = Journal(tmp_path / "ev.jsonl", kind="evidence", attrs={"data_status": "MOCK"}, clock=FixedClock(T0))
        j.append("evidence", {"kind": "holdout", "ref_id": "h", "available_at": T0, "data_status": "REAL"})
        with pytest.raises(ProvenanceError):
            EvidenceRegistry(Journal(tmp_path / "ev.jsonl", kind="evidence", attrs={"data_status": "MOCK"}, clock=FixedClock(T0)))

    def test_validated_knowledge_needs_new_testing_evidence_to_be_superseded(self):          # 20
        chain = (EvidenceRef("walk_forward", "w", T0, dataset_id="d"),
                 EvidenceRef("adversarial", "a", T0, dataset_id="d", strategy="c"),
                 EvidenceRef("holdout", "h", T0, dataset_id="dh", strategy="c"))
        old = entry(MemoryKind.VALIDATED_FINDING, chain, conf=0.9)
        for new_refs in ([], list(chain), [EvidenceRef("trade_review", "r", T0)], [EvidenceRef("decision", "d1", T0)]):
            with pytest.raises(LifecycleError):
                _check_validated_supersession(old, new_refs)
        _check_validated_supersession(old, [EvidenceRef("holdout", "h2", T0, dataset_id="dh2", strategy="c")])

    def test_old_doctrine_in_journal_is_quarantined_on_reload(self, tmp_path):
        """Entries written by pre-1.1 code (MOCK validated finding, duplicate refs) are never served."""
        reg, _ = stores(tmp_path, "MOCK")
        chain = independent_chain(reg)
        j = Journal(tmp_path / "mem.jsonl", kind="memory", attrs={"mode": "PAPER", "data_status": "MOCK"},
                    clock=FixedClock(T0 + timedelta(days=1)))
        bad = [entry(MemoryKind.VALIDATED_FINDING, chain, conf=0.9),
               entry(MemoryKind.REJECTED_HYPOTHESIS, (chain[0], chain[0]), conf=0.5, statement="dup")]
        for e in bad:
            j.append("memory", {"entry_id": e.entry_id, "entry": e})
        mem = MemoryStore(Journal(tmp_path / "mem.jsonl", kind="memory", attrs={"mode": "PAPER", "data_status": "MOCK"},
                                  clock=FixedClock(T0)), reg)
        assert len(mem) == 0 and set(mem.quarantined) == {e.entry_id for e in bad}
        assert mem.query(T0 + timedelta(days=2)) == []


def test_memory_integrity_survives_restart(tmp_path):                                          # 22
    reg, mem = stores(tmp_path, "MOCK")
    r1 = reg.register("experiment", "e1", T0, dataset_id="ds_1", strategy="trend")
    h = mem.add(entry(MemoryKind.HYPOTHESIS, conf=0.2, statement="first idea"))
    rid = mem.add(entry(MemoryKind.REJECTED_HYPOTHESIS, (r1,), conf=0.8, supersedes=h))
    reg2, mem2 = stores(tmp_path, "MOCK")
    got = mem2.get(rid)
    assert got == mem.get(rid)
    assert got.evidence[0].data_status == "MOCK" and got.evidence[0].dataset_id == "ds_1" and got.confidence == 0.8
    assert [e.kind for e in mem2.query(T0 + timedelta(days=2))] == [MemoryKind.REJECTED_HYPOTHESIS]
    assert not mem2.quarantined


def test_mock_recurring_mistake_is_kept_but_never_doctrine(tmp_path):                           # 21
    from ati.agent.loop import AutonomousLoop
    from tests.rig import MOCK_SCRIPT, install_champion, make_system, run_until
    script = MOCK_SCRIPT | {"post_trade_reviewer": json.dumps(
        {"process_quality": "BAD_PROCESS", "notes": "n", "possible_mistake": "entered against the higher-timeframe trend"})}
    s, clock = make_system(tmp_path / "st", script=script)
    install_champion(s)
    loop = AutonomousLoop(s)
    exits = []
    run_until(loop, clock, lambda r: exits.extend(a for a in r.actions if "exit" in a) or len(exits) >= 2, max_ticks=600)
    kinds = [e.kind for e in s.memory.query(clock.now())]
    assert MemoryKind.MISTAKE not in kinds
    recurring = [e for e in s.memory.query(clock.now()) if "recurring possible mistake" in e.statement]
    assert recurring and recurring[0].kind is MemoryKind.HYPOTHESIS and "[MOCK, not doctrine]" in recurring[0].statement
    assert len({r.ref_id for r in recurring[0].evidence}) == len(recurring[0].evidence) >= 2
