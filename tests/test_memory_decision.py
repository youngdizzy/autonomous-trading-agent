from datetime import timedelta
from decimal import Decimal

import pytest

from ati.core.errors import LifecycleError, LookaheadError
from ati.core.time import FixedClock
from ati.decision.records import DecisionLog, DecisionRecord, FinalDecision, make_decision_id
from ati.ledger.journal import Journal
from ati.memory.evidence import EvidenceRef, EvidenceRegistry
from ati.core.errors import MemoryIntegrityError
from ati.memory.store import MemoryEntry, MemoryKind, MemoryStore, _check_independence, _evidence_ok
from tests.helpers import T0

T1 = T0 + timedelta(days=1)


@pytest.fixture
def clock():
    return FixedClock(T1)


@pytest.fixture
def reg(tmp_path, clock):
    return EvidenceRegistry(Journal(tmp_path / "ev.jsonl", kind="evidence", attrs={}, clock=clock))


@pytest.fixture
def mem(tmp_path, clock, reg):
    return MemoryStore(Journal(tmp_path / "mem.jsonl", kind="memory", attrs={}, clock=clock), reg)


def entry(kind, refs=(), conf=0.3, at=T1, statement="trend persistence", **kw):
    return MemoryEntry(kind, statement, at, "test", tuple(refs), conf, **kw)


class TestMemory:
    def test_hypothesis_needs_no_evidence_but_is_capped(self, mem):
        mem.add(entry(MemoryKind.HYPOTHESIS, conf=0.5))
        with pytest.raises(ValueError):
            entry(MemoryKind.HYPOTHESIS, conf=0.9)  # confidence cannot replace evidence

    def test_fabricated_evidence_rejected(self, mem):
        fake = EvidenceRef("experiment", "not-registered", T0)
        with pytest.raises(KeyError):
            mem.add(entry(MemoryKind.FINDING, [fake]))

    def test_evidence_must_match_registration(self, mem, reg):
        reg.register("experiment", "e1", T0)
        with pytest.raises(ValueError):
            mem.add(entry(MemoryKind.FINDING, [EvidenceRef("experiment", "e1", T0 - timedelta(hours=1))]))

    @pytest.mark.parametrize("kind,kinds", [
        (MemoryKind.FINDING, ["decision"]),
        (MemoryKind.VALIDATED_FINDING, ["walk_forward", "adversarial"]),  # no holdout
        (MemoryKind.REJECTED_HYPOTHESIS, ["decision"]),
        (MemoryKind.MISTAKE, ["trade_review"]),  # not recurring
        (MemoryKind.LESSON, ["trade_review"]),
    ])
    def test_evidence_requirements(self, mem, reg, kind, kinds):
        refs = [reg.register(k, f"{k}-{i}", T0) for i, k in enumerate(kinds)]
        with pytest.raises(ValueError):
            mem.add(entry(kind, refs, conf=0.3))

    def test_validated_finding_requires_full_chain(self, mem, reg):
        # 1.1: this store has no data category (attrs={}), so doctrine is refused outright (defect B4).
        refs = [reg.register(k, k, T0) for k in ("walk_forward", "holdout", "adversarial")]
        with pytest.raises(MemoryIntegrityError):
            mem.add(entry(MemoryKind.VALIDATED_FINDING, refs, conf=0.6))
        # The structural chain itself (all three kinds, independent datasets) is satisfied by canonical refs:
        chain = [EvidenceRef("walk_forward", "w", T0, dataset_id="ds_dev"),
                 EvidenceRef("adversarial", "a", T0, dataset_id="ds_dev", strategy="s1"),
                 EvidenceRef("holdout", "h", T0, dataset_id="ds_hold", strategy="s1")]
        assert _evidence_ok(MemoryKind.VALIDATED_FINDING, tuple(chain)) is None
        _check_independence(chain)

    def test_mistake_requires_recurrence(self, mem, reg):
        refs = [reg.register("trade_review", f"r{i}", T0) for i in range(2)]
        assert _evidence_ok(MemoryKind.MISTAKE, tuple(refs)) is None       # two distinct reviews: recurring
        assert _evidence_ok(MemoryKind.MISTAKE, tuple(refs[:1])) is not None
        with pytest.raises(MemoryIntegrityError):                          # 1.1: no doctrine in a category-less store
            mem.add(entry(MemoryKind.MISTAKE, refs, conf=0.5))

    def test_rejected_hypotheses_retained_and_found(self, tmp_path, mem, reg, clock):
        ref = reg.register("experiment", "e-bad", T0)
        eid = mem.add(entry(MemoryKind.REJECTED_HYPOTHESIS, [ref], conf=0.9, statement="RSI mean reversion works",
                            strategy_kind="rsi"))
        assert not hasattr(mem, "delete") and not hasattr(mem, "remove")
        again = MemoryStore(Journal(tmp_path / "mem.jsonl", kind="memory", attrs={}, clock=clock), reg)
        assert again.get(eid).kind is MemoryKind.REJECTED_HYPOTHESIS
        assert again.prior_rejections("works: RSI mean-reversion", "rsi")

    def test_supersession_preserves_history(self, mem, reg):
        first = mem.add(entry(MemoryKind.HYPOTHESIS, conf=0.2, at=T0 + timedelta(hours=1)))
        ref = reg.register("experiment", "e2", T0 + timedelta(hours=2))
        mem.add(entry(MemoryKind.REJECTED_HYPOTHESIS, [ref], conf=0.5, at=T0 + timedelta(hours=3), supersedes=first))
        assert [e.kind for e in mem.query(T1)] == [MemoryKind.REJECTED_HYPOTHESIS]
        assert len(mem.query(T1, include_superseded=True)) == 2
        with pytest.raises(LifecycleError):
            mem.add(entry(MemoryKind.HYPOTHESIS, conf=0.1, statement="other", supersedes=first))

    def test_rejected_hypothesis_needs_oos_evidence_to_revive(self, mem, reg):
        ref = reg.register("experiment", "e3", T0)
        rid = mem.add(entry(MemoryKind.REJECTED_HYPOTHESIS, [ref], conf=0.9))
        with pytest.raises(LifecycleError):
            mem.add(entry(MemoryKind.HYPOTHESIS, conf=0.2, statement="try again", supersedes=rid))

    def test_point_in_time_query(self, mem):
        mem.add(entry(MemoryKind.HYPOTHESIS, at=T0 + timedelta(hours=1), statement="early"))
        mem.add(entry(MemoryKind.HYPOTHESIS, at=T0 + timedelta(hours=5), statement="late"))
        assert [e.statement for e in mem.query(T0 + timedelta(hours=2))] == ["early"]

    def test_memory_cannot_cite_future_evidence(self, reg):
        ref = reg.register("experiment", "future", T1 + timedelta(hours=1))
        with pytest.raises(LookaheadError):
            entry(MemoryKind.FINDING, [ref], conf=0.5, at=T1)


def record(**kw):
    base = dict(
        decision_id=make_decision_id("h", "BTC/USD", T0), timestamp=T1, symbol="BTC/USD", strategy_id="trend",
        strategy_version=1, strategy_hash="h", mode="PAPER", data_status="MOCK", market_context=(("last_price", "1"),),
        available_information_cutoff=T0, signal="LONG", thesis="t", invalidation_condition="i",
        expected_risk=Decimal(10), expected_reward=None, estimated_cost=Decimal(1), proposed_size=None,
        approved_size=Decimal("0.1"), risk_constraints=(), risk_limits_hash="x", evidence_references=(),
        research_references=(), confidence=0.5, reasoning_tier="DEEP", adversarial_objections=(),
        adversarial_verdict="NO_OBJECTION", final_decision=FinalDecision.EXECUTE, reason="ok")
    return DecisionRecord(**(base | kw))


class TestDecisionRecords:
    def test_deterministic_ids(self):
        assert make_decision_id("h", "BTC/USD", T0) == make_decision_id("h", "BTC/USD", T0)
        assert make_decision_id("h", "BTC/USD", T0) != make_decision_id("h", "BTC/USD", T0, "exit:stop")

    def test_immutable(self):
        r = record()
        with pytest.raises(Exception):
            r.thesis = "changed"

    def test_cutoff_and_evidence_timing(self, reg):
        with pytest.raises(LookaheadError):
            record(available_information_cutoff=T1 + timedelta(seconds=1))
        late = reg.register("experiment", "late", T0 + timedelta(hours=1))
        with pytest.raises(LookaheadError):
            record(evidence_references=(late,))

    def test_execute_requires_thesis_and_invalidation(self):
        with pytest.raises(ValueError):
            record(thesis="")
        with pytest.raises(ValueError):
            record(invalidation_condition=" ")
        record(thesis="", invalidation_condition="", final_decision=FinalDecision.NO_TRADE, approved_size=Decimal(0))

    def test_no_chain_of_thought_field_and_length_caps(self):
        assert "reasoning" not in DecisionRecord.__dataclass_fields__
        assert "chain_of_thought" not in DecisionRecord.__dataclass_fields__
        with pytest.raises(ValueError):
            record(thesis="x" * 601)

    def test_log_is_append_only_and_registers_evidence(self, tmp_path, clock, reg):
        log = DecisionLog(Journal(tmp_path / "d.jsonl", kind="decisions", attrs={}, clock=clock), reg)
        r = record()
        log.record(r)
        with pytest.raises(ValueError):
            log.record(r)
        assert r.decision_id in reg
        reopened = DecisionLog(Journal(tmp_path / "d.jsonl", kind="decisions", attrs={}, clock=clock), reg)
        assert r.decision_id in reopened
