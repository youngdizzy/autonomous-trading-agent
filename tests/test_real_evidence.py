"""Phase 3 — REAL-data operationalization & evidence accumulation: test matrix A–AS.

Kraken is unreachable from this environment (proxy 403), so no REAL data exists and none is created. The
positive paths drive the *production* Kraken adapter → payload archive → candle store → dataset → research code
through a Kraken-shaped feed that declares MOCK, so everything they produce is MOCK. REAL appears only in inputs
that must be rejected and in empty, REAL-bound state directories used to prove categories cannot be swapped.
"""

import json
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from ati.agent.reasoning import Budget, ScriptedReasoningClient
from ati.core.errors import (DataIntegrityError, JournalWriteError, MalformedResponse, ModeMismatch,
                             ProviderUnavailable, StateLocked)
from ati.core.lock import StateLock
from ati.core.time import FixedClock, from_epoch
from ati.data.dataset import Dataset
from ati.ledger.journal import Journal, decode
from ati.market import accumulate as acc
from ati.market.archive import CONFLICT_ACK
from ati.market.health import contiguous_runs, research_readiness, series_health, verify_sealed_holdouts
from ati.market.kraken import KrakenPublicOHLC
from ati.market.mock import MockProvider
from ati.market.models import DataStatus, Timeframe
from ati.market.provider import UrllibTransport
from ati.market.store import CandleStore
from ati.market.validation import validate_series
from ati.research import protocol as P
from ati.research.hypothesis import ResearchLog
from ati.research.workflow import research_preconditions
from ati.system import build_paper_system
from tests.helpers import T0, make_candle
from tests.rig import MOCK_SCRIPT, make_system
from tests.test_company_control import MECHANICS, Boom, plane, reply, with_history
from tests.test_self_improvement import designed

H1, H4 = Timeframe.H1, Timeframe.H4
KEYS = {"XBTUSD": ("BTC/USD", "XXBTZUSD"), "ETHUSD": ("ETH/USD", "XETHZUSD")}
TF = {"60": H1, "240": H4}


class MultiSeriesKrakenFeed:
    """MOCK transport answering in Kraken's documented OHLC shape for BTC/ETH × 1h/4h (forming final row,
    ``last`` marker). Declares MOCK: everything the production adapter derives from it is MOCK."""

    data_status = DataStatus.MOCK

    def __init__(self, clock, seed=11):
        self.clock, self.mock = clock, MockProvider(seed, clock, epoch=T0)
        self.override = None          # callable(payload, symbol, tf) -> payload
        self.fail = None              # callable(symbol, tf) -> exception or None
        self.calls = []

    def get_json(self, url, params, timeout_s):
        symbol, key = KEYS[params["pair"]]
        tf = TF[params["interval"]]
        self.calls.append((symbol, tf))
        if self.fail and (exc := self.fail(symbol, tf)):
            raise exc
        since = int(params["since"]) + 1
        start = from_epoch(since - since % tf.seconds + (tf.seconds if since % tf.seconds else 0))
        start = max(start, T0)
        candles = self.mock.fetch_candles(symbol, tf, start, self.clock.now() + tf.delta)
        rows = [[int(c.open_time.timestamp()), str(c.open), str(c.high), str(c.low), str(c.close), str(c.close),
                 str(c.volume), 1] for c in candles]
        payload = {"error": [], "result": {key: rows, "last": rows[-2][0] if len(rows) > 1 else 0}}
        if self.override:
            payload = self.override(payload, symbol, tf)
        raw = json.dumps(payload).encode()
        return json.loads(raw), raw


def kraken(tmp_path, hours=3000, clock=None, feed=None):
    clock = clock or FixedClock(T0 + timedelta(hours=hours))
    feed = feed or MultiSeriesKrakenFeed(clock)
    s, _ = make_system(tmp_path / "st", provider=KrakenPublicOHLC(feed, clock), clock=clock)
    return s, clock, feed


def by(results):
    return {(r.symbol, r.timeframe): r for r in results}


def snapshot(s):
    return {(sym, tf.value): [(c.content_key(), c.provenance, c.received_at, c.status) for c in s.store.series("kraken", sym, tf)]
            for sym, tf in acc.SERIES}


def dataset_ids(s):
    return {(sym, tf.value): Dataset.build(s.store.series("kraken", sym, tf), data_version="v", realization="observed").dataset_id
            for sym, tf in acc.SERIES if s.store.series("kraken", sym, tf)}


# --- Provider A–E ---------------------------------------------------------------------------------------------------
class TestProvider:
    def test_A_provenance_comes_from_the_response_and_transport(self, tmp_path):
        s, clock, feed = kraken(tmp_path)
        acc.accumulate(s)
        payloads = [decode(e.payload) for e in s.evidence.journal.entries("provider_payload")]
        assert len(payloads) == 4 and all(p["status"] == "MOCK" for p in payloads)     # the transport's category
        for sym, tf in acc.SERIES:
            series = s.store.series("kraken", sym, tf)
            assert series and all(c.provider == "kraken" and c.status is DataStatus.MOCK for c in series)
            assert all(c.provenance.raw_sha256 in {p["raw_sha256"] for p in payloads} for c in series)
            assert all(f"interval={60 if tf is H1 else 240}" in c.provenance.method for c in series)
        assert UrllibTransport.data_status is DataStatus.REAL and feed.data_status is DataStatus.MOCK

    def test_B_wrong_pair_rejected(self, tmp_path):
        s, _, feed = kraken(tmp_path)
        feed.override = lambda p, sym, tf: {"error": [], "result": {"XETHZUSD" if sym == "BTC/USD" else "XXBTZUSD":
                                                                    next(v for k, v in p["result"].items() if k != "last"),
                                                                    "last": p["result"]["last"]}}
        r = by(acc.accumulate(s))
        assert all(x.status == "REJECTED" and "pair" in x.detail for x in r.values())
        assert not any(s.store.series("kraken", sym, tf) for sym, tf in acc.SERIES)
        assert len(list(s.evidence.journal.entries("provider_payload_rejected"))) == 4     # archived AND rejected

    def test_C_wrong_timeframe_rejected_even_when_short(self, tmp_path):
        s, clock, feed = kraken(tmp_path)

        def four_hour_rows_for_a_1h_request(p, sym, tf):
            if tf is H1:
                key = next(k for k in p["result"] if k != "last")
                t = [r for r in p["result"][key] if r[0] % 14400 == 0][-3:]
                p["result"][key], p["result"]["last"] = t, t[-2][0]
            return p
        feed.override = four_hour_rows_for_a_1h_request
        r = by(acc.accumulate(s, series=(("BTC/USD", H1),)))
        assert r[("BTC/USD", "1h")].status == "REJECTED" and "not 1h data" in r[("BTC/USD", "1h")].detail

        def hourly_rows_for_a_4h_request(p, sym, tf):
            if tf is H4:
                key = next(k for k in p["result"] if k != "last")
                p["result"][key] = [[r[0] + 3600] + r[1:] for r in p["result"][key]]
            return p
        feed.override = hourly_rows_for_a_4h_request
        r = by(acc.accumulate(s, series=(("BTC/USD", H4),)))
        assert r[("BTC/USD", "4h")].status == "REJECTED" and "ALIGNMENT" in r[("BTC/USD", "4h")].detail

    @pytest.mark.parametrize("mutate,needle", [
        (lambda row: row.__setitem__(2, None), "numeric"),                   # missing field value
        (lambda row: row.pop(), "malformed"),                                  # missing column
        (lambda row: row.__setitem__(4, "NaN"), ""),                           # Y: nonfinite
        (lambda row: row.__setitem__(4, "Infinity"), ""),                      # Y: nonfinite
        (lambda row: row.__setitem__(6, "-1"), "VOLUME"),                      # Z: negative volume
        (lambda row: (row.__setitem__(2, "1"), row.__setitem__(3, "999999")), "OHLC"),   # X: high < low
        (lambda row: row.__setitem__(4, str(Decimal(row[2]) * 2)), "OHLC"),    # X: close outside high/low
    ])
    def test_D_X_Y_Z_malformed_values_rejected(self, tmp_path, mutate, needle):
        s, _, feed = kraken(tmp_path)

        def corrupt(p, sym, tf):
            key = next(k for k in p["result"] if k != "last")
            mutate(p["result"][key][5])
            return p
        feed.override = corrupt
        r = by(acc.accumulate(s, series=(("BTC/USD", H1),)))[("BTC/USD", "1h")]
        assert r.status == "REJECTED" and needle in r.detail and r.added == 0
        assert not s.store.series("kraken", "BTC/USD", H1)

    def test_E_unfinished_candle_never_stored(self, tmp_path):
        s, clock, feed = kraken(tmp_path)
        clock.advance(timedelta(minutes=30))             # the current hour is forming
        acc.accumulate(s, series=(("BTC/USD", H1),))
        series = s.store.series("kraken", "BTC/USD", H1)
        assert series and all(c.is_closed and c.close_time <= clock.now() for c in series)
        assert series[-1].close_time <= clock.now() - timedelta(minutes=30)
        with pytest.raises(DataIntegrityError):
            CandleStore().ingest([make_candle(0, closed=False)])


# --- Persistence F–H ---------------------------------------------------------------------------------------------------
class TestPersistence:
    def test_F_G_H_datasets_hashes_and_provenance_reconstruct_exactly(self, tmp_path):
        s, clock, _ = kraken(tmp_path)
        acc.accumulate(s)
        clock.advance(timedelta(hours=9))
        acc.accumulate(s)
        before, ids = snapshot(s), dataset_ids(s)
        s2, _, _ = kraken(tmp_path, clock=clock)                     # restart: store rebuilt from the archive only
        assert snapshot(s2) == before and dataset_ids(s2) == ids
        for sym, tf in acc.SERIES:
            s2.archive.verify_derivation(Dataset.build(s2.store.series("kraken", sym, tf), data_version="v",
                                                       realization="observed"))
        assert [r["status"] for r in acc.history(s2, "BTC/USD", H1)] == ["ACCUMULATED", "ACCUMULATED"]


# --- Accumulation I–Q ----------------------------------------------------------------------------------------------------
class TestAccumulation:
    def test_I_J_K_first_exact_and_partial_overlap(self, tmp_path):
        s, clock, _ = kraken(tmp_path)
        first = by(acc.accumulate(s))
        assert all(r.status == "ACCUMULATED" and r.added > 0 for r in first.values())
        again = by(acc.accumulate(s))                                            # J: exact overlap
        assert all(r.status == "NO_NEW_DATA" and r.added == 0 for r in again.values())
        assert all(r.candles_after == first[k].candles_after for k, r in again.items())
        old = {k: Dataset.build(s.store.series("kraken", *k2), data_version="v", realization="observed")
               for k, k2 in zip(first, acc.SERIES)}
        clock.advance(timedelta(hours=8))                                        # K: partial overlap
        grown = by(acc.accumulate(s))
        assert grown[("BTC/USD", "1h")].added == 8 and grown[("BTC/USD", "4h")].added == 2
        for k, k2 in zip(first, acc.SERIES):
            new = Dataset.build(s.store.series("kraken", *k2), data_version="v", realization="observed")
            new.verify_extension_of(old[k])                                      # legitimate evolution
            assert new.dataset_id != old[k].dataset_id
            assert len({c.open_time for c in new.candles}) == len(new)            # no duplicates

    def test_L_conflicting_overlap_fails_closed_durably(self, tmp_path):
        s, clock, feed = kraken(tmp_path)
        acc.accumulate(s)
        before = snapshot(s)
        clock.advance(timedelta(hours=1))

        def rewrite_btc_1h(p, sym, tf):
            if (sym, tf) == ("BTC/USD", H1):
                row = p["result"]["XXBTZUSD"][10]
                row[4] = str(Decimal(row[4]) + 1)
                row[2] = str(max(Decimal(row[2]), Decimal(row[4])))
            return p
        feed.override = rewrite_btc_1h
        r = by(acc.accumulate(s))
        assert r[("BTC/USD", "1h")].status == "DATA_CONFLICT" and r[("BTC/USD", "1h")].added == 0
        assert r[("ETH/USD", "1h")].status == "ACCUMULATED"                     # other series unaffected
        assert snapshot(s)[("BTC/USD", "1h")] == before[("BTC/USD", "1h")]      # never overwritten
        h = series_health(s, "BTC/USD", H1)
        assert h["state"] == "BLOCKED" and h["checks"]["conflicts"].startswith("DATA_CONFLICT")
        feed.override = None
        clock.advance(timedelta(hours=1))
        assert by(acc.accumulate(s))[("BTC/USD", "1h")].status == "BLOCKED"      # blocked until acknowledged
        s2, _, _ = kraken(tmp_path, clock=clock)                                 # durable across restart
        assert s2.archive.conflicts() and by(acc.accumulate(s2))[("BTC/USD", "1h")].status == "BLOCKED"
        with pytest.raises(PermissionError):
            s2.archive.acknowledge_conflict(s2.archive.conflicts()[0]["raw_sha256"], "ok")
        s2.archive.acknowledge_conflict(s2.archive.conflicts()[0]["raw_sha256"], CONFLICT_ACK)
        after = by(acc.accumulate(s2))[("BTC/USD", "1h")]
        assert after.status == "ACCUMULATED"
        series = s2.store.series("kraken", "BTC/USD", H1)
        recorded = {k[0]: k for k, *_ in before[("BTC/USD", "1h")]}
        assert all(c.content_key() == recorded[c.open_time] for c in series if c.open_time in recorded)

    def test_M_provider_failure_leaves_evidence_intact(self, tmp_path):
        s, clock, feed = kraken(tmp_path)
        acc.accumulate(s)
        before = snapshot(s)
        feed.fail = lambda sym, tf: ProviderUnavailable("simulated outage")
        clock.advance(timedelta(hours=3))
        r = by(acc.accumulate(s))
        assert all(x.status == "PROVIDER_FAILURE" and x.added == 0 for x in r.values())
        assert snapshot(s) == before
        s.evidence.journal.verify()

    def test_N_O_partial_symbol_and_timeframe_failure_isolated(self, tmp_path):
        s, clock, feed = kraken(tmp_path)
        feed.fail = lambda sym, tf: ProviderUnavailable("eth down") if sym == "ETH/USD" else None
        r = by(acc.accumulate(s))
        assert r[("BTC/USD", "1h")].status == r[("BTC/USD", "4h")].status == "ACCUMULATED"
        assert r[("ETH/USD", "1h")].status == r[("ETH/USD", "4h")].status == "PROVIDER_FAILURE"
        feed.fail = lambda sym, tf: ProviderUnavailable("4h down") if tf is H4 else None
        clock.advance(timedelta(hours=4))
        r = by(acc.accumulate(s))
        assert r[("ETH/USD", "1h")].status == "ACCUMULATED" and r[("BTC/USD", "1h")].added == 4
        assert r[("BTC/USD", "4h")].status == "PROVIDER_FAILURE" and r[("BTC/USD", "4h")].candles_after > 0

    def test_P_concurrent_accumulation_cannot_interleave(self, tmp_path):
        state = tmp_path / "st"
        with StateLock(state):
            with pytest.raises(StateLocked):
                with StateLock(state):
                    pass
        s, clock, _ = kraken(tmp_path)
        other = Journal(s.evidence.journal.path, kind="evidence", attrs=s.evidence.journal.attrs, clock=clock)
        acc.accumulate(s, series=(("BTC/USD", H1),))                             # first writer appends
        with pytest.raises(JournalWriteError):
            other.append("accumulation_run", {"x": 1})                           # stale second writer refused
        Journal(s.evidence.journal.path, kind="evidence", attrs=s.evidence.journal.attrs, clock=clock).verify()

    def test_Q_restart_resumes_from_persisted_evidence(self, tmp_path):
        s, clock, feed = kraken(tmp_path)
        acc.accumulate(s)
        n = {k: len(v) for k, v in snapshot(s).items()}
        clock.advance(timedelta(hours=12))
        s2, _, _ = kraken(tmp_path, clock=clock)
        r = by(acc.accumulate(s2))
        assert r[("BTC/USD", "1h")].candles_before == n[("BTC/USD", "1h")] and r[("BTC/USD", "1h")].added == 12
        assert r[("ETH/USD", "4h")].added == 3


# --- Timeframe R–T ------------------------------------------------------------------------------------------------------
class TestTimeframes:
    def test_R_S_T_series_are_isolated_and_aligned(self, tmp_path):
        s, _, _ = kraken(tmp_path)
        acc.accumulate(s)
        for sym, tf in acc.SERIES:
            series = s.store.series("kraken", sym, tf)
            assert all(c.timeframe is tf and c.symbol == sym for c in series)
            assert all(int(c.open_time.timestamp()) % tf.seconds == 0 for c in series)
        h1, h4 = s.store.series("kraken", "BTC/USD", H1), s.store.series("kraken", "BTC/USD", H4)
        span = lambda xs: xs[-1].close_time - xs[0].open_time                           # noqa: E731
        assert span(h4) == 4 * span(h1)                                                  # same bar count, 4× the time
        assert all(int(c.open_time.timestamp()) % 14400 == 0 for c in h4)
        assert not {id(c) for c in h1} & {id(c) for c in h4}
        with pytest.raises(DataIntegrityError):                                  # a misaligned 4h bar cannot exist
            make_candle(tf=H4, open_time=T0 + timedelta(hours=1))                          # 01:00 is not a 4h boundary


# --- Health U–AB ---------------------------------------------------------------------------------------------------------
class TestHealth:
    def test_U_V_duplicates_and_non_monotonic_detected(self):
        c = [make_candle(i) for i in range(4)]
        with pytest.raises(DataIntegrityError, match="duplicate"):
            validate_series(c + [c[-1]], symbol="BTC/USD", timeframe=H1)
        with pytest.raises(DataIntegrityError, match="after"):
            validate_series([c[1], c[0]], symbol="BTC/USD", timeframe=H1)

    def test_W_gap_detected_reported_and_never_bridged(self, tmp_path):
        s, clock, feed = kraken(tmp_path)

        def drop_three_hours(p, sym, tf):
            if (sym, tf) == ("BTC/USD", H1):
                rows = p["result"]["XXBTZUSD"]
                del rows[100:103]
            return p
        feed.override = drop_three_hours
        acc.accumulate(s, series=(("BTC/USD", H1),))
        h = series_health(s, "BTC/USD", H1)
        assert h["checks"]["gaps"].startswith("UNEXPLAINED_GAP 3") and h["gaps"][0]["missing_intervals"] == 3
        series = s.store.series("kraken", "BTC/USD", H1)
        assert len(contiguous_runs(series)) == 2                                 # windows split, never bridged
        full = Dataset.build(series, data_version="v", realization="observed")
        fails = research_preconditions(s, full, P.base_definition(clock.now()), hypothesis_id="H-gap", min_candles=0,
                                       require_market_data=False)
        assert any(f.startswith("DATA_GAP") for f in fails)

    def test_AA_stale_series_is_not_ready(self, tmp_path):
        s, clock, _ = kraken(tmp_path)
        acc.accumulate(s)
        assert series_health(s, "BTC/USD", H1)["state"] == "PASS"
        clock.advance(timedelta(hours=5))
        h = series_health(s, "BTC/USD", H1)
        assert h["state"] == "NOT_READY" and h["checks"]["freshness"].startswith("INCOMPLETE_CURRENT")

    def test_AB_mixed_provenance_fails_closed(self):
        store = CandleStore()
        store.ingest([make_candle(0)])
        with pytest.raises(DataIntegrityError, match="STATUS_MIX|MOCK"):
            store.ingest([make_candle(1, status=DataStatus.SYNTHETIC)])
        with pytest.raises(DataIntegrityError):
            validate_series([make_candle(0), make_candle(1, status=DataStatus.SYNTHETIC)], symbol="BTC/USD", timeframe=H1)


# --- Holdout AC–AG -------------------------------------------------------------------------------------------------------
def extend(s, clock, frm, to):
    clock.advance(timedelta(hours=to - frm))
    s.store.ingest(c for c in s.provider.fetch_candles("BTC/USD", H1, T0 + timedelta(hours=frm), T0 + timedelta(hours=to))
                   if c.is_closed)


class TestHoldout:
    def test_AC_AD_AE_three_sequential_windows_keep_every_holdout_sealed(self, tmp_path, monkeypatch):
        from ati.company import budget
        monkeypatch.setattr(budget, "POLICY", budget.BudgetPolicy(max_research_runs_per_day=10))
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS, script={"company": reply("NO_TRADE")})
        with_history(s, 3200)
        ends = 3200
        for i, to in enumerate((3200, 6300, 9400)):
            if to != ends:
                extend(s, clock, ends, to)
                ends = to
            s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed(f"H-win-{i}", statement=f"window idea {i}"))
            out = cp.run_cycle()
            assert out.detail.get("research_status") == "COMPLETED", out.detail
        seals = [decode(e.payload) for e in s.research_journal.entries("holdout_sealed")]
        assert len(seals) == 3
        periods = [(x["development"]["start"], x["development"]["end"], x["holdout_identity"]["start"],
                    x["holdout_identity"]["end"]) for x in seals]
        for k in range(1, 3):
            assert periods[k][0] >= periods[k - 1][3]                  # AE: development k starts after holdout k-1
        all_hold = [(p[2], p[3]) for p in periods]
        for k, (ds, de, _, _) in enumerate(periods):                   # no development overlaps any earlier holdout
            assert not any(ds < he and de > hs for hs, he in all_hold[:k])
        assert [h["state"] for h in verify_sealed_holdouts(s)] == ["INTACT"] * 3     # AC: commitments re-verified
        # a holdout is opened only when development passed, and never more than once per seal
        accesses = [e.payload for e in s.research_journal.entries("holdout_access")]
        assert len(accesses) <= 3 and len({a["prereg_hash"] for a in accesses}) == len(accesses)

    def test_AC_mutated_holdout_is_detected(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                             script={"company": reply("RESEARCH_REQUEST", designed("H-mut"))})
        with_history(s, 3200)
        assert cp.run_cycle().detail["research_status"] == "COMPLETED"
        seal = decode(next(s.research_journal.entries("holdout_sealed")).payload)["holdout_identity"]
        s2, _ = make_system(tmp_path / "st2", clock=clock)                        # a store whose history differs
        from dataclasses import replace
        victim = next(c for c in s.store.series("mock", "BTC/USD", H1) if c.open_time >= seal["start"])
        s2.store.ingest(replace(c, volume=c.volume + 1) if c.open_time == victim.open_time else c
                        for c in s.store.series("mock", "BTC/USD", H1))
        fake = SimpleNamespace(research_journal=s.research_journal, store=s2.store)
        assert [h["state"] for h in verify_sealed_holdouts(fake)] == ["MUTATED"]

    def test_AF_AG_holdout_redacted_and_cannot_motivate(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", hours=3200, policies=MECHANICS,
                             script={"company": reply("RESEARCH_REQUEST", designed("H-red"))})
        with_history(s, 3200)
        cp.run_cycle()
        seen = []
        real = s.reasoning.complete
        s.reasoning.complete = lambda role, rid, prompt: (seen.append(prompt), real(role, rid, prompt))[1]
        s.reasoning.script["company"] = reply("NO_TRADE")
        clock.advance(timedelta(hours=1))
        cp.run_cycle()
        seal = decode(next(s.research_journal.entries("holdout_sealed")).payload)
        assert seal["holdout_dataset_id"] not in seen[-1] and "SEALED_HOLDOUT" in seen[-1]
        s.evidence.register("holdout", "hold_ag", clock.now() - timedelta(hours=1), "holdout")
        s.reasoning.script["company"] = reply("RESEARCH_REQUEST", designed("H-ag", evidence_refs=["hold_ag"]))
        clock.advance(timedelta(hours=1))
        assert "cannot motivate" in cp.run_cycle().detail["reason"]


# --- REAL/MOCK AH–AL -------------------------------------------------------------------------------------------------------
def real_bound(state, clock):
    """An empty REAL-bound state directory: the network transport declares REAL; no request is made, no data exists."""
    return build_paper_system(state, clock, KrakenPublicOHLC(UrllibTransport(), clock),
                              ScriptedReasoningClient(MOCK_SCRIPT, Budget(1)), data_status=DataStatus.REAL)


class TestCategories:
    def test_AH_real_plus_mock_is_refused(self, tmp_path):
        clock = FixedClock(T0 + timedelta(hours=10))
        with pytest.raises(ModeMismatch):
            build_paper_system(tmp_path / "a", clock, MockProvider(1, clock, epoch=T0),
                               ScriptedReasoningClient(MOCK_SCRIPT, Budget(1)), data_status=DataStatus.REAL)
        s = real_bound(tmp_path / "b", clock)
        s.provider.transport = MultiSeriesKrakenFeed(clock)                     # a MOCK feed behind a REAL system
        r = by(acc.accumulate(s, series=(("BTC/USD", H1),)))[("BTC/USD", "1h")]
        assert r.status == "REJECTED" and "STATUS_MIX" in r.detail and not s.store.series("kraken", "BTC/USD", H1)

    def test_AI_AJ_AK_category_is_bound_to_the_state_and_survives_restart(self, tmp_path):
        clock = FixedClock(T0 + timedelta(hours=3000))
        mock_state, real_state = tmp_path / "mock", tmp_path / "real"
        s, _, _ = kraken(tmp_path, clock=clock)                                   # MOCK archive with data
        acc.accumulate(s, series=(("BTC/USD", H1),))
        with pytest.raises(ModeMismatch):                                         # AJ: stale MOCK archive cannot reopen REAL
            real_bound(tmp_path / "st", clock)
        real_bound(real_state, clock)                                             # empty REAL-bound state
        with pytest.raises(ModeMismatch):                                         # AK: REAL state cannot reopen as MOCK
            make_system(real_state, clock=clock)
        s2, _, _ = kraken(tmp_path, clock=clock)                                  # AI: restart keeps the category
        assert s2.data_status is DataStatus.MOCK and all(c.status is DataStatus.MOCK
                                                         for c in s2.store.series("kraken", "BTC/USD", H1))
        a = Dataset.build([make_candle(i) for i in range(5)], data_version="v", realization="r")
        b = Dataset.build([make_candle(i, status=DataStatus.SYNTHETIC) for i in range(5)], data_version="v", realization="r")
        assert a.dataset_id != b.dataset_id                                       # identity changes with category

    def test_AL_claude_cannot_set_the_category(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", script={"company": reply("RESEARCH_REQUEST", designed("H-cat") | {"data_status": "REAL"})})
        out = cp.run_cycle()
        assert out.status == "FAILED" and s.data_status is DataStatus.MOCK


# --- Research AM–AP -----------------------------------------------------------------------------------------------------------
class TestResearchReadiness:
    def stub(self, n, status=DataStatus.REAL):
        """Logic-only stub (never persisted, never evidence): a store with n contiguous bars and a category."""
        candles = [make_candle(i) for i in range(n)]
        store = SimpleNamespace(series=lambda p, sym, tf: candles if (sym, tf) == ("BTC/USD", H1) else [])
        return SimpleNamespace(data_status=status, store=store, provider=SimpleNamespace(name="mock", realization="r"))

    @pytest.mark.parametrize("n,expected", [(0, "REAL_DATA_UNAVAILABLE"), (2999, "INSUFFICIENT_REAL_CANDLES"),
                                            (3000, "VALIDATION_PENDING"), (3001, "VALIDATION_PENDING")])
    def test_AM_AN_threshold_states(self, n, expected):
        r = research_readiness(self.stub(n), "BTC/USD", H1, {"state": "PASS"}, P.MIN_CANDLES)
        assert r["state"] == expected and P.MIN_CANDLES == 3000
        assert "FAILED" not in r["state"]                                       # insufficient ≠ failed validation

    def test_AM_readiness_states_for_mock_and_blocked(self):
        assert research_readiness(self.stub(5000, DataStatus.MOCK), "BTC/USD", H1, {"state": "PASS"}, 3000)["state"] \
            == "NOT_APPLICABLE_MOCK"
        assert research_readiness(self.stub(5000), "BTC/USD", H1, {"state": "BLOCKED"}, 3000)["state"] == "BLOCKED"
        assert research_readiness(self.stub(5000), "ETH/USD", H4, {"state": "PASS"}, 3000)["state"] == "REAL_DATA_UNAVAILABLE"

    def test_AN_control_plane_window_enforces_3000(self, tmp_path):
        cp, s, _ = plane(tmp_path / "st", hours=3200)
        with_history(s, 3200)
        series = s.store.series("mock", "BTC/USD", H1)
        assert isinstance(cp._window(series[-3000:], False), tuple)
        assert "2999 bars < 3000" in cp._window(series[-2999:], False)

    def test_AO_real_research_prerequisites(self, tmp_path):
        s, clock, _ = kraken(tmp_path, hours=3200)
        acc.accumulate(s, series=(("BTC/USD", H1),))
        full = Dataset.build(s.store.series("kraken", "BTC/USD", H1), data_version="v", realization="observed")
        fails = research_preconditions(s, full, P.base_definition(clock.now()), hypothesis_id="H-ao",
                                       min_candles=P.MIN_CANDLES, require_market_data=True)
        assert any("MOCK data where market data is required" in f for f in fails)
        assert any(f.startswith("INSUFFICIENT DATA") for f in fails)               # 720 bars < 3000

    def test_AP_outcomes_carry_the_system_category_and_learning_is_category_bound(self, tmp_path):
        cp, s, clock = plane(tmp_path / "st", script={"company": reply("RESUME")})
        cp.run_cycle()
        assert all(o.data_category == "MOCK" for o in cp.learning.outcomes.values())
        attrs = dict(cp.learning.journal.attrs)
        assert attrs["data_status"] == "MOCK"
        with pytest.raises(ModeMismatch):                                          # a REAL learning chain cannot open it
            Journal(cp.learning.journal.path, kind="learning", attrs=attrs | {"data_status": "REAL"}, clock=clock)


# --- Recovery AQ–AS ----------------------------------------------------------------------------------------------------------
class TestRecovery:
    def test_AQ_crash_after_payload_persisted_before_parsing(self, tmp_path, monkeypatch):
        s, clock, _ = kraken(tmp_path)
        import ati.market.kraken as k
        real = k.parse_ohlc
        state = {"n": 0}

        def crash(*a, **kw):
            state["n"] += 1
            if state["n"] == 1:
                raise Boom("crash after the raw payload was archived")
            return real(*a, **kw)
        monkeypatch.setattr(k, "parse_ohlc", crash)
        with pytest.raises(Boom):
            acc.accumulate(s, series=(("BTC/USD", H1),))
        monkeypatch.undo()
        s2, _, _ = kraken(tmp_path, clock=clock)                  # replay re-derives from the archived bytes
        n = len(s2.store.series("kraken", "BTC/USD", H1))
        assert n > 0 and len(s2.archive) == 1
        r = by(acc.accumulate(s2, series=(("BTC/USD", H1),)))[("BTC/USD", "1h")]
        assert r.status == "NO_NEW_DATA" and len(s2.store.series("kraken", "BTC/USD", H1)) == n

    def test_AQ2_crash_after_store_before_run_record(self, tmp_path, monkeypatch):
        s, clock, _ = kraken(tmp_path)
        real = s.evidence.journal.append

        def crash(type_, payload):
            if type_ == "accumulation_run":
                raise Boom("crash before the run record")
            return real(type_, payload)
        monkeypatch.setattr(s.evidence.journal, "append", crash)
        with pytest.raises(Boom):
            acc.accumulate(s, series=(("BTC/USD", H1),))
        s2, _, _ = kraken(tmp_path, clock=clock)
        r = by(acc.accumulate(s2, series=(("BTC/USD", H1),)))[("BTC/USD", "1h")]
        assert r.status == "NO_NEW_DATA" and r.candles_before > 0                 # nothing lost, nothing duplicated

    def test_AR_crash_mid_run_between_series(self, tmp_path, monkeypatch):
        s, clock, feed = kraken(tmp_path)
        feed.fail = lambda sym, tf: Boom("process killed") if (sym, tf) == ("ETH/USD", H1) else None
        with pytest.raises(Boom):
            acc.accumulate(s)
        s2, _, feed2 = kraken(tmp_path, clock=clock)
        r = by(acc.accumulate(s2))
        assert r[("BTC/USD", "1h")].status == "NO_NEW_DATA" and r[("ETH/USD", "1h")].status == "ACCUMULATED"
        for sym, tf in acc.SERIES:
            series = s2.store.series("kraken", sym, tf)
            assert len({c.open_time for c in series}) == len(series)

    def test_AS_repeated_runs_are_idempotent(self, tmp_path):
        s, clock, _ = kraken(tmp_path)
        acc.accumulate(s)
        snap = snapshot(s)
        assert all(r.status == "NO_NEW_DATA" for r in acc.accumulate(s))   # first repeat: overlap-window request
        n_payloads = len(s.archive)
        for _ in range(3):
            assert all(r.status == "NO_NEW_DATA" for r in acc.accumulate(s))
        assert snapshot(s) == snap and len(s.archive) == n_payloads          # identical bytes: receipts only
        s.evidence.journal.verify()
