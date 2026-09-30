from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from ati.core.errors import DatasetIntegrityError, HistoricalConflictError, HoldoutViolation, LookaheadError
from ati.data.dataset import Dataset, DatasetRegistry, Partition
from ati.temporal.features import atr, sma
from ati.temporal.pit import InformationItem, InformationSet, PointInTimeView, assert_available
from tests.helpers import T0, make_candle, mock_dataset


class TestDatasetIdentity:
    def test_hash_is_deterministic(self):
        assert mock_dataset(200).dataset_id == mock_dataset(200).dataset_id
        assert mock_dataset(200).identity.content_sha256 == mock_dataset(200).identity.content_sha256

    def test_content_change_changes_hash(self):
        a = mock_dataset(200, seed=1)
        b = mock_dataset(200, seed=2)
        assert a.identity.content_sha256 != b.identity.content_sha256

    def test_single_value_change_changes_hash(self):
        base = [make_candle(i) for i in range(5)]
        changed = list(base)
        changed[3] = make_candle(3, c="100.51")
        a = Dataset.build(base, data_version="v", realization="r")
        b = Dataset.build(changed, data_version="v", realization="r")
        assert a.identity.content_sha256 != b.identity.content_sha256

    def test_receipt_time_does_not_change_content_hash(self):
        a = Dataset.build([make_candle(i) for i in range(5)], data_version="v", realization="r")
        b = Dataset.build([make_candle(i, received_at=T0 + timedelta(days=9)) for i in range(5)],
                          data_version="v", realization="r")
        assert a.identity.content_sha256 == b.identity.content_sha256

    def test_identity_binds_version_realization_partition(self):
        c = [make_candle(i) for i in range(5)]
        ids = {Dataset.build(c, data_version=v, realization=r, partition=p).dataset_id
               for v in ("v1", "v2") for r in ("r1", "r2") for p in (Partition.FULL, Partition.DEVELOPMENT)}
        assert len(ids) == 8

    def test_in_memory_mutation_detected(self):
        ds = mock_dataset(100)
        ds.verify()
        object.__setattr__(ds.candles[10], "close", ds.candles[10].close + Decimal("1"))
        with pytest.raises(DatasetIntegrityError):
            ds.verify()

    def test_on_disk_mutation_detected(self, tmp_path):
        ds = mock_dataset(50)
        path = tmp_path / "ds.json"
        ds.save(path)
        assert Dataset.load(path, expected_id=ds.dataset_id).dataset_id == ds.dataset_id
        text = path.read_text()
        first_close = str(ds.candles[0].close)
        path.write_text(text.replace(f'"c": "{first_close}"', '"c": "' + str(ds.candles[0].close + 1) + '"', 1))
        with pytest.raises((DatasetIntegrityError, Exception)):
            Dataset.load(path, expected_id=ds.dataset_id)

    def test_reproducible_roundtrip(self, tmp_path):
        ds = mock_dataset(80)
        ds.save(tmp_path / "a.json")
        again = Dataset.load(tmp_path / "a.json")
        assert again.identity == ds.identity

    def test_registry_conflict_on_same_slice_different_content(self):
        reg = DatasetRegistry()
        base = [make_candle(i) for i in range(5)]
        reg.register(Dataset.build(base, data_version="v", realization="r"))
        changed = list(base)
        changed[2] = make_candle(2, c="100.4")
        with pytest.raises(HistoricalConflictError):
            reg.register(Dataset.build(changed, data_version="v", realization="r"))

    def test_open_candle_not_allowed_in_dataset(self):
        with pytest.raises(Exception):
            Dataset.build([make_candle(0), make_candle(1, closed=False)], data_version="v", realization="r")


class TestTemporal:
    def test_view_excludes_candle_not_yet_closed(self):
        ds = mock_dataset(100)
        cutoff = T0 + timedelta(hours=10, minutes=59)
        view = ds.view_at(cutoff)
        assert len(view) == 10  # bars closing 01:00..10:00
        assert view.latest.open_time == T0 + timedelta(hours=9)  # the 10:00 bar closes 11:00: invisible
        assert view.latest.close_time <= cutoff

    def test_view_at_exact_close_includes_that_candle(self):
        ds = mock_dataset(100)
        view = ds.view_at(T0 + timedelta(hours=10))
        assert len(view) == 10

    def test_view_cannot_be_constructed_with_future_candle(self):
        ds = mock_dataset(100)
        with pytest.raises(LookaheadError):
            PointInTimeView(ds.candles[:12], T0 + timedelta(hours=11), ds.dataset_id)

    def test_view_holds_no_reference_to_future(self):
        ds = mock_dataset(100)
        view = ds.view_at(T0 + timedelta(hours=20), lookback=5)
        assert len(view.candles) == 5
        assert not hasattr(view, "_dataset")
        assert max(c.available_at for c in view.candles) <= view.cutoff

    def test_features_do_not_change_when_future_changes(self):
        """The strongest lookahead test: altering every future candle must not change any
        feature computed at the cutoff."""
        base = [make_candle(i, c=str(100 + (i % 3) * 0.1)) for i in range(60)]
        future_changed = base[:30] + [make_candle(i, o="100", h="200", l="99", c="199") for i in range(30, 60)]
        a = Dataset.build(base, data_version="v", realization="r")
        b = Dataset.build(future_changed, data_version="v", realization="r")
        cutoff = T0 + timedelta(hours=30)
        va, vb = a.view_at(cutoff), b.view_at(cutoff)
        assert sma(va, 10) == sma(vb, 10)
        assert atr(va, 14) == atr(vb, 14)
        assert va.candles == vb.candles

    def test_information_set_rejects_future_items(self):
        cutoff = T0 + timedelta(hours=5)
        info = InformationSet(cutoff, [
            InformationItem("news-1", "news", T0 + timedelta(hours=4), "past"),
            InformationItem("news-2", "news", T0 + timedelta(hours=6), "future"),
            InformationItem("research-9", "research", T0 + timedelta(hours=5, seconds=1), "future"),
        ])
        assert [i.item_id for i in info.items()] == ["news-1"]
        with pytest.raises(LookaheadError):
            info.get("news-2")
        with pytest.raises(LookaheadError):
            info.get("research-9")

    def test_assert_available(self):
        assert_available([T0], T0)
        with pytest.raises(LookaheadError):
            assert_available([T0 + timedelta(seconds=1)], T0)


def test_holdout_dataset_refused_for_development():
    ds = mock_dataset(100)
    dev, hold = ds.split(T0 + timedelta(hours=70))
    assert dev.partition is Partition.DEVELOPMENT and hold.partition is Partition.HOLDOUT
    assert dev.identity.end <= hold.identity.start
    with pytest.raises(HoldoutViolation):
        hold.require_not_holdout("optimization")
    dev.require_not_holdout("optimization")
