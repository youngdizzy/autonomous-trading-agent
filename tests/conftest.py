import pytest

from ati.core.time import FixedClock
from ati.ledger.journal import Journal
from tests.helpers import T0


@pytest.fixture(autouse=True)
def _isolated_holdout_seals():
    """Sealed holdout periods are process-wide by design; isolate them per test."""
    from ati.data.dataset import _clear_sealed_ranges_for_tests

    _clear_sealed_ranges_for_tests()
    yield
    _clear_sealed_ranges_for_tests()


@pytest.fixture
def clock():
    return FixedClock(T0)


@pytest.fixture
def journal(tmp_path, clock):
    return Journal(tmp_path / "research.jsonl", kind="research", attrs={"data_status": "MOCK"}, clock=clock)
