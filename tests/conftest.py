import pytest

from ati.core.time import FixedClock
from ati.ledger.journal import Journal
from tests.helpers import T0


@pytest.fixture
def clock():
    return FixedClock(T0)


@pytest.fixture
def journal(tmp_path, clock):
    return Journal(tmp_path / "research.jsonl", kind="research", attrs={"data_status": "MOCK"}, clock=clock)
