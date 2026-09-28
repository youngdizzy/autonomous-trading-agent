"""Live broker adapter boundary.

STATUS: NOT IMPLEMENTED (by design for Foundation 1.0). LIVE_TRADING is False and no brokerage
credentials exist. A future live adapter must implement ``ati.execution.broker.Broker`` (idempotent
submit on client order id, order lookup by client order id, account snapshot) and pass the same
contract and chaos suites as ``PaperBroker`` before any real-money milestone is considered.
"""

from __future__ import annotations

from ati.config import OperatingMode, assert_mode_permitted


def create_live_broker(*_args, **_kwargs):
    assert_mode_permitted(OperatingMode.LIVE)  # raises LiveTradingDisabled in this build
    raise NotImplementedError("NOT IMPLEMENTED: live broker adapter")
