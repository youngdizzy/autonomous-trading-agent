"""System-wide operating mode and the LIVE trading hard switch.

``LIVE_TRADING`` is a module constant, not configuration read from the environment: enabling real
money must be a reviewed code change, never a runtime flag that an agent, a file, or an environment
variable can flip.
"""

from __future__ import annotations

from enum import Enum

from ati.core.errors import LiveTradingDisabled

LIVE_TRADING: bool = False


class OperatingMode(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


def assert_mode_permitted(mode: OperatingMode) -> None:
    """Raise unless ``mode`` may be used in this build."""
    if not isinstance(mode, OperatingMode):
        raise TypeError(f"mode must be OperatingMode, got {type(mode).__name__}")
    if mode is OperatingMode.LIVE and not LIVE_TRADING:
        raise LiveTradingDisabled("LIVE mode requested but LIVE_TRADING is False in this build")
