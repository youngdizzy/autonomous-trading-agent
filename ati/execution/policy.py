"""Execution modes and pre-submission policy. Operator configuration, fixed at system construction — never read
from Claude output, a journal, a file an agent can write, or the environment.

  OBSERVE             Claude reads state and proposes; no order is ever sent (every submission is refused).
  PAPER               risk-approved orders go to the paper venue (the only venue this build can construct).
  ASSISTED            new-risk orders additionally need a journaled operator approval for their decision id.
  AUTONOMOUS_LIMITED  new-risk orders additionally need to fit the configured autonomous limits (tighter only).

The live state is not a mode anybody selects: ``live_state()`` is LIVE_DISABLED whenever ``ati.config.LIVE_TRADING``
is False (a module constant; enabling it is a reviewed code change), and no live adapter exists in this build.
Risk-reducing orders (sells of a held position) are never blocked by ASSISTED/AUTONOMOUS_LIMITED: deterministic
exits must not wait for a human. They are still blocked by OBSERVE, a stale account and an unhealthy broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from enum import Enum

from ati.config import LIVE_TRADING

APPROVAL_ACK = "OPERATOR: I approve this order for submission"


class ExecutionMode(str, Enum):
    OBSERVE = "OBSERVE"
    PAPER = "PAPER"
    ASSISTED = "ASSISTED"
    AUTONOMOUS_LIMITED = "AUTONOMOUS_LIMITED"


@dataclass(frozen=True)
class ExecutionPolicy:
    mode: ExecutionMode = ExecutionMode.PAPER
    max_account_age: timedelta = timedelta(minutes=5)     # last successful reconciliation must be this recent
    # AUTONOMOUS_LIMITED only (never looser than RiskLimits, which the risk engine has already applied)
    autonomous_symbols: frozenset[str] = field(default_factory=frozenset)
    autonomous_max_order_notional: Decimal = Decimal("0")
    autonomous_max_orders_per_day: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ExecutionMode):
            raise TypeError("execution mode must be an ExecutionMode")
        if self.max_account_age <= timedelta(0):
            raise ValueError("max_account_age must be positive")
        if self.autonomous_max_order_notional < 0 or self.autonomous_max_orders_per_day < 0:
            raise ValueError("autonomous limits cannot be negative")


def live_state() -> str:
    """LIVE_DISABLED unless the build constant allows live trading (it does not in this build)."""
    return "LIVE_ENABLED" if LIVE_TRADING else "LIVE_DISABLED"
