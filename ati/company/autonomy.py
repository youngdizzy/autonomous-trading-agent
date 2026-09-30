"""Autonomy levels. The level a company may operate at is derived from code constants, never from
configuration Claude can reach and never from research outcomes.

  RESEARCH_AUTONOMY  analyse, research, hypothesise, request experiments
  PAPER_AUTONOMY     additionally propose paper trades (through the existing risk/execution path)
  SUPERVISED_LIVE    future stage — requires LIVE_TRADING (a reviewed code change) AND external authorization
  FULL_LIVE          future stage — never enabled in this build

A research result, a promotion, a memory entry or a Claude action cannot raise the level: the only inputs
would be ``ati.config.LIVE_TRADING`` (a module constant) and an external authorization mechanism that does
not exist in this build; the ceiling is therefore fixed at PAPER_AUTONOMY.
"""

from __future__ import annotations

from enum import IntEnum

from ati.core.errors import LiveTradingDisabled


class Autonomy(IntEnum):
    RESEARCH_AUTONOMY = 1
    PAPER_AUTONOMY = 2
    SUPERVISED_LIVE = 3
    FULL_LIVE = 4


def maximum_permitted() -> Autonomy:
    # Constant in this build. SUPERVISED_LIVE would need LIVE_TRADING (a reviewed code change) AND an external
    # authorization mechanism, which is NOT IMPLEMENTED — so even LIVE_TRADING=True could not raise this.
    return Autonomy.PAPER_AUTONOMY


def require(level: Autonomy) -> Autonomy:
    if not isinstance(level, Autonomy):
        raise TypeError("autonomy level must be an Autonomy value")
    if level > maximum_permitted():
        raise LiveTradingDisabled(f"{level.name} is not permitted in this build (maximum {maximum_permitted().name})")
    return level
