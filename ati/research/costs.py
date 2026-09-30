"""Execution cost model.

The defaults are ASSUMPTIONS, not measurements: they are intentionally conservative placeholders
until real fill data exists. Every backtest records the cost model it used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from ati.core.types import Side


@dataclass(frozen=True)
class CostModel:
    fee_rate: Decimal = Decimal("0.0026")        # ASSUMPTION: taker fee per side
    half_spread_rate: Decimal = Decimal("0.0002")  # ASSUMPTION: half the quoted spread
    slippage_rate: Decimal = Decimal("0.0003")   # ASSUMPTION: adverse move beyond the spread
    max_participation: Decimal = Decimal("0.10")  # max fraction of a bar's volume one fill may take

    @property
    def round_trip_rate(self) -> Decimal:
        return 2 * (self.fee_rate + self.half_spread_rate + self.slippage_rate)

    def fill_price(self, mid: Decimal, side: Side) -> Decimal:
        adj = self.half_spread_rate + self.slippage_rate
        return mid * (1 + adj) if side is Side.BUY else mid * (1 - adj)

    def scaled(self, factor: Decimal) -> "CostModel":
        return replace(self, fee_rate=self.fee_rate * factor, half_spread_rate=self.half_spread_rate * factor,
                       slippage_rate=self.slippage_rate * factor)

    @classmethod
    def frictionless(cls) -> "CostModel":
        return cls(Decimal(0), Decimal(0), Decimal(0), Decimal("1e9"))
