"""Tradable instrument universe. A symbol not in the universe does not exist to the system."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal


@dataclass(frozen=True)
class Instrument:
    symbol: str
    base: str
    quote: str
    price_tick: Decimal
    lot_step: Decimal
    min_qty: Decimal
    correlation_group: str

    def floor_qty(self, qty: Decimal) -> Decimal:
        if qty <= 0:
            return Decimal(0)
        return (qty / self.lot_step).to_integral_value(rounding=ROUND_DOWN) * self.lot_step


class Universe:
    def __init__(self, instruments: list[Instrument]):
        self._by_symbol: dict[str, Instrument] = {}
        for inst in instruments:
            if inst.symbol in self._by_symbol:
                raise ValueError(f"duplicate instrument {inst.symbol}")
            self._by_symbol[inst.symbol] = inst

    def __contains__(self, symbol: object) -> bool:
        return isinstance(symbol, str) and symbol in self._by_symbol

    def get(self, symbol: str) -> Instrument:
        try:
            return self._by_symbol[symbol]
        except KeyError:
            raise KeyError(f"unknown symbol {symbol!r}") from None

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_symbol))


def default_universe() -> Universe:
    """Foundation universe: spot crypto majors. Correlation grouping is a conservative assumption
    (all crypto majors treated as one correlated group)."""
    return Universe(
        [
            Instrument("BTC/USD", "BTC", "USD", Decimal("0.1"), Decimal("0.00001"), Decimal("0.0001"), "crypto_major"),
            Instrument("ETH/USD", "ETH", "USD", Decimal("0.01"), Decimal("0.0001"), Decimal("0.001"), "crypto_major"),
        ]
    )
