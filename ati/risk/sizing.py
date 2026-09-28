"""Position sizing. One implementation, used by both the backtester and the live/paper risk
engine, so research results and trading behavior size identically."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal


def floor_to_step(qty: Decimal, step: Decimal) -> Decimal:
    if qty <= 0:
        return Decimal(0)
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step


def per_unit_risk(entry: Decimal, stop: Decimal, round_trip_cost_rate: Decimal) -> Decimal:
    """Loss per unit if the stop is hit, including round-trip costs (fees + spread + slippage)."""
    if stop >= entry:
        raise ValueError("long stop must be below entry")
    return (entry - stop) + entry * round_trip_cost_rate


def risk_based_qty(risk_budget: Decimal, entry: Decimal, stop: Decimal, round_trip_cost_rate: Decimal) -> Decimal:
    if risk_budget <= 0:
        return Decimal(0)
    return risk_budget / per_unit_risk(entry, stop, round_trip_cost_rate)
