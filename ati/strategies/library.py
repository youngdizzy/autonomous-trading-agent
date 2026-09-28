"""Reference strategy logic. Deliberately minimal: these exist to exercise the research and
validation machinery, not as claims of edge."""

from __future__ import annotations

from decimal import Decimal

from ati.strategies.base import ParamSpec, Signal, StrategyLogic, Target, register_logic
from ati.temporal.features import atr, sma
from ati.temporal.pit import PointInTimeView


@register_logic
class MovingAverageCrossover(StrategyLogic):
    """Long while fast SMA > slow SMA; stop = close − k·ATR (ratcheted up by the backtester)."""

    kind = "ma_crossover"
    params = {
        "fast": ParamSpec(int, 2, 200),
        "slow": ParamSpec(int, 3, 500),
        "atr_period": ParamSpec(int, 2, 100),
        "stop_atr": ParamSpec(float, 0.5, 10.0),
    }

    @classmethod
    def check_relations(cls, values):
        if values["fast"] >= values["slow"]:
            raise ValueError("ma_crossover: fast must be < slow")

    @classmethod
    def lookback(cls, values):
        return max(values["slow"], values["atr_period"] + 1)

    @classmethod
    def generate(cls, view: PointInTimeView, values, in_position: bool) -> Signal:
        if len(view) < cls.lookback(values):
            return Signal(Target.FLAT, None, "insufficient history", view.cutoff)
        fast, slow, vol = sma(view, values["fast"]), sma(view, values["slow"]), atr(view, values["atr_period"])
        close = view.latest.close
        if fast > slow and vol and vol > 0:
            stop = close - Decimal(str(values["stop_atr"])) * vol
            if stop > 0:
                return Signal(Target.LONG, stop, f"fast {fast:.2f} > slow {slow:.2f}", view.cutoff)
        return Signal(Target.FLAT, None, "fast <= slow", view.cutoff)


@register_logic
class BuyAndHold(StrategyLogic):
    """Benchmark: always long with a wide fixed-fraction stop."""

    kind = "buy_and_hold"
    params = {"stop_fraction": ParamSpec(float, 0.01, 0.9)}

    @classmethod
    def lookback(cls, values):
        return 1

    @classmethod
    def generate(cls, view: PointInTimeView, values, in_position: bool) -> Signal:
        if len(view) < 1:
            return Signal(Target.FLAT, None, "no data", view.cutoff)
        close = view.latest.close
        return Signal(Target.LONG, close * (1 - Decimal(str(values["stop_fraction"]))), "benchmark", view.cutoff)
