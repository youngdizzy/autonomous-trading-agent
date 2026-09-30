"""Pre-declared protocol for the first research run on REAL data.

Declared and committed before any real market data has entered the system, so git history
timestamps it ahead of the evidence it will be judged on. Changing anything here after real data
exists means a new protocol id (a new hypothesis), which increases the recorded number of hypotheses
tested. Nothing here was chosen by looking at results: the strategy, grid and criteria are the
Foundation 1.0 reference values, unchanged.
"""

from __future__ import annotations

from datetime import datetime

from ati.market.models import Timeframe
from ati.research.hypothesis import Criterion
from ati.strategies import library  # noqa: F401  (registers strategy logic)
from ati.strategies.base import StrategyDefinition

PROTOCOL_ID = "REAL-PROTOCOL-001"
HYPOTHESIS_ID = "H-trend-real-001"
STATEMENT = "Moving-average trend persistence yields positive expectancy after costs"
SYMBOL = "BTC/USD"
TIMEFRAME = Timeframe.H1
BASE_PARAMS = {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0}
GRID = [BASE_PARAMS | {"fast": f, "slow": sl} for f in (5, 10, 20) for sl in (50, 100)]
CRITERIA = (Criterion("net_pnl", ">", 0.0), Criterion("expectancy_r", ">", 0.0))
TRAIN_BARS = 800
TEST_BARS = 400
HOLDOUT_FRACTION = 0.2
# 3000 hourly bars ≈ 125 days. Kraken's public OHLC endpoint returns at most 720 recent bars, so this
# requires accumulating archived payloads over time; until then the run is NOT_RUN (insufficient data).
MIN_CANDLES = 3000


def base_definition(created_at: datetime) -> StrategyDefinition:
    return StrategyDefinition.create("trend", 1, "ma_crossover", BASE_PARAMS, TIMEFRAME, created_at,
                                     description="Foundation 1.0 reference trend rule (unmodified)")
