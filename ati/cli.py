"""Command line.

    python -m ati demo   --state-dir DIR   end-to-end research + paper loop on MOCK data, then status
    python -m ati verify --state-dir DIR   verify every journal's hash chain

Everything the demo produces is MOCK: generated prices, scripted reasoning. It exercises the
mechanics; it is not evidence about any market.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from ati.agent.loop import AutonomousLoop
from ati.agent.reasoning import Budget, ScriptedReasoningClient
from ati.core.errors import AtiError
from ati.core.time import UTC, FixedClock
from ati.data.dataset import Dataset
from ati.ledger.journal import Journal
from ati.market.mock import MockProvider
from ati.market.models import DataStatus, Timeframe
from ati.monitoring.status import build_status, render
from ati.research.adversarial import AdversarialPolicy
from ati.research.hypothesis import Criterion
from ati.research.workflow import run_research_cycle
from ati.strategies import library  # noqa: F401
from ati.strategies.base import StrategyDefinition
from ati.system import build_paper_system
from ati.validation.promotion import PromotionPolicy

DEMO_START = datetime(2024, 1, 1, tzinfo=UTC)
DEMO_SEED = 11


def _packet(prompt: str) -> dict:
    return json.loads(prompt.split("\n\nPACKET:\n\n", 1)[1].split("\n\n<<<UNTRUSTED_DATA", 1)[0])


def mock_primary(prompt: str) -> str:
    p = _packet(prompt)
    if p["signal"]["target"] != "LONG":
        return json.dumps({"action": "NO_TRADE", "reason": "no long signal"})
    return json.dumps({
        "action": "PROPOSE_TRADE", "symbol": p["symbol"], "side": "BUY", "strategy_key": p["strategy"]["key"],
        "entry_price": p["recent_closes"][-1], "stop_price": p["signal"]["stop"],
        "thesis": "[MOCK reasoning] champion trend signal active; follow it at deterministic size",
        "invalidation_condition": "close below the strategy stop or fast SMA crossing below slow SMA",
        "confidence": 0.5,
    })


MOCK_SCRIPT = {
    "primary_decision": mock_primary,
    "adversarial_reviewer": json.dumps({"verdict": "CHALLENGE", "objections": [
        "[MOCK] evidence is from MOCK data only", "[MOCK] single-instrument trend rule; regime dependence likely"]}),
    "post_trade_reviewer": json.dumps({"process_quality": "INSUFFICIENT_INFORMATION",
                                       "notes": "[MOCK] outcome not used to judge process", "possible_mistake": None}),
}


def demo(state_dir: Path, ticks: int) -> int:
    history = 3000
    research_end = DEMO_START + Timeframe.H1.delta * history
    clock = FixedClock(research_end)
    provider = MockProvider(DEMO_SEED, clock, epoch=DEMO_START)
    reasoning = ScriptedReasoningClient(MOCK_SCRIPT, Budget(6))
    system = build_paper_system(state_dir, clock, provider, reasoning, data_status=DataStatus.MOCK)

    print("== RESEARCH (MOCK data) ==")
    full = Dataset.build(provider.fetch_candles("BTC/USD", Timeframe.H1, DEMO_START, research_end),
                         data_version="mock-v1", realization=provider.realization)
    base = StrategyDefinition.create("trend", 1, "ma_crossover", {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0},
                                     Timeframe.H1, clock.now(), description="reference trend rule")
    grid = [base.param_dict | {"fast": f, "slow": sl} for f in (5, 10, 20) for sl in (50, 100)]
    result = run_research_cycle(
        system, full, full.candles[int(history * 0.8)].open_time, hypothesis_id="H-trend-001",
        statement="Moving-average trend persistence yields positive expectancy after costs",
        base=base, grid=grid, criteria=(Criterion("net_pnl", ">", 0.0), Criterion("expectancy_r", ">", 0.0)),
        train_bars=800, test_bars=400, adversarial_policy=AdversarialPolicy(allow_non_market_data=True),
        promotion_policy=PromotionPolicy(allow_mock_evidence=True))
    print(f"  development verdict : {result.dev_verdict.value}")
    print(f"  challenger          : {result.challenger_key}")
    print(f"  adversarial blocking: {result.adversarial_blocking}")
    print(f"  holdout verdict     : {result.holdout_verdict and result.holdout_verdict.value}")
    if result.promotion:
        print(f"  promotion           : {'APPROVED' if result.promotion.approved else 'DENIED'}")
        for r in result.promotion.reasons:
            print(f"    - {r}")

    print(f"\n== PAPER LOOP ({ticks} hourly ticks, MOCK data, MOCK reasoning) ==")
    loop = AutonomousLoop(system)
    for _ in range(ticks):
        clock.advance(timedelta(hours=1))
        rep = loop.tick()
        if rep.actions or rep.errors:
            print(f"  tick {rep.tick}: recon={rep.reconciliation} actions={rep.actions} errors={rep.errors}")
    print("\n== STATUS ==")
    print(render(build_status(system, loop)))
    return 0


def verify(state_dir: Path) -> int:
    ok = True
    for path in sorted(state_dir.glob("*.jsonl")):
        try:
            head = json.loads(path.read_text().splitlines()[0])
            j = Journal(path, kind=head["payload"]["kind"], attrs=head["payload"]["attrs"], clock=FixedClock(DEMO_START))
            print(f"OK        {path.name}: {len(j)} entries, head {j.head_hash[:16]}")
        except (AtiError, OSError, ValueError, KeyError, IndexError) as exc:
            ok = False
            print(f"CORRUPT   {path.name}: {exc}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ati")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--state-dir", type=Path, required=True)
    d.add_argument("--ticks", type=int, default=72)
    v = sub.add_parser("verify")
    v.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        if args.state_dir.exists() and any(args.state_dir.iterdir()):
            print("state dir not empty; use a fresh directory for the demo", file=sys.stderr)
            return 2
        return demo(args.state_dir, args.ticks)
    return verify(args.state_dir)


if __name__ == "__main__":
    sys.exit(main())
