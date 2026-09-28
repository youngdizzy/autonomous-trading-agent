"""Wiring. Builds one paper-trading system rooted in a state directory. All durable state is
append-only journals plus the kill-switch file and the simulated venue's own state file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from ati.agent.pipeline import DecisionPipeline
from ati.agent.reasoning import ReasoningClient
from ati.config import OperatingMode, assert_mode_permitted
from ati.core.errors import DataIntegrityError
from ati.core.time import Clock
from ati.decision.records import DecisionLog
from ati.execution.engine import ExecutionEngine
from ati.execution.paper import PaperBroker, Quote
from ati.ledger.journal import Journal
from ati.market.models import DataStatus, Timeframe
from ati.market.provider import MarketDataProvider
from ati.market.store import CandleStore
from ati.market.universe import Universe, default_universe
from ati.memory.evidence import EvidenceRegistry
from ati.memory.store import MemoryStore
from ati.research.costs import CostModel
from ati.risk.engine import ApprovalAuthority, RiskEngine, RiskLimits
from ati.risk.killswitch import KillSwitch
from ati.security.secrets import SecretGuard
from ati.strategies.registry import StrategyRegistry


@dataclass
class System:
    state_dir: Path
    clock: Clock
    mode: OperatingMode
    data_status: DataStatus
    timeframe: Timeframe
    symbols: tuple[str, ...]
    universe: Universe
    provider: MarketDataProvider
    store: CandleStore
    costs: CostModel
    limits: RiskLimits
    guard: SecretGuard
    kill_switch: KillSwitch
    authority: ApprovalAuthority
    risk: RiskEngine
    broker: PaperBroker
    execution: ExecutionEngine
    evidence: EvidenceRegistry
    memory: MemoryStore
    decisions: DecisionLog
    strategies: StrategyRegistry
    research_journal: Journal
    reasoning: ReasoningClient
    pipeline: DecisionPipeline


def build_paper_system(state_dir: Path | str, clock: Clock, provider: MarketDataProvider, reasoning: ReasoningClient, *,
                       symbols: tuple[str, ...] = ("BTC/USD",), timeframe: Timeframe = Timeframe.H1,
                       data_status: DataStatus, initial_cash: Decimal = Decimal("100000"),
                       limits: RiskLimits = RiskLimits(), costs: CostModel = CostModel(),
                       guard: SecretGuard | None = None, universe: Universe | None = None) -> System:
    mode = OperatingMode.PAPER
    assert_mode_permitted(mode)
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    guard = guard or SecretGuard()
    universe = universe or default_universe()
    attrs = {"mode": mode.value, "data_status": data_status.value}
    store = CandleStore()

    def quotes(symbol: str) -> Quote:
        series = store.series(provider.name, symbol, timeframe)
        if not series:
            raise DataIntegrityError("NO_DATA", f"no market data for {symbol}")
        last = series[-1]
        return Quote(symbol, last.close, last.close_time, sum((c.volume for c in series[-24:]), Decimal(0)), last.status)

    kill = KillSwitch(state / "kill_switch.json", clock)
    authority = ApprovalAuthority()
    risk = RiskEngine(limits, universe, kill, authority, clock)
    broker = PaperBroker(quotes, costs, initial_cash, data_status, clock, state_path=state / "paper_venue.json")
    execution = ExecutionEngine(broker, state / "execution.jsonl", authority, clock, mode=mode, data_status=data_status,
                                initial_cash=initial_cash, guard=guard)
    evidence = EvidenceRegistry(Journal(state / "evidence.jsonl", kind="evidence", attrs=attrs, clock=clock, guard=guard))
    memory = MemoryStore(Journal(state / "memory.jsonl", kind="memory", attrs=attrs, clock=clock, guard=guard), evidence)
    decisions = DecisionLog(Journal(state / "decisions.jsonl", kind="decisions", attrs=attrs, clock=clock, guard=guard), evidence)
    research_journal = Journal(state / "research.jsonl", kind="research", attrs=attrs, clock=clock, guard=guard)
    strategies = StrategyRegistry(research_journal)
    pipeline = DecisionPipeline(clock=clock, universe=universe, risk=risk, execution=execution, decisions=decisions,
                                evidence=evidence, memory=memory, reasoning=reasoning, guard=guard)
    return System(state, clock, mode, data_status, timeframe, tuple(symbols), universe, provider, store, costs, limits,
                  guard, kill, authority, risk, broker, execution, evidence, memory, decisions, strategies,
                  research_journal, reasoning, pipeline)


DEFAULT_HISTORY_BARS = 300
STALE_AFTER = timedelta(hours=2)
