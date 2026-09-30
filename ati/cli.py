"""Command line.

    python -m ati demo   --state-dir DIR   end-to-end research + paper loop on MOCK data, then status
    python -m ati verify --state-dir DIR   verify every journal's hash chain
    python -m ati ingest-kraken --state-dir DIR   read-only Kraken OHLC fetch → REAL payload archive
    python -m ati research-real --state-dir DIR   the pre-declared REAL research protocol, run once
    python -m ati accumulate --state-dir DIR --data kraken|mock   BTC/ETH × 1h/4h incremental accumulation
    python -m ati data-health --state-dir DIR --data kraken|mock  read-only dataset scorecard and health
    python -m ati company cycle|status|pause|resume --state-dir DIR --data mock|kraken
    python -m ati intake-vault --state-dir DIR --data mock|kraken --corpus PATH --manifest FILE --commit SHA
                                                  one bounded company control-plane cycle (no loop, no daemon)

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


def _real_system(state_dir: Path):
    """PAPER system bound to REAL data from Kraken over the real network transport. Read-only market
    data; LIVE_TRADING stays False; reasoning via the Claude-native file exchange."""
    from ati.agent.reasoning import FileExchangeClient
    from ati.core.time import SystemClock
    from ati.market.kraken import KrakenPublicOHLC
    from ati.market.provider import UrllibTransport

    clock = SystemClock()
    provider = KrakenPublicOHLC(UrllibTransport(), clock)
    # The category comes from the transport (the network transport declares REAL); it is not asserted here.
    return build_paper_system(state_dir, clock, provider, FileExchangeClient(state_dir / "exchange"),
                              data_status=provider.data_status)


def ingest_kraken(state_dir: Path, symbol: str) -> int:
    from ati.core.errors import ProviderError

    s = _real_system(state_dir)
    now = s.clock.now()
    before = len(s.store.series("kraken", symbol, Timeframe.H1))
    try:
        candles = s.provider.fetch_candles(symbol, Timeframe.H1, now - timedelta(hours=720), now)
    except ProviderError as exc:
        print(f"LIVE_CONNECTIVITY = BLOCKED\n  endpoint : https://api.kraken.com/0/public/OHLC\n"
              f"  error    : {type(exc).__name__}: {exc}\n  cause    : {exc.__cause__!r}\n  nothing was ingested")
        return 3
    added = s.archive.ingest(s.store, candles)
    series = s.store.series("kraken", symbol, Timeframe.H1)
    print(f"LIVE_CONNECTIVITY = VERIFIED\n  payloads archived : {len(s.archive)}\n  candles received  : {len(candles)} "
          f"(closed {sum(c.is_closed for c in candles)})\n  new closed candles: {added}\n"
          f"  series            : {before} -> {len(series)} candles, {series[0].open_time} .. {series[-1].close_time}")
    return 0


def research_real(state_dir: Path) -> int:
    from ati.research.hypothesis import ResearchLog
    from ati.research import protocol as P

    s = _real_system(state_dir)
    series = s.store.series("kraken", P.SYMBOL, P.TIMEFRAME)
    print(f"== REAL RESEARCH ({P.PROTOCOL_ID}) ==")
    if not series:
        s.research_journal.append("research_not_run", {"hypothesis_id": P.HYPOTHESIS_ID, "reasons": ["no archived REAL data"]})
        print("REAL RESEARCH RUN = NOT_RUN\n  - no archived REAL data (run `ati ingest-kraken`; currently BLOCKED by egress policy)")
        return 4
    full = Dataset.build(series, data_version="kraken-ohlc-archive", realization="observed")
    print(f"  dataset  : {full.dataset_id} ({len(full)} candles {full.identity.start} .. {full.identity.end})")
    boundary = full.candles[int(len(full) * (1 - P.HOLDOUT_FRACTION))].open_time
    from ati.research.protocols import REGISTRY
    proto = REGISTRY[P.PROTOCOL_ID]
    if not ResearchLog(s.research_journal).was_tested(P.HYPOTHESIS_ID):
        from ati.research.protocols import deployment
        s.research_journal.append("protocol_run", {"protocol_id": proto.protocol_id, "protocol_hash": proto.protocol_hash,
                                                   **{k: v for k, v in deployment(proto).items()
                                                      if k not in ("protocol_id", "protocol_hash", "symbol", "timeframe")},
                                                   "strategy_id": proto.strategy_id, "strategy_version": proto.strategy_version,
                                                   "kind": "validation", "hypothesis_id": P.HYPOTHESIS_ID,
                                                   "experiment_id": None, "symbol": proto.symbol,
                                                   "timeframe": proto.timeframe.value, "dataset_id": full.dataset_id})
    result = run_research_cycle(s, full, boundary, hypothesis_id=P.HYPOTHESIS_ID, statement=P.STATEMENT,
                                base=P.base_definition(s.clock.now()), grid=P.GRID, criteria=P.CRITERIA,
                                train_bars=P.TRAIN_BARS, test_bars=P.TEST_BARS, min_candles=P.MIN_CANDLES)
    if result.status == "NOT_RUN":
        print("REAL RESEARCH RUN = NOT_RUN")
        for r in result.reasons:
            print(f"  - {r}")
        return 4
    print(f"  development verdict : {result.dev_verdict.value}\n  challenger          : {result.challenger_key}\n"
          f"  adversarial blocking: {result.adversarial_blocking}\n"
          f"  holdout verdict     : {result.holdout_verdict and result.holdout_verdict.value}\n"
          f"  promotion           : {result.promotion and ('APPROVED' if result.promotion.approved else 'DENIED')}")
    return 0


def _data_system(state_dir: Path, data: str):
    """``kraken``: the REAL system (network transport; REAL only if the provider actually answers).
    ``mock``: MOCK provider — no archive, nothing persisted, history regenerated deterministically."""
    from ati.agent.reasoning import FileExchangeClient
    from ati.core.time import SystemClock

    if data == "kraken":
        return _real_system(state_dir)
    clock = SystemClock()
    provider = MockProvider(DEMO_SEED, clock, epoch=DEMO_START)
    return build_paper_system(state_dir, clock, provider, FileExchangeClient(state_dir / "exchange"),
                              data_status=provider.data_status)


def accumulate_cmd(state_dir: Path, data: str) -> int:
    from ati.core.errors import AtiError as _AtiError
    from ati.core.lock import StateLock
    from ati.market.accumulate import accumulate

    try:
        with StateLock(state_dir):
            s = _data_system(state_dir, data)
            results = accumulate(s)
    except _AtiError as exc:
        print(f"ACCUMULATION REFUSED — nothing written: {type(exc).__name__}: {exc}")
        return 5
    print(f"== ACCUMULATION ({s.data_status.value} via {s.provider.name}) ==")
    for r in results:
        print(json.dumps(r.__dict__, default=str))
    return 0 if all(r.status in ("ACCUMULATED", "NO_NEW_DATA") for r in results) else 3


def intake_vault_cmd(state_dir: Path, data: str, corpus_dir: Path, manifest: Path, repository: str, commit: str,
                     batch: str, salt: str, per_language: str) -> int:
    """Pilot intake of an external strategy corpus (text only; nothing external is executed). Output: one normalized
    record per selected source and the research-universe record. Research of COMPATIBLE candidates is a separate,
    explicit step through the existing workflow — never automatic."""
    from ati.core.errors import AtiError as _AtiError
    from ati.core.lock import StateLock
    from ati.intake.corpus import ExternalIntake
    from ati.intake.source import GitCorpus

    quota = {k: int(v) for k, v in (item.split("=") for item in per_language.split(",") if item)}
    if sum(quota.values()) > 25:
        print("REFUSED: a pilot is at most 25 strategies")
        return 2
    try:
        with StateLock(state_dir):
            s = _data_system(state_dir, data)
            corpus = GitCorpus(corpus_dir, repository, commit, manifest)
            intake = ExternalIntake(s.research_journal)
            chosen, census = intake.select_pilot(corpus, quota, salt)
            records = [intake.ingest(a) for a in corpus.read_many(chosen)]
            universe = intake.record_universe(corpus, batch, chosen, census, salt, quota)
    except _AtiError as exc:
        print(f"INTAKE REFUSED: {type(exc).__name__}: {exc}")
        return 5
    keep = ("external_strategy_id", "source_path", "source_hash", "source_language", "strategy_family",
            "normalization_status", "compatibility_status", "tradetown_strategy_id", "normalized_strategy_hash")
    print(json.dumps({"records": [{k: r.get(k) for k in keep} | {"reasons": [x["state"] + ": " + x["reason"]
                                                                               for x in r["reasons"]]}
                                  for r in records],
                      "universe": intake.universe_summary(universe["universe_id"]), "census": census},
                     indent=2, default=str))
    return 0


def data_health_cmd(state_dir: Path, data: str) -> int:
    from ati.core.errors import AtiError as _AtiError
    from ati.market.accumulate import SERIES
    from ati.market.health import scorecard, verify_sealed_holdouts

    try:
        s = _data_system(state_dir, data)
    except _AtiError as exc:
        print(f"DATA STATE UNTRUSTWORTHY: {type(exc).__name__}: {exc}")
        return 5
    from ati.market.health import readiness_table
    from ati.research.protocols import REGISTRY

    print(json.dumps({"category": s.data_status.value, "provider": s.provider.name,
                      "readiness_table": readiness_table(s, SERIES),
                      "protocols": [p.describe() for p in REGISTRY.values()],
                      "datasets": scorecard(s, SERIES),
                      "sealed_holdouts": verify_sealed_holdouts(s),
                      "open_conflicts": s.archive.conflicts()}, indent=2, default=str))
    return 0


def _company(state_dir: Path, data: str):
    """Control plane over a PAPER system. ``kraken``: REAL data via the network transport (BLOCKED here).
    ``mock``: MOCK data (fixed seed and epoch so the realization is stable across invocations)."""
    from ati.agent.reasoning import FileExchangeClient
    from ati.company.control import CompanyControlPlane
    from ati.core.time import SystemClock

    if data == "kraken":
        return CompanyControlPlane(_real_system(state_dir))
    clock = SystemClock()
    provider = MockProvider(DEMO_SEED, clock, epoch=DEMO_START)
    system = build_paper_system(state_dir, clock, provider, FileExchangeClient(state_dir / "exchange"),
                                data_status=provider.data_status)
    return CompanyControlPlane(system)


def company(cmd: str, state_dir: Path, data: str, ack: str | None, max_cycles: int = 1) -> int:
    from ati.core.errors import AtiError as _AtiError

    from ati.core.lock import StateLock

    if cmd in ("cycle", "run", "pause", "resume"):   # writers hold the state lock for the whole command
        try:
            with StateLock(state_dir):
                return _company_cmd(cmd, state_dir, data, ack, max_cycles)
        except _AtiError as exc:
            print(f"COMPANY REFUSED — nothing written: {type(exc).__name__}: {exc}")
            return 5
    return _company_cmd(cmd, state_dir, data, ack, max_cycles)


def _company_cmd(cmd: str, state_dir: Path, data: str, ack: str | None, max_cycles: int) -> int:
    from ati.core.errors import AtiError as _AtiError

    try:
        cp = _company(state_dir, data)
    except _AtiError as exc:  # corrupt or contradictory company state: fail closed, do nothing
        print(f"COMPANY STATE UNTRUSTWORTHY — refusing to act: {type(exc).__name__}: {exc}")
        return 5
    if cmd == "pause":
        cp.pause("operator", "operator pause")
    elif cmd == "resume":
        try:
            cp.resume(ack or "")
        except (PermissionError, _AtiError) as exc:
            print(f"resume refused: {exc}")
            return 6
    elif cmd == "report":
        from ati.company.intelligence import report as intelligence_report

        print(json.dumps(intelligence_report(cp), indent=2, default=str))
        return 0
    elif cmd == "readiness":
        from ati.company.readiness import report

        print(json.dumps(report(cp), indent=2, default=str))
        return 0
    elif cmd == "run":
        # Scheduler entry point: bounded cycles; stops at the first cycle that needs Claude or does not complete.
        # A future scheduler calls this; it never becomes an authority itself.
        for _ in range(max(1, min(max_cycles, 24))):
            outcome = cp.run_cycle()
            print(json.dumps({"cycle_id": outcome.cycle_id, "status": outcome.status, "action": outcome.action}))
            if outcome.status not in ("COMPLETED",):
                break
    elif cmd == "cycle":
        outcome = cp.run_cycle()
        print(json.dumps({"cycle_id": outcome.cycle_id, "status": outcome.status, "action": outcome.action,
                          "detail": outcome.detail, "data_state": outcome.health.get("data_state")},
                         indent=2, default=str))
        if outcome.status == "AWAITING_CLAUDE":
            print(f"request: {state_dir / 'exchange' / 'requests'}  (write the response JSON to exchange/responses/"
                  "with the same file stem)")
    last = cp.cycles.get(cp.last_finished) if cp.last_finished else None
    print(f"company state: {cp.state.value}  paused: {cp.paused}  running: {cp.running}  "
          f"last cycle: {last.cycle_id + ' ' + str(last.status) if last else None}")
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
    k = sub.add_parser("ingest-kraken", help="read-only Kraken OHLC fetch into the REAL payload archive")
    k.add_argument("--state-dir", type=Path, required=True)
    k.add_argument("--symbol", default="BTC/USD")
    r = sub.add_parser("research-real", help="run the pre-declared REAL research protocol once")
    r.add_argument("--state-dir", type=Path, required=True)
    for name, text in (("accumulate", "BTC/ETH × 1h/4h incremental accumulation through the payload archive"),
                       ("data-health", "read-only dataset scorecard, health and sealed-holdout verification")):
        a = sub.add_parser(name, help=text)
        a.add_argument("--state-dir", type=Path, required=True)
        a.add_argument("--data", choices=["mock", "kraken"], required=True)
    iv = sub.add_parser("intake-vault", help="pilot intake of an external strategy corpus (research candidates only)")
    iv.add_argument("--state-dir", type=Path, required=True)
    iv.add_argument("--data", choices=["mock", "kraken"], required=True)
    iv.add_argument("--corpus", type=Path, required=True, help="git checkout of the corpus repository")
    iv.add_argument("--manifest", type=Path, required=True,
                    help="`git -C CORPUS ls-tree -r -z COMMIT` output (run by the operator; the system never shells out)")
    iv.add_argument("--repository", default="brainbrick-trades/The-Quant-Trading-Vault")
    iv.add_argument("--commit", required=True, help="full 40-hex commit to read sources at")
    iv.add_argument("--batch", default="pilot-1")
    iv.add_argument("--salt", default="tradetown-vault-pilot-1")
    iv.add_argument("--per-language", default="PineScript=14,javascript=4,python=3,MyLanguage=2,cpp=1")
    c = sub.add_parser("company", help="company control plane: one bounded cycle, status, pause, resume")
    c.add_argument("action", choices=["cycle", "run", "status", "readiness", "report", "pause", "resume"])
    c.add_argument("--state-dir", type=Path, required=True)
    c.add_argument("--data", choices=["mock", "kraken"], required=True)
    c.add_argument("--ack", default=None, help="operator acknowledgement phrase (resume only)")
    c.add_argument("--max-cycles", type=int, default=6, help="run: at most this many cycles (capped at 24)")
    args = parser.parse_args(argv)
    if args.cmd == "company":
        return company(args.action, args.state_dir, args.data, args.ack, args.max_cycles)
    if args.cmd == "ingest-kraken":
        from ati.core.lock import StateLock
        with StateLock(args.state_dir):
            return ingest_kraken(args.state_dir, args.symbol)
    if args.cmd == "intake-vault":
        return intake_vault_cmd(args.state_dir, args.data, args.corpus, args.manifest, args.repository, args.commit,
                                args.batch, args.salt, args.per_language)
    if args.cmd == "accumulate":
        return accumulate_cmd(args.state_dir, args.data)
    if args.cmd == "data-health":
        return data_health_cmd(args.state_dir, args.data)
    if args.cmd == "research-real":
        return research_real(args.state_dir)
    if args.cmd == "demo":
        if args.state_dir.exists() and any(args.state_dir.iterdir()):
            print("state dir not empty; use a fresh directory for the demo", file=sys.stderr)
            return 2
        return demo(args.state_dir, args.ticks)
    return verify(args.state_dir)


if __name__ == "__main__":
    sys.exit(main())
