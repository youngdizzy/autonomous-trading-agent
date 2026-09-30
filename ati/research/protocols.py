"""Research protocol registry — BTC/USD and ETH/USD × 1h and 4h.

``REAL-PROTOCOL-001`` (BTC/USD 1h) is the pre-declared protocol in ``ati.research.protocol``; it is reproduced
here *from that module*, so its constants are defined once. The other three protocols declare the equivalent
requirements for the remaining approved datasets. Nothing new is invented:

  strategy        the same Foundation 1.0 reference rule (``ma_crossover`` with ``BASE_PARAMS``), on the protocol's
                  timeframe — parameters are in bars, exactly as the reference protocol states them
  grid, criteria, training/test windows, holdout fraction, 3,000-candle minimum
                  the reference protocol's values
  evidence floor  the existing AdversarialPolicy / PromotionPolicy defaults (the same objects the research
                  workflow uses)
  validation      the objective contract's required verdicts, then the promotion gate

A protocol's identity (``protocol_hash``) is the hash of every material requirement: changing any of them
produces a different identity, so a protocol can never be silently mutated under an old id. Research runs record
the id *and* hash they ran under (``protocol_run`` in the research journal).

Executability is a *state*, not a requirement (it is excluded from the identity hash). Since Phase 4B the
strategy registry keys definitions by ``strategy_id@vN/<timeframe>`` and scopes lifecycle/champion state by
research dimension, so every protocol resolves to its own timeframe deployment and all four are routable. That
says nothing about data: without REAL candles every protocol is still REAL_DATA_UNAVAILABLE.

Holdout: protocols state that a sealed holdout is required and how large it is. They never carry holdout
identities, candles, outcomes or metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ati.core.canonical import sha256_hex
from ati.market.models import MARKET_EVIDENCE_STATUSES, Timeframe
from ati.research import protocol as P
from ati.research.adversarial import AdversarialPolicy
from ati.strategies.base import StrategyDefinition
from ati.validation.promotion import PromotionPolicy

_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)   # creation time is excluded from definition hashes


@dataclass(frozen=True)
class ResearchProtocol:
    protocol_id: str
    hypothesis_id: str
    statement: str
    symbol: str
    timeframe: Timeframe
    strategy_id: str
    strategy_version: int
    strategy_kind: str
    base_params: tuple[tuple[str, object], ...]
    grid: tuple[tuple[tuple[str, object], ...], ...]
    criteria: tuple[tuple[str, str, float], ...]
    min_candles: int
    holdout_fraction: float
    train_bars: int
    test_bars: int
    min_oos_trades: int                  # evidence floor: development out-of-sample trades
    min_holdout_trades: int              # evidence floor: holdout trades
    min_positive_fold_fraction: float
    validation_stages: tuple[str, ...]
    permitted_provenance: tuple[str, ...]
    # state, not identity
    executable: bool = field(default=False, compare=False)
    not_executable_reason: str = field(default="", compare=False)

    @property
    def strategy(self) -> StrategyDefinition:
        return StrategyDefinition.create(self.strategy_id, self.strategy_version, self.strategy_kind,
                                         dict(self.base_params), self.timeframe, _EPOCH,
                                         description=f"{self.protocol_id} reference strategy")

    @property
    def strategy_fingerprint(self) -> str:
        return self.strategy.definition_hash

    @property
    def protocol_hash(self) -> str:
        material = {k: getattr(self, k) for k in self.__dataclass_fields__
                    if k not in ("executable", "not_executable_reason")}
        return sha256_hex(material | {"strategy_fingerprint": self.strategy_fingerprint})

    def describe(self) -> dict:
        """Read-only metadata (no holdout identities, candles, outcomes or metrics)."""
        return {"protocol_id": self.protocol_id, "protocol_hash": self.protocol_hash, "symbol": self.symbol,
                "timeframe": self.timeframe.value, "hypothesis_id": self.hypothesis_id, "statement": self.statement,
                "strategy": {"strategy_id": self.strategy_id, "version": self.strategy_version, "kind": self.strategy_kind,
                             "params": dict(self.base_params), "fingerprint": self.strategy_fingerprint,
                             "registry_key": self.strategy.registry_key,
                             "behavior_fingerprint": self.strategy.behavior_fingerprint,
                             "code_hash": self.strategy.code_hash},
                "min_real_candles": self.min_candles,
                "development_holdout": {"holdout_fraction": self.holdout_fraction, "train_bars": self.train_bars,
                                        "test_bars": self.test_bars, "sealed_holdout_required": True},
                "evidence_floor": {"min_oos_trades": self.min_oos_trades, "min_holdout_trades": self.min_holdout_trades,
                                   "min_positive_fold_fraction": self.min_positive_fold_fraction},
                "success_criteria": [list(c) for c in self.criteria],
                "validation_stages": list(self.validation_stages), "permitted_provenance": list(self.permitted_provenance),
                "executable": self.executable, "not_executable_reason": self.not_executable_reason or None,
                "not_ready_when": ["no REAL candles (REAL_DATA_UNAVAILABLE)",
                                   f"fewer than {self.min_candles} contiguous, closed, unsealed, conflict-free REAL "
                                   "candles (INSUFFICIENT_REAL_CANDLES)",
                                   "series health FAIL or BLOCKED (gaps, conflicts, mixed provenance, mutated holdout)"]}


def _protocol(protocol_id: str, hypothesis_id: str, symbol: str, timeframe: Timeframe, executable: bool,
              reason: str = "") -> ResearchProtocol:
    adv, promo = AdversarialPolicy(), PromotionPolicy()
    from ati.company.objectives import CONTRACT

    return ResearchProtocol(
        protocol_id=protocol_id, hypothesis_id=hypothesis_id, statement=P.STATEMENT, symbol=symbol, timeframe=timeframe,
        strategy_id="trend", strategy_version=1, strategy_kind="ma_crossover",
        base_params=tuple(sorted(P.BASE_PARAMS.items())), grid=tuple(tuple(sorted(g.items())) for g in P.GRID),
        criteria=tuple((c.metric, c.op, c.threshold) for c in P.CRITERIA), min_candles=P.MIN_CANDLES,
        holdout_fraction=P.HOLDOUT_FRACTION, train_bars=P.TRAIN_BARS, test_bars=P.TEST_BARS,
        min_oos_trades=max(adv.min_oos_trades, promo.min_oos_trades), min_holdout_trades=promo.min_holdout_trades,
        min_positive_fold_fraction=promo.min_positive_fold_fraction,
        validation_stages=tuple(CONTRACT.required_verdicts) + ("promotion_gate:APPROVED",),
        permitted_provenance=tuple(sorted(s.value for s in MARKET_EVIDENCE_STATUSES)),
        executable=executable, not_executable_reason=reason)


REGISTRY: dict[str, ResearchProtocol] = {p.protocol_id: p for p in (
    _protocol(P.PROTOCOL_ID, P.HYPOTHESIS_ID, P.SYMBOL, P.TIMEFRAME, True),
    _protocol("REAL-PROTOCOL-002", "H-trend-real-002", "BTC/USD", Timeframe.H4, True),
    _protocol("REAL-PROTOCOL-003", "H-trend-real-003", "ETH/USD", Timeframe.H1, True),
    _protocol("REAL-PROTOCOL-004", "H-trend-real-004", "ETH/USD", Timeframe.H4, True),
)}


def deployment(protocol: ResearchProtocol) -> dict:
    """Research deployment identity: the strategy on this protocol's timeframe, evaluated on its symbol."""
    s = protocol.strategy
    return {"registry_key": s.registry_key, "strategy_fingerprint": s.definition_hash,
            "behavior_fingerprint": s.behavior_fingerprint, "symbol": protocol.symbol,
            "timeframe": protocol.timeframe.value, "protocol_id": protocol.protocol_id,
            "protocol_hash": protocol.protocol_hash}


def for_series(symbol: str, timeframe: Timeframe) -> ResearchProtocol | None:
    return next((p for p in REGISTRY.values() if (p.symbol, p.timeframe) == (symbol, timeframe)), None)


def identity_mismatches(protocol: ResearchProtocol, symbol: str, timeframe, strategy: StrategyDefinition) -> list[str]:
    """Identity layer only: does (symbol, timeframe, strategy deployment) belong to this protocol? Evidence
    (provenance, candle count, health, holdout) is checked separately by ``compatibility``."""
    reasons: list[str] = []
    if symbol != protocol.symbol:
        reasons.append(f"symbol {symbol} ≠ protocol {protocol.symbol}")
    if getattr(timeframe, "value", None) is None or timeframe is not protocol.timeframe:
        reasons.append(f"timeframe {getattr(timeframe, 'value', timeframe)} ≠ protocol {protocol.timeframe.value}")
    expected = protocol.strategy
    if (strategy.strategy_id, strategy.version, strategy.kind, strategy.params, strategy.timeframe) != \
            (expected.strategy_id, expected.version, expected.kind, expected.params, expected.timeframe):
        reasons.append("strategy deployment does not match the protocol (identity or timeframe differs)")
    elif strategy.code_hash != expected.code_hash or strategy.definition_hash != protocol.strategy_fingerprint:
        reasons.append("strategy fingerprint does not match the protocol (logic source or lineage differs)")
    return reasons


def compatibility(protocol: ResearchProtocol, dataset, strategy: StrategyDefinition, system) -> list[str]:
    """Reasons ``dataset`` + ``strategy`` cannot satisfy ``protocol`` (empty = compatible). Provenance is proven
    against the payload archive — the dataset's own label, or any caller string, is never taken as proof."""
    from ati.market.health import series_health, verify_sealed_holdouts

    ident = dataset.identity
    reasons: list[str] = identity_mismatches(protocol, ident.symbol, ident.timeframe, strategy)
    if ident.status.value not in protocol.permitted_provenance:
        reasons.append(f"provenance {ident.status.value} is not permitted (requires {list(protocol.permitted_provenance)})")
    else:
        try:
            system.archive.verify_market_provenance(dataset)      # the archive, not the label, is the proof
        except Exception as exc:
            reasons.append(f"provenance not proven by the payload archive: {exc}"[:200])
    if ident.gaps:
        reasons.append(f"dataset has {ident.gaps} missing interval(s)")
    if ident.n_candles < protocol.min_candles:
        reasons.append(f"{ident.n_candles} candles < {protocol.min_candles}")
    if ident.symbol == protocol.symbol and ident.timeframe is protocol.timeframe:
        health = series_health(system, ident.symbol, ident.timeframe)
        if health["state"] in ("FAIL", "BLOCKED"):
            reasons.append(f"series health {health['state']}")
    if any(h["state"] == "MUTATED" for h in verify_sealed_holdouts(system)):
        reasons.append("a sealed holdout no longer matches its commitment")
    try:
        dataset.require_not_holdout("protocol compatibility")
    except Exception as exc:
        reasons.append(f"dataset overlaps a sealed holdout: {exc}"[:200])
    return reasons
