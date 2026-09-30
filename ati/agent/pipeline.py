"""Decision pipeline for one trade candidate.

    champion signal (deterministic, point-in-time)
      → Claude primary decision  (untrusted text → schema-validated TradeProposal | NoTrade | ResearchRequest)
      → Claude adversarial review (can only BLOCK or annotate)
      → deterministic RiskEngine (sizes or rejects; signs approvals)
      → DecisionRecord journaled (always, before any order)
      → ExecutionEngine.submit (only with a signed approval)

Claude cannot loosen anything deterministic: for longs the effective stop is the *higher* (tighter)
of Claude's stop and the strategy's stop; Claude's quantity can only lower the risk engine's size;
an invalid, pending or over-budget reasoning step results in no trade. Exits are deterministic and
never wait for reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from ati.agent.reasoning import ReasoningBudgetExceeded, ReasoningClient, ReasoningPending
from ati.agent.roles import ADVERSARIAL, PRIMARY, Tier, build_prompt
from ati.agent.schema import (NoTrade, ResearchRequest, TradeProposal, ValidationContext, parse_decision_output,
                              parse_review_output)
from ati.core.errors import LookaheadError, SchemaViolation
from ati.core.time import Clock, to_iso
from ati.core.types import Side
from ati.decision.records import DecisionLog, DecisionRecord, FinalDecision, make_decision_id
from ati.execution.broker import Order
from ati.execution.engine import ExecutionEngine
from ati.market.universe import Universe
from ati.memory.evidence import EvidenceRegistry
from ati.memory.store import MemoryKind, MemoryStore
from ati.risk.engine import MarketSnapshot, OrderRequest, PortfolioSnapshot, RiskEngine, RiskVerdict
from ati.security.secrets import SecretGuard
from ati.security.untrusted import UntrustedText
from ati.strategies.base import Signal, StrategyDefinition
from ati.temporal.features import atr, sma
from ati.temporal.pit import PointInTimeView


class Outcome(str, Enum):
    ALREADY_DECIDED = "ALREADY_DECIDED"
    PENDING_REASONING = "PENDING_REASONING"
    RECORDED = "RECORDED"


@dataclass(frozen=True)
class PipelineResult:
    outcome: Outcome
    decision: DecisionRecord | None = None
    verdict: RiskVerdict | None = None
    order: Order | None = None


class DecisionPipeline:
    def __init__(self, *, clock: Clock, universe: Universe, risk: RiskEngine, execution: ExecutionEngine,
                 decisions: DecisionLog, evidence: EvidenceRegistry, memory: MemoryStore, reasoning: ReasoningClient,
                 guard: SecretGuard):
        self.clock = clock
        self.universe = universe
        self.risk = risk
        self.execution = execution
        self.decisions = decisions
        self.evidence = evidence
        self.memory = memory
        self.reasoning = reasoning
        self.guard = guard

    # --- context packet -----------------------------------------------------------------------
    def packet(self, strategy: StrategyDefinition, view: PointInTimeView, signal: Signal, pf: PortfolioSnapshot,
               mkt: MarketSnapshot) -> dict:
        cutoff = view.cutoff
        params = strategy.param_dict
        mem = self.memory.query(cutoff)
        return {
            "mode": pf.mode.value,
            "data_status": mkt.status.value,
            "notice": f"{pf.mode.value} trading on {mkt.status.value} data",
            "cutoff": to_iso(cutoff),
            "symbol": mkt.symbol,
            "timeframe": mkt.timeframe.value,
            "recent_closes": [str(c) for c in view.closes(20)],
            "features": {
                "sma_fast": str(sma(view, params["fast"])) if "fast" in params else None,
                "sma_slow": str(sma(view, params["slow"])) if "slow" in params else None,
                "atr": str(atr(view, params.get("atr_period", 14))),
            },
            "strategy": {"key": strategy.key, "hash": strategy.definition_hash, "params": params},
            "signal": {"target": signal.target.value, "stop": str(signal.stop_price), "reason": signal.reason},
            "portfolio": {"equity": str(pf.equity), "cash": str(pf.cash), "position": str(pf.qty(mkt.symbol))},
            "risk_limits": {"max_risk_per_trade_fraction": str(self.risk.limits.max_risk_per_trade_fraction),
                            "note": "sizing is computed by the deterministic risk engine"},
            "memory": [{"kind": m.kind.value, "statement": m.statement, "confidence": m.confidence}
                       for m in mem[-20:]],
            "prior_rejections": [m.statement for m in mem if m.kind is MemoryKind.REJECTED_HYPOTHESIS][-10:],
            "evidence_ids": [],
        }

    # --- entry ------------------------------------------------------------------------------------
    def entry(self, strategy: StrategyDefinition, view: PointInTimeView, pf: PortfolioSnapshot, mkt: MarketSnapshot,
              untrusted: list[UntrustedText] | None = None) -> PipelineResult:
        cutoff = view.cutoff
        signal = strategy.signal(view, pf.qty(mkt.symbol) > 0)
        decision_id = make_decision_id(strategy.definition_hash, mkt.symbol, cutoff, "entry")
        if decision_id in self.decisions:
            return PipelineResult(Outcome.ALREADY_DECIDED)
        base = dict(decision_id=decision_id, strategy=strategy, cutoff=cutoff, signal=signal, mkt=mkt, pf=pf)
        packet = self.packet(strategy, view, signal, pf, mkt)
        ctx = ValidationContext(
            universe=self.universe, allowed_strategy_keys=frozenset({strategy.key}),
            last_prices={mkt.symbol: mkt.last_price},
            evidence_exists=lambda r: r in self.evidence and self.evidence.resolve(r).available_at <= cutoff,
            account_state_known=pf.account_state_known)
        try:
            raw = self.reasoning.complete(PRIMARY.name, decision_id, build_prompt(PRIMARY, packet, untrusted, self.guard))
        except ReasoningPending:
            return PipelineResult(Outcome.PENDING_REASONING)
        except ReasoningBudgetExceeded as exc:
            return self._record(**base, final=FinalDecision.NO_TRADE, reason=f"reasoning unavailable: {exc}")
        try:
            out = parse_decision_output(raw, ctx)
        except SchemaViolation as exc:
            return self._record(**base, final=FinalDecision.INVALID_REASONING_OUTPUT, reason=str(exc)[:600])
        if isinstance(out, NoTrade):
            return self._record(**base, final=FinalDecision.NO_TRADE, reason=out.reason, confidence=out.confidence,
                                refs=out.evidence_refs)
        if isinstance(out, ResearchRequest):
            return self._record(**base, final=FinalDecision.NO_TRADE, reason=f"research requested: {out.question}"[:600])
        assert isinstance(out, TradeProposal)
        return self._route(strategy, view, pf, mkt, out, untrusted, decision_id, signal, packet, base)

    def route_proposal(self, strategy: StrategyDefinition, view: PointInTimeView, pf: PortfolioSnapshot,
                       mkt: MarketSnapshot, proposal: TradeProposal,
                       untrusted: list[UntrustedText] | None = None) -> PipelineResult:
        """Route an already schema-validated proposal (e.g. a company TRADE_PROPOSAL) through exactly the
        same adversarial review → risk → decision record → execution path as ``entry``. Same deterministic
        decision id, so a proposal and an entry for the same information can never both execute."""
        cutoff = view.cutoff
        signal = strategy.signal(view, pf.qty(mkt.symbol) > 0)
        decision_id = make_decision_id(strategy.definition_hash, mkt.symbol, cutoff, "entry")
        if decision_id in self.decisions:
            return PipelineResult(Outcome.ALREADY_DECIDED)
        base = dict(decision_id=decision_id, strategy=strategy, cutoff=cutoff, signal=signal, mkt=mkt, pf=pf)
        packet = self.packet(strategy, view, signal, pf, mkt)
        return self._route(strategy, view, pf, mkt, proposal, untrusted, decision_id, signal, packet, base)

    def _route(self, strategy, view, pf, mkt, out: TradeProposal, untrusted, decision_id, signal, packet,
               base) -> PipelineResult:
        if out.symbol != mkt.symbol:
            return self._record(**base, final=FinalDecision.INVALID_REASONING_OUTPUT,
                                reason=f"proposal symbol {out.symbol} differs from candidate {mkt.symbol}")
        stop = out.stop_price
        if out.side is Side.BUY and signal.stop_price is not None:
            stop = max(stop, signal.stop_price)  # Claude may tighten, never loosen

        review_packet = packet | {"proposal": {"side": out.side.value, "entry": str(out.entry_price), "stop": str(stop),
                                               "thesis": out.thesis, "invalidation": out.invalidation_condition,
                                               "confidence": out.confidence}}
        try:
            review = parse_review_output(self.reasoning.complete(
                ADVERSARIAL.name, decision_id, build_prompt(ADVERSARIAL, review_packet, untrusted, self.guard)))
        except ReasoningPending:
            return PipelineResult(Outcome.PENDING_REASONING)
        except (SchemaViolation, ReasoningBudgetExceeded) as exc:
            return self._record(**base, final=FinalDecision.INVALID_REASONING_OUTPUT, proposal=out, stop=stop,
                                reason=f"adversarial review unusable: {exc}"[:600])
        if review.verdict == "BLOCK":
            return self._record(**base, final=FinalDecision.BLOCKED_BY_REVIEW, proposal=out, stop=stop, review=review,
                                reason="blocked by adversarial review")

        request = OrderRequest(decision_id, strategy.key, mkt.symbol, out.side, out.entry_price, stop, out.proposed_qty)
        verdict = self.risk.evaluate(request, pf, mkt)
        final = FinalDecision.EXECUTE if verdict.approved else FinalDecision.REJECTED_BY_RISK
        reason = "risk approved" if verdict.approved else "; ".join(f"{c.name}: {c.detail}" for c in verdict.failed)[:600]
        result = self._record(**base, final=final, proposal=out, stop=stop, review=review, verdict=verdict,
                              reason=reason, confidence=out.confidence, refs=out.evidence_refs)
        if result.decision is not None and result.decision.final_decision is FinalDecision.EXECUTE:
            order = self.execution.submit(verdict)
            return PipelineResult(Outcome.RECORDED, result.decision, verdict, order)
        return result

    # --- deterministic exit -----------------------------------------------------------------------
    def exit(self, strategy_key: str, strategy_hash: str, why: str, cutoff, pf: PortfolioSnapshot,
             mkt: MarketSnapshot) -> PipelineResult:
        decision_id = make_decision_id(strategy_hash, mkt.symbol, cutoff, f"exit:{why}")
        if decision_id in self.decisions:
            return PipelineResult(Outcome.ALREADY_DECIDED)
        request = OrderRequest(decision_id, strategy_key, mkt.symbol, Side.SELL, mkt.last_price, None, None)
        verdict = self.risk.evaluate(request, pf, mkt)
        rec = DecisionRecord(
            decision_id=decision_id, timestamp=self.clock.now(), symbol=mkt.symbol,
            strategy_id=strategy_key.split("@")[0], strategy_version=int(strategy_key.split("@v")[1]),
            strategy_hash=strategy_hash, mode=pf.mode.value, data_status=mkt.status.value,
            market_context=self._context(mkt), available_information_cutoff=cutoff, signal=f"EXIT ({why})",
            thesis=f"deterministic exit: {why}", invalidation_condition="not applicable: risk-reducing exit",
            expected_risk=Decimal(0), expected_reward=None, estimated_cost=self._cost(mkt, verdict.qty),
            proposed_size=None, approved_size=verdict.qty, risk_constraints=verdict.checks,
            risk_limits_hash=verdict.limits_hash, evidence_references=(), research_references=(), confidence=None,
            reasoning_tier=Tier.NONE.value, adversarial_objections=(), adversarial_verdict="NOT_APPLICABLE",
            final_decision=FinalDecision.EXECUTE if verdict.approved else FinalDecision.REJECTED_BY_RISK,
            reason="risk approved exit" if verdict.approved else "; ".join(c.detail for c in verdict.failed)[:600])
        self.decisions.record(rec)
        order = self.execution.submit(verdict) if verdict.approved else None
        return PipelineResult(Outcome.RECORDED, rec, verdict, order)

    # --- helpers ------------------------------------------------------------------------------------
    @staticmethod
    def _context(mkt: MarketSnapshot) -> tuple[tuple[str, str], ...]:
        return (("last_price", str(mkt.last_price)), ("last_close_time", to_iso(mkt.last_close_time)),
                ("recent_volume", str(mkt.recent_volume)), ("est_spread_rate", str(mkt.est_spread_rate)),
                ("est_slippage_rate", str(mkt.est_slippage_rate)))

    def _cost(self, mkt: MarketSnapshot, qty: Decimal) -> Decimal:
        return qty * mkt.last_price * (self.risk.limits.fee_rate + mkt.est_spread_rate + mkt.est_slippage_rate)

    def _record(self, *, decision_id, strategy: StrategyDefinition, cutoff, signal: Signal, mkt: MarketSnapshot,
                pf: PortfolioSnapshot, final: FinalDecision, reason: str, proposal: TradeProposal | None = None,
                stop=None, review=None, verdict: RiskVerdict | None = None, confidence=None,
                refs: tuple[str, ...] = ()) -> PipelineResult:
        evidence = tuple(self.evidence.resolve(r) for r in refs if r in self.evidence)
        entry = proposal.entry_price if proposal else mkt.last_price
        qty = verdict.qty if verdict else Decimal(0)
        try:
            rec = DecisionRecord(
                decision_id=decision_id, timestamp=self.clock.now(), symbol=mkt.symbol,
                strategy_id=strategy.strategy_id, strategy_version=strategy.version, strategy_hash=strategy.definition_hash,
                mode=pf.mode.value, data_status=mkt.status.value, market_context=self._context(mkt),
                available_information_cutoff=cutoff,
                signal=f"{signal.target.value} stop={signal.stop_price} ({signal.reason})"[:600],
                thesis=proposal.thesis if proposal else "", invalidation_condition=proposal.invalidation_condition if proposal else "",
                expected_risk=verdict.max_loss if verdict else Decimal(0),
                expected_reward=((proposal.target_price - entry) * qty) if proposal and proposal.target_price else None,
                estimated_cost=self._cost(mkt, qty), proposed_size=proposal.proposed_qty if proposal else None,
                approved_size=qty, risk_constraints=verdict.checks if verdict else (),
                risk_limits_hash=verdict.limits_hash if verdict else self.risk.limits.limits_hash,
                evidence_references=evidence, research_references=(strategy.definition_hash,),
                confidence=confidence, reasoning_tier=Tier.DEEP.value,
                adversarial_objections=review.objections if review else (),
                adversarial_verdict=review.verdict if review else "NOT_REACHED",
                final_decision=final, reason=reason)
        except (LookaheadError, ValueError) as exc:
            rec = DecisionRecord(
                decision_id=decision_id, timestamp=self.clock.now(), symbol=mkt.symbol, strategy_id=strategy.strategy_id,
                strategy_version=strategy.version, strategy_hash=strategy.definition_hash, mode=pf.mode.value,
                data_status=mkt.status.value, market_context=self._context(mkt), available_information_cutoff=cutoff,
                signal=f"{signal.target.value}", thesis="", invalidation_condition="", expected_risk=Decimal(0),
                expected_reward=None, estimated_cost=Decimal(0), proposed_size=None, approved_size=Decimal(0),
                risk_constraints=(), risk_limits_hash=self.risk.limits.limits_hash, evidence_references=(),
                research_references=(), confidence=None, reasoning_tier=Tier.DEEP.value, adversarial_objections=(),
                adversarial_verdict="NOT_REACHED", final_decision=FinalDecision.INVALID_REASONING_OUTPUT,
                reason=f"decision record rejected: {exc}"[:600])
            verdict = None
        self.decisions.record(rec)
        return PipelineResult(Outcome.RECORDED, rec, verdict if rec.final_decision is FinalDecision.EXECUTE else None)
