"""Claude output validation.

Claude's output is untrusted text until it is parsed into one of a closed set of typed results.
There is no path from Claude text to a command: unknown keys, unknown actions and any field outside
the schema are rejected, and nothing in a parsed result is ever executed — it is only passed to the
deterministic risk engine as a request.

Accepted envelope: exactly one JSON object, optionally inside a single ```json fence. Max 16 KiB.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from ati.core.errors import SchemaViolation
from ati.core.types import Side
from ati.market.universe import Universe

MAX_BYTES = 16 * 1024
MAX_TEXT = 600
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*)\n\s*```\s*$", re.DOTALL)

ACTIONS = frozenset({"PROPOSE_TRADE", "NO_TRADE", "REQUEST_RESEARCH"})
_KEYS = {
    "PROPOSE_TRADE": ({"action", "symbol", "side", "strategy_key", "entry_price", "stop_price", "thesis",
                       "invalidation_condition", "confidence"},
                      {"proposed_qty", "target_price", "evidence_refs"}),
    "NO_TRADE": ({"action", "reason"}, {"confidence", "evidence_refs"}),
    "REQUEST_RESEARCH": ({"action", "question"}, {"rationale"}),
}


@dataclass(frozen=True)
class TradeProposal:
    symbol: str
    side: Side
    strategy_key: str
    entry_price: Decimal
    stop_price: Decimal | None
    proposed_qty: Decimal | None
    target_price: Decimal | None
    thesis: str
    invalidation_condition: str
    confidence: float
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class NoTrade:
    reason: str
    confidence: float | None
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class ResearchRequest:
    question: str
    rationale: str


@dataclass(frozen=True)
class ReviewResult:
    verdict: str  # NO_OBJECTION | CHALLENGE | BLOCK
    objections: tuple[str, ...]


@dataclass(frozen=True)
class PostTradeReview:
    process_quality: str  # GOOD_PROCESS | BAD_PROCESS | INSUFFICIENT_INFORMATION
    notes: str
    possible_mistake: str | None


@dataclass(frozen=True)
class ValidationContext:
    universe: Universe
    allowed_strategy_keys: frozenset[str]
    last_prices: dict[str, Decimal]
    evidence_exists: Callable[[str], bool]
    account_state_known: bool
    max_price_deviation: Decimal = Decimal("0.20")


def load_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise SchemaViolation("output must be text")
    if len(raw.encode("utf-8")) > MAX_BYTES:
        raise SchemaViolation("output too large")
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        obj = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(SchemaViolation(f"non-finite {c}")))
    except SchemaViolation:
        raise
    except ValueError as exc:
        raise SchemaViolation(f"malformed JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise SchemaViolation("output must be a JSON object")
    return obj


def _text(obj: dict, key: str, required: bool = True, limit: int = MAX_TEXT) -> str:
    value = obj.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise SchemaViolation(f"{key} must be a non-empty string")
    if len(value) > limit:
        raise SchemaViolation(f"{key} exceeds {limit} characters")
    return value.strip()


def _decimal(obj: dict, key: str, required: bool = True) -> Decimal | None:
    value = obj.get(key)
    if value is None:
        if required:
            raise SchemaViolation(f"{key} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise SchemaViolation(f"{key} must be a number or numeric string")
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaViolation(f"{key} is not finite")
    try:
        d = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise SchemaViolation(f"{key} is not numeric") from exc
    if not d.is_finite() or d <= 0:
        raise SchemaViolation(f"{key} must be positive and finite")
    return d


def _confidence(obj: dict, required: bool) -> float | None:
    value = obj.get("confidence")
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 <= value <= 1):
        raise SchemaViolation("confidence must be a number in [0, 1]")
    return float(value)


def _refs(obj: dict, ctx: ValidationContext) -> tuple[str, ...]:
    refs = obj.get("evidence_refs", [])
    if not isinstance(refs, list) or len(refs) > 20 or not all(isinstance(r, str) for r in refs):
        raise SchemaViolation("evidence_refs must be a list of up to 20 strings")
    missing = [r for r in refs if not ctx.evidence_exists(r)]
    if missing:
        raise SchemaViolation(f"unregistered evidence cited: {missing[:3]}")
    return tuple(refs)


def _check_keys(obj: dict, action: str) -> None:
    required, optional = _KEYS[action]
    missing = required - obj.keys()
    extra = obj.keys() - required - optional
    if missing:
        raise SchemaViolation(f"{action}: missing required fields {sorted(missing)}")
    if extra:
        raise SchemaViolation(f"{action}: unrecognized fields {sorted(extra)}")


def parse_decision_output(raw: str, ctx: ValidationContext) -> TradeProposal | NoTrade | ResearchRequest:
    obj = load_json_object(raw)
    action = obj.get("action")
    if action not in ACTIONS:
        raise SchemaViolation(f"unrecognized action {action!r}")
    _check_keys(obj, action)
    if action == "NO_TRADE":
        return NoTrade(_text(obj, "reason"), _confidence(obj, False), _refs(obj, ctx))
    if action == "REQUEST_RESEARCH":
        return ResearchRequest(_text(obj, "question"), _text(obj, "rationale", required=False))

    if not ctx.account_state_known:
        raise SchemaViolation("trade proposals are refused while account state is unknown")
    symbol = obj["symbol"]
    if symbol not in ctx.universe:
        raise SchemaViolation(f"unknown symbol {symbol!r}")
    try:
        side = Side(obj["side"])
    except ValueError:
        raise SchemaViolation(f"invalid side {obj['side']!r}") from None
    strategy_key = obj["strategy_key"]
    if strategy_key not in ctx.allowed_strategy_keys:
        raise SchemaViolation(f"strategy {strategy_key!r} is not active for trading")
    entry = _decimal(obj, "entry_price")
    last = ctx.last_prices.get(symbol)
    if last is None:
        raise SchemaViolation(f"no current price for {symbol}")
    if abs(entry / last - 1) > ctx.max_price_deviation:
        raise SchemaViolation(f"impossible entry price {entry} vs last {last}")
    stop = _decimal(obj, "stop_price", required=(side is Side.BUY))
    if side is Side.BUY and stop >= entry:
        raise SchemaViolation("long stop must be below entry")
    target = _decimal(obj, "target_price", required=False)
    qty = _decimal(obj, "proposed_qty", required=False)
    return TradeProposal(symbol, side, strategy_key, entry, stop, qty, target, _text(obj, "thesis"),
                         _text(obj, "invalidation_condition"), _confidence(obj, True), _refs(obj, ctx))


def parse_review_output(raw: str) -> ReviewResult:
    obj = load_json_object(raw)
    if obj.keys() != {"verdict", "objections"}:
        raise SchemaViolation("review must contain exactly verdict and objections")
    if obj["verdict"] not in ("NO_OBJECTION", "CHALLENGE", "BLOCK"):
        raise SchemaViolation("invalid review verdict")
    objections = obj["objections"]
    if not isinstance(objections, list) or len(objections) > 10:
        raise SchemaViolation("objections must be a list of at most 10 strings")
    for o in objections:
        if not isinstance(o, str) or not o.strip() or len(o) > 300:
            raise SchemaViolation("each objection must be a 1..300 character string")
    return ReviewResult(obj["verdict"], tuple(o.strip() for o in objections))


def parse_post_trade_output(raw: str) -> PostTradeReview:
    obj = load_json_object(raw)
    if not obj.keys() <= {"process_quality", "notes", "possible_mistake"} or "process_quality" not in obj:
        raise SchemaViolation("post-trade review fields invalid")
    if obj["process_quality"] not in ("GOOD_PROCESS", "BAD_PROCESS", "INSUFFICIENT_INFORMATION"):
        raise SchemaViolation("invalid process_quality")
    mistake = obj.get("possible_mistake")
    if mistake is not None and (not isinstance(mistake, str) or len(mistake) > 300):
        raise SchemaViolation("possible_mistake must be a short string or null")
    return PostTradeReview(obj["process_quality"], _text(obj, "notes", required=False), mistake or None)
