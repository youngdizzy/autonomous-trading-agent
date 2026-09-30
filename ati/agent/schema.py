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
from dataclasses import dataclass, field
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


# --- Company control-plane contract (Company 1.0 Phase 1) ------------------------------------------------
#
# Envelope (exactly these keys): {"request_id", "cycle_id", "action", "reason", "payload"}.
# The response is bound to one request/cycle; the action is one of a closed vocabulary; each action has a
# closed payload. TRADE_PROPOSAL payloads are validated by the existing trade validator above. Free text is
# data, and text that looks executable (shell, code, filesystem paths) is refused outright.

COMPANY_ACTIONS = ("NO_TRADE", "TRADE_PROPOSAL", "RESEARCH_REQUEST", "PAUSE", "REQUEST_DATA",
                   "REVIEW_POSITION", "REVIEW_RISK", "REVIEW_SYSTEM")
READ_ONLY_ACTIONS = frozenset({"NO_TRADE", "REQUEST_DATA", "REVIEW_POSITION", "REVIEW_RISK", "REVIEW_SYSTEM"})
_COMPANY_ENVELOPE = frozenset({"request_id", "cycle_id", "action", "reason", "payload"})
_COMPANY_PAYLOAD = {
    "NO_TRADE": (frozenset(), frozenset()),
    "TRADE_PROPOSAL": (frozenset({"symbol", "side", "strategy_key", "entry_price", "stop_price", "thesis",
                                  "invalidation_condition", "confidence"}),
                       frozenset({"proposed_qty", "target_price", "evidence_refs"})),
    "RESEARCH_REQUEST": (frozenset({"hypothesis_id", "protocol_id", "question", "statement", "strategy_key",
                                    "evidence_requested", "success_criteria"}),
                         frozenset({"scope", "motivation", "expected_mechanism", "evidence_refs",
                                    "learning_candidate_id", "experiment"})),
    "PAUSE": (frozenset(), frozenset()),
    "REQUEST_DATA": (frozenset({"need"}), frozenset({"symbol"})),
    "REVIEW_POSITION": (frozenset(), frozenset({"symbol"})),
    "REVIEW_RISK": (frozenset(), frozenset()),
    "REVIEW_SYSTEM": (frozenset(), frozenset()),
}
RESEARCH_EVIDENCE_TYPES = frozenset({"walk_forward", "robustness", "adversarial", "holdout"})
# Experiment design (optional part of a RESEARCH_REQUEST). All six types are representable; which ones the
# control plane can *execute* is decided there (unimplemented executors are BLOCKED, never approximated).
EXPERIMENT_TYPES = frozenset({"SINGLE_VARIABLE", "INTERACTION", "STRUCTURAL", "REGIME", "EXECUTION", "RISK"})
_EXPERIMENT_FIELDS = frozenset({"type", "independent_variables", "dependent_variable", "controls",
                                "failure_criteria", "stopping_criteria"})
_EXPERIMENT_OPTIONAL = frozenset({"candidate_params"})
# Evidence a hypothesis may cite. Holdout and promotion outcomes are evaluation results: citing them as
# motivation for a new hypothesis would turn the holdout into training feedback.
_NON_CITABLE_EVIDENCE = frozenset({"holdout", "promotion"})
_CANDIDATE_ID = re.compile(r"^lc_[0-9a-f]{20}$")
_HYPOTHESIS_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")
_EXECUTABLE = [re.compile(p, re.IGNORECASE) for p in (
    r"`", r"\$\(", r"\$\{", r"\bsudo\b", r"\brm\s+-", r"\bchmod\b", r"\bcurl\s+\S", r"\bwget\s+\S",
    r"\bimport\s+[A-Za-z_]", r"\bfrom\s+[A-Za-z_.]+\s+import\b", r"\bsubprocess\b", r"\bos\.(system|popen|remove)",
    r"\beval\s*\(", r"\bexec\s*\(", r"__\w+__", r"\.\./", r"(^|[\s\"'=])/(etc|bin|usr|root|home|tmp|var|proc)/",
    r"#!\s*/", r"<script", r"\|\s*(sh|bash|python)\b", r";\s*(sh|bash|python|rm)\b")]


@dataclass(frozen=True)
class ResearchSpec:
    hypothesis_id: str
    protocol_id: str
    question: str
    statement: str
    strategy_key: str
    evidence_requested: tuple[str, ...]
    success_criteria: tuple[tuple[str, str, float], ...]
    scope: str
    motivation: str = ""
    expected_mechanism: str = ""
    evidence_refs: tuple[str, ...] = ()
    learning_candidate_id: str | None = None
    experiment: "ExperimentSpec | None" = None


@dataclass(frozen=True)
class ExperimentSpec:
    type: str
    independent_variables: tuple[str, ...]
    dependent_variable: str
    controls: str
    failure_criteria: str
    stopping_criteria: str
    candidate_params: tuple[tuple[str, object], ...] | None   # None = the protocol baseline itself


@dataclass(frozen=True)
class CompanyAction:
    action: str
    request_id: str
    cycle_id: str
    reason: str
    trade: TradeProposal | None = None
    research: ResearchSpec | None = None
    symbol: str | None = None
    need: str | None = None


@dataclass(frozen=True)
class CompanyContext:
    request_id: str
    cycle_id: str
    trade: ValidationContext
    protocols: dict[str, str]            # protocol_id → the strategy key it is locked to
    protocol_params: dict[str, dict] = field(default_factory=dict)   # protocol_id → baseline parameters
    evidence_kind: Callable[[str], str | None] = lambda ref: None    # ref → evidence kind (None = unknown)


def _refuse_executable(value, where: str = "response") -> None:
    if isinstance(value, str):
        for pattern in _EXECUTABLE:
            if pattern.search(value):
                raise SchemaViolation(f"{where}: executable/command-like or path content refused")
    elif isinstance(value, dict):
        for k, v in value.items():
            _refuse_executable(k, where)
            _refuse_executable(v, f"{where}.{k}")
    elif isinstance(value, list):
        for v in value:
            _refuse_executable(v, where)


def parse_company_response(raw: str, ctx: CompanyContext) -> CompanyAction:
    obj = load_json_object(raw)
    if set(obj) != _COMPANY_ENVELOPE:
        raise SchemaViolation(f"envelope must contain exactly {sorted(_COMPANY_ENVELOPE)}")
    if obj["request_id"] != ctx.request_id or obj["cycle_id"] != ctx.cycle_id:
        raise SchemaViolation("response is bound to a different request or cycle")
    action = obj["action"]
    if action not in COMPANY_ACTIONS:
        raise SchemaViolation(f"unrecognized company action {action!r}")
    payload = obj["payload"]
    if not isinstance(payload, dict):
        raise SchemaViolation("payload must be an object")
    required, optional = _COMPANY_PAYLOAD[action]
    missing, extra = required - payload.keys(), payload.keys() - required - optional
    if missing or extra:
        raise SchemaViolation(f"{action}: missing {sorted(missing)} / unrecognized {sorted(extra)} payload fields")
    _refuse_executable(obj)
    reason = _text(obj, "reason")
    common = dict(action=action, request_id=ctx.request_id, cycle_id=ctx.cycle_id, reason=reason)
    if action == "TRADE_PROPOSAL":
        proposal = parse_decision_output(json.dumps({"action": "PROPOSE_TRADE", **payload}), ctx.trade)
        return CompanyAction(**common, trade=proposal, symbol=proposal.symbol)
    if action == "RESEARCH_REQUEST":
        return CompanyAction(**common, research=_research_spec(payload, ctx))
    if action in ("REQUEST_DATA", "REVIEW_POSITION"):
        symbol = payload.get("symbol")
        if symbol is not None and symbol not in ctx.trade.universe:
            raise SchemaViolation(f"unknown symbol {symbol!r}")
        need = _text(payload, "need", limit=300) if action == "REQUEST_DATA" else None
        return CompanyAction(**common, symbol=symbol, need=need)
    return CompanyAction(**common)


def _research_spec(p: dict, ctx: CompanyContext) -> ResearchSpec:
    hid = p["hypothesis_id"]
    if not isinstance(hid, str) or not _HYPOTHESIS_ID.match(hid):
        raise SchemaViolation("hypothesis_id must be 1-48 of [A-Za-z0-9_-] (stages are system-assigned)")
    if p["protocol_id"] not in ctx.protocols:
        raise SchemaViolation(f"unknown research protocol {p['protocol_id']!r}")
    if p["strategy_key"] != ctx.protocols[p["protocol_id"]]:
        raise SchemaViolation("strategy_key does not match the protocol's locked strategy")
    ev = p["evidence_requested"]
    if not isinstance(ev, list) or not ev or not set(ev) <= RESEARCH_EVIDENCE_TYPES or len(set(ev)) != len(ev):
        raise SchemaViolation(f"evidence_requested must be distinct values from {sorted(RESEARCH_EVIDENCE_TYPES)}")
    crit = p["success_criteria"]
    if not isinstance(crit, list) or len(crit) > 8:
        raise SchemaViolation("success_criteria must be a list of at most 8 criteria")
    parsed = []
    for c in crit:
        if not isinstance(c, dict) or set(c) != {"metric", "op", "threshold"}:
            raise SchemaViolation("each criterion is exactly {metric, op, threshold}")
        t = c["threshold"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            raise SchemaViolation("criterion threshold must be a finite number")
        from ati.research.hypothesis import Criterion  # local import: research depends on nothing here
        try:
            Criterion(c["metric"], c["op"], float(t))
        except (ValueError, TypeError) as exc:
            raise SchemaViolation(f"invalid criterion: {exc}") from None
        parsed.append((c["metric"], c["op"], float(t)))
    refs = p.get("evidence_refs", [])
    if not isinstance(refs, list) or len(refs) > 12 or not all(isinstance(r, str) for r in refs) \
            or len(set(refs)) != len(refs):
        raise SchemaViolation("evidence_refs must be a list of at most 12 distinct strings")
    for r in refs:
        kind = ctx.evidence_kind(r)
        if kind is None or not ctx.trade.evidence_exists(r):
            raise SchemaViolation(f"evidence ref {r!r} does not resolve to point-in-time evidence")
        if kind in _NON_CITABLE_EVIDENCE:
            raise SchemaViolation(f"evidence ref {r!r} is {kind} evidence: holdout/promotion outcomes cannot "
                                  "motivate a new hypothesis")
    lc = p.get("learning_candidate_id")
    if lc is not None and (not isinstance(lc, str) or not _CANDIDATE_ID.match(lc)):
        raise SchemaViolation("learning_candidate_id must be a learning candidate id (lc_ + 20 hex)")
    experiment = _experiment(p["experiment"], ctx.protocol_params.get(p["protocol_id"], {})) \
        if "experiment" in p else None
    return ResearchSpec(hid, p["protocol_id"], _text(p, "question"), _text(p, "statement"), p["strategy_key"],
                        tuple(ev), tuple(parsed), _text(p, "scope", required=False, limit=300),
                        _text(p, "motivation", required=False, limit=400),
                        _text(p, "expected_mechanism", required=False, limit=400), tuple(refs), lc, experiment)


def _experiment(e, base: dict) -> ExperimentSpec:
    """Structure and internal consistency only. The executor decides what may run."""
    if not isinstance(e, dict):
        raise SchemaViolation("experiment must be an object")
    missing, extra = _EXPERIMENT_FIELDS - e.keys(), e.keys() - _EXPERIMENT_FIELDS - _EXPERIMENT_OPTIONAL
    if missing or extra:
        raise SchemaViolation(f"experiment: missing {sorted(missing)} / unrecognized {sorted(extra)} fields")
    kind = e["type"]
    if kind not in EXPERIMENT_TYPES:
        raise SchemaViolation(f"experiment type must be one of {sorted(EXPERIMENT_TYPES)}")
    iv = e["independent_variables"]
    if not isinstance(iv, list) or not iv or len(iv) > 4 or not all(isinstance(v, str) and len(v) <= 40 for v in iv) \
            or len(set(iv)) != len(iv):
        raise SchemaViolation("independent_variables must be 1-4 distinct short names")
    from ati.research.hypothesis import Criterion  # local import: research depends on nothing here
    dv = e["dependent_variable"]
    try:
        Criterion(dv, ">", 0.0)
    except (ValueError, TypeError):
        raise SchemaViolation(f"dependent_variable {dv!r} is not a recorded metric") from None
    texts = {k: _text(e, k, limit=300) for k in ("controls", "failure_criteria", "stopping_criteria")}
    params = None
    if "candidate_params" in e:
        cp = e["candidate_params"]
        if not isinstance(cp, dict) or not base or set(cp) != set(base):
            raise SchemaViolation("candidate_params must give every baseline parameter (and nothing else)")
        cp = dict(cp)
        for k, v in cp.items():
            if isinstance(base[k], float) and isinstance(v, int) and not isinstance(v, bool):
                cp[k] = v = float(v)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) \
                    or type(v) is not type(base[k]) or v <= 0:
                raise SchemaViolation(f"candidate_params.{k} must be a finite positive {type(base[k]).__name__}")
        changed = sorted(k for k in cp if cp[k] != base[k])
        if kind == "SINGLE_VARIABLE" and len(changed) != 1:
            raise SchemaViolation(f"SINGLE_VARIABLE changes exactly one parameter (changed: {changed})")
        if kind == "INTERACTION" and len(changed) < 2:
            raise SchemaViolation(f"INTERACTION changes at least two parameters (changed: {changed})")
        if kind in ("SINGLE_VARIABLE", "INTERACTION") and sorted(iv) != changed:
            raise SchemaViolation(f"independent_variables {sorted(iv)} must name exactly the changed parameters {changed}")
        if kind == "REGIME" and changed:
            raise SchemaViolation("REGIME experiments hold parameters fixed (the regime is the variable)")
        params = tuple(sorted(cp.items()))
    elif kind in ("SINGLE_VARIABLE", "INTERACTION"):
        raise SchemaViolation(f"{kind} experiments require candidate_params")
    return ExperimentSpec(kind, tuple(iv), dv, texts["controls"], texts["failure_criteria"], texts["stopping_criteria"],
                          params)
