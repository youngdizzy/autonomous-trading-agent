"""Normalization and compatibility: external strategy → (maybe) a TradeTown research candidate.

A candidate is COMPATIBLE only when a *mapping rule* proves that its trading rules are the rules of an existing
registered TradeTown logic kind. There is exactly one rule in version ``map-1``:

  ma_crossover (``ati.strategies.library.MovingAverageCrossover``): long while SMA(close, fast) > SMA(close, slow);
  exit when fast <= slow; stop = close − k·ATR, ratcheted upward while long, where TradeTown's ATR is the *simple
  mean* of true range. The recognised Pine form (inputs may replace the integer/float literals)::

      F = sma(close, a)          S = sma(close, b)          A = sma(tr(true), n)     (or sma(tr, n))
      T := strategy.position_size > 0 ? max(nz(T[1]), close - k * A) : close - k * A
      strategy.entry(ID, strategy.long, when = F > S)
      strategy.close(ID, when = F <= S)
      strategy.exit(X, from_entry = ID, stop = T)

Anything else is refused with an explicit state — the parser does not "clean up" a strategy until it fits:

  PARSE_FAILED                  the file does not follow the Vault format
  UNSAFE                        dynamic code execution, shell access or credential handling in the source
  UNSUPPORTED_LANGUAGE          no normalizer for the source language (v1 normalizes Pine Script only)
  INCOMPATIBLE                  not a strategy (indicator script / no entry rules) or parameters outside TradeTown's
                                bounds for the mapped logic
  UNSUPPORTED_EXECUTION_MODEL   shorts, pyramiding, fill-on-close / intrabar recalculation, raw orders
  UNSUPPORTED_MARKET            derivatives, or an asset outside BTC/ETH
  UNSUPPORTED_TIMEFRAME         timeframe not stated, not 1h/4h, or other-timeframe data requested
  UNSUPPORTED_INDICATOR         indicators with no TradeTown equivalent — including Pine ``atr`` (Wilder RMA), which
                                is *not* TradeTown's simple-mean ATR
  SEMANTICS_UNCERTAIN           statements or functions whose meaning the parser cannot establish
  PARTIALLY_COMPATIBLE          everything is representable in principle, but the rules differ from the mapping rule
  COMPATIBLE                    exact mapping-rule match

The primary state is the first failing check in that order; every failing reason is recorded. Only COMPATIBLE
yields TradeTown identity. The strategy description is metadata; the source block is the specification of record.
External execution assumptions (sizing, capital, commission, slippage) are recorded but never adopted: TradeTown's
own backtester, costs and risk sizing decide how rules are executed.
"""

from __future__ import annotations

import re

from ati.core.canonical import sha256_hex
from ati.intake.source import SourceArtifact
from ati.intake.vault import PARSER_VERSION, SECURITY_PATTERNS, UNSAFE_CATEGORIES, ParsedSpec, VaultParseError, parse

MAPPING_VERSION = "map-1"
STATES = ("PARSE_FAILED", "UNSAFE", "UNSUPPORTED_LANGUAGE", "INCOMPATIBLE", "UNSUPPORTED_EXECUTION_MODEL",
          "UNSUPPORTED_MARKET", "UNSUPPORTED_TIMEFRAME", "UNSUPPORTED_INDICATOR", "SEMANTICS_UNCERTAIN",
          "PARTIALLY_COMPATIBLE", "COMPATIBLE")
RESEARCHABLE = frozenset({"COMPATIBLE"})
TIMEFRAMES = {"1h": "1h", "4h": "4h"}
ASSETS = {"BTC": "BTC/USD", "ETH": "ETH/USD"}

KNOWN_INDICATORS = frozenset({"sma", "ema", "wma", "rma", "vwma", "hma", "alma", "swma", "linreg", "rsi", "macd",
                              "stoch", "atr", "bb", "bbw", "cci", "adx", "dmi", "mfi", "obv", "vwap", "supertrend",
                              "sar", "highest", "lowest", "highestbars", "lowestbars", "stdev", "variance", "change",
                              "mom", "roc", "pivothigh", "pivotlow", "percentrank", "correlation", "kc", "kcw",
                              "cmo", "cog", "dev", "tsi", "wpr", "williams_r", "median", "mode", "range", "cum",
                              "falling", "rising", "barssince", "valuewhen", "security", "tr"})
SUPPORTED_INDICATORS = frozenset({"sma", "tr"})       # tr only inside sma(tr, n) — TradeTown's simple-mean ATR
HELPERS = frozenset({"max", "min", "nz", "na", "abs", "iff", "int", "float", "bool", "round", "input", "input.int",
                     "input.float", "input.bool", "input.source", "input.string", "input.timeframe", "tostring",
                     "timestamp", "crossover", "crossunder", "cross", "sign", "sqrt", "pow", "log", "exp", "avg",
                     "sum", "color.new", "color.rgb", "time", "year", "month", "dayofmonth", "dayofweek", "hour",
                     "minute",
                     # presentation only: never part of trading rules
                     "plot", "plotshape", "plotchar", "plotarrow", "hline", "fill", "bgcolor", "barcolor", "label.new",
                     "line.new", "box.new", "table.new", "alert", "alertcondition"})


def _reason(state: str, text: str) -> dict:
    return {"state": state, "reason": text}


def _lit(value: str | None, inputs: dict[str, str], kind=int):
    if value is None:
        return None
    v = inputs.get(value, value).strip()
    try:
        return kind(v) if kind is float or re.fullmatch(r"\d+", v) else None
    except ValueError:
        return None


def _plain(cond: str) -> str:
    return re.sub(r"[()\s]", "", cond)


def market_scope(parsed: ParsedSpec) -> tuple[str, list[str], list[dict]]:
    """(market_type, symbol_scope, reasons) from the external backtest configuration."""
    ex = (parsed.backtest or {}).get("exchanges")
    if not ex:
        return "UNKNOWN", [], []
    if not isinstance(ex, list) or not all(isinstance(e, dict) for e in ex):
        return "UNKNOWN", [], [_reason("UNSUPPORTED_MARKET", "external market configuration is unparseable")]
    reasons, symbols, types = [], [], set()
    for e in ex:
        eid, cur = str(e.get("eid", "UNKNOWN")), str(e.get("currency", "UNKNOWN"))
        kind = "perpetual_futures" if eid.startswith("Futures_") else "spot"
        types.add(kind)
        symbols.append(f"{eid}:{cur}")
        if kind != "spot":
            reasons.append(_reason("UNSUPPORTED_MARKET", f"{eid} {cur} is a derivatives market; TradeTown researches "
                                                         "spot BTC/USD and ETH/USD, long/flat"))
        elif cur.split("_")[0] not in ASSETS:
            reasons.append(_reason("UNSUPPORTED_MARKET", f"asset {cur} outside the approved universe (BTC, ETH)"))
    if len(ex) > 1:
        reasons.append(_reason("UNSUPPORTED_MARKET", f"{len(ex)} instruments; TradeTown strategies are single-series"))
    return "+".join(sorted(types)), symbols, reasons


def _pine_checks(parsed: ParsedSpec) -> tuple[list[dict], dict | None]:
    p = parsed.pine
    assert p is not None
    reasons: list[dict] = []
    if p.header_kind != "strategy":
        reasons.append(_reason("INCOMPATIBLE", "indicator script: declares no strategy" if p.header_kind == "indicator"
                               else "no strategy() declaration: the script is not a runnable strategy as published"))
    if not p.entries:
        reasons.append(_reason("INCOMPATIBLE", "no strategy.entry: no entry rule to research"))
    if any(e["side"] == "short" for e in p.entries):
        reasons.append(_reason("UNSUPPORTED_EXECUTION_MODEL", "short entries; TradeTown researches long/flat only"))
    if any(e["side"] == "UNKNOWN" for e in p.entries):
        reasons.append(_reason("SEMANTICS_UNCERTAIN", "entry direction could not be established"))
    if p.orders:
        reasons.append(_reason("UNSUPPORTED_EXECUTION_MODEL", f"raw order management ({len(p.orders)} statement(s))"))
    pyramiding = _lit(p.strategy_kwargs.get("pyramiding"), p.inputs)
    if pyramiding is not None and pyramiding > 1:
        reasons.append(_reason("UNSUPPORTED_EXECUTION_MODEL", f"pyramiding={pyramiding}; TradeTown holds one position"))
    for flag in ("process_orders_on_close", "calc_on_order_fills", "calc_on_every_tick"):
        if p.strategy_kwargs.get(flag, "").strip() == "true":
            reasons.append(_reason("UNSUPPORTED_EXECUTION_MODEL", f"{flag}=true changes the fill model"))
    if "security" in p.calls:
        reasons.append(_reason("UNSUPPORTED_TIMEFRAME", "requests other-timeframe/other-symbol data (security)"))
    if "atr" in p.calls:
        reasons.append(_reason("UNSUPPORTED_INDICATOR", "Pine atr() is Wilder's RMA of true range; TradeTown's ATR "
                                                        "is a simple mean — not equivalent"))
    unsupported = sorted(c for c in p.calls if c in KNOWN_INDICATORS - SUPPORTED_INDICATORS - {"atr", "security"})
    if unsupported:
        reasons.append(_reason("UNSUPPORTED_INDICATOR", f"no TradeTown equivalent for: {', '.join(unsupported)}"))
    unknown = sorted(c for c in p.calls if c not in KNOWN_INDICATORS and c not in HELPERS
                     and not c.startswith("strategy."))
    if unknown:
        reasons.append(_reason("SEMANTICS_UNCERTAIN", f"unrecognised functions: {', '.join(unknown[:8])}"))
    if p.unrecognized:
        reasons.append(_reason("SEMANTICS_UNCERTAIN", f"{len(p.unrecognized)} statement(s) not understood, e.g. "
                                                      f"{p.unrecognized[0][:80]!r}"))
    mapped, differences = match_ma_crossover(p)
    if mapped is None:
        reasons.append(_reason("PARTIALLY_COMPATIBLE", "rules differ from mapping rule ma_crossover/"
                                                       f"{MAPPING_VERSION}: {'; '.join(differences)[:300]}"))
    return reasons, mapped


def match_ma_crossover(p) -> tuple[dict | None, list[str]]:
    diffs: list[str] = []
    a = p.assignments
    smas = {n: e for n, e in a.items() if re.fullmatch(r"sma\(close,\w+\)", e)}
    atrs = {n: e for n, e in a.items() if re.fullmatch(r"sma\(tr(\(true\))?,\w+\)", e)}
    if len(p.entries) != 1 or p.entries[0]["side"] != "long":
        return None, ["needs exactly one long entry"]
    entry = p.entries[0]
    m = re.fullmatch(r"(\w+)>(\w+)", _plain(entry["condition"]))
    if not m or m.group(1) not in smas or m.group(2) not in smas:
        diffs.append("entry is not the state condition SMA(fast) > SMA(slow)")
        return None, diffs + (["entry uses a crossover event (TradeTown re-enters while fast > slow)"]
                              if "cross" in entry["condition"] else [])
    fast_n, slow_n = m.group(1), m.group(2)
    if len(p.closes) != 1 or p.closes[0]["id"] != entry["id"] or \
            _plain(p.closes[0]["condition"]) not in (f"{fast_n}<={slow_n}", f"{slow_n}>={fast_n}"):
        diffs.append("exit is not strategy.close(entry, when = fast <= slow)")
    if len(atrs) != 1:
        diffs.append("no simple-mean ATR (sma(tr(true), n)) for the stop")
    if len(p.exits) != 1 or p.exits[0]["from_entry"] != entry["id"] or \
            set(p.exits[0]["kwargs"]) - {"from_entry", "stop"} or "stop" not in p.exits[0]["kwargs"]:
        diffs.append("stop is not a single strategy.exit(from_entry = entry, stop = T)")
    if diffs:
        return None, diffs
    atr_name = next(iter(atrs))
    t = p.exits[0]["kwargs"]["stop"].strip()
    ratchet = re.fullmatch(rf"strategy\.position_size>0\?max\(nz\({t}\[1\]\),close-(\w+(?:\.\d+)?)\*{atr_name}\):"
                           rf"close-(\w+(?:\.\d+)?)\*{atr_name}", a.get(t, ""))
    if not ratchet or ratchet.group(1) != ratchet.group(2):
        return None, ["stop is not close − k·ATR ratcheted upward while long"]
    extra = set(a) - {fast_n, slow_n, atr_name, t}
    if extra or set(p.reassigned) - {t}:
        return None, [f"additional state not in the mapping rule: {', '.join(sorted(extra | set(p.reassigned) - {t}))}"]
    if any("else" in x["condition"] for x in (entry, p.closes[0], p.exits[0])):
        return None, ["rules inside else-branches"]
    params = {"fast": _lit(re.search(r",(\w+)\)", smas[fast_n]).group(1), p.inputs),
              "slow": _lit(re.search(r",(\w+)\)", smas[slow_n]).group(1), p.inputs),
              "atr_period": _lit(re.search(r",(\w+)\)", atrs[atr_name]).group(1), p.inputs),
              "stop_atr": _lit(ratchet.group(1), p.inputs, float)}
    if any(v is None for v in params.values()):
        return None, ["a parameter is not a literal or input default"]
    return {"kind": "ma_crossover", "params": params}, []


def classify(parsed: ParsedSpec) -> dict:
    reasons: list[dict] = []
    mapped = None
    hits = {k: v for k, v in parsed.security.items() if SECURITY_PATTERNS[k][0] in UNSAFE_CATEGORIES}
    if hits:
        reasons.append(_reason("UNSAFE", f"source contains {', '.join(sorted(hits))}; not processed further"))
    elif parsed.language != "pine":
        execution = sorted(k for k in parsed.security if SECURITY_PATTERNS[k][0] == "EXECUTION")
        reasons.append(_reason("UNSUPPORTED_LANGUAGE", f"{parsed.language_label}: no normalizer (v1 normalizes Pine "
                                                       f"Script only)" + (f"; execution code present: "
                                                                          f"{', '.join(execution)}" if execution else "")))
    else:
        _, _, market_reasons = market_scope(parsed)
        reasons += market_reasons
        period = (parsed.backtest or {}).get("period")
        if period not in TIMEFRAMES:
            reasons.append(_reason("UNSUPPORTED_TIMEFRAME", f"timeframe {period or 'not stated'}; approved: 1h, 4h"))
        pine_reasons, mapped = _pine_checks(parsed)
        reasons += pine_reasons
        if mapped is not None:
            from ati.market.models import Timeframe
            from ati.strategies.base import LOGIC_REGISTRY
            try:
                LOGIC_REGISTRY[mapped["kind"]].validate_params(mapped["params"])
            except ValueError as exc:
                reasons.append(_reason("INCOMPATIBLE", f"parameters outside TradeTown bounds: {exc}"))
            if period in TIMEFRAMES:
                mapped["timeframe"] = Timeframe(TIMEFRAMES[period]).value
    state = min((r["state"] for r in reasons), key=STATES.index, default="COMPATIBLE")
    return {"compatibility_status": state, "reasons": reasons,
            "mapping": mapped if state == "COMPATIBLE" else None}


def _definition(record: dict):
    from datetime import datetime, timezone

    from ati.market.models import Timeframe
    from ati.strategies.base import StrategyDefinition

    m = record["mapping"]
    return StrategyDefinition.create(record["tradetown_strategy_id"], 1, m["kind"], m["params"], Timeframe(m["timeframe"]),
                                     datetime(2024, 1, 1, tzinfo=timezone.utc),
                                     description=f"external {record['external_strategy_id']} ({MAPPING_VERSION})")


def build_record(artifact: SourceArtifact) -> dict:
    """The normalized external strategy record. Deterministic: same bytes + same parser/mapping version → same record."""
    base = artifact.identity() | {"parser_version": PARSER_VERSION, "mapping_version": MAPPING_VERSION}
    try:
        parsed = parse(artifact.raw)
    except VaultParseError as exc:
        record = base | {"normalization_status": "PARSE_FAILED", "compatibility_status": "PARSE_FAILED",
                         "reasons": [_reason("PARSE_FAILED", str(exc))], "external_claims": [],
                         "tradetown_strategy_id": None, "tradetown_strategy_version": None,
                         "normalized_strategy_hash": None, "mapping": None}
        return record | {"record_hash": sha256_hex(record)}
    decision = classify(parsed)
    p = parsed.pine
    market_type, symbols, _ = market_scope(parsed) if parsed.language == "pine" else ("UNKNOWN", [], [])
    record = base | {
        "source_language": parsed.language, "source_name": parsed.name, "source_author": parsed.author,
        "source_description": parsed.description[:600], "source_detail_url": parsed.detail_url,
        "source_last_modified": parsed.last_modified, "strategy_family": parsed.family,
        "market_type": market_type, "asset_class": "crypto" if symbols else "UNKNOWN",
        "symbol_scope": symbols or ["UNKNOWN"],
        "timeframe_scope": (parsed.backtest or {}).get("period", "UNKNOWN"),
        "indicators": sorted(c for c in (p.calls if p else ()) if c in KNOWN_INDICATORS) or ["UNKNOWN"],
        "parameters": [{"name": a, "default": d, "description": t} for a, d, t in parsed.arguments]
        + ([{"name": k, "default": v, "description": "pine input"} for k, v in sorted(p.inputs.items())] if p else []),
        "entry_rules": [e["text"] for e in p.entries][:10] if p else ["UNKNOWN"],
        "exit_rules": [c["text"] for c in p.closes][:10] if p else ["UNKNOWN"],
        "stop_rules": [x["text"] for x in p.exits if {"stop", "loss", "trail_points", "trail_offset", "trail_price"}
                       & set(x["kwargs"])][:10] if p else ["UNKNOWN"],
        "target_rules": [x["text"] for x in p.exits if {"limit", "profit"} & set(x["kwargs"])][:10] if p else ["UNKNOWN"],
        "execution_assumptions": {"status": "RECORDED_NOT_ADOPTED", "strategy_settings": dict(sorted(p.strategy_kwargs.items()))}
        if p else {"status": "UNKNOWN"},
        "risk_assumptions": "UNKNOWN (not stated in a machine-readable form)",
        "security_findings": dict(sorted(parsed.security.items())),
        "external_claims": list(parsed.claims),
        "normalization_status": "NORMALIZED" if decision["compatibility_status"] == "COMPATIBLE" else "NOT_NORMALIZED",
        "compatibility_status": decision["compatibility_status"], "reasons": decision["reasons"],
        "mapping": decision["mapping"], "tradetown_strategy_id": None, "tradetown_strategy_version": None,
        "normalized_strategy_hash": None}
    if record["compatibility_status"] == "COMPATIBLE":
        record["tradetown_strategy_id"] = "ext-" + artifact.external_strategy_id[4:16]
        record["tradetown_strategy_version"] = 1
        record["normalized_strategy_hash"] = _definition(record).definition_hash
        record["symbol_mapping"] = ("the rule, not the external instrument, is the hypothesis: TradeTown researches it "
                                    "on its own BTC/USD or ETH/USD series; the external instrument "
                                    f"({', '.join(symbols) or 'UNKNOWN'}) stays recorded as claim context")
    return record | {"record_hash": sha256_hex(record)}


def definition(record: dict):
    """The TradeTown StrategyDefinition of a COMPATIBLE record (raises for anything else)."""
    if record.get("compatibility_status") not in RESEARCHABLE or not record.get("mapping"):
        raise ValueError(f"{record.get('external_strategy_id')} is {record.get('compatibility_status')}: "
                         "not a research candidate")
    d = _definition(record)
    if d.definition_hash != record["normalized_strategy_hash"]:
        raise ValueError("normalized definition no longer matches the recorded hash (logic code or mapping changed)")
    return d
