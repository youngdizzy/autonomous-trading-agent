"""Read-only health of accumulated market series, sealed-holdout verification, and the dataset scorecard.

States are explicit and never softened: PASS · FAIL · BLOCKED · NOT_READY · NOT_RUN. Nothing here repairs,
fills, interpolates, resamples, or rewrites evidence — it only reports.

Gap policy (crypto trades continuously, so a missing BTC/ETH interval is taken seriously):
  UNEXPLAINED_GAP     intervals missing *between* stored candles. Never filled; reported with locations. Research
                      windows are contiguous runs only, so a gap splits windows and can never be bridged.
  INCOMPLETE_CURRENT  the latest closed candle is older than two intervals: accumulation is behind (provider
                      outage, blocked network, or no run) — the series is STALE, not corrupt.
The system cannot tell an exchange outage from a lost response; both are reported as gaps, never repaired.
"""

from __future__ import annotations

from datetime import datetime

from ati.core.errors import AtiError, DataIntegrityError
from ati.data.dataset import Dataset, Partition, content_hash, sealed_ranges
from ati.ledger.journal import decode
from ati.market.models import MARKET_EVIDENCE_STATUSES, DataStatus, Timeframe
from ati.market.validation import validate_series

STATES = ("PASS", "FAIL", "BLOCKED", "NOT_READY", "NOT_RUN")


def gaps(candles) -> list[dict]:
    out = []
    for a, b in zip(candles, candles[1:]):
        missing = int((b.open_time - a.open_time).total_seconds() // a.timeframe.seconds) - 1
        if missing > 0:
            out.append({"after": a.open_time.isoformat(), "before": b.open_time.isoformat(), "missing_intervals": missing})
    return out


def contiguous_runs(candles) -> list[list]:
    runs, current = [], []
    for c in candles:
        if current and (c.open_time - current[-1].open_time) != c.timeframe.delta:
            runs.append(current)
            current = []
        current.append(c)
    if current:
        runs.append(current)
    return runs


def verify_sealed_holdouts(system) -> list[dict]:
    """Every sealed holdout is re-derived from the stored series and compared with the commitment journaled
    when it was sealed: identical content → INTACT; different → MUTATED (FAIL); absent → UNVERIFIABLE."""
    s = system
    out = []
    for e in s.research_journal.entries("holdout_sealed"):
        p = decode(e.payload)
        h = p["holdout_identity"]
        tf = Timeframe(h["timeframe"])
        candles = [c for c in s.store.series(h["provider"], h["symbol"], tf)
                   if c.open_time >= h["start"] and c.close_time <= h["end"]]
        row = {"holdout": "SEALED_HOLDOUT", "symbol": h["symbol"], "timeframe": h["timeframe"], "n_candles": h["n_candles"],
               "sealed_at": e.at}
        if not candles:
            out.append(row | {"state": "UNVERIFIABLE", "detail": "holdout candles are not in the store"})
        elif len(candles) != h["n_candles"] or content_hash(candles) != p["holdout_commitment"]:
            out.append(row | {"state": "MUTATED", "detail": "stored candles differ from the sealed commitment"})
        else:
            rebuilt = Dataset.build(candles, data_version=h["data_version"], realization=h["realization"],
                                    partition=Partition.HOLDOUT)
            same = rebuilt.dataset_id == p["holdout_dataset_id"]
            out.append(row | {"state": "INTACT" if same else "MUTATED",
                              "detail": "content and identity match the sealed commitment" if same else "identity differs"})
    return out


def series_health(system, symbol: str, tf: Timeframe, now: datetime | None = None) -> dict:
    s = system
    now = now or s.clock.now()
    candles = s.store.series(s.provider.name, symbol, tf)
    checks: dict[str, str] = {}
    category = s.data_status.value
    if not candles:
        state = "NOT_READY"
        detail = ("REAL_DATA_UNAVAILABLE" if s.data_status in MARKET_EVIDENCE_STATUSES else f"no {category} candles")
        # the category describes stored candles; with none there is no category to report (only the binding)
        return {"symbol": symbol, "timeframe": tf.value, "category": f"NO_DATA (system bound to {category})",
                "state": state, "detail": detail,
                "checks": {"data": "NO_CANDLES"}, "gaps": [], "conflicts": []}
    statuses = {c.status for c in candles}
    if len(statuses) > 1 or statuses != {s.data_status}:
        category = "MIXED"
    try:
        validate_series(candles, symbol=symbol, timeframe=tf, provider=s.provider.name)
        checks["series"] = "PASS (symbol, timeframe, provider, category, closed, unique, monotonic)"
    except DataIntegrityError as exc:
        checks["series"] = f"FAIL {exc.code}: {exc}"[:200]
    misaligned = [c for c in candles if int(c.open_time.timestamp()) % tf.seconds]
    checks["alignment"] = "PASS" if not misaligned else f"FAIL {len(misaligned)} misaligned"
    found_gaps = gaps(candles)
    checks["gaps"] = "PASS" if not found_gaps else f"UNEXPLAINED_GAP {sum(g['missing_intervals'] for g in found_gaps)} " \
                                                   f"interval(s) in {len(found_gaps)} place(s)"
    age = now - candles[-1].close_time
    stale = age > tf.delta * 2
    checks["freshness"] = "PASS" if not stale else f"INCOMPLETE_CURRENT latest close {candles[-1].close_time.isoformat()} " \
                                                   f"({age} ago)"
    conflicts = [c for c in s.archive.conflicts() if (c["symbol"], c["timeframe"]) == (symbol, tf.value)]
    checks["conflicts"] = "PASS" if not conflicts else f"DATA_CONFLICT {len(conflicts)} unacknowledged"
    if s.data_status in MARKET_EVIDENCE_STATUSES:
        try:
            s.archive.verify_derivation(Dataset.build(candles, data_version="health", realization="observed"))
            checks["provenance"] = "PASS every candle re-derived from its archived provider payload"
        except AtiError as exc:
            checks["provenance"] = f"FAIL {exc}"[:200]
    else:
        checks["provenance"] = f"NOT_RUN ({category}: no market provenance to verify)"
    holdouts = [h for h in verify_sealed_holdouts(s) if (h["symbol"], h["timeframe"]) == (symbol, tf.value)]
    checks["sealed_holdouts"] = "PASS" if all(h["state"] != "MUTATED" for h in holdouts) else "FAIL holdout mutated"
    if category == "MIXED" or any(v.startswith("FAIL") for v in checks.values()):
        state = "FAIL"
    elif conflicts:
        state = "BLOCKED"
    elif stale:
        state = "NOT_READY"
    else:
        state = "PASS"
    return {"symbol": symbol, "timeframe": tf.value, "category": category, "state": state, "checks": checks,
            "gaps": found_gaps[:10], "conflicts": conflicts}


# --- readiness contract ---------------------------------------------------------------------------------------
# Overall states, most to least blocking. They are never collapsed into one another:
#   REAL_DATA_UNAVAILABLE      no usable REAL candles for the series (provider unreachable / nothing accumulated)
#   BLOCKED                    candles exist but health is FAIL/BLOCKED (gaps in the window, conflicts, MIXED, ...)
#   INSUFFICIENT_REAL_CANDLES  REAL candles exist but fewer than the protocol's contiguous, unsealed minimum
#   VALIDATION_PENDING         enough REAL candles; the protocol's validation has not run, or has not finished
#   INSUFFICIENT_EVIDENCE      validation ran, but a stage ended below its evidence/trade floor (never "failed")
#   VALIDATION_FAILED          validation ran and a required stage actually failed its criteria
#   VALIDATION_PASSED          validation ran and every required evidence stage passed (promotion is separate)
#   NOT_APPLICABLE_MOCK / NOT_APPLICABLE_<category>   the data is not market evidence: no REAL readiness at all
VALIDATION_STATES = ("NOT_RUN", "VALIDATION_PENDING", "INSUFFICIENT_EVIDENCE", "VALIDATION_FAILED", "VALIDATION_PASSED")


def validation_outcome(dev: str | None, holdout: str | None, adversarial: list[str] | None) -> str:
    """Pure: required stages are walk_forward_oos, adversarial and holdout (objective contract). Inputs are the
    *recorded* verdicts (PASS / FAIL / INSUFFICIENT_EVIDENCE / NOT_AUTOMATED), None when a stage has no record."""
    if dev is None:
        return "NOT_RUN"
    if dev == "FAIL":
        return "VALIDATION_FAILED"
    if dev == "INSUFFICIENT_EVIDENCE":
        return "INSUFFICIENT_EVIDENCE"
    if holdout is None or adversarial is None:       # development passed; the run has not reached its end
        return "VALIDATION_PENDING"
    if holdout == "FAIL" or "FAIL" in adversarial:
        return "VALIDATION_FAILED"
    if holdout == "INSUFFICIENT_EVIDENCE" or "INSUFFICIENT_EVIDENCE" in adversarial:
        return "INSUFFICIENT_EVIDENCE"
    return "VALIDATION_PASSED" if holdout == "PASS" else "VALIDATION_PENDING"


def classify(category: str, n_candles: int, latest_contiguous: int, required: int, health_state: str,
             validation: str) -> dict:
    """Pure readiness contract. ``category`` is the *verified* category of the stored candles (``MIXED`` when they
    disagree); ``validation`` is ``validation_outcome`` of the protocol's latest recorded run (market data only)."""
    market = category in {s.value for s in MARKET_EVIDENCE_STATUSES}
    if category == "MIXED":
        real_data, overall = "MIXED_INVALID", "BLOCKED"
    elif not market and n_candles and category not in ("", "NONE"):
        real_data = overall = "NOT_APPLICABLE_MOCK" if category == "MOCK" else f"NOT_APPLICABLE_{category}"
    elif n_candles == 0:
        real_data = overall = "REAL_DATA_UNAVAILABLE"
    else:
        real_data = f"REAL_DATA_PRESENT ({n_candles} candles)"
        overall = None
    candles = ("NOT_APPLICABLE" if overall and overall != "REAL_DATA_UNAVAILABLE" and not overall.startswith("BLOCK")
               else "REAL_DATA_UNAVAILABLE" if n_candles == 0 or real_data == "MIXED_INVALID"
               else "SUFFICIENT" if latest_contiguous >= required else "INSUFFICIENT_REAL_CANDLES")
    evidence = ("INSUFFICIENT_EVIDENCE" if validation == "INSUFFICIENT_EVIDENCE" else
                "SUFFICIENT" if validation in ("VALIDATION_FAILED", "VALIDATION_PASSED") else "NOT_EVALUATED")
    if overall is None:
        if health_state in ("FAIL", "BLOCKED"):
            overall = "BLOCKED"
        elif validation not in ("NOT_RUN",):
            overall = validation            # a recorded validation outcome is a fact; it is reported as recorded
        elif latest_contiguous < required:
            overall = "INSUFFICIENT_REAL_CANDLES"
        else:
            overall = "VALIDATION_PENDING"
    if not market:
        evidence = validation_col = "NOT_APPLICABLE"
    else:
        validation_col = validation
    return {"state": overall, "real_data": real_data, "candle_readiness": candles, "evidence_readiness": evidence,
            "validation": validation_col}


def protocol_validation(system, protocol_id: str) -> dict:
    """Latest recorded validation run for a protocol, derived from the research journal only."""
    from ati.research.hypothesis import ResearchLog

    journal = getattr(system, "research_journal", None)
    if journal is None:
        return {"outcome": "NOT_RUN", "hypothesis_id": None, "runs": 0}
    runs = [decode(e.payload) for e in journal.entries("protocol_run")
            if e.payload["protocol_id"] == protocol_id and e.payload.get("kind") == "validation"]
    if not runs:
        return {"outcome": "NOT_RUN", "hypothesis_id": None, "runs": 0}
    log = ResearchLog(journal)
    h = runs[-1]["hypothesis_id"]

    def verdict(hid):
        rows = [e for e in log.experiments if e["hypothesis_id"] == hid and not str(e["stage"]).startswith("diagnostic")]
        return str(getattr(rows[-1]["verdict"], "value", rows[-1]["verdict"])) if rows else None
    dev, hold = verdict(h), verdict(f"{h}:holdout")
    adversarial = None
    if log.status(f"{h}:holdout") != "UNKNOWN":
        challenger = log.get(f"{h}:holdout").strategy_hash
        reports = [decode(e.payload)["report"] for e in journal.entries("adversarial_report")
                   if e.payload["report"]["strategy_hash"] == challenger]
        if reports:
            adversarial = [str(getattr(o["verdict"], "value", o["verdict"])) for o in reports[-1]["objections"]]
    promos = [decode(e.payload)["record"] for e in journal.entries("promotion_decision")]
    promo = next((p for p in reversed(promos) if log.status(f"{h}:holdout") != "UNKNOWN"
                  and p["challenger_hash"] == log.get(f"{h}:holdout").strategy_hash), None)
    return {"outcome": validation_outcome(dev, hold, adversarial), "hypothesis_id": h, "runs": len(runs),
            "protocol_hash": runs[-1]["protocol_hash"], "stages": {"walk_forward_oos": dev or "NOT_RUN",
                                                                 "adversarial": "RECORDED" if adversarial else "NOT_RUN",
                                                                 "holdout": hold or "NOT_RUN"},
            "promotion_gate": ("APPROVED" if promo["approved"] else "DENIED") if promo else "NOT_RUN"}


def research_readiness(system, symbol: str, tf: Timeframe, health: dict, min_candles: int | None = None) -> dict:
    """Readiness of one series against its declared protocol (``ati.research.protocols``)."""
    from ati.research.protocols import for_series

    s = system
    protocol = for_series(symbol, tf)
    required = min_candles if min_candles is not None else (protocol.min_candles if protocol else 0)
    candles = s.store.series(s.provider.name, symbol, tf)
    sealed = sealed_ranges(s.provider.name, symbol, tf, getattr(s.provider, "realization", "observed"))
    unsealed = [c for c in candles if not any(c.open_time < end and c.close_time > start for start, end in sealed)]
    runs = contiguous_runs(unsealed)
    latest = len(runs[-1]) if runs else 0
    statuses = {c.status for c in candles}
    category = ("MIXED" if len(statuses) > 1 or (statuses and statuses != {s.data_status}) else
                s.data_status.value if candles else "NONE")
    if not candles and s.data_status not in MARKET_EVIDENCE_STATUSES:
        category = s.data_status.value
    validation = protocol_validation(s, protocol.protocol_id) if protocol else {"outcome": "NOT_RUN"}
    market = category in {x.value for x in MARKET_EVIDENCE_STATUSES} or (category == "NONE")
    result = classify(category, len(candles) if category != "NONE" else 0, latest, required, health.get("state", "NOT_RUN"),
                      validation["outcome"] if market else "NOT_RUN")
    if not candles and s.data_status not in MARKET_EVIDENCE_STATUSES:
        result = {"state": "NOT_APPLICABLE_MOCK" if s.data_status is DataStatus.MOCK else f"NOT_APPLICABLE_{category}",
                  "real_data": "NOT_APPLICABLE", "candle_readiness": "NOT_APPLICABLE",
                  "evidence_readiness": "NOT_APPLICABLE", "validation": "NOT_APPLICABLE"}
    if not market and validation["outcome"] != "NOT_RUN":
        result["non_market_mechanics_outcome"] = f"[{category}] {validation['outcome']} (not market evidence)"
    return result | {"latest_contiguous_unsealed_candles": latest, "required": required,
                     "shortfall": max(required - latest, 0),
                     "protocol": protocol.protocol_id if protocol else None,
                     "protocol_hash": protocol.protocol_hash if protocol else None,
                     "protocol_executable": protocol.executable if protocol else False,
                     "validation_detail": validation}


def scorecard(system, series, champion_fingerprint: str | None = None) -> list[dict]:
    """Phase-13 dataset scorecard (read-only), one row per accumulated series."""
    from ati.market.accumulate import history
    from ati.research import protocol as P
    from ati.research.protocols import for_series

    s = system
    rows = []
    for symbol, tf in series:
        candles = s.store.series(s.provider.name, symbol, tf)
        h = series_health(s, symbol, tf)
        proto = for_series(symbol, tf)
        sealed = sealed_ranges(s.provider.name, symbol, tf, getattr(s.provider, "realization", "observed"))
        holdout_n = sum(1 for c in candles if any(c.open_time < end and c.close_time > start for start, end in sealed))
        runs = history(s, symbol, tf)
        receipts = s.archive.receipts_for(s.provider.name, symbol, tf)
        market = s.data_status in MARKET_EVIDENCE_STATUSES
        ds_id = Dataset.build(candles, data_version="scorecard", realization=getattr(s.provider, "realization",
                                                                                    "observed")).dataset_id if candles else None
        rows.append({
            "provider": s.provider.name, "symbol": symbol, "timeframe": tf.value, "category": h["category"],
            "earliest": candles[0].open_time.isoformat() if candles else "NOT_AVAILABLE",
            "latest": candles[-1].close_time.isoformat() if candles else "NOT_AVAILABLE",
            "total_candles": len(candles),
            "development_candles": len(candles) - holdout_n,     # every candle outside a sealed holdout
            "sealed_holdout_candles": holdout_n,
            "dataset_hash": ds_id or "NOT_AVAILABLE",
            "strategy_fingerprint": {"protocol_reference": proto.strategy_fingerprint if proto else "NO_PROTOCOL_DECLARED",
                                     "champion": champion_fingerprint if (symbol, tf) == (P.SYMBOL, P.TIMEFRAME)
                                     and champion_fingerprint else "NO_CHAMPION"},
            "protocol": proto.protocol_id if proto else "NO_PROTOCOL_DECLARED",
            "first_accumulation": runs[0]["at"].isoformat() if runs else "NOT_RUN",
            "latest_accumulation": {"at": runs[-1]["at"].isoformat(), "status": runs[-1]["status"]} if runs else "NOT_RUN",
            "health": h["state"], "gap_status": h["checks"].get("gaps", "NOT_RUN"),
            "conflict_status": h["checks"].get("conflicts", "NOT_RUN"),
            "persistence": (f"ARCHIVED ({len(receipts)} provider receipts; re-derived on restart)" if receipts else
                            ("NO_ARCHIVED_PAYLOADS" if market else f"NOT_PERSISTED ({h['category']}: regenerated)")),
            "research_readiness": research_readiness(s, symbol, tf, h),
        })
    return rows


def readiness_table(system, series) -> list[dict]:
    """Dataset | Protocol | REAL data | Candle readiness | Evidence readiness | Validation — actual states only."""
    from ati.research.protocols import for_series

    rows = []
    for symbol, tf in series:
        h = series_health(system, symbol, tf)
        r = research_readiness(system, symbol, tf, h)
        proto = for_series(symbol, tf)
        rows.append({"dataset": f"{symbol} {tf.value}",
                     "protocol": f"{proto.protocol_id} (declared{', executable' if proto.executable else ', not executable'})"
                     if proto else "NO_PROTOCOL_DECLARED",
                     "real_data": r["real_data"], "candle_readiness": r["candle_readiness"],
                     "evidence_readiness": r["evidence_readiness"], "validation": r["validation"], "overall": r["state"]})
    return rows
