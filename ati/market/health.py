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


def research_readiness(system, symbol: str, tf: Timeframe, health: dict, min_candles: int) -> dict:
    """REAL_DATA_UNAVAILABLE · INSUFFICIENT_REAL_CANDLES · VALIDATION_PENDING · BLOCKED · NOT_APPLICABLE_MOCK ·
    NO_PROTOCOL_DECLARED. Insufficient evidence is never reported as a failed validation."""
    from ati.research import protocol as P

    s = system
    candles = s.store.series(s.provider.name, symbol, tf)
    sealed = sealed_ranges(s.provider.name, symbol, tf, getattr(s.provider, "realization", "observed"))
    unsealed = [c for c in candles if not any(c.open_time < end and c.close_time > start for start, end in sealed)]
    runs = contiguous_runs(unsealed)
    latest = len(runs[-1]) if runs else 0
    protocol = (symbol, tf) == (P.SYMBOL, P.TIMEFRAME)
    if s.data_status not in MARKET_EVIDENCE_STATUSES:
        state = "NOT_APPLICABLE_MOCK" if s.data_status is DataStatus.MOCK else f"NOT_APPLICABLE_{s.data_status.value}"
    elif not candles:
        state = "REAL_DATA_UNAVAILABLE"
    elif health["state"] in ("FAIL", "BLOCKED"):
        state = "BLOCKED"
    elif latest < min_candles:
        state = "INSUFFICIENT_REAL_CANDLES"
    elif not protocol:
        state = "NO_PROTOCOL_DECLARED"
    else:
        state = "VALIDATION_PENDING"
    return {"state": state, "latest_contiguous_unsealed_candles": latest, "required": min_candles,
            "shortfall": max(min_candles - latest, 0), "protocol": P.PROTOCOL_ID if protocol else None}


def scorecard(system, series, champion_fingerprint: str | None = None) -> list[dict]:
    """Phase-13 dataset scorecard (read-only), one row per accumulated series."""
    from ati.market.accumulate import history
    from ati.research import protocol as P

    s = system
    rows = []
    for symbol, tf in series:
        candles = s.store.series(s.provider.name, symbol, tf)
        h = series_health(s, symbol, tf)
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
            "strategy_fingerprint": champion_fingerprint if (symbol, tf) == (P.SYMBOL, P.TIMEFRAME) and champion_fingerprint
            else ("NO_CHAMPION" if (symbol, tf) == (P.SYMBOL, P.TIMEFRAME) else "NO_PROTOCOL_DECLARED"),
            "first_accumulation": runs[0]["at"].isoformat() if runs else "NOT_RUN",
            "latest_accumulation": {"at": runs[-1]["at"].isoformat(), "status": runs[-1]["status"]} if runs else "NOT_RUN",
            "health": h["state"], "gap_status": h["checks"].get("gaps", "NOT_RUN"),
            "conflict_status": h["checks"].get("conflicts", "NOT_RUN"),
            "persistence": (f"ARCHIVED ({len(receipts)} provider receipts; re-derived on restart)" if receipts else
                            ("NO_ARCHIVED_PAYLOADS" if market else f"NOT_PERSISTED ({h['category']}: regenerated)")),
            "research_readiness": research_readiness(s, symbol, tf, h, P.MIN_CANDLES),
        })
    return rows
