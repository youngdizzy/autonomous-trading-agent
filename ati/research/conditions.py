"""Future-condition intelligence: assumption monitors.

This does not predict anything. Each monitor asks one question: *what evidence would tell us that an
assumption behind our research is becoming invalid?* It compares a recent window with a reference window
of the same point-in-time data and reports

    STABLE | SHIFT_DETECTED | INSUFFICIENT_EVIDENCE | NOT_AVAILABLE

with the statistic, the threshold, and which assumption a shift would undermine. Thresholds are code.
A SHIFT is a learning outcome (it can seed a research question); it never changes a strategy or a limit.
Floats are statistics only.
"""

from __future__ import annotations

import math
from decimal import Decimal

RECENT = 168          # one week of hourly bars
REFERENCE = 720       # the preceding month

ASSUMPTIONS = {
    "volatility": "volatility is similar to the period the strategy was researched on (stop distance, sizing)",
    "trend_range": "the market's trend/range character is unchanged (trend-following needs persistent moves)",
    "liquidity": "traded volume supports the modelled participation and slippage",
    "distribution": "the return distribution is stationary enough for past evidence to apply",
    "structural_break": "the mean hourly return has not shifted abruptly",
    "correlation": "cross-asset correlation is unchanged (diversification assumptions)",
    "execution_cost": "realized paper fills cost no more than the modelled spread + slippage",
}


def _row(name, status, statistic=None, threshold=None, detail=""):
    return {"assumption": name, "status": status, "statistic": statistic, "threshold": threshold,
            "invalidates": ASSUMPTIONS[name], "detail": detail}


def _returns(candles) -> list[float]:
    closes = [float(c.close) for c in candles]
    return [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]


def _sd(x: list[float]) -> float:
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def _efficiency(rets: list[float]) -> float:
    total = sum(abs(r) for r in rets)
    return abs(sum(rets)) / total if total > 0 else 0.0


def _ks(a: list[float], b: list[float]) -> float:
    a, b = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    return d


def monitor(candles_by_symbol: dict, fills: list[tuple[Decimal, Decimal, Decimal]] | None = None,
            modelled_cost_rate: Decimal | None = None) -> list[dict]:
    """``candles_by_symbol``: closed candles, oldest first, point-in-time (nothing after the cutoff).
    ``fills``: (reference_price, fill_price, qty) for paper BUY entries, if any."""
    out: list[dict] = []
    sym = sorted(candles_by_symbol)[0] if candles_by_symbol else None
    candles = list(candles_by_symbol.get(sym, ())) if sym else []
    if len(candles) < RECENT + REFERENCE + 1:
        for name in ("volatility", "trend_range", "liquidity", "distribution", "structural_break"):
            out.append(_row(name, "INSUFFICIENT_EVIDENCE", detail=f"{len(candles)} bars < {RECENT + REFERENCE + 1}"))
    else:
        window = candles[-(RECENT + REFERENCE + 1):]
        rets = _returns(window)
        ref, rec = rets[:REFERENCE], rets[REFERENCE:]
        ratio = _sd(rec) / _sd(ref) if _sd(ref) > 0 else None
        out.append(_row("volatility", "INSUFFICIENT_EVIDENCE" if ratio is None else
                        "SHIFT_DETECTED" if ratio > 1.5 or ratio < 1 / 1.5 else "STABLE",
                        None if ratio is None else round(ratio, 4), "ratio outside [0.667, 1.5]"))
        e_ref, e_rec = _efficiency(ref[-RECENT:]), _efficiency(rec)
        state = lambda e: "TREND" if e >= 0.3 else "RANGE"   # noqa: E731
        out.append(_row("trend_range", "SHIFT_DETECTED" if state(e_ref) != state(e_rec) else "STABLE",
                        round(e_rec, 4), "efficiency ratio 0.3 separates TREND/RANGE",
                        f"reference {state(e_ref)} → recent {state(e_rec)}"))
        vols_ref = sorted(float(c.volume) for c in window[1:REFERENCE + 1])
        vols_rec = sorted(float(c.volume) for c in window[REFERENCE + 1:])
        med_ref, med_rec = vols_ref[len(vols_ref) // 2], vols_rec[len(vols_rec) // 2]
        lr = med_rec / med_ref if med_ref > 0 else None
        out.append(_row("liquidity", "INSUFFICIENT_EVIDENCE" if lr is None else
                        "SHIFT_DETECTED" if lr < 0.5 else "STABLE", None if lr is None else round(lr, 4),
                        "median volume ratio < 0.5"))
        d = _ks(ref, rec)
        crit = 1.628 * math.sqrt((len(ref) + len(rec)) / (len(ref) * len(rec)))   # alpha = 0.01
        out.append(_row("distribution", "SHIFT_DETECTED" if d > crit else "STABLE", round(d, 4),
                        round(crit, 4), "two-sample Kolmogorov-Smirnov, alpha 0.01"))
        se = math.sqrt(_sd(ref) ** 2 / len(ref) + _sd(rec) ** 2 / len(rec))
        t = (sum(rec) / len(rec) - sum(ref) / len(ref)) / se if se > 0 else 0.0
        out.append(_row("structural_break", "SHIFT_DETECTED" if abs(t) > 3.0 else "STABLE", round(t, 4), "|t| > 3",
                        "Welch t on mean hourly return, recent vs reference"))
    if len(candles_by_symbol) < 2:
        out.append(_row("correlation", "NOT_AVAILABLE", detail="needs at least two symbols"))
    else:
        a, b = (list(candles_by_symbol[k]) for k in sorted(candles_by_symbol)[:2])
        n = min(len(a), len(b))
        if n < RECENT + REFERENCE + 1:
            out.append(_row("correlation", "INSUFFICIENT_EVIDENCE", detail=f"{n} aligned bars"))
        else:
            ra, rb = _returns(a[-(RECENT + REFERENCE + 1):]), _returns(b[-(RECENT + REFERENCE + 1):])
            c_ref, c_rec = _corr(ra[:REFERENCE], rb[:REFERENCE]), _corr(ra[REFERENCE:], rb[REFERENCE:])
            out.append(_row("correlation", "SHIFT_DETECTED" if abs(c_rec - c_ref) > 0.4 else "STABLE",
                            round(c_rec - c_ref, 4), "|Δcorrelation| > 0.4"))
    if not fills or modelled_cost_rate is None:
        out.append(_row("execution_cost", "NOT_AVAILABLE", detail="no paper fills with a reference price"))
    else:
        notional = sum(float(ref * q) for ref, _, q in fills)
        slip = sum(float((px - ref) * q) for ref, px, q in fills) / notional if notional > 0 else 0.0
        limit = float(modelled_cost_rate) * 2
        status = "INSUFFICIENT_EVIDENCE" if len(fills) < 5 else ("SHIFT_DETECTED" if slip > limit else "STABLE")
        out.append(_row("execution_cost", status, round(slip, 6), f"> 2× modelled {float(modelled_cost_rate):.5f}",
                        f"{len(fills)} fills"))
    return out


def _corr(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx, vy = math.sqrt(sum((a - mx) ** 2 for a in x)), math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0
