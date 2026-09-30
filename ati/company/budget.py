"""Research budget: bounded research resources, derived from the journals (never from a counter that a
restart could reset). The point is not to stop research arbitrarily; it is to make "keep trying variants
until something looks good" structurally impossible. Limits are code; Claude cannot change them."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ati.core.canonical import sha256_hex
from ati.ledger.journal import decode
from ati.research.hypothesis import ResearchLog


@dataclass(frozen=True)
class BudgetPolicy:
    max_root_hypotheses: int = 12          # distinct hypotheses ever tested (multiple-testing exposure)
    max_variants_per_baseline: int = 4     # derived candidate definitions per baseline fingerprint
    max_holdout_evaluations: int = 6       # holdout periods consumed in total
    max_research_runs_per_day: int = 2     # research executions started by the company per UTC day

    @property
    def policy_hash(self) -> str:
        return sha256_hex(self)


POLICY = BudgetPolicy()


def usage(system, company_journal, now: datetime, baseline_hash: str | None = None) -> dict:
    s = system
    log = ResearchLog(s.research_journal)
    holdouts = sum(1 for _ in s.research_journal.entries("holdout_access"))
    variants = 0
    if baseline_hash:
        variants = sum(1 for k in s.strategies.keys() if s.strategies.get(k).parent_hash == baseline_hash)
    today = now.date().isoformat()
    runs_today = sum(1 for e in company_journal.entries("cycle_step")
                     if e.payload["step"] == "research_invoked" and e.at[:10] == today)
    return {"root_hypotheses_tested": log.hypotheses_tested, "experiments": len(log.experiments),
            "holdout_evaluations": holdouts, "variants_of_baseline": variants, "research_runs_today": runs_today,
            "policy": {k: v for k, v in POLICY.__dict__.items()}, "policy_hash": POLICY.policy_hash}


def check(u: dict, new_variant: bool) -> list[str]:
    """Reasons a new research run is not permitted (empty = within budget)."""
    reasons = []
    if u["root_hypotheses_tested"] >= POLICY.max_root_hypotheses:
        reasons.append(f"root hypothesis budget exhausted ({u['root_hypotheses_tested']}/{POLICY.max_root_hypotheses})")
    if u["holdout_evaluations"] >= POLICY.max_holdout_evaluations:
        reasons.append(f"holdout budget exhausted ({u['holdout_evaluations']}/{POLICY.max_holdout_evaluations})")
    if new_variant and u["variants_of_baseline"] >= POLICY.max_variants_per_baseline:
        reasons.append(f"variant budget for this baseline exhausted ({u['variants_of_baseline']}/"
                       f"{POLICY.max_variants_per_baseline})")
    if u["research_runs_today"] >= POLICY.max_research_runs_per_day:
        reasons.append(f"daily research budget exhausted ({u['research_runs_today']}/{POLICY.max_research_runs_per_day})")
    return reasons
