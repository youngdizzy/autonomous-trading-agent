"""Research budget: bounded research resources, derived from the journals (never from a counter that a
restart could reset). The point is not to stop research arbitrarily; it is to make "keep trying variants
until something looks good" structurally impossible. Limits are code; Claude cannot change them.

Tracked: root hypotheses (multiple-testing exposure), experiments, variants per baseline, tests of the same
idea (saturation), holdout periods consumed, research runs per day, compute units per day (bars × configs
evaluated), and distinct datasets used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ati.core.canonical import sha256_hex
from ati.research.hypothesis import ResearchLog


@dataclass(frozen=True)
class BudgetPolicy:
    max_root_hypotheses: int = 12          # distinct hypotheses ever tested (multiple-testing exposure)
    max_variants_per_baseline: int = 4     # derived candidate definitions per baseline fingerprint
    max_holdout_evaluations: int = 6       # holdout periods consumed in total
    max_research_runs_per_day: int = 2     # research executions started by the company per UTC day
    max_tests_per_idea: int = 3            # hypotheses with the same statement: beyond this, the idea is saturated
    max_compute_units_per_day: int = 200_000   # Σ (development bars × configurations) for designed runs per day

    @property
    def policy_hash(self) -> str:
        return sha256_hex(self)


POLICY = BudgetPolicy()


def _norm(statement: str) -> str:
    return " ".join(statement.lower().split())


def usage(system, company_journal, now: datetime, baseline_hash: str | None = None,
          statement: str | None = None) -> dict:
    s = system
    log = ResearchLog(s.research_journal)
    holdouts = sum(1 for _ in s.research_journal.entries("holdout_access"))
    variants = 0
    if baseline_hash:
        variants = sum(1 for k in s.strategies.keys() if s.strategies.get(k).parent_hash == baseline_hash)
    today = now.date().isoformat()
    runs_today = sum(1 for e in company_journal.entries("cycle_step")
                     if e.payload["step"] == "research_invoked" and e.at[:10] == today)
    compute_today = sum(int(e.payload.get("compute_units", 0)) for e in s.research_journal.entries("experiment_design")
                        if e.at[:10] == today)
    tested_roots = {e["hypothesis_id"].split(":", 1)[0] for e in log.experiments}
    same_idea = 0
    if statement is not None:
        same_idea = sum(1 for h in tested_roots if _norm(log.get(h).statement) == _norm(statement))
    rejected = sum(1 for h in tested_roots if log.status(h) in ("TESTED:FAIL", "TESTED:INSUFFICIENT_EVIDENCE"))
    designed = {e.payload["hypothesis_id"] for e in s.research_journal.entries("experiment_design")
                if e.payload.get("candidate")}
    grid_challengers = {e.payload["hypothesis_id"] for e in s.research_journal.entries("candidate_lineage")} - designed
    return {"root_hypotheses_tested": log.hypotheses_tested, "experiments": len(log.experiments),
            "experiments_run": len(log.experiments), "candidates_generated": len(designed) + len(grid_challengers),
            "rejected_hypotheses": rejected, "holdout_evaluations": holdouts, "variants_of_baseline": variants,
            "tests_of_this_idea": same_idea, "research_runs_today": runs_today, "compute_units_today": compute_today,
            "datasets_used": len({e["dataset_id"] for e in log.experiments}),
            "policy": {k: v for k, v in POLICY.__dict__.items()}, "policy_hash": POLICY.policy_hash}


def check(u: dict, new_variant: bool, compute_units: int = 0) -> list[str]:
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
    if u.get("tests_of_this_idea", 0) >= POLICY.max_tests_per_idea:
        reasons.append(f"IDEA SATURATED: this statement has been tested {u['tests_of_this_idea']} ways; another "
                       "positive result would be unreliable (multiple testing) — a genuinely new idea is required")
    if u.get("compute_units_today", 0) + compute_units > POLICY.max_compute_units_per_day:
        reasons.append(f"daily compute budget exhausted ({u.get('compute_units_today', 0)} + {compute_units} > "
                       f"{POLICY.max_compute_units_per_day})")
    return reasons
