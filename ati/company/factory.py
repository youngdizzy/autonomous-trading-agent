"""Candidate strategy factory — lineage and evaluation trail.

Candidates are generated only by the existing research workflow (``registry.register`` / ``registry.derive``
inside ``run_research_cycle``) from a pre-registered hypothesis; this module never creates, mutates or
promotes a strategy. It records a lineage record per candidate and derives, from the journals alone, how
far the candidate has progressed:

    HYPOTHESIS → CANDIDATE_GENERATED → DEVELOPMENT → WFO → ROBUSTNESS → ADVERSARIAL → HOLDOUT → VALIDATION
    → CHALLENGER → PROMOTION_REVIEW

A stage is "reached" only if the journal holds the corresponding record for this exact fingerprint. The
registry's own lifecycle (CANDIDATE / CHALLENGER / CHAMPION / REJECTED / RETIRED) stays the authority for
what a strategy may do.
"""

from __future__ import annotations

from ati.ledger.journal import decode
from ati.research.hypothesis import ResearchLog

STAGES = ("HYPOTHESIS", "CANDIDATE_GENERATED", "DEVELOPMENT", "WFO", "ROBUSTNESS", "ADVERSARIAL", "HOLDOUT",
          "VALIDATION", "CHALLENGER", "PROMOTION_REVIEW")
_ROBUSTNESS_QUESTIONS = ("doubled costs", "parameter changes", "execution delay", "different regimes", "liquidity")


def _v(value) -> str:
    return str(getattr(value, "value", value))


def experiment_id(hypothesis_id: str, design_hash: str) -> str:
    from ati.core.canonical import sha256_hex
    return "exp_" + sha256_hex({"h": hypothesis_id, "d": design_hash})[:20]


def record_lineage(system, definition, *, hypothesis_id: str, experiment_id: str, dev_dataset_id: str,
                   full_dataset_id: str, baseline_hash: str | None) -> bool:
    """Idempotent: one lineage record per (fingerprint, hypothesis)."""
    s = system
    for e in s.research_journal.entries("candidate_lineage"):
        if e.payload["fingerprint"] == definition.definition_hash and e.payload["hypothesis_id"] == hypothesis_id:
            return False
    s.research_journal.append("candidate_lineage", {
        "fingerprint": definition.definition_hash, "key": definition.key, "parent_fingerprint": definition.parent_hash,
        "baseline_fingerprint": baseline_hash, "hypothesis_id": hypothesis_id, "experiment_id": experiment_id,
        "datasets": {"development": dev_dataset_id, "full": full_dataset_id}, "params": definition.param_dict,
        "code_hash": definition.code_hash, "kind": definition.kind,
        "provenance": {"data_status": s.data_status.value, "provider": s.provider.name, "generated_by": "research_workflow"}})
    return True


def stages(system, fingerprint: str) -> dict:
    """Derived evaluation trail for one candidate fingerprint."""
    s = system
    lineage = [e.payload for e in s.research_journal.entries("candidate_lineage") if e.payload["fingerprint"] == fingerprint]
    log = ResearchLog(s.research_journal)
    reached: dict[str, str] = {}
    key = None
    for k in s.strategies.keys():
        if s.strategies.get(k).definition_hash == fingerprint:
            key = k
            reached["CANDIDATE_GENERATED"] = k
    hyps = {l["hypothesis_id"] for l in lineage}
    for h in hyps:
        if log.status(h) != "UNKNOWN":
            reached["HYPOTHESIS"] = h
        rows = [e for e in log.experiments if e["hypothesis_id"] == h and e["stage"] == "walk_forward_oos"]
        if rows:
            reached["DEVELOPMENT"] = rows[-1]["dataset_id"]
            reached["WFO"] = _v(rows[-1]["verdict"])
    advs = [decode(e.payload)["report"] for e in s.research_journal.entries("adversarial_report")
            if e.payload["report"]["strategy_hash"] == fingerprint]
    if advs:
        objections = advs[-1]["objections"]
        robust = [o for o in objections if any(q in o["question"] for q in _ROBUSTNESS_QUESTIONS)]
        reached["ROBUSTNESS"] = "FAIL" if any(o["verdict"] == "FAIL" for o in robust) else "NON_BLOCKING"
        blocking = any(o["verdict"] in ("FAIL", "INSUFFICIENT_EVIDENCE") for o in objections)
        reached["ADVERSARIAL"] = "BLOCKING" if blocking else "NON_BLOCKING"
    holds = [e.payload for e in s.research_journal.entries("holdout_access") if e.payload["strategy_hash"] == fingerprint]
    if holds:
        rows = [e for e in log.experiments if e["stage"] == "holdout" and log.get(e["hypothesis_id"]).strategy_hash == fingerprint]
        reached["HOLDOUT"] = _v(rows[-1]["verdict"]) if rows else "ACCESSED (no result recorded)"
    promos = [decode(e.payload)["record"] for e in s.research_journal.entries("promotion_decision")
              if e.payload["record"]["challenger_hash"] == fingerprint]
    if promos:
        reached["VALIDATION"] = "APPROVED" if promos[-1]["approved"] else "DENIED"
        reached["PROMOTION_REVIEW"] = promos[-1]["record_hash"][:16]
    if key is not None and any(h[0] == key and h[2] == "CHALLENGER" for h in s.strategies.history):
        reached["CHALLENGER"] = "entered"
    furthest = max((STAGES.index(k) for k in reached), default=-1)
    return {"fingerprint": fingerprint, "key": key, "lineage": lineage,
            "registry_state": s.strategies.state(key).value if key else "NOT_AVAILABLE",
            "stages": {st: reached.get(st, "NOT_REACHED") for st in STAGES},
            "furthest_stage": STAGES[furthest] if furthest >= 0 else "NOT_AVAILABLE"}
