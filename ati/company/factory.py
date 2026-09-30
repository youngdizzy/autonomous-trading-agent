"""Candidate strategy factory — lineage and evaluation trail.

Candidates are generated only by the existing research workflow (``registry.register`` / ``registry.derive``
inside ``run_research_cycle``) from a pre-registered hypothesis; this module never creates, mutates or
promotes a strategy. It records a lineage record per candidate and derives, from the journals alone, how
far the candidate has progressed:

    HYPOTHESIS → CANDIDATE_GENERATED → DEVELOPMENT_TEST → WALK_FORWARD → ROBUSTNESS → ADVERSARIAL → HOLDOUT
    → VALIDATION → CHALLENGER → PROMOTION_REVIEW

A stage is "reached" only if the journal holds the corresponding record for this exact fingerprint. The
registry's own lifecycle (CANDIDATE / CHALLENGER / CHAMPION / REJECTED / RETIRED) stays the authority for
what a strategy may do.
"""

from __future__ import annotations

from ati.ledger.journal import decode
from ati.research.hypothesis import ResearchLog

STAGES = ("HYPOTHESIS", "CANDIDATE_GENERATED", "DEVELOPMENT_TEST", "WALK_FORWARD", "ROBUSTNESS", "ADVERSARIAL",
          "HOLDOUT", "VALIDATION", "CHALLENGER", "PROMOTION_REVIEW")
_ROBUSTNESS_QUESTIONS = ("doubled costs", "parameter changes", "execution delay", "different regimes", "liquidity")


def _v(value) -> str:
    return str(getattr(value, "value", value))


def candidate_id(fingerprint: str, hypothesis_id: str) -> str:
    from ati.core.canonical import sha256_hex
    return "cand_" + sha256_hex({"f": fingerprint, "h": hypothesis_id})[:20]


def promotion_id(record_hash: str) -> str:
    return "prom_" + record_hash[:20]


def attempts(system, hypothesis_id: str) -> dict:
    """"How many attempts did we make before finding this result?" — counted from the research journal up to
    (not including) this hypothesis's first pre-registration. Failures are counted, never hidden."""
    s = system
    entries = list(s.research_journal.entries())
    root = hypothesis_id.split(":", 1)[0]
    start = next((e.seq for e in entries if e.type == "preregistration"
                  and e.payload["prereg"]["hypothesis_id"].split(":", 1)[0] == root), None)
    before = [e for e in entries if start is None or e.seq < start]
    exps = [decode(e.payload) for e in before if e.type == "experiment"]
    roots = {x["hypothesis_id"].split(":", 1)[0] for x in exps}
    failed = {x["hypothesis_id"].split(":", 1)[0] for x in exps if _v(x["verdict"]) in ("FAIL", "INSUFFICIENT_EVIDENCE")}
    statements = {}
    for e in before:
        if e.type == "preregistration":
            statements[e.payload["prereg"]["hypothesis_id"].split(":", 1)[0]] = e.payload["prereg"]["statement"]
    this = next((e.payload["prereg"]["statement"] for e in entries if e.type == "preregistration"
                 and e.payload["prereg"]["hypothesis_id"] == root), None)
    norm = lambda x: " ".join(str(x).lower().split())   # noqa: E731
    return {"root_hypotheses_tested_before": len(roots), "experiments_before": len(exps),
            "failed_or_insufficient_before": len(failed & roots),
            "candidates_generated_before": sum(1 for e in before if e.type == "candidate_lineage"),
            "rejected_strategies_before": sum(1 for e in before if e.type == "strategy_lifecycle" and e.payload["to"] == "REJECTED"),
            "holdout_evaluations_before": sum(1 for e in before if e.type == "holdout_access"),
            "same_idea_tested_before": sum(1 for h, st in statements.items() if h in roots and this and norm(st) == norm(this)),
            "independent_datasets_before": len({x["dataset_id"] for x in exps})}


def review(system, fingerprint: str) -> dict:
    """Champion/challenger evidence profile for one fingerprint — every dimension separately, NOT_AVAILABLE when
    the evidence does not exist. Informational: promotion is decided only by the promotion gate."""
    s = system
    log = ResearchLog(s.research_journal)
    hyps = [e.payload["prereg"]["hypothesis_id"] for e in s.research_journal.entries("preregistration")
            if e.payload["prereg"]["strategy_hash"] == fingerprint]
    roots = {h.split(":", 1)[0] for h in hyps}   # a challenger is locked by "<root>:holdout"; its WFO row is the root's
    wf = [x for x in log.experiments if x["stage"] == "walk_forward_oos" and x["hypothesis_id"] in roots]
    hold = [x for x in log.experiments if x["stage"] == "holdout" and x["hypothesis_id"] in hyps]
    advs = [decode(e.payload)["report"] for e in s.research_journal.entries("adversarial_report")
            if e.payload["report"]["strategy_hash"] == fingerprint]
    objections = {o["question"]: _v(o["verdict"]) for o in (advs[-1]["objections"] if advs else [])}

    def pick(word):
        return next((v for q, v in objections.items() if word in q), "NOT_AVAILABLE")

    m = wf[-1]["metrics"] if wf else None
    hm = hold[-1]["metrics"] if hold else None
    get = lambda d, k: d.get(k) if d else "NOT_AVAILABLE"   # noqa: E731
    return {"fingerprint": fingerprint,
            "development": {"return": get(m, "net_return"), "drawdown": get(m, "max_drawdown"),
                            "sharpe": get(m, "sharpe_annualized"), "trade_count": get(m, "n_trades"),
                            "costs": get(m, "cost_share_of_gross"), "volatility": "NOT_AVAILABLE (not measured)"},
            "walk_forward": _v(wf[-1]["verdict"]) if wf else "NOT_RUN",
            "robustness": {"costs": pick("costs"), "timing": pick("delay"), "liquidity": pick("liquidity")},
            "adversarial": ("BLOCKING" if any(v in ("FAIL", "INSUFFICIENT_EVIDENCE") for v in objections.values())
                            else "NON_BLOCKING") if objections else "NOT_RUN",
            "regime_behavior": pick("regimes"), "parameter_sensitivity": pick("parameter"),
            "holdout": {"verdict": _v(hold[-1]["verdict"]), "return": get(hm, "net_return"),
                        "trade_count": get(hm, "n_trades")} if hold else "NOT_RUN",
            "attempts": attempts(s, hyps[0]) if hyps else "NOT_AVAILABLE"}


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
        "candidate_id": candidate_id(definition.definition_hash, hypothesis_id), "attempts": attempts(s, hypothesis_id),
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
            reached["DEVELOPMENT_TEST"] = rows[-1]["dataset_id"]
            reached["WALK_FORWARD"] = _v(rows[-1]["verdict"])
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
        reached["PROMOTION_REVIEW"] = "promotion_id " + promotion_id(promos[-1]["record_hash"])
    if key is not None and any(h[0] == key and h[2] == "CHALLENGER" for h in s.strategies.history):
        reached["CHALLENGER"] = "entered"
    furthest = max((STAGES.index(k) for k in reached), default=-1)
    return {"fingerprint": fingerprint, "key": key, "lineage": lineage,
            "registry_state": s.strategies.state(key).value if key else "NOT_AVAILABLE",
            "stages": {st: reached.get(st, "NOT_REACHED") for st in STAGES},
            "furthest_stage": STAGES[furthest] if furthest >= 0 else "NOT_AVAILABLE"}
