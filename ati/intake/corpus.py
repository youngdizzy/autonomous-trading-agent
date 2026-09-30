"""Intake ledger: external sources, normalized records and research universes, in the research journal.

Journal record types (append-only, hash-chained with the rest of the research history):

  external_source     one per external_strategy_id: repository, commit, path, url, git blob id, SHA-256, size.
                      Raw bytes are not copied (the corpus is re-readable at the exact commit, and ``verify`` re-derives
                      the hash from it); the Vault's third-party source text is not redistributed.
  external_intake     one per (external_strategy_id, record_hash): the normalized record from ``build_record``.
  research_universe   one per selection batch: corpus size, selection method, every considered id and its state.

Idempotence: re-importing the same bytes at the same commit/path is a no-op. The same (repository, commit, path)
yielding different bytes is impossible for an honest corpus and raises ``ExternalSourceError`` — the old identity is
never overwritten. A changed file (a later commit) is a new identity; the old one remains.

Research: a COMPATIBLE record becomes a ``StrategyDefinition`` and goes through the *existing*
``run_research_cycle`` (pre-registration with the intake id in ``observation_refs``, walk-forward on the development
partition, adversarial checks, one sealed-holdout evaluation, promotion gate). Grid = the external parameters only:
the corpus is not used as an optimization loop. Criteria, windows and the candle minimum are the research protocol's
for that (symbol, timeframe) — never lowered for external candidates.
"""

from __future__ import annotations

import hashlib
from collections import Counter

from ati.core.canonical import sha256_hex
from ati.core.errors import ExternalSourceError
from ati.intake.normalize import RESEARCHABLE, build_record, definition
from ati.intake.source import GitCorpus, SourceArtifact
from ati.ledger.journal import decode

SELECTION_METHOD = ("deterministic pseudo-random order by sha256(salt + path), stratified by source language; "
                    "blind to names, descriptions, parameters and external performance claims")


class ExternalIntake:
    def __init__(self, research_journal):
        self.journal = research_journal
        self.sources: dict[str, dict] = {}
        self.records: dict[str, dict] = {}
        self._by_location: dict[tuple[str, str, str], str] = {}
        self.universes: dict[str, dict] = {}
        for e in research_journal.entries():
            if e.type == "external_source":
                p = decode(e.payload)
                self.sources[p["external_strategy_id"]] = p
                self._by_location[(p["source_repository"], p["source_commit"], p["source_path"])] = p["source_hash"]
            elif e.type == "external_intake":
                p = decode(e.payload)
                if sha256_hex({k: v for k, v in p["record"].items() if k != "record_hash"}) != p["record"]["record_hash"]:
                    raise ExternalSourceError(f"{p['record']['external_strategy_id']}: journaled record hash mismatch")
                self.records[p["record"]["external_strategy_id"]] = p["record"]
            elif e.type == "research_universe":
                p = decode(e.payload)
                self.universes[p["universe_id"]] = p

    # ------------------------------------------------------------------------------------------ import
    def ingest(self, artifact: SourceArtifact) -> dict:
        """Record one untrusted source and its normalized record. Returns the record (existing one if duplicate)."""
        if hashlib.sha256(artifact.raw).hexdigest() != artifact.source_hash:
            raise ExternalSourceError(f"{artifact.path}: bytes do not match the artifact's source hash")
        loc = (artifact.repository, artifact.commit, artifact.path)
        known = self._by_location.get(loc)
        if known is not None and known != artifact.source_hash:
            raise ExternalSourceError(f"{artifact.repository}@{artifact.commit[:12]}:{artifact.path} was recorded with "
                                      f"hash {known[:12]}, now {artifact.source_hash[:12]}; recorded identity kept")
        xid = artifact.external_strategy_id
        if xid not in self.sources:
            ident = artifact.identity()
            self.journal.append("external_source", ident)
            self.sources[xid] = ident
            self._by_location[loc] = artifact.source_hash
        record = build_record(artifact)
        existing = self.records.get(xid)
        if existing is not None and existing["record_hash"] == record["record_hash"]:
            return existing
        self.journal.append("external_intake", {"record": record})
        self.records[xid] = record
        return record

    def select_pilot(self, corpus: GitCorpus, per_language: dict[str, int], salt: str, prefix: str = "strategies/",
                     exclude: tuple[str, ...] = ("strategies/README.md",)) -> tuple[list[str], dict]:
        """Deterministic, claim-blind pilot selection. Reads only each file's ``> Source (<Language>)`` header."""
        paths = [p for p in corpus.list(prefix) if p.endswith(".md") and p not in exclude]
        strata: dict[str, list[str]] = {}
        for art in corpus.read_many(paths):
            label = next((line[10:-1] for line in art.raw.decode("utf-8", "replace").splitlines()
                          if line.startswith("> Source (") and line.endswith(")")), "UNKNOWN")
            strata.setdefault(label, []).append(art.path)
        chosen: list[str] = []
        for label in sorted(strata):
            ranked = sorted(strata[label], key=lambda p: hashlib.sha256(f"{salt}:{p}".encode()).hexdigest())
            chosen += ranked[:per_language.get(label, 0)]
        census = {"corpus_files": len(paths), "by_language": {k: len(v) for k, v in sorted(strata.items())}}
        return chosen, census

    def record_universe(self, corpus: GitCorpus, batch: str, selected: list[str], census: dict, salt: str,
                        per_language: dict[str, int]) -> dict:
        ids = []
        for path in selected:
            xid = next((i for i, s in self.sources.items() if (s["source_commit"], s["source_path"]) ==
                        (corpus.commit, path)), None)
            if xid is None or xid not in self.records:
                raise ExternalSourceError(f"{path} has not been ingested; a universe lists only recorded candidates")
            ids.append(xid)
        states = Counter(self.records[i]["compatibility_status"] for i in ids)
        body = {"corpus": {"repository": corpus.repository, "commit": corpus.commit, **census},
                "selection_batch": batch, "selection_method": SELECTION_METHOD, "selection_salt": salt,
                "per_language": dict(sorted(per_language.items())),
                "candidates": [{"external_strategy_id": i, "source_path": self.records[i]["source_path"],
                                "compatibility_status": self.records[i]["compatibility_status"]} for i in ids],
                "number_of_candidates_considered": len(ids),
                "number_of_candidates_accepted": sum(n for s, n in states.items() if s in RESEARCHABLE),
                "number_of_candidates_rejected": sum(n for s, n in states.items() if s not in RESEARCHABLE),
                "states": dict(sorted(states.items()))}
        uid = "universe_" + sha256_hex(body)[:20]
        if uid not in self.universes:
            self.journal.append("research_universe", {"universe_id": uid, **body})
            self.universes[uid] = {"universe_id": uid, **body}
        return self.universes[uid]

    # ------------------------------------------------------------------------------------------ audit
    def universe_summary(self, universe_id: str) -> dict:
        """Derived, never stored: how far each candidate in a universe got in the existing research workflow."""
        u = self.universes[universe_id]
        ids = {c["external_strategy_id"] for c in u["candidates"]}
        prereg = [e.payload["prereg"] for e in self.journal.entries("preregistration")
                  if ids & set(e.payload["prereg"]["observation_refs"])]
        return {"universe_id": universe_id, "corpus_files": u["corpus"]["corpus_files"],
                "considered": u["number_of_candidates_considered"], "accepted": u["number_of_candidates_accepted"],
                "rejected": u["number_of_candidates_rejected"],
                "researched": len({p["hypothesis_id"].split(":", 1)[0] for p in prereg}),
                "rejection_reasons": dict(Counter(r["state"] for i in ids for r in self.records[i]["reasons"])),
                "states": u["states"]}

    def verify(self, corpus: GitCorpus) -> list[str]:
        """Re-read every recorded source of this corpus commit; report any identity that no longer re-derives."""
        problems = []
        mine = [s for s in self.sources.values() if (s["source_repository"], s["source_commit"]) ==
                (corpus.repository, corpus.commit)]
        for art, s in zip(corpus.read_many([s["source_path"] for s in mine]), mine):
            if (art.source_hash, art.git_blob, art.external_strategy_id) != \
                    (s["source_hash"], s["source_git_blob"], s["external_strategy_id"]):
                problems.append(f"{s['source_path']}: recorded identity does not re-derive")
        return problems


def research_external(system, intake: ExternalIntake, external_strategy_id: str, full, *, protocol,
                      adversarial_policy=None, promotion_policy=None):
    """Research one COMPATIBLE external candidate with the existing workflow, under the protocol of the dataset's
    research dimension. Returns the workflow's ``CycleResult``. Refuses anything that is not COMPATIBLE.

    The holdout boundary is the protocol's (``holdout_fraction`` of ``full``, as the control plane computes it); the
    caller cannot choose it. ``run_research_cycle`` seals it before any development experiment runs."""
    from ati.research.adversarial import AdversarialPolicy
    from ati.research.hypothesis import Criterion
    from ati.research.workflow import run_research_cycle
    from ati.validation.promotion import PromotionPolicy

    record = intake.records[external_strategy_id]
    base = definition(record)                                     # raises unless COMPATIBLE and hash-consistent
    if (full.identity.symbol, full.identity.timeframe) != (protocol.symbol, protocol.timeframe) \
            or base.timeframe is not protocol.timeframe:
        raise ValueError("dataset, strategy and protocol must share one research dimension")
    boundary = full.candles[int(len(full) * (1 - protocol.holdout_fraction))].open_time
    statement = (f"External strategy {record['source_name']!r} ({record['source_repository']}@"
                 f"{record['source_commit'][:12]}:{record['source_path']}) has out-of-sample edge on "
                 f"{protocol.symbol} {protocol.timeframe.value} under TradeTown's own backtest")
    return run_research_cycle(
        system, full, boundary, hypothesis_id=f"H-{external_strategy_id}-{protocol.protocol_id}", statement=statement,
        base=base, grid=[base.param_dict],
        criteria=tuple(Criterion(m, op, t) for m, op, t in protocol.criteria),
        train_bars=protocol.train_bars, test_bars=protocol.test_bars,
        adversarial_policy=adversarial_policy or AdversarialPolicy(),
        promotion_policy=promotion_policy or PromotionPolicy(), min_candles=protocol.min_candles,
        observation_refs=(external_strategy_id, record["record_hash"]))
