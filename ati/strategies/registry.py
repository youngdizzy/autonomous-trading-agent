"""Strategy registry and lifecycle.

- Registration is append-only: a (strategy_id, version) can never be re-registered with different
  content. Modifying a strategy means ``derive`` → a new version in CANDIDATE state.
- The CHAMPION can only be changed by ``apply_promotion`` with an *approved* ``PromotionRecord``
  from the promotion gate. There is no other path.
- Every lifecycle transition is appended to the journal when one is attached.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from ati.core.errors import LifecycleError, PromotionDenied, StrategyImmutableError
from ati.ledger.journal import Journal
from ati.strategies.base import StrategyDefinition


class Lifecycle(str, Enum):
    CANDIDATE = "CANDIDATE"
    CHALLENGER = "CHALLENGER"
    CHAMPION = "CHAMPION"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


_ALLOWED = {
    Lifecycle.CANDIDATE: {Lifecycle.CHALLENGER, Lifecycle.REJECTED},
    Lifecycle.CHALLENGER: {Lifecycle.REJECTED},  # → CHAMPION only via apply_promotion
    Lifecycle.CHAMPION: {Lifecycle.RETIRED},     # retired only when replaced via apply_promotion
    Lifecycle.RETIRED: set(),
    # A rejection is evidence about one hypothesis, not a permanent ban on the definition: a *new*
    # pre-registered hypothesis that re-tests the identical definition and survives development may
    # re-admit it as a challenger — the same gate a new candidate passes. The earlier rejection stays
    # in the append-only lifecycle history and in its immutable promotion record.
    Lifecycle.REJECTED: {Lifecycle.CHALLENGER},
}


class StrategyRegistry:
    def __init__(self, journal: Journal | None = None):
        self._defs: dict[str, StrategyDefinition] = {}
        self._state: dict[str, Lifecycle] = {}
        self._history: list[tuple[str, str, str, str]] = []
        self.journal = journal
        if journal is not None:
            self._replay(journal)

    def _replay(self, journal: Journal) -> None:
        """Rebuild from the hash-chained journal. A definition whose logic source has changed since it
        was recorded fails to load (code_hash mismatch) rather than silently meaning something new."""
        from ati.ledger.journal import decode
        from ati.market.models import Timeframe

        for entry in journal.entries():
            p = decode(entry.payload)
            if entry.type == "strategy_registered":
                d = p["definition"]
                definition = StrategyDefinition(d["strategy_id"], d["version"], d["kind"],
                                                tuple((k, v) for k, v in d["params"]), Timeframe(d["timeframe"]),
                                                d["created_at"], d["parent_hash"], d["description"], d["code_hash"])
                if definition.definition_hash != p["definition_hash"]:
                    raise StrategyImmutableError(f"{definition.key}: journaled definition hash mismatch")
                self._defs[definition.key] = definition
            elif entry.type == "strategy_lifecycle":
                self._state[p["key"]] = Lifecycle(p["to"])
                self._history.append((p["key"], p["from"], p["to"], p["reason"]))

    def _log(self, key: str, old: str, new: str, why: str) -> None:
        self._history.append((key, old, new, why))
        if self.journal is not None:
            self.journal.append("strategy_lifecycle", {"key": key, "from": old, "to": new, "reason": why})

    def register(self, definition: StrategyDefinition) -> StrategyDefinition:
        existing = self._defs.get(definition.key)
        if existing is not None:
            if existing.definition_hash != definition.definition_hash:
                raise StrategyImmutableError(f"{definition.key} already registered with different content")
            return existing
        self._defs[definition.key] = definition
        self._state[definition.key] = Lifecycle.CANDIDATE
        if self.journal is not None:
            self.journal.append("strategy_registered", {"key": definition.key, "definition_hash": definition.definition_hash,
                                                         "definition": definition})
        self._log(definition.key, "-", Lifecycle.CANDIDATE.value, "registered")
        return definition

    def derive(self, parent_key: str, params: dict[str, object], created_at: datetime, description: str) -> StrategyDefinition:
        parent = self.get(parent_key)
        version = 1 + max(d.version for d in self._defs.values() if d.strategy_id == parent.strategy_id)
        child = StrategyDefinition.create(parent.strategy_id, version, parent.kind, params, parent.timeframe,
                                          created_at, parent.definition_hash, description)
        return self.register(child)

    def get(self, key: str) -> StrategyDefinition:
        try:
            return self._defs[key]
        except KeyError:
            raise KeyError(f"unknown strategy {key}") from None

    def state(self, key: str) -> Lifecycle:
        return self._state[key]

    def transition(self, key: str, new: Lifecycle, reason: str) -> None:
        old = self._state[key]
        if new not in _ALLOWED[old]:
            raise LifecycleError(f"{key}: {old.value} → {new.value} not allowed via transition()")
        self._state[key] = new
        self._log(key, old.value, new.value, reason)

    def apply_promotion(self, record) -> None:
        from ati.validation.promotion import PromotionRecord

        if not isinstance(record, PromotionRecord) or not record.verify_integrity():
            raise PromotionDenied("apply_promotion requires an intact PromotionRecord")
        if not record.approved:
            raise PromotionDenied(f"promotion of {record.challenger_key} was denied: {record.reasons}")
        key = record.challenger_key
        if self.get(key).definition_hash != record.challenger_hash:
            raise PromotionDenied("promotion record does not match registered challenger definition")
        if self._state[key] is not Lifecycle.CHALLENGER:
            raise PromotionDenied(f"{key} is {self._state[key].value}, not CHALLENGER")
        current = self.champion()
        if (current.key if current else None) != record.champion_key:
            raise PromotionDenied("promotion record was decided against a different champion")
        if current is not None:
            self._state[current.key] = Lifecycle.RETIRED
            self._log(current.key, Lifecycle.CHAMPION.value, Lifecycle.RETIRED.value, f"replaced by {key}")
        self._state[key] = Lifecycle.CHAMPION
        self._log(key, Lifecycle.CHALLENGER.value, Lifecycle.CHAMPION.value, f"promotion {record.record_hash[:12]}")

    def champion(self) -> StrategyDefinition | None:
        champs = [k for k, s in self._state.items() if s is Lifecycle.CHAMPION]
        if len(champs) > 1:
            raise LifecycleError("more than one champion")
        return self._defs[champs[0]] if champs else None

    def keys(self, state: Lifecycle | None = None) -> list[str]:
        return [k for k, s in self._state.items() if state is None or s is state]

    @property
    def history(self) -> tuple:
        return tuple(self._history)
