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


APPROVED_TIMEFRAMES = ("1h", "4h")       # the approved research dimensions; nothing else is registrable
LEGACY_SCOPE = "LEGACY_UNSCOPED"         # lifecycle entries recorded before scoping (no symbol): never a live scope


class StrategyRegistry:
    """Definitions are keyed by ``registry_key`` (``strategy_id@vN/<timeframe>``), so one strategy can exist on 1h
    and 4h without collision. Lifecycle state (CANDIDATE/CHALLENGER/CHAMPION/RETIRED/REJECTED) is scoped by
    *research dimension*: (registry_key, symbol) — the timeframe is part of the key. There is no global champion:
    ``champion(symbol, timeframe)`` answers for one dimension only, and every lifecycle question requires the
    symbol explicitly (no implicit default).

    Legacy journals (before scoping) are replayed without guessing: an old ``strategy_id@vN`` key maps to exactly
    one recorded definition, whose timeframe is part of its recorded content; old lifecycle entries carry no
    symbol and are kept under ``LEGACY_UNSCOPED``, which is never a live research dimension.
    """

    def __init__(self, journal: Journal | None = None):
        self._defs: dict[str, StrategyDefinition] = {}
        self._state: dict[tuple[str, str], Lifecycle] = {}          # (registry_key, symbol) → state
        self._history: list[tuple[str, str, str, str, str]] = []    # (registry_key, from, to, reason, symbol)
        self._legacy_alias: dict[str, str] = {}                     # old strategy_id@vN → registry_key
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
                self._defs[definition.registry_key] = definition
                if p["key"] != definition.registry_key:              # legacy entry: key without timeframe
                    self._legacy_alias[p["key"]] = definition.registry_key
            elif entry.type == "strategy_lifecycle":
                key = p["key"] if "/" in p["key"] else self._legacy_alias.get(p["key"], p["key"])
                symbol = p.get("symbol", LEGACY_SCOPE)
                if p["from"] != "-":                                   # registration entries carry no lifecycle state
                    self._state[(key, symbol)] = Lifecycle(p["to"])
                self._history.append((key, p["from"], p["to"], p["reason"], symbol))

    @property
    def legacy_records(self) -> dict:
        """What was recorded before scoping, classified explicitly (never re-labelled)."""
        return {"legacy_keys": dict(self._legacy_alias),
                "legacy_unscoped_states": {k: s.value for (k, sym), s in self._state.items() if sym == LEGACY_SCOPE}}

    def _log(self, key: str, symbol: str, old: str, new: str, why: str) -> None:
        self._history.append((key, old, new, why, symbol))
        if self.journal is not None:
            self.journal.append("strategy_lifecycle", {"key": key, "symbol": symbol, "from": old, "to": new,
                                                       "reason": why})

    def register(self, definition: StrategyDefinition) -> StrategyDefinition:
        if definition.timeframe.value not in APPROVED_TIMEFRAMES:
            raise LifecycleError(f"{definition.registry_key}: timeframe {definition.timeframe.value} is not an approved "
                                 f"research timeframe {APPROVED_TIMEFRAMES}")
        key = definition.registry_key
        existing = self._defs.get(key)
        if existing is not None:
            if existing.definition_hash != definition.definition_hash:
                raise StrategyImmutableError(f"{key} already registered with different content")
            return existing
        self._defs[key] = definition
        if self.journal is not None:
            self.journal.append("strategy_registered", {"key": key, "definition_hash": definition.definition_hash,
                                                         "definition": definition})
        self._log(key, "*", "-", Lifecycle.CANDIDATE.value, "registered")   # definition-level audit entry, no scope
        return definition

    def derive(self, parent_key: str, params: dict[str, object], created_at: datetime, description: str) -> StrategyDefinition:
        parent = self.get(parent_key)
        version = 1 + max(d.version for d in self._defs.values()
                          if d.strategy_id == parent.strategy_id and d.timeframe is parent.timeframe)
        child = StrategyDefinition.create(parent.strategy_id, version, parent.kind, params, parent.timeframe,
                                          created_at, parent.definition_hash, description)
        return self.register(child)

    def get(self, key: str) -> StrategyDefinition:
        if "/" not in key:
            raise KeyError(f"{key!r} has no timeframe: registry lookups need strategy_id@vN/<timeframe>")
        try:
            return self._defs[key]
        except KeyError:
            raise KeyError(f"unknown strategy {key}") from None

    def definitions(self, timeframe=None) -> list[str]:
        """Registry keys of every registered definition (optionally one timeframe)."""
        return [k for k, d in self._defs.items() if timeframe is None or d.timeframe is timeframe]

    @staticmethod
    def _symbol(symbol: str) -> str:
        if not isinstance(symbol, str) or not symbol or symbol == LEGACY_SCOPE or symbol == "*":
            raise LifecycleError(f"a research dimension needs an explicit symbol, got {symbol!r}")
        return symbol

    def state(self, key: str, symbol: str) -> Lifecycle:
        self.get(key)
        return self._state.get((key, self._symbol(symbol)), Lifecycle.CANDIDATE)

    def transition(self, key: str, symbol: str, new: Lifecycle, reason: str) -> None:
        old = self.state(key, symbol)
        if new not in _ALLOWED[old]:
            raise LifecycleError(f"{key} [{symbol}]: {old.value} → {new.value} not allowed via transition()")
        self._state[(key, symbol)] = new
        self._log(key, symbol, old.value, new.value, reason)

    def apply_promotion(self, record) -> None:
        from ati.validation.promotion import PromotionRecord

        if not isinstance(record, PromotionRecord) or not record.verify_integrity():
            raise PromotionDenied("apply_promotion requires an intact PromotionRecord")
        if not record.approved:
            raise PromotionDenied(f"promotion of {record.challenger_key} was denied: {record.reasons}")
        if not record.symbol or not record.timeframe:
            raise PromotionDenied("promotion record has no research dimension (symbol, timeframe)")
        key, symbol = record.challenger_key, record.symbol
        try:
            challenger = self.get(key)
        except KeyError as exc:
            raise PromotionDenied(str(exc)) from None
        if challenger.definition_hash != record.challenger_hash:
            raise PromotionDenied("promotion record does not match registered challenger definition")
        if challenger.timeframe.value != record.timeframe:
            raise PromotionDenied("promotion record was decided for a different timeframe")
        if self.state(key, symbol) is not Lifecycle.CHALLENGER:
            raise PromotionDenied(f"{key} is {self.state(key, symbol).value} for {symbol}, not CHALLENGER")
        current = self.champion(symbol, challenger.timeframe)
        if (current.registry_key if current else None) != record.champion_key:
            raise PromotionDenied("promotion record was decided against a different champion")
        if current is not None:
            self._state[(current.registry_key, symbol)] = Lifecycle.RETIRED
            self._log(current.registry_key, symbol, Lifecycle.CHAMPION.value, Lifecycle.RETIRED.value, f"replaced by {key}")
        self._state[(key, symbol)] = Lifecycle.CHAMPION
        self._log(key, symbol, Lifecycle.CHALLENGER.value, Lifecycle.CHAMPION.value, f"promotion {record.record_hash[:12]}")

    def champion(self, symbol: str, timeframe) -> StrategyDefinition | None:
        """The champion of ONE research dimension (symbol, timeframe). Never global, never a fallback."""
        self._symbol(symbol)
        if getattr(timeframe, "value", None) not in APPROVED_TIMEFRAMES:
            raise LifecycleError(f"champion lookup needs an approved Timeframe, got {timeframe!r}")
        champs = [k for (k, sym), s in self._state.items()
                  if s is Lifecycle.CHAMPION and sym == symbol and self._defs[k].timeframe is timeframe]
        if len(champs) > 1:
            raise LifecycleError(f"more than one champion for {symbol} {timeframe.value}")
        return self._defs[champs[0]] if champs else None

    def keys(self, state: Lifecycle | None = None, symbol: str | None = None) -> list[str]:
        """Without ``state``: every registered definition. With ``state``: definitions in that state for ``symbol``
        (a lifecycle state only exists within a research dimension, so the symbol is required)."""
        if state is None:
            return self.definitions()
        self._symbol(symbol)
        return [k for k in self._defs if self.state(k, symbol) is state]

    @property
    def history(self) -> tuple:
        return tuple(self._history)
