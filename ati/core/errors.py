"""Error taxonomy. Every safety-relevant failure has its own type so callers can fail closed
precisely instead of catching broad exceptions."""

from __future__ import annotations


class AtiError(Exception):
    """Base for all system errors."""


class DataIntegrityError(AtiError):
    """Market data failed validation. ``code`` is a stable machine-readable reason."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


class HistoricalConflictError(AtiError):
    """Previously recorded history disagrees with newly received history. Fail closed."""


class ProvenanceError(AtiError):
    """Data claims a category (e.g. REAL) that its recorded provider provenance does not support."""


class DatasetIntegrityError(AtiError):
    """A dataset's content no longer matches its recorded identity."""


class LookaheadError(AtiError):
    """An attempt was made to access information not available at the decision cutoff."""


class HoldoutViolation(AtiError):
    """Holdout data was requested by a development process, or the holdout budget is spent."""


class StrategyImmutableError(AtiError):
    """An attempt was made to change a registered strategy definition in place."""


class ResearchIntegrityError(AtiError):
    """The research journal contradicts itself (e.g. two different pre-registrations for one id)."""


class MemoryIntegrityError(AtiError, ValueError):
    """A memory record is not supported by independent, correctly categorised canonical evidence."""


class LifecycleError(AtiError):
    """Illegal strategy / memory lifecycle transition."""


class PromotionDenied(AtiError):
    """A promotion was attempted without an approved promotion record."""


class SchemaViolation(AtiError):
    """External (e.g. Claude) output failed schema validation."""


class JournalCorruption(AtiError):
    """The append-only journal failed hash-chain verification or is truncated."""


class JournalWriteError(AtiError):
    """A durable write failed. Callers must not proceed as if it succeeded."""


class ModeMismatch(AtiError):
    """Paper/live/backtest state or data categories were mixed."""


class LiveTradingDisabled(AtiError):
    """LIVE mode is disabled in this build."""


class ExecutionHalted(AtiError):
    """Execution is halted until state is reconciled."""


class ApprovalInvalid(AtiError):
    """A risk approval was missing, forged, altered, expired, or already consumed."""


class BrokerError(AtiError):
    """Base for broker adapter failures."""


class BrokerTimeout(BrokerError):
    """No response: the order's true state is UNKNOWN."""


class BrokerUnavailable(BrokerError):
    """Broker could not be reached. For submissions the order's state is UNKNOWN."""


class BrokerRejected(BrokerError):
    """Broker explicitly and verifiably rejected the order."""


class ProviderError(AtiError):
    """Base for market data provider failures."""


class ProviderUnavailable(ProviderError):
    pass


class RateLimited(ProviderError):
    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class MalformedResponse(ProviderError):
    pass


class SecretLeakError(AtiError):
    """A registered secret was about to be persisted, logged, or sent to a prompt."""
