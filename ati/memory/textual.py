"""News / text evidence — structure only. No news source is wired in this build (NOT IMPLEMENTED).

Text is kept in strictly separated layers so that a model's reading of an article can never pass for
what the article said, and an article can never pass for a market fact:

    RAW_SOURCE            the retrieved bytes (hash + text), with SOURCE_ID and SOURCE_TIMESTAMP
    EXTRACTED_FACTS       verbatim quotations of RAW_SOURCE (checked: each must occur in the raw text)
    MODEL_INTERPRETATION  what a model thinks the text means — labelled with the model, never a fact
    HYPOTHESIS            a research idea citing the layers above; it enters research only through a
                          pre-registered RESEARCH_REQUEST like any other hypothesis

Independence: every fact and interpretation derived from one SOURCE_ID counts as ONE source, however
many interpretations exist. Point-in-time: a layer is available no earlier than the raw source was both
published and retrieved. Raw text is untrusted data: it is never parsed for instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ati.core.canonical import sha256_text
from ati.core.errors import TextEvidenceError
from ati.core.time import ensure_utc

LAYERS = ("RAW_SOURCE", "EXTRACTED_FACTS", "MODEL_INTERPRETATION", "HYPOTHESIS")


@dataclass(frozen=True)
class RawSource:
    source_id: str
    publisher: str
    source_timestamp: datetime     # when the publisher says it was published
    retrieved_at: datetime         # when this system obtained it
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_timestamp", ensure_utc(self.source_timestamp))
        object.__setattr__(self, "retrieved_at", ensure_utc(self.retrieved_at))
        if not self.source_id or not self.publisher or not self.text:
            raise TextEvidenceError("raw source needs source_id, publisher and text")

    @property
    def raw_sha256(self) -> str:
        return sha256_text(self.text)

    @property
    def available_at(self) -> datetime:
        return max(self.source_timestamp, self.retrieved_at)


@dataclass(frozen=True)
class ExtractedFact:
    source_id: str
    quote: str                     # must occur verbatim in the raw text
    layer: str = "EXTRACTED_FACTS"


@dataclass(frozen=True)
class Interpretation:
    source_id: str
    model: str
    text: str
    produced_at: datetime
    layer: str = "MODEL_INTERPRETATION"


class TextEvidence:
    def __init__(self):
        self.sources: dict[str, RawSource] = {}
        self.facts: list[ExtractedFact] = []
        self.interpretations: list[Interpretation] = []

    def add_source(self, src: RawSource) -> str:
        existing = self.sources.get(src.source_id)
        if existing is not None and existing.raw_sha256 != src.raw_sha256:
            raise TextEvidenceError(f"{src.source_id}: raw source changed after it was recorded (append a new id)")
        self.sources[src.source_id] = src
        return src.raw_sha256

    def add_fact(self, fact: ExtractedFact) -> None:
        src = self.sources.get(fact.source_id)
        if src is None:
            raise TextEvidenceError(f"fact cites unknown source {fact.source_id}")
        if not fact.quote.strip() or fact.quote not in src.text:
            raise TextEvidenceError("an extracted fact must be a verbatim quotation of the raw source; "
                                    "anything else is interpretation")
        self.facts.append(fact)

    def add_interpretation(self, interp: Interpretation) -> None:
        src = self.sources.get(interp.source_id)
        if src is None:
            raise TextEvidenceError(f"interpretation cites unknown source {interp.source_id}")
        if ensure_utc(interp.produced_at) < src.available_at:
            raise TextEvidenceError("interpretation predates its source's availability (look-ahead)")
        self.interpretations.append(interp)

    def independent_sources(self, source_ids: list[str], as_of: datetime) -> int:
        """Distinct source ids available at ``as_of``. Many interpretations of one article are one source."""
        as_of = ensure_utc(as_of)
        return len({sid for sid in source_ids if sid in self.sources and self.sources[sid].available_at <= as_of})

    def context(self, as_of: datetime) -> list[dict]:
        """Point-in-time, layer-labelled view for a prompt. Raw text is marked untrusted data."""
        as_of = ensure_utc(as_of)
        out = []
        for sid, src in sorted(self.sources.items()):
            if src.available_at > as_of:
                continue
            out.append({"source_id": sid, "publisher": src.publisher, "source_timestamp": src.source_timestamp.isoformat(),
                        "RAW_SOURCE": {"sha256": src.raw_sha256, "untrusted_text": src.text[:1000]},
                        "EXTRACTED_FACTS": [f.quote for f in self.facts if f.source_id == sid],
                        "MODEL_INTERPRETATION": [{"model": i.model, "text": i.text} for i in self.interpretations
                                                 if i.source_id == sid and ensure_utc(i.produced_at) <= as_of]})
        return out
