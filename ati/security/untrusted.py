"""Untrusted content handling (prompt-injection defense).

External content — web pages, news, API responses, documents, messages, files, and Claude's own
prior outputs — is *data*. The defense is structural, not heuristic:

1. System/role instructions are constant strings in code; they are never built from external text.
2. External text is carried as ``UntrustedText`` and only ever rendered inside a fenced data block
   whose delimiter cannot be forged by the content (delimiter-like sequences are neutralized).
3. Claude output is never executed: it is parsed into a closed schema (``ati.agent.schema``) with an
   explicit action whitelist. An injected instruction can at most produce a proposal, which still
   has to pass schema validation and the deterministic risk engine.
4. Injection heuristics below only *flag* content for the record; safety never depends on them.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_FENCE_OPEN = "<<<UNTRUSTED_DATA source={source}>>>"
_FENCE_CLOSE = "<<<END_UNTRUSTED_DATA>>>"
_FENCE_TOKEN = re.compile(r"<<<\s*/?\s*(END_)?UNTRUSTED_DATA", re.IGNORECASE)

_INJECTION_HINTS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
        r"disregard (the )?(system|previous) (prompt|instructions)",
        r"you are now",
        r"new instructions?:",
        r"system prompt",
        r"override (the )?risk",
        r"disable (the )?(kill.?switch|risk)",
        r"set LIVE_TRADING",
        r"execute (this|the following) (command|order)",
        r"</?(system|assistant|user)>",
    )
]


@dataclass(frozen=True)
class UntrustedText:
    source: str
    text: str

    def flags(self) -> list[str]:
        normalized = unicodedata.normalize("NFKC", self.text)
        return [p.pattern for p in _INJECTION_HINTS if p.search(normalized)]


def _neutralize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return _FENCE_TOKEN.sub("[fence-token-removed]", text)


def fence(item: UntrustedText, max_chars: int = 4000) -> str:
    body = _neutralize(item.text)[:max_chars]
    source = re.sub(r"[^A-Za-z0-9_.:/-]", "_", item.source)[:80]
    return f"{_FENCE_OPEN.format(source=source)}\n{body}\n{_FENCE_CLOSE}"
