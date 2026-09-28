"""Secret isolation.

Secrets are only ever held in ``SecretValue`` (whose repr/str never reveal the value) and are
registered with a ``SecretGuard``. Every durable write (journal) and every prompt passes through
the guard, which refuses content containing a registered secret or a well-known credential shape.
Secrets are loaded from the environment by explicit name only — the system never scans or dumps
the environment.
"""

from __future__ import annotations

import os
import re

from ati.core.errors import SecretLeakError

_CREDENTIAL_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


class SecretValue:
    __slots__ = ("_value", "name")

    def __init__(self, name: str, value: str):
        if not isinstance(value, str) or len(value) < 8:
            raise ValueError("secret values must be strings of at least 8 characters")
        self.name = name
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"SecretValue({self.name}=***)"

    __str__ = __repr__

    def __reduce__(self):  # refuse pickling
        raise TypeError("SecretValue cannot be serialized")


class SecretGuard:
    def __init__(self) -> None:
        self._values: list[str] = []

    def register(self, secret: SecretValue) -> SecretValue:
        self._values.append(secret.reveal())
        return secret

    def scan(self, text: str, where: str = "output") -> None:
        for value in self._values:
            if value in text:
                raise SecretLeakError(f"registered secret detected in {where}")
        for pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(text):
                raise SecretLeakError(f"credential-shaped string detected in {where}")

    def redact(self, text: str) -> str:
        for value in self._values:
            text = text.replace(value, "[REDACTED]")
        for pattern in _CREDENTIAL_PATTERNS:
            text = pattern.sub("[REDACTED]", text)
        return text


def load_secret(name: str, guard: SecretGuard, environ: dict[str, str] | None = None) -> SecretValue | None:
    """Load one explicitly named secret. Returns None if absent — callers decide whether that
    blocks them. Never logs the value."""
    env = os.environ if environ is None else environ
    value = env.get(name)
    if value is None:
        return None
    return guard.register(SecretValue(name, value))
