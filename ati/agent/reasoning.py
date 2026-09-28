"""Reasoning clients: how the deterministic system obtains Claude's output.

- ``FileExchangeClient`` (Claude-native): the system writes a request file; a Claude session
  (e.g. woken by a scheduled Routine) reads it and writes a response file; the next tick picks it
  up. No server, no API key, no extra infrastructure. While a response is pending the loop takes
  no new risk.
- ``ScriptedReasoningClient`` (MOCK): deterministic responses for tests and demos.
- A direct Anthropic API client is NOT IMPLEMENTED: no project API key is provisioned
  (BLOCKED — CREDENTIALS). It would implement the same protocol.

All clients enforce a per-tick call budget; exceeding it raises and the pipeline defaults to
NO_TRADE.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Protocol

from ati.core.errors import AtiError


class ReasoningPending(AtiError):
    """The reasoning request was issued but no response exists yet."""


class ReasoningBudgetExceeded(AtiError):
    pass


class ReasoningClient(Protocol):
    label: str

    def complete(self, role: str, request_id: str, prompt: str) -> str: ...


class Budget:
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.used = 0

    def spend(self) -> None:
        if self.used >= self.max_calls:
            raise ReasoningBudgetExceeded(f"reasoning budget of {self.max_calls} calls exhausted")
        self.used += 1

    def reset(self) -> None:
        self.used = 0


class ScriptedReasoningClient:
    """MOCK reasoning. ``script`` maps role → response text or callable(prompt) → text."""

    label = "MOCK"

    def __init__(self, script: dict[str, str | Callable[[str], str]], budget: Budget | None = None):
        self.script = script
        self.budget = budget or Budget(10)
        self.calls: list[tuple[str, str]] = []

    def complete(self, role: str, request_id: str, prompt: str) -> str:
        self.budget.spend()
        self.calls.append((role, request_id))
        response = self.script.get(role)
        if response is None:
            raise ReasoningPending(f"no scripted response for {role}")
        return response(prompt) if callable(response) else response


class FileExchangeClient:
    label = "CLAUDE_FILE_EXCHANGE"

    def __init__(self, root: Path | str, budget: Budget | None = None):
        self.root = Path(root)
        (self.root / "requests").mkdir(parents=True, exist_ok=True)
        (self.root / "responses").mkdir(parents=True, exist_ok=True)
        self.budget = budget or Budget(10)

    @staticmethod
    def _name(role: str, request_id: str) -> str:
        return f"{role}__{hashlib.sha256(request_id.encode()).hexdigest()[:16]}"

    def complete(self, role: str, request_id: str, prompt: str) -> str:
        name = self._name(role, request_id)
        response = self.root / "responses" / f"{name}.json"
        if response.exists():
            self.budget.spend()
            return response.read_text(encoding="utf-8")
        request = self.root / "requests" / f"{name}.md"
        if not request.exists():
            request.write_text(prompt, encoding="utf-8")
        raise ReasoningPending(f"awaiting Claude response at {response}")
