"""Constrained generation engine.

Holds the (already trimmed) product automaton for one request and runs the
generate loop.  At every step the candidate set is intersected with the
automaton's allowed tokens, so a chosen token can always still complete into
a valid output.  Cancellation and deadlines are checked between steps, so a
cancelled request frees its worker promptly.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .model import ToyModel
from .regex import DFA
from .vocab import STOP


class GenerationOutcome(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class GenerationResult:
    outcome: GenerationOutcome
    text: str
    tokens_generated: int


class ConstrainedSession:
    """Per-request generation state.  Never shared across requests."""

    def __init__(
        self,
        automaton: DFA,
        model: ToyModel,
        max_tokens: int,
        cancel_event: threading.Event,
        deadline: float | None,
        step_hook: Callable[[int, str], None] | None = None,
    ) -> None:
        if automaton.is_empty():
            raise ValueError("cannot generate from an empty constraint automaton")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self._automaton = automaton
        self._model = model
        self._max_tokens = max_tokens
        self._cancel_event = cancel_event
        self._deadline = deadline
        self._step_hook = step_hook

    def run(self) -> GenerationResult:
        state = self._automaton.start
        text = ""
        for step_index in range(self._max_tokens):
            if self._cancel_event.is_set():
                return GenerationResult(GenerationOutcome.CANCELLED, text, step_index)
            if self._deadline is not None and time.monotonic() > self._deadline:
                return GenerationResult(GenerationOutcome.TIMEOUT, text, step_index)
            if self._step_hook is not None:
                self._step_hook(step_index, text)

            allowed = sorted(self._automaton.allowed_tokens(state))
            candidates = list(allowed)
            if self._automaton.is_accept(state):
                candidates.append(STOP)
            if not candidates:
                # Unreachable for a trimmed automaton; guard anyway.
                return GenerationResult(GenerationOutcome.INCOMPLETE, text, step_index)

            logits = self._model.logits(text, candidates)
            best = max(range(len(candidates)), key=lambda i: logits[i])
            choice = candidates[best]
            if choice == STOP:
                return GenerationResult(GenerationOutcome.COMPLETED, text, step_index)
            nxt = self._automaton.step(state, choice)
            if nxt is None:  # pragma: no cover - defensive
                return GenerationResult(GenerationOutcome.INCOMPLETE, text, step_index)
            state = nxt
            text += choice

        if self._automaton.is_accept(state):
            return GenerationResult(
                GenerationOutcome.COMPLETED, text, self._max_tokens
            )
        return GenerationResult(GenerationOutcome.INCOMPLETE, text, self._max_tokens)
