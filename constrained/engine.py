"""Generation engine: request registry, cancellation, timeouts, decoding loop.

Every request runs in its own worker with its own constraint states, so
concurrent requests are fully isolated and a failing request cannot affect
others.  Cancellation and deadlines are checked at every decoding step, so a
cancelled request releases its compute slot promptly.  The registry is
in-memory only: after a process restart no request exists, therefore an
unfinished request can never be reported as completed.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

from constrained.constraint import CompiledConstraint
from constrained.errors import ErrorCode
from constrained.model import MockModel
from constrained.vocab import Vocab


class RequestStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_STATUSES = frozenset(
    {
        RequestStatus.COMPLETED,
        RequestStatus.FAILED,
        RequestStatus.CANCELLED,
        RequestStatus.TIMED_OUT,
    }
)

#: Failure reasons surfaced on failed generation requests.
FAIL_MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"
FAIL_INTERNAL = "internal"


@dataclass
class GenerationRequest:
    request_id: str
    tenant_id: str
    constraint_ids: list[str]
    prompt: str
    max_tokens: int
    timeout_s: float | None
    status: RequestStatus = RequestStatus.QUEUED
    output_text: str | None = None
    error: dict | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def public_view(self) -> dict:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "constraint_ids": list(self.constraint_ids),
            "output_text": self.output_text,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class RequestRegistry:
    """In-memory, tenant-scoped registry of generation requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, GenerationRequest] = {}

    def add(self, request: GenerationRequest) -> None:
        with self._lock:
            self._by_id[request.request_id] = request

    def get(self, tenant_id: str, request_id: str) -> GenerationRequest | None:
        with self._lock:
            request = self._by_id.get(request_id)
        # Tenant isolation: a request owned by another tenant is
        # indistinguishable from a missing one.
        if request is None or request.tenant_id != tenant_id:
            return None
        return request


class GenerationEngine:
    def __init__(
        self,
        vocab: Vocab,
        model_factory: Callable[[int], MockModel] | None = None,
        max_workers: int = 8,
    ) -> None:
        self._vocab = vocab
        self._model_factory = model_factory or (
            lambda seed: MockModel(seed=seed, eos_token_id=vocab.eos_id)
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="gen"
        )
        self.registry = RequestRegistry()

    def submit(
        self,
        tenant_id: str,
        constraints: list[CompiledConstraint],
        prompt: str,
        max_tokens: int,
        timeout_s: float | None,
    ) -> GenerationRequest:
        request = GenerationRequest(
            request_id=f"r_{uuid.uuid4().hex}",
            tenant_id=tenant_id,
            constraint_ids=[c.constraint_id for c in constraints],
            prompt=prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        self.registry.add(request)
        self._executor.submit(self._run, request, list(constraints))
        return request

    def cancel(self, request: GenerationRequest) -> bool:
        """Signal cancellation; returns False if the request already finished."""
        if request.status in TERMINAL_STATUSES:
            return False
        request.cancel_event.set()
        return True

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------

    def _run(self, request: GenerationRequest, constraints: list[CompiledConstraint]) -> None:
        vocab = self._vocab
        request.status = RequestStatus.RUNNING
        seed = int(request.request_id[2:18], 16)
        model = self._model_factory(seed)
        states = [c.initial_state() for c in constraints]
        deadline = (
            time.monotonic() + request.timeout_s if request.timeout_s is not None else None
        )
        tokens: list[int] = []
        try:
            while True:
                if request.cancel_event.is_set():
                    self._finish(request, RequestStatus.CANCELLED)
                    return
                if deadline is not None and time.monotonic() > deadline:
                    self._finish(request, RequestStatus.TIMED_OUT)
                    return
                if len(tokens) >= request.max_tokens:
                    if all(st.is_accepting() for st in states):
                        request.output_text = "".join(vocab.tokens[t] for t in tokens)
                        self._finish(request, RequestStatus.COMPLETED)
                    else:
                        request.error = {
                            "code": ErrorCode.INVALID_REQUEST,
                            "reason": FAIL_MAX_TOKENS_EXCEEDED,
                            "message": "max_tokens reached before the output could "
                            "satisfy all constraints",
                        }
                        self._finish(request, RequestStatus.FAILED)
                    return
                allowed = self._allowed_tokens(states)
                if not allowed:
                    # Unreachable for satisfiable constraints thanks to
                    # liveness pruning; kept as a defensive guard.
                    request.error = {
                        "code": ErrorCode.INTERNAL,
                        "reason": FAIL_INTERNAL,
                        "message": "no token can continue the output",
                    }
                    self._finish(request, RequestStatus.FAILED)
                    return
                logits = model.logits(tuple(tokens), vocab.size)
                best = max(allowed, key=lambda t: logits[t])
                if best == vocab.eos_id:
                    request.output_text = "".join(vocab.tokens[t] for t in tokens)
                    self._finish(request, RequestStatus.COMPLETED)
                    return
                token = vocab.tokens[best]
                tokens.append(best)
                states = [st.advance(token) for st in states]
        except Exception as exc:  # never let one request affect others
            request.error = {
                "code": ErrorCode.INTERNAL,
                "reason": FAIL_INTERNAL,
                "message": f"generation failed: {exc}",
            }
            self._finish(request, RequestStatus.FAILED)

    def _allowed_tokens(self, states) -> set[int]:
        vocab = self._vocab
        if states:
            allowed = set(states[0].allowed_tokens(vocab))
            for st in states[1:]:
                allowed &= st.allowed_tokens(vocab)
        else:
            allowed = set(range(vocab.size)) - {vocab.eos_id}
        if all(st.is_accepting() for st in states):
            allowed.add(vocab.eos_id)
        return allowed

    @staticmethod
    def _finish(request: GenerationRequest, status: RequestStatus) -> None:
        request.status = status
        request.finished_at = time.time()
