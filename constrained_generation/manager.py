"""Request manager: admission, concurrency, isolation, cancellation.

Each submitted request gets its own session, cancel event and constraint
cursor; requests never share mutable state.  Compiled automata are immutable
and cached across requests.  Tenant isolation is enforced on every access:
a request is only visible to the tenant that created it.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .engine import ConstrainedSession, GenerationOutcome
from .errors import (
    ConstraintContradictoryError,
    ConstraintUnsatisfiableError,
    ErrorCode,
    RequestConflictError,
    RequestInvalidError,
    RequestNotFoundError,
)
from .model import ToyModel
from .regex import DFA, product
from .registry import ConstraintRegistry
from .schema import compile_definition, parse_definition
from .store import RequestRecord, RequestStatus, RequestStore
from .vocab import ALPHABET

MAX_CONSTRAINTS_PER_REQUEST = 8
MAX_TOKENS_LIMIT = 4096
MAX_TIMEOUT_MS = 60_000


@dataclass(frozen=True)
class GenerationParams:
    constraints: list[Any]
    max_tokens: int = 512
    timeout_ms: int = 10_000
    seed: int = 0
    wait: bool = True


class GenerationManager:
    def __init__(
        self,
        store: RequestStore,
        registry: ConstraintRegistry | None = None,
        max_workers: int = 8,
    ) -> None:
        self._store = store
        self._registry = registry if registry is not None else ConstraintRegistry()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="cg-worker"
        )
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()

    # ------------------------------------------------------------------
    # admission
    # ------------------------------------------------------------------

    def _validate_params(self, params: GenerationParams) -> None:
        if not isinstance(params.constraints, list) or not params.constraints:
            raise RequestInvalidError("'constraints' must be a non-empty array")
        if len(params.constraints) > MAX_CONSTRAINTS_PER_REQUEST:
            raise RequestInvalidError(
                f"at most {MAX_CONSTRAINTS_PER_REQUEST} constraints per request"
            )
        if not (1 <= params.max_tokens <= MAX_TOKENS_LIMIT):
            raise RequestInvalidError(
                f"'max_tokens' must be within 1..{MAX_TOKENS_LIMIT}"
            )
        if not (1 <= params.timeout_ms <= MAX_TIMEOUT_MS):
            raise RequestInvalidError(
                f"'timeout_ms' must be within 1..{MAX_TIMEOUT_MS}"
            )

    def _compile_constraints(self, raw_constraints: list[Any]) -> DFA:
        dfas: list[DFA] = []
        for raw in raw_constraints:
            definition = parse_definition(raw)
            dfa = self._registry.get_or_compile(
                definition.canonical_key(),
                lambda d=definition: _compile_to_dfa(d),
            )
            if dfa.is_empty():
                raise ConstraintUnsatisfiableError(
                    "constraint cannot be satisfied by any output",
                    details={"constraint": definition.canonical_key()},
                )
            dfas.append(dfa)
        combined = product(dfas)
        if combined.is_empty():
            raise ConstraintContradictoryError(
                "constraints are mutually contradictory: no output can satisfy "
                "all of them at once",
                details={"constraint_count": len(dfas)},
            )
        return combined

    def submit(self, tenant_id: str, params: GenerationParams) -> RequestRecord:
        """Validate constraints, then enqueue generation.

        Constraint problems raise before any state is persisted, so a rejected
        request never appears in the ledger.
        """
        self._validate_params(params)
        automaton = self._compile_constraints(params.constraints)

        now = time.time()
        request_id = uuid.uuid4().hex
        record = RequestRecord(
            request_id=request_id,
            tenant_id=tenant_id,
            status=RequestStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._cancel_events[request_id] = cancel_event
        self._store.create(record)

        deadline = time.monotonic() + params.timeout_ms / 1000.0
        future = self._executor.submit(
            self._execute, request_id, automaton, params, cancel_event, deadline
        )
        if params.wait:
            # The session enforces its own deadline; the small margin covers
            # queueing delay so waiters never block forever.
            remaining = max(deadline - time.monotonic(), 0.0) + 5.0
            try:
                future.result(timeout=remaining)
            except Exception:  # noqa: BLE001 - status is read from the store
                pass
        return self._store.get(request_id)

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        request_id: str,
        automaton: DFA,
        params: GenerationParams,
        cancel_event: threading.Event,
        deadline: float,
    ) -> None:
        try:
            self._store.transition(request_id, RequestStatus.RUNNING)
            session = ConstrainedSession(
                automaton=automaton,
                model=ToyModel(seed=params.seed),
                max_tokens=params.max_tokens,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            result = session.run()
            if result.outcome == GenerationOutcome.COMPLETED:
                self._store.transition(
                    request_id, RequestStatus.COMPLETED, result=result.text
                )
            elif result.outcome == GenerationOutcome.CANCELLED:
                self._store.transition(
                    request_id,
                    RequestStatus.CANCELLED,
                    error=_error_payload(
                        ErrorCode.GENERATION_CANCELLED, "request was cancelled"
                    ),
                )
            elif result.outcome == GenerationOutcome.TIMEOUT:
                self._store.transition(
                    request_id,
                    RequestStatus.FAILED,
                    error=_error_payload(
                        ErrorCode.GENERATION_TIMEOUT, "generation deadline exceeded"
                    ),
                )
            else:
                self._store.transition(
                    request_id,
                    RequestStatus.FAILED,
                    error=_error_payload(
                        ErrorCode.GENERATION_INCOMPLETE,
                        "token budget exhausted before a complete output was produced",
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - never leak worker failures
            self._store.transition(
                request_id,
                RequestStatus.FAILED,
                error=_error_payload(ErrorCode.INTERNAL, f"internal error: {exc}"),
            )
        finally:
            with self._cancel_lock:
                self._cancel_events.pop(request_id, None)

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------

    def get(self, tenant_id: str, request_id: str) -> RequestRecord:
        """Fetch a request owned by ``tenant_id`` (404 otherwise)."""
        record = self._store.get(request_id)
        if record is None or record.tenant_id != tenant_id:
            raise RequestNotFoundError("request not found")
        return record

    def cancel(self, tenant_id: str, request_id: str) -> RequestRecord:
        """Request cancellation of a running/pending request."""
        record = self.get(tenant_id, request_id)
        if record.status in (
            RequestStatus.COMPLETED,
            RequestStatus.FAILED,
            RequestStatus.CANCELLED,
            RequestStatus.ABORTED,
        ):
            raise RequestConflictError(
                f"request already in terminal status {record.status.value!r}"
            )
        with self._cancel_lock:
            event = self._cancel_events.get(request_id)
        if event is not None:
            event.set()
        return self._store.get(request_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _compile_to_dfa(definition) -> DFA:
    from .regex import compile_regex

    return compile_regex(compile_definition(definition), ALPHABET)


def _error_payload(code: ErrorCode, message: str) -> dict:
    return {"code": code.value, "message": message, "details": {}}
