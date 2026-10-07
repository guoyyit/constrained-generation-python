"""HTTP service layer: FastAPI app exposing constrained generation.

Error convention (applied uniformly, including request-validation errors)::

    {"error": {"code": <ErrorCode>, "message": <str>, "reason": <optional str>}}

Authentication is via the ``X-API-Key`` header; each API key maps to one
tenant.  Constraints and generation requests are strictly tenant-scoped:
cross-tenant access is indistinguishable from a missing resource (404).
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from typing import Any, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from constrained.constraint import CompiledConstraint, ConstraintCache
from constrained.engine import GenerationEngine
from constrained.errors import ConstraintError, ErrorCode
from constrained.vocab import default_vocab

#: Bounded-work limits for the service layer.
MAX_CONSTRAINTS_PER_REQUEST = 8
MAX_TOKENS_LIMIT = 4096
MAX_TIMEOUT_MS = 120_000
MAX_CONSTRAINTS_PER_TENANT = 512

_DEFAULT_API_KEYS = {"dev-key-a": "tenant-a", "dev-key-b": "tenant-b"}


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        message: str,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.reason = reason


def _error_body(code: ErrorCode, message: str, reason: str | None = None) -> dict:
    error: dict[str, Any] = {"code": code, "message": message}
    if reason is not None:
        error["reason"] = reason
    return {"error": error}


class ConstraintStore:
    """Tenant-scoped registry of compiled constraints (bounded per tenant)."""

    def __init__(self, max_per_tenant: int = MAX_CONSTRAINTS_PER_TENANT) -> None:
        self._max_per_tenant = max_per_tenant
        self._lock = threading.Lock()
        self._by_tenant: dict[str, OrderedDict[str, CompiledConstraint]] = {}

    def register(self, tenant_id: str, compiled: CompiledConstraint) -> str:
        with self._lock:
            entries = self._by_tenant.setdefault(tenant_id, OrderedDict())
            entries[compiled.constraint_id] = compiled
            entries.move_to_end(compiled.constraint_id)
            while len(entries) > self._max_per_tenant:
                entries.popitem(last=False)
        return compiled.constraint_id

    def get(self, tenant_id: str, constraint_id: str) -> CompiledConstraint | None:
        with self._lock:
            entries = self._by_tenant.get(tenant_id)
            if entries is None:
                return None
            compiled = entries.get(constraint_id)
            if compiled is not None:
                entries.move_to_end(constraint_id)
            return compiled

    def list(self, tenant_id: str) -> list[CompiledConstraint]:
        with self._lock:
            entries = self._by_tenant.get(tenant_id)
            return list(entries.values()) if entries else []


class ServiceState:
    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        model_factory=None,
    ) -> None:
        self.vocab = default_vocab()
        self.cache = ConstraintCache()
        self.engine = GenerationEngine(vocab=self.vocab, model_factory=model_factory)
        self.constraints = ConstraintStore()
        self.api_keys = dict(api_keys if api_keys is not None else _DEFAULT_API_KEYS)


def _load_api_keys() -> dict[str, str]:
    raw = os.environ.get("CONSTRAINED_API_KEYS")
    if not raw:
        return dict(_DEFAULT_API_KEYS)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return dict(_DEFAULT_API_KEYS)
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        return dict(_DEFAULT_API_KEYS)
    return parsed


class ConstraintCreate(BaseModel):
    kind: Literal["regex", "json_schema"]
    definition: Any


class ConstraintView(BaseModel):
    constraint_id: str
    kind: str
    definition_hash: str


class GenerationCreate(BaseModel):
    constraint_ids: list[str] = Field(
        default_factory=list, max_length=MAX_CONSTRAINTS_PER_REQUEST
    )
    prompt: str = ""
    max_tokens: int = Field(default=256, ge=1, le=MAX_TOKENS_LIMIT)
    timeout_ms: int | None = Field(default=None, ge=1, le=MAX_TIMEOUT_MS)


def create_app(
    api_keys: dict[str, str] | None = None, model_factory=None
) -> FastAPI:
    state = ServiceState(
        api_keys if api_keys is not None else _load_api_keys(),
        model_factory=model_factory,
    )
    app = FastAPI(title="constrained-generation")
    app.state.service = state

    @app.exception_handler(ApiError)
    async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message, exc.reason),
        )

    @app.exception_handler(ConstraintError)
    async def _constraint_error_handler(_: Request, exc: ConstraintError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_body(ErrorCode.INVALID_CONSTRAINT, exc.message, exc.reason),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_body(
                ErrorCode.INVALID_REQUEST,
                "request validation failed",
                reason=json.dumps(exc.errors(), default=str)[:500],
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_error_body(ErrorCode.INTERNAL, f"unexpected error: {exc}"),
        )

    def require_tenant(request: Request) -> str:
        key = request.headers.get("x-api-key")
        tenant = state.api_keys.get(key) if key else None
        if tenant is None:
            raise ApiError(401, ErrorCode.UNAUTHORIZED, "missing or invalid API key")
        return tenant

    @app.post("/v1/constraints", status_code=201)
    def create_constraint(
        body: ConstraintCreate, tenant: str = Depends(require_tenant)
    ) -> ConstraintView:
        compiled = state.cache.get_or_compile(body.kind, body.definition)
        state.constraints.register(tenant, compiled)
        return ConstraintView(
            constraint_id=compiled.constraint_id,
            kind=compiled.kind,
            definition_hash=compiled.definition_hash,
        )

    @app.get("/v1/constraints")
    def list_constraints(tenant: str = Depends(require_tenant)) -> list[ConstraintView]:
        return [
            ConstraintView(
                constraint_id=c.constraint_id,
                kind=c.kind,
                definition_hash=c.definition_hash,
            )
            for c in state.constraints.list(tenant)
        ]

    @app.post("/v1/generations", status_code=202)
    def create_generation(body: GenerationCreate, tenant: str = Depends(require_tenant)):
        constraints: list[CompiledConstraint] = []
        for constraint_id in body.constraint_ids:
            compiled = state.constraints.get(tenant, constraint_id)
            if compiled is None:
                raise ApiError(
                    404,
                    ErrorCode.NOT_FOUND,
                    f"constraint {constraint_id!r} not found",
                )
            constraints.append(compiled)
        request = state.engine.submit(
            tenant_id=tenant,
            constraints=constraints,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            timeout_s=body.timeout_ms / 1000 if body.timeout_ms is not None else None,
        )
        return request.public_view()

    @app.get("/v1/generations/{request_id}")
    def get_generation(request_id: str, tenant: str = Depends(require_tenant)):
        request = state.engine.registry.get(tenant, request_id)
        if request is None:
            raise ApiError(404, ErrorCode.NOT_FOUND, f"request {request_id!r} not found")
        return request.public_view()

    @app.post("/v1/generations/{request_id}/cancel", status_code=202)
    def cancel_generation(request_id: str, tenant: str = Depends(require_tenant)):
        request = state.engine.registry.get(tenant, request_id)
        if request is None:
            raise ApiError(404, ErrorCode.NOT_FOUND, f"request {request_id!r} not found")
        if not state.engine.cancel(request):
            raise ApiError(
                409,
                ErrorCode.CONFLICT,
                f"request {request_id!r} is already {request.status}",
            )
        return request.public_view()

    return app
