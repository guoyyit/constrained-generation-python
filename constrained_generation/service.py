"""FastAPI surface for constrained generation.

Error convention (shared by every endpoint): failures return
``{"error": {"code", "message", "details"}}`` with an HTTP status derived
from the error class, so clients can distinguish rejection classes
mechanically.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .errors import ConstrainedGenerationError, RequestInvalidError
from .manager import GenerationManager, GenerationParams
from .store import RequestRecord, RequestStore


class GenerateRequest(BaseModel):
    constraints: list[Any] = Field(..., description="constraint definitions")
    max_tokens: int = 512
    timeout_ms: int = 10_000
    seed: int = 0
    wait: bool = True


def _record_view(record: RequestRecord) -> dict:
    return {
        "request_id": record.request_id,
        "status": record.status.value,
        "result": record.result,
        "error": record.error,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def create_app(data_dir: str | None = None) -> FastAPI:
    """Build the application.  ``data_dir`` enables the persistent ledger."""
    ledger_path = Path(data_dir) / "ledger.jsonl" if data_dir else None
    manager = GenerationManager(RequestStore(ledger_path))

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(title="constrained-generation", lifespan=_lifespan)
    app.state.manager = manager

    @app.exception_handler(ConstrainedGenerationError)
    async def _cg_error_handler(_: Request, exc: ConstrainedGenerationError):
        return JSONResponse(status_code=exc.http_status, content=exc.to_envelope())

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError):
        error = RequestInvalidError(
            "malformed request body", details={"issues": exc.errors()}
        )
        return JSONResponse(status_code=error.http_status, content=error.to_envelope())

    def _tenant(x_tenant_id: str | None) -> str:
        if not x_tenant_id:
            raise RequestInvalidError("missing 'X-Tenant-Id' header")
        return x_tenant_id

    @app.post("/v1/generations", status_code=200)
    def submit(
        body: GenerateRequest,
        x_tenant_id: str | None = Header(default=None),
    ) -> dict:
        tenant = _tenant(x_tenant_id)
        record = manager.submit(
            tenant,
            GenerationParams(
                constraints=body.constraints,
                max_tokens=body.max_tokens,
                timeout_ms=body.timeout_ms,
                seed=body.seed,
                wait=body.wait,
            ),
        )
        return _record_view(record)

    @app.get("/v1/generations/{request_id}")
    def status(request_id: str, x_tenant_id: str | None = Header(default=None)) -> dict:
        tenant = _tenant(x_tenant_id)
        return _record_view(manager.get(tenant, request_id))

    @app.post("/v1/generations/{request_id}/cancel")
    def cancel(request_id: str, x_tenant_id: str | None = Header(default=None)) -> dict:
        tenant = _tenant(x_tenant_id)
        return _record_view(manager.cancel(tenant, request_id))

    return app
