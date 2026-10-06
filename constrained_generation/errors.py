"""Error taxonomy for constrained generation.

Every failure surfaced to callers carries a stable machine-readable ``code``
so that clients can distinguish rejection classes without parsing messages.
"""
from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    # Constraint definition rejections (raised before any generation starts).
    CONSTRAINT_INVALID = "constraint/invalid"
    CONSTRAINT_UNSUPPORTED = "constraint/unsupported"
    CONSTRAINT_UNSATISFIABLE = "constraint/unsatisfiable"
    CONSTRAINT_CONTRADICTORY = "constraint/contradictory"
    # Request-level rejections.
    REQUEST_INVALID = "request/invalid"
    REQUEST_NOT_FOUND = "request/not_found"
    REQUEST_CONFLICT = "request/conflict"
    # Generation-time failures (recorded on the request status).
    GENERATION_TIMEOUT = "generation/timeout"
    GENERATION_CANCELLED = "generation/cancelled"
    GENERATION_INCOMPLETE = "generation/incomplete"
    GENERATION_ABORTED = "generation/aborted"
    INTERNAL = "internal"


class ConstrainedGenerationError(Exception):
    """Base error carrying a stable :class:`ErrorCode`."""

    code: ErrorCode = ErrorCode.INTERNAL
    http_status: int = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_envelope(self) -> dict:
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": self.details,
            }
        }


class ConstraintInvalidError(ConstrainedGenerationError):
    code = ErrorCode.CONSTRAINT_INVALID
    http_status = 400


class ConstraintUnsupportedError(ConstrainedGenerationError):
    code = ErrorCode.CONSTRAINT_UNSUPPORTED
    http_status = 400


class ConstraintUnsatisfiableError(ConstrainedGenerationError):
    code = ErrorCode.CONSTRAINT_UNSATISFIABLE
    http_status = 422


class ConstraintContradictoryError(ConstrainedGenerationError):
    code = ErrorCode.CONSTRAINT_CONTRADICTORY
    http_status = 422


class RequestInvalidError(ConstrainedGenerationError):
    code = ErrorCode.REQUEST_INVALID
    http_status = 400


class RequestNotFoundError(ConstrainedGenerationError):
    code = ErrorCode.REQUEST_NOT_FOUND
    http_status = 404


class RequestConflictError(ConstrainedGenerationError):
    code = ErrorCode.REQUEST_CONFLICT
    http_status = 409
