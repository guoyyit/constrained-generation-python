"""Error types and failure classification for constrained generation.

All domain errors raised by this package use :class:`ConstraintError` for
constraint-definition problems and carry a machine-distinguishable
:class:`FailureReason`.  The service layer maps these to the shared error
envelope defined in :mod:`constrained.service`.
"""

from __future__ import annotations

from enum import StrEnum


class FailureReason(StrEnum):
    """Distinguishable reasons a constraint definition is rejected."""

    #: The definition cannot be parsed / is malformed.
    INVALID_DEFINITION = "invalid_definition"
    #: The definition uses constructs this engine does not support.
    UNSUPPORTED = "unsupported"
    #: The definition is internally contradictory (e.g. minimum > maximum).
    CONTRADICTORY = "contradictory"
    #: The definition is well-formed but cannot match any output.
    UNSATISFIABLE = "unsatisfiable"
    #: The definition exceeds bounded resource limits (states, repetitions).
    TOO_COMPLEX = "too_complex"


class ErrorCode(StrEnum):
    """Stable error codes used in the API error envelope."""

    INVALID_CONSTRAINT = "INVALID_CONSTRAINT"
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    INTERNAL = "INTERNAL"


class ConstraintError(Exception):
    """Raised when a constraint definition must be rejected before generation."""

    def __init__(self, reason: FailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


class RequestError(Exception):
    """Raised for request-level failures (bad input, conflicts)."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
