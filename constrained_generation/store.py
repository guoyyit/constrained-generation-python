"""Request ledger with restart recovery.

Every state change is appended to a JSONL ledger (when a path is configured)
so a process restart can rebuild in-memory state.  At load time any request
left in a non-terminal status is swept to ``aborted``: an interrupted request
must never be served as a completed result after a restart.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from .errors import ErrorCode


class RequestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"


TERMINAL_STATUSES: frozenset[RequestStatus] = frozenset(
    {
        RequestStatus.COMPLETED,
        RequestStatus.FAILED,
        RequestStatus.CANCELLED,
        RequestStatus.ABORTED,
    }
)


@dataclass
class RequestRecord:
    request_id: str
    tenant_id: str
    status: RequestStatus
    created_at: float
    updated_at: float
    result: str | None = None
    error: dict | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RequestRecord":
        return cls(
            request_id=data["request_id"],
            tenant_id=data["tenant_id"],
            status=RequestStatus(data["status"]),
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
            result=data.get("result"),
            error=data.get("error"),
        )


class RequestStore:
    def __init__(self, ledger_path: Path | None) -> None:
        self._ledger_path = ledger_path
        self._records: dict[str, RequestRecord] = {}
        self._lock = threading.Lock()
        if ledger_path is not None:
            self._load(ledger_path)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = RequestRecord.from_dict(json.loads(line))
                    self._records[record.request_id] = record
        # Restart recovery: interrupted requests are aborted, never completed.
        for record in list(self._records.values()):
            if record.status not in TERMINAL_STATUSES:
                record.status = RequestStatus.ABORTED
                record.error = {
                    "code": ErrorCode.GENERATION_ABORTED.value,
                    "message": "request interrupted by process restart",
                    "details": {},
                }
                record.result = None
                record.updated_at = time.time()
                self._append(record)

    def _append(self, record: RequestRecord) -> None:
        if self._ledger_path is None:
            return
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------

    def create(self, record: RequestRecord) -> None:
        with self._lock:
            self._records[record.request_id] = record
            self._append(record)

    def transition(
        self,
        request_id: str,
        status: RequestStatus,
        *,
        result: str | None = None,
        error: dict | None = None,
    ) -> RequestRecord:
        with self._lock:
            record = self._records[request_id]
            record.status = status
            record.updated_at = time.time()
            if result is not None:
                record.result = result
            if error is not None:
                record.error = error
            self._append(record)
            return record

    def get(self, request_id: str) -> RequestRecord | None:
        with self._lock:
            return self._records.get(request_id)
