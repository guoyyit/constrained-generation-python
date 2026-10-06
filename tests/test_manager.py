"""Manager: concurrency isolation, cancellation, timeout, cache reuse."""
import concurrent.futures
import json
import time

import pytest

from constrained_generation.errors import (
    ConstraintContradictoryError,
    ConstraintUnsatisfiableError,
    RequestConflictError,
    RequestInvalidError,
    RequestNotFoundError,
)
from constrained_generation.manager import GenerationManager, GenerationParams
from constrained_generation.registry import ConstraintRegistry
from constrained_generation.store import RequestStatus, RequestStore

LONG = {"kind": "regex", "body": "x([a-z]{32}){60}y"}  # ~1922 tokens, no early stop
INTEGER = {"kind": "json_schema", "body": {"type": "integer"}}


@pytest.fixture()
def manager():
    mgr = GenerationManager(RequestStore(None))
    yield mgr
    mgr.shutdown()


def wait_terminal(mgr, tenant, request_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = mgr.get(tenant, request_id)
        if record.status not in (RequestStatus.PENDING, RequestStatus.RUNNING):
            return record
        time.sleep(0.01)
    raise AssertionError("request did not reach a terminal status in time")


def test_submit_completes(manager):
    record = manager.submit("t1", GenerationParams(constraints=[INTEGER], seed=1))
    assert record.status == RequestStatus.COMPLETED
    assert isinstance(json.loads(record.result), int)


def test_contradictory_constraints_rejected_before_generation(manager):
    with pytest.raises(ConstraintContradictoryError) as excinfo:
        manager.submit(
            "t1",
            GenerationParams(
                constraints=[
                    {"kind": "regex", "body": '"a+"'},
                    {"kind": "regex", "body": '"b+"'},
                ]
            ),
        )
    assert excinfo.value.code.value == "constraint/contradictory"


def test_unsatisfiable_constraint_rejected(manager):
    with pytest.raises(ConstraintUnsatisfiableError):
        manager.submit(
            "t1",
            GenerationParams(constraints=[{"kind": "json_schema", "body": {"enum": []}}]),
        )


def test_invalid_request_params_rejected(manager):
    with pytest.raises(RequestInvalidError):
        manager.submit("t1", GenerationParams(constraints=[]))
    with pytest.raises(RequestInvalidError):
        manager.submit("t1", GenerationParams(constraints=[INTEGER], max_tokens=0))
    with pytest.raises(RequestInvalidError):
        manager.submit("t1", GenerationParams(constraints=[INTEGER], timeout_ms=0))


def test_tenant_isolation(manager):
    record = manager.submit("t1", GenerationParams(constraints=[INTEGER]))
    with pytest.raises(RequestNotFoundError):
        manager.get("t2", record.request_id)
    with pytest.raises(RequestNotFoundError):
        manager.cancel("t2", record.request_id)


def test_cancel_terminal_request_conflicts(manager):
    record = manager.submit("t1", GenerationParams(constraints=[INTEGER]))
    with pytest.raises(RequestConflictError):
        manager.cancel("t1", record.request_id)


def test_cancellation(manager):
    record = manager.submit(
        "t1", GenerationParams(constraints=[LONG], max_tokens=4096, wait=False)
    )
    # The request may still be queued or already running; cancel either way.
    try:
        manager.cancel("t1", record.request_id)
    except RequestConflictError:
        pass  # already finished before the cancel landed
    final = wait_terminal(manager, "t1", record.request_id)
    assert final.status in (RequestStatus.CANCELLED, RequestStatus.COMPLETED)
    if final.status == RequestStatus.CANCELLED:
        assert final.error["code"] == "generation/cancelled"
        assert final.result is None


def test_timeout(manager):
    record = manager.submit(
        "t1",
        GenerationParams(constraints=[LONG], max_tokens=4096, timeout_ms=1, wait=False),
    )
    final = wait_terminal(manager, "t1", record.request_id)
    assert final.status == RequestStatus.FAILED
    assert final.error["code"] == "generation/timeout"
    assert final.result is None


def test_concurrent_requests_are_isolated(manager):
    """Many concurrent requests across tenants; each sees only its own state,
    and one request's failure does not affect the others."""
    schemas = [
        {"kind": "json_schema", "body": {"type": "integer"}},
        {"kind": "json_schema", "body": {"type": "string", "maxLength": 4}},
        {"kind": "regex", "body": '"(ab)+?"'},
        {"kind": "json_schema", "body": {"enum": []}},  # always rejected
    ]

    def one(i):
        tenant = f"tenant-{i % 4}"
        constraint = schemas[i % len(schemas)]
        try:
            record = manager.submit(
                tenant, GenerationParams(constraints=[constraint], seed=i, wait=True)
            )
            return tenant, record, None
        except ConstraintUnsatisfiableError as exc:
            return tenant, None, exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(one, range(32)))

    completed = 0
    for i, (tenant, record, error) in enumerate(outcomes):
        if i % len(schemas) == 3:
            assert isinstance(error, ConstraintUnsatisfiableError)
            continue
        assert record.status == RequestStatus.COMPLETED, record.error
        completed += 1
        # cross-tenant access is denied for every request
        with pytest.raises(RequestNotFoundError):
            manager.get("intruder", record.request_id)
    assert completed == 24


def test_compiled_constraints_are_cached():
    registry = ConstraintRegistry()
    mgr = GenerationManager(RequestStore(None), registry=registry)
    try:
        for seed in range(5):
            record = mgr.submit(
                "t1", GenerationParams(constraints=[INTEGER], seed=seed)
            )
            assert record.status == RequestStatus.COMPLETED
        assert len(registry) == 1  # compiled once, reused across requests
    finally:
        mgr.shutdown()
