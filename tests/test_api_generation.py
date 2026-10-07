"""Generation: constrained decoding, isolation, cancellation, timeouts."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import jsonschema
import pytest

from constrained.schema import schema_to_regex
from constrained.service import create_app
from fastapi.testclient import TestClient
from tests.conftest import TENANT_A, TENANT_B, slow_model_factory, wait_terminal

PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 30},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "emails": {
            "type": "array",
            "items": {"type": "string", "minLength": 3},
            "maxItems": 2,
        },
        "vip": {"type": "boolean"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


def _register(client, headers, kind, definition) -> str:
    resp = client.post(
        "/v1/constraints", headers=headers, json={"kind": kind, "definition": definition}
    )
    assert resp.status_code == 201, resp.json()
    return resp.json()["constraint_id"]


def _generate(client, headers, constraint_ids, **kwargs) -> dict:
    resp = client.post(
        "/v1/generations",
        headers=headers,
        json={"constraint_ids": constraint_ids, **kwargs},
    )
    assert resp.status_code == 202, resp.json()
    return wait_terminal(client, resp.json()["request_id"], headers)


def test_constrained_output_matches_full_validation(client) -> None:
    """Every constrained result must pass a full, after-the-fact validation
    against the very same structure definition."""
    cid = _register(client, TENANT_A, "json_schema", PERSON)
    pattern = re.compile(schema_to_regex(PERSON))
    for _ in range(10):
        view = _generate(client, TENANT_A, [cid], max_tokens=512)
        assert view["status"] == "completed", view
        text = view["output_text"]
        assert pattern.fullmatch(text), text
        jsonschema.validate(json.loads(text), PERSON)


def test_regex_constraint_output_matches(client) -> None:
    cid = _register(client, TENANT_A, "regex", "(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/[0-9]{4}")
    for _ in range(5):
        view = _generate(client, TENANT_A, [cid], max_tokens=64)
        assert view["status"] == "completed"
        assert re.fullmatch(
            r"(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/[0-9]{4}", view["output_text"]
        )


def test_multiple_constraints_apply_as_intersection(client) -> None:
    c1 = _register(client, TENANT_A, "regex", "[0-9]+")
    c2 = _register(client, TENANT_A, "regex", "[0-9]{4}")
    c3 = _register(client, TENANT_A, "regex", "(?:[0-9]{2})+")
    for _ in range(5):
        view = _generate(client, TENANT_A, [c1, c2, c3], max_tokens=64)
        assert view["status"] == "completed"
        text = view["output_text"]
        assert re.fullmatch(r"[0-9]+", text)
        assert re.fullmatch(r"[0-9]{4}", text)
        assert re.fullmatch(r"(?:[0-9]{2})+", text)


def test_max_tokens_exceeded_fails_without_partial_result(client) -> None:
    cid = _register(
        client, TENANT_A, "json_schema", {"type": "string", "minLength": 50}
    )
    view = _generate(client, TENANT_A, [cid], max_tokens=10)
    assert view["status"] == "failed"
    assert view["error"]["reason"] == "max_tokens_exceeded"
    assert view["output_text"] is None


def test_concurrent_requests_are_isolated(client) -> None:
    """Concurrent requests across tenants and constraints must not interfere;
    a failing request must not affect any other request."""
    person_cid = _register(client, TENANT_A, "json_schema", PERSON)
    date_cid = _register(client, TENANT_B, "regex", "[0-9]{4}-[0-9]{2}-[0-9]{2}")
    long_string_cid = _register(
        client, TENANT_A, "json_schema", {"type": "string", "minLength": 100}
    )
    person_rx = re.compile(schema_to_regex(PERSON))
    date_rx = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

    def work(i: int):
        if i % 5 == 4:
            # deliberately failing request: budget too small for the constraint
            return ("fail", _generate(client, TENANT_A, [long_string_cid], max_tokens=3))
        if i % 2 == 0:
            return ("person", _generate(client, TENANT_A, [person_cid], max_tokens=512))
        return ("date", _generate(client, TENANT_B, [date_cid], max_tokens=64))

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(work, range(20)))

    assert len(results) == 20
    for kind, view in results:
        if kind == "fail":
            assert view["status"] == "failed"
            assert view["error"]["reason"] == "max_tokens_exceeded"
        else:
            assert view["status"] == "completed", view
            if kind == "person":
                assert person_rx.fullmatch(view["output_text"])
                jsonschema.validate(json.loads(view["output_text"]), PERSON)
            else:
                assert date_rx.fullmatch(view["output_text"])


def test_cancellation(slow_client) -> None:
    cid = _register(
        slow_client, TENANT_A, "json_schema", {"type": "string", "minLength": 200}
    )
    resp = slow_client.post(
        "/v1/generations",
        headers=TENANT_A,
        json={"constraint_ids": [cid], "max_tokens": 512},
    )
    request_id = resp.json()["request_id"]
    resp = slow_client.post(f"/v1/generations/{request_id}/cancel", headers=TENANT_A)
    assert resp.status_code in (202, 409)  # may have finished only if very fast
    view = wait_terminal(slow_client, request_id, TENANT_A)
    assert view["status"] in ("cancelled", "completed")
    if resp.status_code == 202:
        assert view["status"] == "cancelled"
        assert view["output_text"] is None
    # a finished request cannot be cancelled
    resp = slow_client.post(f"/v1/generations/{request_id}/cancel", headers=TENANT_A)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "CONFLICT"


def test_timeout(slow_client) -> None:
    cid = _register(
        slow_client, TENANT_A, "json_schema", {"type": "string", "minLength": 200}
    )
    view = _generate(slow_client, TENANT_A, [cid], max_tokens=512, timeout_ms=20)
    assert view["status"] == "timed_out"
    assert view["output_text"] is None


def test_cross_tenant_request_access_is_not_found(client) -> None:
    cid = _register(client, TENANT_A, "regex", "a+")
    resp = client.post(
        "/v1/generations", headers=TENANT_A, json={"constraint_ids": [cid]}
    )
    request_id = resp.json()["request_id"]
    for other in (TENANT_B, {}):
        assert client.get(f"/v1/generations/{request_id}", headers=other).status_code in (
            401,
            404,
        )
        assert client.post(
            f"/v1/generations/{request_id}/cancel", headers=other
        ).status_code in (401, 404)


def test_restart_never_reports_unfinished_as_completed() -> None:
    """State is in-memory only: after a restart the request is gone, so it can
    never be mistaken for a completed one."""
    app1 = create_app(model_factory=slow_model_factory())
    with TestClient(app1) as client1:
        cid = _register(
            client1, TENANT_A, "json_schema", {"type": "string", "minLength": 200}
        )
        resp = client1.post(
            "/v1/generations",
            headers=TENANT_A,
            json={"constraint_ids": [cid], "max_tokens": 512},
        )
        request_id = resp.json()["request_id"]
    # simulate a process restart with a fresh application instance
    app2 = create_app()
    with TestClient(app2) as client2:
        resp = client2.get(f"/v1/generations/{request_id}", headers=TENANT_A)
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_generation_with_unknown_constraint_rejected(client) -> None:
    resp = client.post(
        "/v1/generations",
        headers=TENANT_A,
        json={"constraint_ids": ["c_missing"]},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_request_limits_enforced(client) -> None:
    resp = client.post(
        "/v1/generations",
        headers=TENANT_A,
        json={"constraint_ids": ["c"] * 9},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"
    resp = client.post(
        "/v1/generations",
        headers=TENANT_A,
        json={"max_tokens": 0},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"
