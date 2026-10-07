"""Constraint registration: rejection classification, caching, tenant isolation."""

from __future__ import annotations

from tests.conftest import TENANT_A, TENANT_B


def test_register_regex_constraint(client) -> None:
    resp = client.post(
        "/v1/constraints",
        headers=TENANT_A,
        json={"kind": "regex", "definition": "[0-9]{4}"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["constraint_id"].startswith("c_")
    assert body["kind"] == "regex"
    assert len(body["definition_hash"]) == 64


def test_identical_definition_shares_compiled_form(client) -> None:
    payload = {"kind": "json_schema", "definition": {"type": "integer", "minimum": 1}}
    first = client.post("/v1/constraints", headers=TENANT_A, json=payload).json()
    second = client.post("/v1/constraints", headers=TENANT_A, json=payload).json()
    assert first["constraint_id"] == second["constraint_id"]
    # only one entry is kept for the tenant
    listed = client.get("/v1/constraints", headers=TENANT_A).json()
    assert len(listed) == 1


def test_constraints_are_tenant_scoped(client) -> None:
    payload = {"kind": "regex", "definition": "a+"}
    mine = client.post("/v1/constraints", headers=TENANT_A, json=payload).json()
    # another tenant cannot see it
    assert client.get("/v1/constraints", headers=TENANT_B).json() == []
    # and cannot use it
    resp = client.post(
        "/v1/generations",
        headers=TENANT_B,
        json={"constraint_ids": [mine["constraint_id"]]},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


REJECTION_CASES = [
    ({"kind": "regex", "definition": "a("}, "invalid_definition"),
    ({"kind": "regex", "definition": "a{2,1}"}, "invalid_definition"),
    ({"kind": "regex", "definition": "a{513}"}, "too_complex"),
    ({"kind": "regex", "definition": 123}, "invalid_definition"),
    ({"kind": "json_schema", "definition": {"type": "integer", "minimum": 5, "maximum": 3}}, "contradictory"),
    ({"kind": "json_schema", "definition": {"enum": []}}, "contradictory"),
    ({"kind": "json_schema", "definition": {"type": "array"}}, "unsupported"),
    ({"kind": "json_schema", "definition": {"type": "string", "pattern": "x"}}, "unsupported"),
    ({"kind": "json_schema", "definition": False}, "unsatisfiable"),
    ({"kind": "json_schema", "definition": {"type": "string", "maxLength": 100000}}, "too_complex"),
    ({"kind": "json_schema", "definition": {"type": "nope"}}, "invalid_definition"),
    ({"kind": "xml_schema", "definition": {}}, "invalid_request"),
]


def test_invalid_constraints_rejected_with_distinguishable_reasons(client) -> None:
    for payload, expected in REJECTION_CASES:
        resp = client.post("/v1/constraints", headers=TENANT_A, json=payload)
        assert resp.status_code == 422, (payload, resp.status_code)
        error = resp.json()["error"]
        if expected == "invalid_request":
            assert error["code"] == "INVALID_REQUEST"
        else:
            assert error["code"] == "INVALID_CONSTRAINT"
            assert error["reason"] == expected, (payload, error)


def test_error_envelope_shape_is_consistent(client) -> None:
    # unauthenticated
    resp = client.post("/v1/constraints", json={"kind": "regex", "definition": "a"})
    assert resp.status_code == 401
    assert set(resp.json()["error"]) >= {"code", "message"}
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"
    # validation failure uses the same envelope
    resp = client.post("/v1/constraints", headers=TENANT_A, json={"kind": "regex"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"
    # unknown routes of known resources
    resp = client.get("/v1/generations/r_doesnotexist", headers=TENANT_A)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"
