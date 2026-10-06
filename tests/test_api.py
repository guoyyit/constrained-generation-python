"""HTTP surface: error envelope consistency, tenant isolation, restart recovery."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from constrained_generation.service import create_app

H1 = {"X-Tenant-Id": "tenant-1"}
H2 = {"X-Tenant-Id": "tenant-2"}
INTEGER = {"kind": "json_schema", "body": {"type": "integer"}}
LONG = {"kind": "regex", "body": "x([a-z]{32}){60}y"}


@pytest.fixture()
def client():
    with TestClient(create_app()) as client:
        yield client


def test_happy_path(client):
    response = client.post(
        "/v1/generations", headers=H1, json={"constraints": [INTEGER], "seed": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert isinstance(json.loads(body["result"]), int)

    fetched = client.get(f"/v1/generations/{body['request_id']}", headers=H1)
    assert fetched.status_code == 200
    assert fetched.json()["result"] == body["result"]


@pytest.mark.parametrize(
    "constraints,expected_code,expected_status",
    [
        ([{"kind": "json_schema", "body": {"type": "nope"}}], "constraint/invalid", 400),
        ([{"kind": "json_schema", "body": {"type": "string", "pattern": "a+"}}], "constraint/unsupported", 400),
        ([{"kind": "json_schema", "body": {"enum": []}}], "constraint/unsatisfiable", 422),
        (
            [{"kind": "regex", "body": '"a+"'}, {"kind": "regex", "body": '"b+"'}],
            "constraint/contradictory",
            422,
        ),
        ([], "request/invalid", 400),
    ],
)
def test_rejection_envelope(client, constraints, expected_code, expected_status):
    response = client.post(
        "/v1/generations", headers=H1, json={"constraints": constraints}
    )
    assert response.status_code == expected_status
    envelope = response.json()
    assert set(envelope) == {"error"}
    assert envelope["error"]["code"] == expected_code
    assert isinstance(envelope["error"]["message"], str)
    assert isinstance(envelope["error"]["details"], dict)


def test_malformed_body_uses_same_envelope(client):
    response = client.post("/v1/generations", headers=H1, json={"constraints": "x"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request/invalid"


def test_missing_tenant_header_rejected(client):
    response = client.post("/v1/generations", json={"constraints": [INTEGER]})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request/invalid"


def test_cross_tenant_access_denied(client):
    created = client.post("/v1/generations", headers=H1, json={"constraints": [INTEGER]})
    request_id = created.json()["request_id"]

    fetched = client.get(f"/v1/generations/{request_id}", headers=H2)
    assert fetched.status_code == 404
    assert fetched.json()["error"]["code"] == "request/not_found"

    cancelled = client.post(f"/v1/generations/{request_id}/cancel", headers=H2)
    assert cancelled.status_code == 404


def test_unknown_request_is_404(client):
    response = client.get("/v1/generations/does-not-exist", headers=H1)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "request/not_found"


def test_cancel_via_api(client):
    created = client.post(
        "/v1/generations",
        headers=H1,
        json={"constraints": [LONG], "max_tokens": 4096, "wait": False},
    )
    request_id = created.json()["request_id"]
    client.post(f"/v1/generations/{request_id}/cancel", headers=H1)
    deadline = time.time() + 10
    while time.time() < deadline:
        body = client.get(f"/v1/generations/{request_id}", headers=H1).json()
        if body["status"] in ("cancelled", "completed", "failed"):
            break
        time.sleep(0.01)
    assert body["status"] in ("cancelled", "completed")
    if body["status"] == "cancelled":
        assert body["error"]["code"] == "generation/cancelled"
        assert body["result"] is None


def test_timeout_via_api(client):
    created = client.post(
        "/v1/generations",
        headers=H1,
        json={"constraints": [LONG], "max_tokens": 4096, "timeout_ms": 1, "wait": False},
    )
    request_id = created.json()["request_id"]
    deadline = time.time() + 10
    while time.time() < deadline:
        body = client.get(f"/v1/generations/{request_id}", headers=H1).json()
        if body["status"] == "failed":
            break
        time.sleep(0.01)
    assert body["status"] == "failed"
    assert body["error"]["code"] == "generation/timeout"


def test_restart_does_not_serve_unfinished_as_completed(tmp_path):
    data_dir = tmp_path / "cgdata"
    data_dir.mkdir()
    ledger = data_dir / "ledger.jsonl"
    now = time.time()
    events = [
        # a request left mid-flight when the process "died"
        {"request_id": "unfinished", "tenant_id": "tenant-1", "status": "running",
         "created_at": now, "updated_at": now, "result": None, "error": None},
        # a request that genuinely completed before the restart
        {"request_id": "finished", "tenant_id": "tenant-1", "status": "completed",
         "created_at": now, "updated_at": now, "result": "42", "error": None},
    ]
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))

    with TestClient(create_app(data_dir=str(data_dir))) as client:
        revived = client.get("/v1/generations/unfinished", headers=H1).json()
        assert revived["status"] == "aborted"
        assert revived["error"]["code"] == "generation/aborted"
        assert revived["result"] is None
        # A completed request keeps its result across the restart.
        finished = client.get("/v1/generations/finished", headers=H1).json()
        assert finished["status"] == "completed"
        assert json.loads(finished["result"]) == 42
        # The sweep is persisted: another restart still shows aborted.
    with TestClient(create_app(data_dir=str(data_dir))) as client2:
        assert client2.get("/v1/generations/unfinished", headers=H1).json()["status"] == "aborted"
