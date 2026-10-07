from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from constrained.service import create_app

TENANT_A = {"X-API-Key": "dev-key-a"}
TENANT_B = {"X-API-Key": "dev-key-b"}


class SlowMockModel:
    """Deterministic model with a per-step delay so cancellation and
    timeouts can be observed deterministically in tests."""

    def __init__(self, seed: int = 0, eos_token_id: int = -1, delay_s: float = 0.005):
        self._seed = seed
        self._eos_token_id = eos_token_id
        self._delay = delay_s

    def logits(self, context: tuple[int, ...], vocab_size: int) -> list[float]:
        time.sleep(self._delay)
        from constrained.model import MockModel

        return MockModel(seed=self._seed, eos_token_id=self._eos_token_id).logits(
            context, vocab_size
        )


def slow_model_factory(delay_s: float = 0.005):
    def factory(seed: int) -> SlowMockModel:
        from constrained.vocab import default_vocab

        return SlowMockModel(seed=seed, eos_token_id=default_vocab().eos_id, delay_s=delay_s)

    return factory


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def slow_client():
    app = create_app(model_factory=slow_model_factory())
    with TestClient(app) as c:
        yield c


def wait_terminal(client: TestClient, request_id: str, headers: dict, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = client.get(f"/v1/generations/{request_id}", headers=headers).json()
        if view["status"] not in ("queued", "running"):
            return view
        time.sleep(0.01)
    raise AssertionError(f"request {request_id} did not reach a terminal state")
