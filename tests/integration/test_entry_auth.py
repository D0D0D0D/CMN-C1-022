# CMN-C1-022 — Entry-adapter authentication tests
#
# Verify the standalone /invoke entry point is fail-CLOSED:
#   - unauthenticated caller (no verified middleware trust, no token) → 401
#   - correct INVOKE_AUTH_TOKEN Bearer → promoted to VERIFIED_EXTERNAL → 200
#   - wrong token → 401
#   - /health stays open (no auth)
#
# These exercise src/entry_adapter/auth.py through the FastAPI app. They require
# the framework wheel (imported transitively by src.api.server); in a wheel-less
# local shell collection errors here are expected and acceptable.

import pytest
from fastapi.testclient import TestClient

TOKEN = "mock-invoke-token-for-testing"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", TOKEN)
    # Import inside the fixture so INVOKE_AUTH_TOKEN is set before app import;
    # the auth core reads os.environ at request time, so import order is not
    # strictly required, but this keeps the token bound for the whole test.
    from src.api.server import app

    return TestClient(app, raise_server_exceptions=False)


_PAYLOAD = {
    "input": (
        '{"candidates": [{"id": "a", "attributes": {"v": 1}}], '
        '"criteria": [{"name": "v", "type": "numeric", "weight": 1.0}]}'
    ),
    "session_id": "t",
}


def test_unauthenticated_invoke_is_rejected(client):
    """No Bearer token and no verified middleware trust → 401 (fail-closed)."""
    resp = client.post("/invoke", json=_PAYLOAD)
    assert resp.status_code == 401


def test_wrong_token_is_rejected(client):
    resp = client.post("/invoke", json=_PAYLOAD, headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


def test_valid_token_is_accepted(client):
    resp = client.post("/invoke", json=_PAYLOAD, headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200


def test_health_is_open(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_server_builds_the_agent_from_config_yaml():
    """The standalone entry point is never exercised by CI otherwise.

    `ScoringRankingGraph()` with no argument discarded config/config.yaml entirely,
    so every runtime tunable in it silently fell back to a framework default. No
    test observed that, because the tests construct the graph themselves.
    """
    import src.api.server as server

    assert server._config["max_retry"] == 3
    assert server._config["timeout_seconds"] == 30
    assert server.agent.config["max_retry"] == 3


def test_server_does_not_hardcode_an_llm():
    """A template names no model and no concrete client; the platform injects one."""
    import src.api.server as server

    assert "llm" not in server._config
    assert server.agent._nodes["main"]._service._llm_scorer is None
