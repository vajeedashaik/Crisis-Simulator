import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_trigger_crisis_without_reset_returns_400(client):
    response = client.post("/trigger-crisis")
    assert response.status_code == 400
    assert "No active episode" in response.json()["detail"]


def test_trigger_crisis_after_reset_injects_one_hazard(client):
    client.post("/reset")
    state_before = client.get("/state").json()
    hazards_before = len(state_before["hazards"])

    response = client.post("/trigger-crisis")
    assert response.status_code == 200
    data = response.json()
    assert len(data["hazards"]) == hazards_before + 1


def test_trigger_crisis_returns_building_state_shape(client):
    client.post("/reset")
    response = client.post("/trigger-crisis")
    data = response.json()
    assert "zones" in data
    assert "people" in data
    assert "hazards" in data
    assert "tick" in data
