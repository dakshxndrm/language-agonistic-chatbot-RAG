"""M3 needs a real LiveKit join token minted server-side — the web client
never sees LIVEKIT_API_SECRET directly."""

from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)


def test_health_reports_resolved_providers():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "llm" in body["providers"]


def test_livekit_token_mints_a_jwt_with_room_grant():
    resp = client.post("/livekit/token")
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"].count(".") == 2   # header.payload.signature
    assert body["room"].startswith("bhasha-")


def test_livekit_token_respects_requested_room_name():
    resp = client.post("/livekit/token", params={"room": "my-room"})
    assert resp.json()["room"] == "my-room"
