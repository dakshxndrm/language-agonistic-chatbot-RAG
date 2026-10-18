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


def test_query_endpoint_returns_valid_payload(monkeypatch):
    from server.rag.retrieve import Hit
    from server.rag.orchestrate import OrchestrationResult
    from server.lang.contract import LangContract

    async def fake_build_contract(q):
        return LangContract(lang="en", script="Latn")

    async def fake_orchestrate(q, question_type="factoid"):
        hits = [Hit(chunk_id="ashoka::0001", article="Ashoka", section="Reign", text="Ashoka was an emperor.", score=0.9)]
        return OrchestrationResult(hits=hits, graph_paths=[], vector_only_hits=hits)

    async def fake_generate(llm, msgs, contract, ctx):
        return "Ashoka was a great Mauryan emperor.", True, 0

    monkeypatch.setattr("server.lang.contract.build_contract", fake_build_contract)
    monkeypatch.setattr("server.rag.orchestrate.orchestrate", fake_orchestrate)
    monkeypatch.setattr("server.lang.contract.generate_with_contract", fake_generate)
    monkeypatch.setattr("server.providers.factory.get_llm", lambda: object())

    resp = client.post("/query", json={"question": "Who was Ashoka?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["question"] == "Who was Ashoka?"
    assert data["answer"] == "Ashoka was a great Mauryan emperor."
    assert "Ashoka" in data["citations"]
    assert "visual_stage" in data


def test_stream_query_endpoint(monkeypatch):
    from server.rag.retrieve import Hit
    from server.rag.orchestrate import OrchestrationResult
    from server.lang.contract import LangContract

    async def fake_build_contract(q):
        return LangContract(lang="en", script="Latn")

    async def fake_orchestrate(q, question_type="factoid"):
        hits = [Hit(chunk_id="ashoka::0001", article="Ashoka", section="Reign", text="Ashoka was an emperor.", score=0.9)]
        return OrchestrationResult(hits=hits, graph_paths=[], vector_only_hits=hits)

    class FakeLLM:
        async def stream(self, msgs, contract, ctx):
            yield "Ashoka "
            yield "was "
            yield "great."

    monkeypatch.setattr("server.lang.contract.build_contract", fake_build_contract)
    monkeypatch.setattr("server.rag.orchestrate.orchestrate", fake_orchestrate)
    monkeypatch.setattr("server.providers.factory.get_llm", lambda: FakeLLM())

    resp = client.post("/query/stream", json={"question": "Who was Ashoka?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "start" in text
    assert "Ashoka " in text
    assert "done" in text


def test_stream_query_endpoint_handles_error(monkeypatch):
    from server.rag.retrieve import Hit
    from server.rag.orchestrate import OrchestrationResult
    from server.lang.contract import LangContract

    async def fake_build_contract(q):
        return LangContract(lang="en", script="Latn")

    async def fake_orchestrate(q, question_type="factoid"):
        hits = [Hit(chunk_id="ashoka::0001", article="Ashoka", section="Reign", text="Ashoka was an emperor.", score=0.9)]
        return OrchestrationResult(hits=hits, graph_paths=[], vector_only_hits=hits)

    class FailingLLM:
        async def stream(self, msgs, contract, ctx):
            yield "Starting..."
            raise RuntimeError("LLM provider unavailable")

    monkeypatch.setattr("server.lang.contract.build_contract", fake_build_contract)
    monkeypatch.setattr("server.rag.orchestrate.orchestrate", fake_orchestrate)
    monkeypatch.setattr("server.providers.factory.get_llm", lambda: FailingLLM())

    resp = client.post("/query/stream", json={"question": "Who was Ashoka?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "error" in text
    assert "LLM provider unavailable" in text
    assert '"event": "done"' not in text


