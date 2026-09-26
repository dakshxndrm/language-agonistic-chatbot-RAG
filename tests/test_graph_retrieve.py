"""C12 (graph only queried on multi_hop/comparative) and C11 (unreachable graph
degrades to no hits, never raises) are the two correctness-critical rules for
graph_retrieve.py — this tests both without a live Neo4j.
"""

from server.rag.graph_retrieve import graph_retrieve


async def test_non_trigger_type_never_touches_the_graph(monkeypatch):
    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("graph provider must not be constructed for a factoid question")

    monkeypatch.setattr("server.rag.graph_retrieve.get_graph_provider", boom)
    hits = await graph_retrieve("Which war made Ashoka give up violence?", "factoid")
    assert hits == []
    assert not called


async def test_unreachable_graph_returns_no_hits(monkeypatch):
    class UnhealthyGraph:
        async def health(self):
            return False

    monkeypatch.setattr("server.rag.graph_retrieve.get_graph_provider", lambda: UnhealthyGraph())
    hits = await graph_retrieve("Who succeeded Aurangzeb?", "multi_hop")
    assert hits == []
