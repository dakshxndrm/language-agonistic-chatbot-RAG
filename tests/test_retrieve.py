"""RRF is the only piece of retrieval logic that can be wrong silently — a bad
fusion still returns plausible chunks. Test it without a DB or a model."""

from server.rag.retrieve import rrf


def row(cid: str) -> dict:
    return {"chunk_id": cid, "article": "A", "section": "S", "text": cid}


def test_rrf_rewards_agreement_between_rankers():
    dense = [row("a"), row("b"), row("c")]
    sparse = [row("c"), row("b"), row("z")]
    fused = rrf([dense, sparse])
    # b (2nd in both) and c (1st + 3rd) both appear twice, so both outrank the
    # singletons a and z. c edges b out: 1/61 + 1/63 > 1/62 + 1/62.
    assert {f["chunk_id"] for f in fused[:2]} == {"b", "c"}
    assert [f["chunk_id"] for f in fused[2:]] == ["a", "z"]
    assert len(fused) == 4          # union, deduplicated by chunk_id
    assert fused == sorted(fused, key=lambda r: r["score"], reverse=True)


def test_rrf_single_ranker_preserves_order():
    dense = [row("a"), row("b"), row("c")]
    assert [f["chunk_id"] for f in rrf([dense])] == ["a", "b", "c"]


def test_rrf_empty():
    assert rrf([[], []]) == []
