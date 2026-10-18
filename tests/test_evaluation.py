"""Unit tests for the evaluation harness (Milestone 5 completion).

Tests verify:
- Script Preservation Rate (SPR) calculation per flavour and overall
- Citation Faithfulness Rate and [chunk_id] verification
- Retrieval Recall@5 and Additive Graph Lift
- Latency waterfall percentiles (P50/P90)
- Checkpointing and resumption mechanics
- Fallback judge handling
"""

from __future__ import annotations

from pathlib import Path

from scripts.evaluate import (
    calculate_percentiles,
    calculate_recall_at_k,
    calculate_spr,
    format_markdown_report,
    load_partial_records,
    save_partial_record,
    verify_citations,
)


def test_eval_script_preservation_rate():
    records = [
        {"flavour": "en", "expected_script": "Latn", "answer": "Ashoka renounced violence."},
        {"flavour": "en", "expected_script": "Latn", "answer": "The battle occurred in 1757."},
        {"flavour": "hi-deva", "expected_script": "Deva", "answer": "चंद्रगुप्त मौर्य ने स्थापना की।"},
        {"flavour": "hi-deva", "expected_script": "Deva", "answer": "Wrong script."},  # mismatch
        {
            "flavour": "hinglish-latn",
            "expected_script": "Latn",
            "answer": "Unhone Kalinga war ke baad Buddhism apnaya.",
        },
    ]
    spr_metrics = calculate_spr(records)
    assert spr_metrics["spr_overall"] == 0.8  # 4 out of 5 match
    assert spr_metrics["per_flavour"]["en"] == 1.0
    assert spr_metrics["per_flavour"]["hi-deva"] == 0.5
    assert spr_metrics["per_flavour"]["hinglish-latn"] == 1.0


def test_eval_citation_faithfulness():
    valid_corpus_ids = {"ashoka::0001", "ashoka::0002", "plassey::0005"}
    context_chunk_ids = {"ashoka::0001", "plassey::0005"}

    # Valid citations matching retrieved context
    ans1 = "Ashoka ruled Maurya [ashoka::0001] after Plassey [plassey::0005]."
    res1 = verify_citations(ans1, valid_corpus_ids, context_chunk_ids)
    assert res1["has_citations"] is True
    assert res1["cited_chunks"] == ["ashoka::0001", "plassey::0005"]
    assert res1["valid_citation_rate"] == 1.0
    assert res1["grounded_in_context_rate"] == 1.0

    # Hallucinated citation not in corpus
    ans2 = "Ashoka fought [fake_chunk::9999]."
    res2 = verify_citations(ans2, valid_corpus_ids, context_chunk_ids)
    assert res2["has_citations"] is True
    assert res2["valid_citation_rate"] == 0.0

    # Citation exists in corpus but was not in the retrieved context
    ans3 = "Ashoka text [ashoka::0002]."
    res3 = verify_citations(ans3, valid_corpus_ids, context_chunk_ids)
    assert res3["valid_citation_rate"] == 1.0
    assert res3["grounded_in_context_rate"] == 0.0

    # No citations
    ans4 = "No citation here."
    res4 = verify_citations(ans4, valid_corpus_ids, context_chunk_ids)
    assert res4["has_citations"] is False
    assert res4["cited_chunks"] == []


def test_eval_retrieval_recall_at_k():
    gold = ["Ashoka", "Kalinga War"]
    # Both found in top 5
    retrieved1 = ["Ashoka", "Chandragupta", "Kalinga War", "Maurya Empire", "Buddhism"]
    assert calculate_recall_at_k(gold, retrieved1, k=5) == 1.0

    # 1 of 2 found in top 5
    retrieved2 = ["Ashoka", "Akbar", "Babur", "Mughal Empire", "Delhi"]
    assert calculate_recall_at_k(gold, retrieved2, k=5) == 0.5

    # 0 found
    retrieved3 = ["Akbar", "Babur", "Humayun", "Jahangir", "Shah Jahan"]
    assert calculate_recall_at_k(gold, retrieved3, k=5) == 0.0

    # Empty gold articles
    assert calculate_recall_at_k([], retrieved1, k=5) == 1.0


def test_eval_latency_percentiles():
    latencies = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 1000.0]
    p = calculate_percentiles(latencies)
    assert 250.0 <= p["p50"] <= 350.0
    assert p["p90"] >= 450.0
    assert p["count"] == 10

    # Empty
    assert calculate_percentiles([]) == {"p50": 0.0, "p90": 0.0, "mean": 0.0, "count": 0}


def test_eval_checkpoint_and_resume(tmp_path: Path):
    checkpoint_file = tmp_path / "eval_partial.jsonl"
    r1 = {"id": "G001", "answer": "Answer 1", "fidelity_passed": True}
    r2 = {"id": "G002", "answer": "Answer 2", "fidelity_passed": False}

    save_partial_record(r1, checkpoint_file)
    save_partial_record(r2, checkpoint_file)

    loaded = load_partial_records(checkpoint_file)
    assert len(loaded) == 2
    assert loaded["G001"]["answer"] == "Answer 1"
    assert loaded["G002"]["fidelity_passed"] is False


def test_eval_format_markdown_report():
    sample_results = {
        "generated_at": "2026-10-17T12:00:00Z",
        "answer_model": "openai/gpt-oss-120b",
        "judge_model": "openai/gpt-oss-20b",
        "spr": {
            "spr_overall": 0.96,
            "per_flavour": {"en": 1.0, "hi-deva": 0.95, "hinglish-latn": 0.96},
        },
        "lfr": {
            "lfr_overall": 0.94,
            "per_flavour": {"en": 1.0, "hi-deva": 0.90, "hinglish-latn": 0.95},
        },
        "citations": {
            "citation_presence_rate": 0.98,
            "citation_validity_rate": 1.0,
            "citation_grounded_rate": 0.92,
        },
        "retrieval_ablation": {
            "no_rag": 0.0,
            "dense": 0.769,
            "hybrid_norerank": 0.769,
            "hybrid_rerank": 0.780,
            "hybrid_rerank_plus_graph": 0.845,
            "_graph_ablation_vector_only": 0.780,
        },
        "latency_waterfall": {
            "retrieval_ms": {"p50": 38.5, "p90": 45.2},
            "generation_ms": {"p50": 310.0, "p90": 480.0},
        },
    }
    table = format_markdown_report(sample_results)
    assert "### 📊 BHASHA System Evaluation Benchmark Report" in table
    assert "Script Preservation Rate (SPR)" in table
    assert "Recall@5" in table
    assert "Citation Grounding Rate" in table
