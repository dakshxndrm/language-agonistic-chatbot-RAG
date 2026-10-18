"""Builds the messages+context fed into lang.contract.generate_with_contract().

Citation is forced two ways: llm_groq.py's system prompt already numbers every
source chunk by chunk_id, and this module's user-facing instruction makes
citing non-optional and forbids fabricating one (C8). Graph paths ride along
as a synthetic context entry so a multi-hop answer can point at the actual
reasoning chain (e.g. "Shah Jahan -FATHER_OF-> Aurangzeb"), not just prose.
"""

from __future__ import annotations

from server.providers.base import GraphHit
from server.rag.orchestrate import OrchestrationResult

ANSWER_INSTRUCTION = """Answer the user's question using ONLY the sources given
to you. Every factual claim you make must be followed by the chunk_id(s) that
support it, like [chunk_id]. If the sources do not contain the answer, say so
plainly instead of guessing — never invent a citation. If the question's
premise is factually wrong according to the sources, correct the premise
before answering."""


def render_graph_paths(graph_paths: list[GraphHit]) -> str:
    if not graph_paths:
        return ""
    lines = ["Additional reasoning chains from the knowledge graph:"]
    for hit in graph_paths[:10]:
        chain = " ".join(step.get("entity") or f"-{step.get('rel')}->" for step in hit.path)
        lines.append(f"{chain} (sources: {', '.join(hit.source_chunk_ids)})")
    return "\n".join(lines)


def build_messages(question: str, history: list[dict] | None = None) -> list[dict]:
    messages = list(history or [])
    messages.append({"role": "user", "content": f"{ANSWER_INSTRUCTION}\n\nQuestion: {question}"})
    return messages


def build_context(result: OrchestrationResult) -> list[dict]:
    """[{"chunk_id", "text"}] — what llm_groq.py's stream() numbers into the
    system prompt for citation."""
    context = [{"chunk_id": h.chunk_id, "text": h.text} for h in result.hits]
    graph_block = render_graph_paths(result.graph_paths)
    if graph_block:
        context.append({"chunk_id": "graph", "text": graph_block})
    return context
