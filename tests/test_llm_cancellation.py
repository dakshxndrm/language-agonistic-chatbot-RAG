"""M2 requires stream() to be genuinely cancellable (base.py's contract, and
CLAUDE.md's barge-in feature depends on it later). This hits the real Groq
API — a fake SSE stream can't prove the httpx connection actually tears down
on cancellation, which is the thing that matters for barge-in.
"""

import asyncio

from server.providers.base import LangContract
from server.providers.llm_groq import GroqLLM


async def test_stream_is_cancellable_mid_response():
    llm = GroqLLM()
    contract = LangContract(lang="en", script="Latn")
    messages = [{"role": "user", "content": "Write a very long, detailed 500-word essay "
                                            "about the history of the Mughal Empire."}]

    chunks: list[str] = []

    async def consume():
        async for tok in llm.stream(messages, contract, []):
            chunks.append(tok)

    task = asyncio.create_task(consume())
    await asyncio.sleep(1.5)  # let a few tokens arrive before cutting it off
    task.cancel()

    try:
        await task
        raise AssertionError("task should have raised CancelledError")
    except asyncio.CancelledError:
        pass

    assert task.cancelled()
    assert len(chunks) > 0          # it had actually started streaming
    assert len(chunks) < 400        # and did not run to completion
