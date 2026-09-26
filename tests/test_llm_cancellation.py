"""M2 requires stream() to be genuinely cancellable (base.py's contract, and
CLAUDE.md's barge-in feature depends on it later).

The mocked test below exercises OUR cancellation-forwarding logic (that
CancelledError propagates cleanly out of the async generator, with no
swallowing or retry-on-cancel) without any network. It cannot prove the
underlying httpx connection actually tears down — that needs a real socket —
so the `live`-marked test below it hits the real Groq API for that specific
guarantee. Run it with `pytest -m live`.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from server.providers.base import LangContract
from server.providers.llm_groq import GroqLLM


class _FakeSSEResponse:
    status_code = 200

    async def aiter_lines(self):
        for i in range(1000):
            await asyncio.sleep(0.01)  # forces a real suspension point per token
            yield f'data: {{"choices":[{{"delta":{{"content":"tok{i} "}}}}]}}'

    def raise_for_status(self):
        pass


async def test_mocked_stream_cancellation_does_not_swallow_cancelled_error():
    llm = GroqLLM(api_key="fake")
    contract = LangContract(lang="en", script="Latn")

    @asynccontextmanager
    async def fake_stream(*args, **kwargs):
        yield _FakeSSEResponse()

    chunks: list[str] = []
    got_first_chunk = asyncio.Event()

    async def consume():
        async for tok in llm.stream([{"role": "user", "content": "hi"}], contract, []):
            chunks.append(tok)
            got_first_chunk.set()

    with patch("httpx.AsyncClient.stream", fake_stream):
        task = asyncio.create_task(consume())
        # Wait for real progress instead of a fixed sleep — deterministic,
        # no race against how fast the event loop happens to schedule this.
        await asyncio.wait_for(got_first_chunk.wait(), timeout=5.0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert task.cancelled()
    assert 0 < len(chunks) < 1000


@pytest.mark.live
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
