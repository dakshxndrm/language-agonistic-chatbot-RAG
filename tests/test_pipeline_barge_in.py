"""M4 acceptance check, at the pipeline-wiring level (not just barge_in.py's
own unit tests): does BhashaBrain actually cancel the in-flight turn and
rewind to only the sentences already pushed downstream, when a
VADUserStartedSpeakingFrame arrives mid-turn? Mocks the RAG/LLM stack (no
network) so this is deterministic — Pipecat's own frame-processor plumbing
is exercised for real, only the slow I/O is stubbed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from pipecat.frames.frames import (
    OutputTransportMessageFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from server.pipeline import BhashaBrain, BhashaSpeakFrame, split_sentences


@dataclass
class _FakeContract:
    lang: str = "en"
    script: str = "Latn"


@dataclass
class _FakeOrchestrationResult:
    hits: list = None

    def __post_init__(self):
        self.hits = self.hits or []


@pytest.fixture
def brain(monkeypatch):
    b = BhashaBrain()
    pushed: list = []

    async def fake_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    b.push_frame = fake_push_frame
    b.broadcast_interruption = _RecordingAsyncMock()

    monkeypatch.setattr("server.pipeline.build_contract",
                        _async_return(_FakeContract()))
    monkeypatch.setattr("server.pipeline.orchestrate",
                        _async_return(_FakeOrchestrationResult()))
    monkeypatch.setattr("server.pipeline.build_messages", lambda q: [])
    monkeypatch.setattr("server.pipeline.build_context", lambda r: [])
    monkeypatch.setattr("server.pipeline.get_llm", lambda: object())

    b._pushed = pushed
    return b


class _RecordingAsyncMock:
    def __init__(self):
        self.calls = 0

    async def __call__(self, *a, **kw):
        self.calls += 1


def _async_return(value):
    async def f(*a, **kw):
        return value
    return f


async def test_full_turn_completes_and_speaks_all_sentences(brain, monkeypatch):
    monkeypatch.setattr(
        "server.pipeline.generate_with_contract",
        _async_return(("Ashoka ruled well. He built many stupas.", True, 0)),
    )

    await brain.process_frame(
        TranscriptionFrame(text="Tell me about Ashoka", user_id="", timestamp="0"),
        FrameDirection.DOWNSTREAM,
    )
    await brain._turn_task  # let the spawned turn task finish

    speak_frames = [f for f in brain._pushed if isinstance(f, BhashaSpeakFrame)]
    assert [f.text for f in speak_frames] == ["Ashoka ruled well.", "He built many stupas."]
    assert brain._controller is None  # cleared once the turn finished normally


async def test_barge_in_mid_turn_rewinds_to_spoken_sentences_only(brain, monkeypatch):
    full_answer = "Ashoka ruled well. He built many stupas. He also spread Buddhism widely."
    release = asyncio.Event()

    async def slow_generate(*a, **kw):
        await release.wait()  # blocks until the test lets it through
        return full_answer, True, 0

    monkeypatch.setattr("server.pipeline.generate_with_contract", slow_generate)

    await brain.process_frame(
        TranscriptionFrame(text="Tell me about Ashoka", user_id="", timestamp="0"),
        FrameDirection.DOWNSTREAM,
    )
    task_before_barge_in = brain._turn_task
    assert task_before_barge_in is not None and not task_before_barge_in.done()

    # User starts talking again before the (still-blocked) LLM call returns —
    # this must cancel the turn outright, before any sentence is ever spoken.
    await brain.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert task_before_barge_in.cancelled()
    assert brain._turn_task is None
    assert brain.broadcast_interruption.calls == 1  # Pipecat's own TTS-flush path was triggered

    barge_in_msgs = [f for f in brain._pushed if isinstance(f, OutputTransportMessageFrame)]
    assert len(barge_in_msgs) == 1
    assert '"barge_in": true' in barge_in_msgs[0].message.lower()


def test_split_sentences_handles_devanagari_danda():
    assert split_sentences("Hello there. How are you?") == ["Hello there.", "How are you?"]
    assert split_sentences("अशोक ने राज किया। उसने बौद्ध धर्म अपनाया।") == [
        "अशोक ने राज किया।", "उसने बौद्ध धर्म अपनाया।",
    ]
    assert split_sentences("") == []
