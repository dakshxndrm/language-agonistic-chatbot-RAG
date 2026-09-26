"""M4 acceptance check: interrupting at 30/60/90% of playback must leave
exactly the spoken prefix behind, and leave no orphan tasks running. Uses a
fake LLM/TTS pair (asyncio.Event-gated, no real audio) so the assertion is
about the cancellation/rewind logic itself, not about timing.
"""

import asyncio

import pytest

from server.barge_in import BargeInController, SpokenTracker


@pytest.mark.parametrize("acked_chars", [30, 60, 90])
async def test_barge_in_rewinds_to_exact_spoken_prefix(acked_chars):
    tracker = SpokenTracker()
    tracker.append_generated("A" * 100)  # the LLM has already generated all 100 chars
    controller = BargeInController(tracker=tracker)

    async def fake_llm_still_generating():
        await asyncio.Event().wait()  # runs until cancelled, like a live stream() call

    async def fake_tts_still_playing():
        await asyncio.Event().wait()  # runs until cancelled, like a live playback task

    llm_task = asyncio.create_task(fake_llm_still_generating())
    tts_task = asyncio.create_task(fake_tts_still_playing())
    controller.bind_llm_task(llm_task)
    controller.bind_tts_task(tts_task)

    # The speaker had actually played `acked_chars` worth of audio before the
    # user barged in — this is the signal barge-in must respect, NOT how much
    # text the LLM has generated (which is 100 in every case here).
    tracker.mark_spoken(acked_chars)

    spoken = await controller.barge_in()

    assert spoken == "A" * acked_chars
    assert len(spoken) < len(tracker.full_text)  # proves it rewound, not just returned everything

    assert llm_task.done() and llm_task.cancelled()
    assert tts_task.done() and tts_task.cancelled()


async def test_barge_in_leaves_no_orphan_tasks():
    before = {t for t in asyncio.all_tasks() if not t.done()}

    tracker = SpokenTracker()
    tracker.append_generated("hello world")
    controller = BargeInController(tracker=tracker)

    async def fake_llm():
        await asyncio.Event().wait()

    async def fake_tts():
        await asyncio.Event().wait()

    llm_task = asyncio.create_task(fake_llm())
    tts_task = asyncio.create_task(fake_tts())
    controller.bind_llm_task(llm_task)
    controller.bind_tts_task(tts_task)
    tracker.mark_spoken(5)

    await controller.barge_in()

    after = {t for t in asyncio.all_tasks() if not t.done()}
    assert after - before == set()  # nothing new left running


async def test_barge_in_is_safe_with_no_tasks_bound():
    tracker = SpokenTracker()
    tracker.append_generated("hello")
    tracker.mark_spoken(3)
    controller = BargeInController(tracker=tracker)

    spoken = await controller.barge_in()
    assert spoken == "hel"
