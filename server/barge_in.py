"""Cancellation bus + conversation-state rewind for true barge-in (§ five
things list, item 5).

The key correctness property: when the user interrupts, the turn's recorded
`spoken_text` (schema.sql) must rewind to what was ACTUALLY played through
the speaker — tracked via TTS playback acks — never to how much text the LLM
had generated. Those two diverge constantly: the LLM can be several sentences
ahead of what the speaker has emitted.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class SpokenTracker:
    """`mark_spoken(upto)` is called by the TTS-sending side of the pipeline
    every time a chunk is handed off to playback — `upto` is the character
    offset into `full_text` that chunk corresponds to. That offset, not the
    length of `full_text` itself, is what barge-in rewinds to."""

    full_text: str = ""
    spoken_upto: int = 0

    def append_generated(self, delta: str) -> None:
        self.full_text += delta

    def mark_spoken(self, upto: int) -> None:
        self.spoken_upto = max(self.spoken_upto, upto)

    @property
    def spoken_text(self) -> str:
        return self.full_text[: self.spoken_upto]


@dataclass
class BargeInController:
    """One instance per in-flight agent turn. Owns the two cancellable tasks
    (LLM generation, TTS playback) and the SpokenTracker between them."""

    tracker: SpokenTracker = field(default_factory=SpokenTracker)
    interrupted: bool = False
    _llm_task: asyncio.Task | None = None
    _tts_task: asyncio.Task | None = None

    def bind_llm_task(self, task: asyncio.Task) -> None:
        self._llm_task = task

    def bind_tts_task(self, task: asyncio.Task) -> None:
        self._tts_task = task

    async def barge_in(self) -> str:
        """Cancels both tasks and waits for both to actually finish before
        returning — no orphan tasks left running after this call — then
        returns the rewound spoken_text."""
        self.interrupted = True
        tasks = [t for t in (self._tts_task, self._llm_task) if t is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        return self.tracker.spoken_text
