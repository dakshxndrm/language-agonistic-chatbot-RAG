"""Pipecat pipeline assembly: LiveKit transport -> Silero VAD -> our ASR
router -> RAG+contract "brain" -> our TTS -> LiveKit output.

M4 barge-in: BhashaBrain owns each turn as an explicit asyncio.Task bound to
a barge_in.BargeInController, and replies are split into sentences (so a
partial answer is possible at all). On VADUserStartedSpeakingFrame while a
turn is in flight, it (1) cancels that task via the controller — stopping
generation and any not-yet-pushed sentences dead, and (2) calls
broadcast_interruption(), Pipecat's own documented mechanism (confirmed in
frame_processor.py: FrameProcessor._start_interruption() unconditionally
cancels+recreates every processor's in-flight task on any InterruptionFrame,
in this installed version — NOT gated by a "allow_interruptions" pipeline
flag, which does not exist on PipelineParams in Pipecat 1.11.0; an earlier
version of this file passed that kwarg and it was silently ignored) — this
is what actually flushes BhashaTTS's in-flight run_tts task.

The "spoken_text" rewind is a real but coarse approximation: a sentence
counts as "spoken" once pushed downstream to TTS, not from a true
audio-playback-position ack (our TTS providers don't expose one — Sarvam's
REST API returns a whole utterance in one shot). Documented, not hidden.

NOT run end-to-end: this needs a live LiveKit room and real audio in/out to
verify, which this environment has no way to drive (§2a — no GPU, no browser
mic here). Every provider this wraps WAS verified live (see
server/providers/*.py's own docstrings); this file is the wiring between
them, reviewed carefully against Pipecat 1.11.0's actual installed API
(including the two corrections above, both found by reading its source, not
guessed), but not run.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    OutputTransportMessageFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from server.barge_in import BargeInController
from server.config import get_settings
from server.lang.contract import build_contract, generate_with_contract
from server.providers._pcm import wav_bytes_to_pcm
from server.providers.asr_router import route_transcribe
from server.providers.factory import get_llm, get_tts
from server.rag.orchestrate import orchestrate
from server.rag.prompt import build_context, build_messages

# Heuristic sentence split (English '.'/'!'/'?', Devanagari '।') — good
# enough to chunk TTS work; not a full multilingual sentence tokenizer.
_SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return parts or ([text] if text.strip() else [])


class BhashaSTT(SegmentedSTTService):
    """Wraps asr_router.route_transcribe(). VADProcessor upstream segments
    the audio; each segment arrives here as a WAV-wrapped buffer (the
    default `wants_wav_segments=True`, since both our cloud ASR providers
    want an uploaded file, not raw PCM)."""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        pcm, sample_rate, _channels = wav_bytes_to_pcm(audio)
        transcript = await route_transcribe(pcm, sample_rate, hint_lang=self._hint_lang)
        if transcript.text.strip():
            yield TranscriptionFrame(
                text=transcript.text,
                user_id="",
                timestamp=str(time.time()),
                language=None,
            )

    def __init__(self, hint_lang: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._hint_lang = hint_lang  # previous turn's detected lang, or None for turn 1


@dataclass
class BhashaSpeakFrame(TTSSpeakFrame):
    """TTSSpeakFrame doesn't carry a language — the contract's target
    lang/script has to ride along so BhashaTTS knows which voice to use."""

    lang: str = "en"


class BhashaBrain(FrameProcessor):
    """RAG + Language Fidelity Contract + barge-in (M4). Forwards every
    frame unchanged. A finalized transcript starts a new turn as its own
    asyncio.Task, bound to a BargeInController; VADUserStartedSpeakingFrame
    while that task is still running triggers a real barge-in."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._controller: BargeInController | None = None
        self._turn_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if (isinstance(frame, VADUserStartedSpeakingFrame)
                and self._turn_task is not None and not self._turn_task.done()):
            spoken = await self._controller.barge_in()  # cancels _run_turn (LLM side)
            await self.broadcast_interruption()          # flushes TTS (Pipecat's own path)
            await self.push_frame(
                OutputTransportMessageFrame(
                    message=json.dumps({"barge_in": True, "spoken_text": spoken})),
                direction,
            )
            self._turn_task, self._controller = None, None

        await self.push_frame(frame, direction)

        if not isinstance(frame, TranscriptionFrame) or not frame.text.strip():
            return

        self._controller = BargeInController()
        self._turn_task = asyncio.create_task(self._run_turn(frame.text, direction))
        self._controller.bind_llm_task(self._turn_task)

    async def _run_turn(self, question: str, direction: FrameDirection) -> None:
        contract = await build_contract(question)
        # question "type" classification (factoid/multi_hop/comparative/...)
        # doesn't exist yet at pipeline runtime — graph augmentation (C12)
        # only ever helps, never breaks a turn, so "factoid" (no graph query)
        # is the safe default until a real classifier is wired in.
        result = await orchestrate(question, question_type="factoid")
        messages = build_messages(question)
        context = build_context(result)

        llm = get_llm()
        text, passed, _repairs = await generate_with_contract(llm, messages, contract, context)
        self._controller.tracker.append_generated(text)

        # web/ renders this as the transcript/lang/script/answer/citations
        # panel — sent as a LiveKit data message, not audio, so it arrives
        # independently of (and typically ahead of) the spoken reply.
        turn_metadata = json.dumps({
            "transcript": question,
            "lang": contract.lang,
            "script": contract.script,
            "answer": text,
            "citations": [h.chunk_id for h in result.hits],
            "fidelity_passed": passed,
        })
        await self.push_frame(OutputTransportMessageFrame(message=turn_metadata), direction)

        spoken_so_far = 0
        for sentence in split_sentences(text):
            await self.push_frame(BhashaSpeakFrame(text=sentence, lang=contract.lang), direction)
            spoken_so_far += len(sentence) + 1
            self._controller.tracker.mark_spoken(min(spoken_so_far, len(text)))

        self._turn_task, self._controller = None, None


class BhashaTTS(TTSService):
    """Wraps providers/factory.get_tts(). Reads the target language off
    BhashaSpeakFrame (stashed in process_frame, since TTSService.run_tts's
    signature is fixed to (text, context_id) — it doesn't get the frame)."""

    def __init__(self, **kwargs):
        provider = get_tts()
        super().__init__(sample_rate=provider.sample_rate, **kwargs)
        self._provider = provider
        self._pending_lang = "en"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, BhashaSpeakFrame):
            self._pending_lang = frame.lang
        await super().process_frame(frame, direction)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        async for pcm_chunk in self._provider.synthesize(text, self._pending_lang):
            yield TTSAudioRawFrame(audio=pcm_chunk, sample_rate=self.sample_rate, num_channels=1)


def build_pipeline(room_name: str, livekit_token: str) -> PipelineTask:
    s = get_settings()
    transport = LiveKitTransport(
        url=s.livekit_url,
        token=livekit_token,
        room_name=room_name,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=s.vad_silence_ms / 1000)),
    )
    pipeline = Pipeline([
        transport.input(),
        vad,
        BhashaSTT(),
        BhashaBrain(),
        BhashaTTS(),
        transport.output(),
    ])
    return PipelineTask(pipeline, params=PipelineParams())


async def run_pipeline(room_name: str, livekit_token: str) -> None:
    task = build_pipeline(room_name, livekit_token)
    runner = PipelineRunner()
    await runner.run(task)
