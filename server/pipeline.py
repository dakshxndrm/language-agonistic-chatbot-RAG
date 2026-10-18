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
    OutputTransportMessageUrgentFrame,
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
from pipecat.transports.livekit.transport import (
    LiveKitInputTransportMessageFrame,
    LiveKitParams,
    LiveKitTransport,
)

from server.barge_in import BargeInController
from server.config import get_settings
from server.lang.contract import build_contract, generate_with_contract
from server.providers._pcm import wav_bytes_to_pcm
from server.providers.asr_router import route_transcribe
from server.providers.factory import get_llm, get_tts
from server.rag.cache import check_semantic_cache, store_semantic_cache
from server.rag.orchestrate import orchestrate
from server.rag.prompt import build_context, build_messages
from server.visual_stage import extract_visual_payload

# Heuristic sentence split (English '.'/'!'/'?', Devanagari '।', double danda '॥', newlines)
_SENTENCE_RE = re.compile(r"(?<=[.!?।॥])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return parts or ([text] if text.strip() else [])


def clean_tts_text(text: str) -> str:
    """Strips citation brackets (e.g. 【ashoka::0000】 or [ashoka::0001]) and markdown
    formatting so TTS speaks natural prose without stumbling on tokens."""
    cleaned = re.sub(r"[\[【][^\]】]+[\]】]", "", text)
    cleaned = cleaned.replace("**", "").replace("*", "").replace("#", "")
    cleaned = re.sub(r"\s+([.,!?।॥])", r"\1", cleaned)
    return cleaned.strip()


def detect_question_type(question: str) -> str:
    """Classifies question intent to trigger graph augmentation when relational or comparative."""
    q_lower = question.lower()
    if any(w in q_lower for w in ["compare", "difference", "vs", "versus", "antar", "tulna"]):
        return "comparative"
    if any(w in q_lower for w in [
        "father", "mother", "son", "daughter", "grandfather", "grandmother",
        "successor", "succeeded", "predecessor", "descendant", "ancestor",
        "lineage", "dynasty", "battle", "war", "defeated", "ruled after", "ruled before",
        "baap", "pita", "beta", "pota", "kiske baad", "kiske pehle", "kiska beta", "kiski beti",
        "vansh", "samrajya", "parivar", "yudh", "ladai", "ladha"
    ]):
        return "multi_hop"
    return "factoid"


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
        self._history: list[dict] = []
        self._current_question: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if (isinstance(frame, VADUserStartedSpeakingFrame)
                and self._turn_task is not None and not self._turn_task.done()):
            spoken = await self._controller.barge_in()  # cancels _run_turn (LLM side)
            await self.broadcast_interruption()          # flushes TTS (Pipecat's own path)
            if self._current_question:
                self._history.append({"role": "user", "content": self._current_question})
                if spoken.strip():
                    self._history.append({"role": "assistant", "content": spoken})
                if len(self._history) > 6:
                    self._history = self._history[-6:]
                self._current_question = None
            barge_payload = json.dumps({"barge_in": True, "spoken_text": spoken})
            await self.push_frame(
                OutputTransportMessageFrame(message=barge_payload),
                direction,
            )
            await self.push_frame(
                OutputTransportMessageUrgentFrame(message=barge_payload),
                direction,
            )
            self._turn_task, self._controller = None, None

        await self.push_frame(frame, direction)

        query_text: str | None = None
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            query_text = frame.text.strip()
        elif isinstance(frame, LiveKitInputTransportMessageFrame):
            msg = frame.message
            if isinstance(msg, dict) and "text" in msg and str(msg["text"]).strip():
                query_text = str(msg["text"]).strip()

        if not query_text:
            return

        self._current_question = query_text
        self._controller = BargeInController()
        self._turn_task = asyncio.create_task(self._run_turn(query_text, direction))
        self._controller.bind_llm_task(self._turn_task)

    async def _run_turn(self, question: str, direction: FrameDirection) -> None:
        # 0. Send immediate feedback that speech was recognized and assistant is thinking
        status_payload = json.dumps({
            "event": "status",
            "status": "thinking",
            "question": question,
        })
        await self.push_frame(OutputTransportMessageFrame(message=status_payload), direction)
        await self.push_frame(OutputTransportMessageUrgentFrame(message=status_payload), direction)

        contract = await build_contract(question)

        # 1. Semantic Cache check (Phase 5): sub-120ms bypass on cosine similarity >= 0.96
        cached_hit = await check_semantic_cache(question)
        if cached_hit:
            text = cached_hit.response_text
            passed = True
            visual_payload = extract_visual_payload(
                question,
                text,
                [],
                [],
            )
            citations = cached_hit.metadata.get("citations", [])
            turn_metadata = json.dumps({
                "event": "answer",
                "transcript": question,
                "lang": contract.lang,
                "script": contract.script,
                "answer": text,
                "citations": citations,
                "fidelity_passed": True,
                "cached": True,
                "similarity": cached_hit.similarity,
                "visual_stage": visual_payload.to_dict(),
            })
            await self.push_frame(OutputTransportMessageFrame(message=turn_metadata), direction)
            await self.push_frame(OutputTransportMessageUrgentFrame(message=turn_metadata), direction)
            for sentence in split_sentences(text):
                clean_s = clean_tts_text(sentence)
                if clean_s:
                    await self.push_frame(
                        BhashaSpeakFrame(text=clean_s, lang=contract.lang), direction
                    )
            self._history.append({"role": "user", "content": question})
            self._history.append({"role": "assistant", "content": text})
            if len(self._history) > 6:
                self._history = self._history[-6:]
            self._current_question = None
            self._turn_task, self._controller = None, None
            return

        # 2. Standard RAG + Generation path with dynamic graph classification and memory
        q_type = detect_question_type(question)
        result = await orchestrate(question, question_type=q_type)
        try:
            messages = build_messages(question, history=self._history)
        except TypeError:
            messages = build_messages(question)
        context = build_context(result)

        llm = get_llm()
        sentences_streamed = []

        async def _stream_sentence(sent: str):
            clean_s = clean_tts_text(sent)
            if not clean_s:
                return
            sentences_streamed.append(clean_s)
            await self.push_frame(BhashaSpeakFrame(text=clean_s, lang=contract.lang), direction)
            self._controller.tracker.mark_spoken(sum(len(s) + 1 for s in sentences_streamed))

        text, passed, _repairs = await generate_with_contract(
            llm, messages, contract, context, on_sentence=_stream_sentence
        )
        self._controller.tracker.append_generated(text)

        # Save to semantic cache asynchronously
        asyncio.create_task(store_semantic_cache(question, text))

        # Generate Visual Stage telemetry
        visual_payload = extract_visual_payload(
            question,
            text,
            getattr(result, "hits", []),
            getattr(result, "graph_paths", []),
        )

        turn_metadata = json.dumps({
            "event": "answer",
            "transcript": question,
            "lang": contract.lang,
            "script": contract.script,
            "answer": text,
            "citations": [h.chunk_id for h in getattr(result, "hits", [])],
            "fidelity_passed": passed,
            "cached": False,
            "visual_stage": visual_payload.to_dict(),
        })
        await self.push_frame(OutputTransportMessageFrame(message=turn_metadata), direction)
        await self.push_frame(OutputTransportMessageUrgentFrame(message=turn_metadata), direction)

        # If on_sentence was not invoked during generation (e.g. test mock), push sentences now
        if not sentences_streamed:
            spoken_so_far = 0
            for sentence in split_sentences(text):
                clean_s = clean_tts_text(sentence)
                if clean_s:
                    await self.push_frame(BhashaSpeakFrame(text=clean_s, lang=contract.lang), direction)
                spoken_so_far += len(sentence) + 1
                self._controller.tracker.mark_spoken(min(spoken_so_far, len(text)))

        # Update conversational memory (sliding window of 6 messages = 3 turns)
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": text})
        if len(self._history) > 6:
            self._history = self._history[-6:]

        self._current_question = None
        self._turn_task, self._controller = None, None


class BhashaTTS(TTSService):
    """Wraps providers/factory.get_tts(). Reads the target language off
    BhashaSpeakFrame (stashed in process_frame, since TTSService.run_tts's
    signature is fixed to (text, context_id) — it doesn't get the frame)."""

    def __init__(self, **kwargs):
        provider = get_tts()
        super().__init__(sample_rate=provider.sample_rate, **kwargs)
        self._provider = provider
        self._sample_rate = provider.sample_rate
        self._pending_lang = "en"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, BhashaSpeakFrame):
            self._pending_lang = frame.lang
        await super().process_frame(frame, direction)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        async for pcm_chunk in self._provider.synthesize(text, self._pending_lang):
            yield TTSAudioRawFrame(
                audio=pcm_chunk, sample_rate=self._provider.sample_rate, num_channels=1
            )


def build_pipeline(room_name: str, livekit_token: str) -> PipelineTask:
    s = get_settings()
    tts = get_tts()
    transport = LiveKitTransport(
        url=s.livekit_url,
        token=livekit_token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=tts.sample_rate,
        ),
    )
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(stop_secs=s.vad_silence_ms / 1000, min_volume=0.05)
        ),
    )
    pipeline = Pipeline([
        transport.input(),
        vad,
        BhashaSTT(),
        BhashaBrain(),
        BhashaTTS(),
        transport.output(),
    ])
    task = PipelineTask(pipeline, params=PipelineParams())

    @transport.event_handler("on_participant_disconnected")
    async def _on_participant_disconnected(transport, participant_id: str):
        await task.cancel()

    return task


async def run_pipeline(room_name: str, livekit_token: str) -> None:
    task = build_pipeline(room_name, livekit_token)
    runner = PipelineRunner()
    await runner.run(task)
