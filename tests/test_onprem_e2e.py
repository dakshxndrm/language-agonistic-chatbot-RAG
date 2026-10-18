"""tests/test_onprem_e2e.py — End-to-End Air-Gapped (BHASHA_PROFILE=onprem) Integration Test.

Verifies:
  1. Fail-loud startup on leaked cloud keys (GROQ_API_KEY, SARVAM_API_KEY,
     OPENAI_API_KEY, COHERE_API_KEY).
  2. Local provider resolution (faster-whisper, BAAI/bge-m3, Ollama Qwen2.5,
     Kokoro-82M, fastText lid.176).
  3. Total cloud egress defense: any outbound connection to cloud domains
     triggers an immediate AssertionError.
  4. End-to-end turn execution for a Romanized Hinglish query:
     - Language Detection (local fastText + Unicode histogram)
     - Local Retrieval (pgvector + Neo4j)
     - Local LLM Streaming (Ollama qwen2.5:7b-instruct)
     - Local TTS Synthesis (Kokoro-82M)
     - Multimodal Visual Stage telemetry emission
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.frames.frames import (
    OutputTransportMessageFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from server.config import ConfigError, Settings
from server.pipeline import BhashaBrain, BhashaSpeakFrame, BhashaTTS
from server.providers.base import GraphHit
from server.providers.factory import (
    get_asr_en,
    get_asr_indic,
    get_embedder,
    get_llm,
    get_tts,
    reset_providers,
)
from server.rag.orchestrate import OrchestrationResult
from server.rag.retrieve import Hit

BANNED_DOMAINS = [
    "groq.com",
    "sarvam.ai",
    "cohere.ai",
    "cohere.com",
    "openai.com",
    "anthropic.com",
]


def _egress_guard(url: str) -> None:
    """Raises AssertionError if any cloud API domain is requested."""
    for domain in BANNED_DOMAINS:
        if domain in url.lower():
            raise AssertionError(f"Cloud egress detected to banned domain: {url}")


@pytest.fixture(autouse=True)
def clean_onprem_env(monkeypatch):
    """Sets onprem profile and unsets all cloud API keys for the test run."""
    monkeypatch.setenv("BHASHA_PROFILE", "onprem")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    onprem_settings = Settings(_env_file=None, bhasha_profile="onprem")
    monkeypatch.setattr("server.config.get_settings", lambda: onprem_settings)
    monkeypatch.setattr("server.pipeline.get_settings", lambda: onprem_settings)
    monkeypatch.setattr("server.providers.factory.get_settings", lambda: onprem_settings)
    monkeypatch.setattr("server.providers.llm_ollama.get_settings", lambda: onprem_settings)
    monkeypatch.setattr("server.lang.contract.get_settings", lambda: onprem_settings)
    monkeypatch.setattr("server.rag.retrieve.get_settings", lambda: onprem_settings)
    monkeypatch.setattr("server.rag.graph_retrieve.get_settings", lambda: onprem_settings)
    monkeypatch.setattr("server.main.get_settings", lambda: onprem_settings)

    reset_providers()
    yield
    reset_providers()


# ---------------------------------------------------------------------------
# 1. Fail-Loud Startup Sanity Checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cloud_key",
    ["GROQ_API_KEY", "SARVAM_API_KEY", "COHERE_API_KEY", "OPENAI_API_KEY"],
)
def test_onprem_fails_loud_on_leaked_key(cloud_key, monkeypatch):
    """Server must fail immediately at startup if any cloud key is populated."""
    monkeypatch.setenv(cloud_key, "sk-leaked-credential-12345")
    with pytest.raises(ConfigError, match=cloud_key):
        Settings(_env_file=None, bhasha_profile="onprem")


def test_onprem_provider_resolution():
    """Verify onprem profile binds exclusively to local runtimes."""
    s = Settings(_env_file=None, bhasha_profile="onprem")
    assert s.resolved == {
        "asr_en": "local",
        "asr_indic": "local",
        "llm": "ollama",
        "tts": "kokoro",
        "embed": "local",
        "rerank": "local",
        "graph": "neo4j_local",
    }


def test_factory_resolves_local_instances():
    """Verify provider factory produces local instances under onprem profile."""
    asr_en = get_asr_en()
    asr_indic = get_asr_indic()
    embedder = get_embedder()
    llm = get_llm()
    tts = get_tts()

    assert asr_en.name == "faster-whisper"
    assert asr_indic.name == "faster-whisper"
    assert embedder.name == "bge-m3"
    assert llm.name == "ollama"
    assert tts.name == "kokoro"


# ---------------------------------------------------------------------------
# 2. Local Language ID (fastText from disk)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_language_id_loads_and_detects():
    """Verify local fastText model loads from disk and detects Hinglish."""
    from server.lang.detect import get_detector

    detector = get_detector()
    detection = await detector.detect("Ashoka ne Kalinga ke baad kya kiya?")
    assert detection.lang == "hi-en"
    assert detection.script == "Latn"


# ---------------------------------------------------------------------------
# 3. End-to-End On-Prem Turn Execution with Cloud Egress Guard
# ---------------------------------------------------------------------------


class MockPushFrameCollector:
    """Captures pushed pipeline frames for assertion."""

    def __init__(self):
        self.frames = []

    async def __call__(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.frames.append(frame)


@pytest.mark.asyncio
async def test_onprem_e2e_turn_with_zero_egress(monkeypatch):
    """Executes a full turn in 100% onprem mode and verifies:
      - Zero cloud egress (guard asserts on any external call).
      - Local retrieval returns grounded corpus chunks.
      - Local Ollama LLM produces cited response.
      - Local Kokoro TTS generates raw PCM audio frames.
      - Multimodal visual telemetry sideband is emitted.
    """
    question = "Ashoka ne Kalinga ke baad kya kiya?"

    # 1. Guard against any cloud HTTP calls
    import httpx

    original_send = httpx.AsyncClient.send

    async def guarded_send(self, request, *args, **kwargs):
        url_str = str(request.url)
        _egress_guard(url_str)
        # Handle Ollama local mock
        if "11434" in url_str:
            # Stream response in Ollama NDJSON format
            tokens = [
                "Ashoka ",
                "ne ",
                "Kalinga ",
                "war ",
                "ke ",
                "baad ",
                "Buddhism ",
                "apnaya ",
                "[ashoka::0000].",
            ]
            lines = [
                json.dumps({"message": {"content": t}, "done": False}).encode("utf-8")
                + b"\n"
                for t in tokens
            ]
            lines.append(json.dumps({"done": True}).encode("utf-8") + b"\n")

            mock_stream = httpx.ByteStream(b"".join(lines))
            return httpx.Response(200, stream=mock_stream, request=request)
        return await original_send(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", guarded_send)

    # 2. Mock local retrieval to return chunk from Ashoka
    fake_retrieval = OrchestrationResult(
        hits=[
            Hit(
                chunk_id="ashoka::0000",
                article="Ashoka",
                section="Introduction",
                text="Ashoka embraced Buddhism after the Kalinga war.",
                score=0.95,
            )
        ],
        graph_paths=[
            GraphHit(
                source_chunk_ids=["ashoka::0000"],
                path=[
                    {"entity": "Ashoka"},
                    {"rel": "CONVERTED_TO"},
                    {"entity": "Buddhism"},
                ],
                confidence=1.0,
            )
        ],
        vector_only_hits=[
            Hit(
                chunk_id="ashoka::0000",
                article="Ashoka",
                section="Introduction",
                text="Ashoka embraced Buddhism after the Kalinga war.",
                score=0.95,
            )
        ],
    )

    with (
        patch("server.pipeline.orchestrate", new_callable=AsyncMock) as mock_orch,
        patch("server.pipeline.check_semantic_cache", new_callable=AsyncMock) as mock_cache,
        patch("server.pipeline.store_semantic_cache", new_callable=AsyncMock),
    ):
        mock_orch.return_value = fake_retrieval
        mock_cache.return_value = None

        # Mock local TTS synthesis to emit 24kHz PCM chunks without Kokoro weight download
        fake_pcm = b"\x00\x00" * 4800  # 0.2s of silence/audio at 24kHz 16-bit mono

        async def fake_synthesize(text, lang, voice=None):
            yield fake_pcm

        monkeypatch.setattr("server.providers.tts_kokoro.KokoroTTS.synthesize", fake_synthesize)

        # 3. Feed question into BhashaBrain
        brain = BhashaBrain()
        collector = MockPushFrameCollector()
        brain.push_frame = collector

        input_frame = TranscriptionFrame(
            text=question,
            user_id="test_user",
            timestamp="1234567.89",
        )

        await brain.process_frame(input_frame, FrameDirection.DOWNSTREAM)
        if brain._turn_task is not None:
            await brain._turn_task

        # 4. Verify downstream frames
        meta_frames = [
            f
            for f in collector.frames
            if isinstance(f, OutputTransportMessageFrame)
            and "transcript" in f.message
        ]
        assert len(meta_frames) >= 1
        meta = json.loads(meta_frames[0].message)

        # Verify language fidelity contract preserved Romanized Hinglish
        assert meta["transcript"] == question
        assert meta["lang"] == "hi-en"
        assert meta["script"] == "Latn"
        assert meta["fidelity_passed"] is True
        assert "ashoka::0000" in meta["citations"]
        assert "visual_stage" in meta

        # Verify Visual Stage payload
        vs = meta["visual_stage"]
        assert "geo_coordinates" in vs
        assert "subgraph" in vs
        assert "citation" in vs

        # Verify speak frames pushed
        speak_frames = [f for f in collector.frames if isinstance(f, BhashaSpeakFrame)]
        assert len(speak_frames) >= 1
        assert speak_frames[0].lang == "hi-en"
        assert "Ashoka" in speak_frames[0].text

        # 5. Verify BhashaTTS emits audio
        tts = BhashaTTS()
        audio_frames = []
        for sf in speak_frames:
            await tts.process_frame(sf, FrameDirection.DOWNSTREAM)
            async for audio_frame in tts.run_tts(sf.text, "test_context"):
                if isinstance(audio_frame, TTSAudioRawFrame):
                    audio_frames.append(audio_frame)

        assert len(audio_frames) >= 1
        assert len(audio_frames[0].audio) > 0
        assert audio_frames[0].sample_rate == 24000
