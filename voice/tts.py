"""
Text-to-speech abstraction layer.
Supports edge-tts (free Microsoft neural voices, local-quality) and ElevenLabs (API, premium quality).
Selected at startup via TTS_ENGINE environment variable.
"""

from __future__ import annotations

import asyncio
import io
import logging

from abc import ABC, abstractmethod

from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


# ── Abstract base ──────────────────────────────────────────────────────────────

class TTSBackend(ABC):
    """Abstract interface that all TTS backends must implement."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert *text* to audio and return it as raw WAV bytes.

        Args:
            text: The string to synthesize.

        Returns:
            WAV audio bytes (RIFF header + PCM data) ready to base64-encode
            and send over the WebSocket ``audio_chunk`` message.

        Raises:
            RuntimeError: If synthesis fails and the error is unrecoverable.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any open connections or resources."""


# ── edge-tts backend ──────────────────────────────────────────────────────────

class EdgeTTSBackend(TTSBackend):
    """TTS via Microsoft Edge's free neural speech API (edge-tts).

    Requires an internet connection. No API key needed.
    Default voice: ``en-US-AriaNeural`` — high quality, natural cadence.

    The voice can be overridden by setting ``ELEVENLABS_VOICE_ID`` to any
    edge-tts voice name (e.g. ``en-GB-SoniaNeural``) in the .env file.
    This field is repurposed here to avoid adding a new env variable.
    """

    _DEFAULT_VOICE = "en-US-AriaNeural"

    def __init__(self) -> None:
        # Reuse ELEVENLABS_VOICE_ID as a generic "voice name" override so the
        # user doesn't need an extra env var just to pick an edge-tts voice.
        self._voice = settings.elevenlabs_voice_id or self._DEFAULT_VOICE
        logger.info("EdgeTTS backend initialised (voice=%s).", self._voice)

    async def synthesize(self, text: str) -> bytes:
        """Stream audio from the Edge TTS service and return WAV bytes.

        Args:
            text: Text to synthesize (any length; edge-tts handles chunking).

        Returns:
            WAV bytes of the synthesized speech.

        Raises:
            RuntimeError: On network or synthesis failure.
        """
        try:
            import edge_tts  # imported lazily to avoid import cost when unused

            communicate = edge_tts.Communicate(text, self._voice)

            # Collect all audio chunks into a buffer
            audio_chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            raw_mp3 = b"".join(audio_chunks)
            if not raw_mp3:
                raise RuntimeError("edge-tts returned empty audio.")

            # Convert MP3 → WAV in-memory via soundfile + io
            wav_bytes = _mp3_to_wav(raw_mp3)
            logger.debug("EdgeTTS synthesized %d chars → %d bytes WAV.", len(text), len(wav_bytes))
            return wav_bytes

        except ImportError:
            raise RuntimeError(
                "edge-tts is not installed. Run: pip install edge-tts==7.0.2"
            )
        except Exception as exc:
            raise RuntimeError(f"EdgeTTS synthesis failed: {exc}") from exc

    async def close(self) -> None:
        """No persistent resources to release."""


# ── ElevenLabs backend ─────────────────────────────────────────────────────────

class ElevenLabsBackend(TTSBackend):
    """TTS via the ElevenLabs API — premium neural voice quality.

    Requires ``ELEVENLABS_API_KEY`` and optionally ``ELEVENLABS_VOICE_ID``
    in .env. Falls back to the first available voice if no ID is set.
    """

    def __init__(self) -> None:
        if not settings.elevenlabs_api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not set. "
                "Add it to .env or switch TTS_ENGINE=edge-tts."
            )
        try:
            from elevenlabs.client import ElevenLabs  # type: ignore[import]
            self._client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        except ImportError:
            raise RuntimeError("elevenlabs package is not installed.")

        self._voice_id: Optional[str] = settings.elevenlabs_voice_id or None
        logger.info(
            "ElevenLabs backend initialised (voice_id=%s).",
            self._voice_id or "auto",
        )

    async def synthesize(self, text: str) -> bytes:
        """Call the ElevenLabs TTS API and return WAV bytes.

        The API call is run in a thread executor so it doesn't block the event loop.

        Args:
            text: Text to synthesize.

        Returns:
            WAV audio bytes.

        Raises:
            RuntimeError: On API or network failure.
        """
        try:
            loop = asyncio.get_event_loop()
            mp3_bytes: bytes = await loop.run_in_executor(
                None, self._synthesize_sync, text
            )
            wav_bytes = _mp3_to_wav(mp3_bytes)
            logger.debug("ElevenLabs synthesized %d chars → %d bytes WAV.", len(text), len(wav_bytes))
            return wav_bytes
        except Exception as exc:
            raise RuntimeError(f"ElevenLabs synthesis failed: {exc}") from exc

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous ElevenLabs call (runs inside executor)."""
        audio_iter = self._client.text_to_speech.convert(
            text=text,
            voice_id=self._voice_id or "Rachel",
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )
        return b"".join(audio_iter)

    async def close(self) -> None:
        """No persistent connections to release."""


# ── TTS facade ────────────────────────────────────────────────────────────────

class TextToSpeech:
    """Public facade that selects and wraps the configured TTS backend.

    Usage::

        tts = TextToSpeech()
        await tts.init()

        wav_bytes = await tts.synthesize("Hello from IRIS.")
        # send wav_bytes as base64 over WebSocket

        await tts.close()
    """

    def __init__(self) -> None:
        self._backend: Optional[TTSBackend] = None

    async def init(self) -> None:
        """Instantiate the backend selected by ``settings.tts_engine``.

        Raises:
            ValueError: If ``TTS_ENGINE`` is set to an unsupported value.
            RuntimeError: If the backend fails to initialise.
        """
        engine = settings.tts_engine.lower().strip()
        logger.info("Initialising TTS engine: %s", engine)

        if engine == "edge-tts" or engine == "edge":
            self._backend = EdgeTTSBackend()
        elif engine == "elevenlabs":
            self._backend = ElevenLabsBackend()
        else:
            raise ValueError(
                f"Unknown TTS_ENGINE '{settings.tts_engine}'. "
                "Valid options: 'edge-tts' | 'elevenlabs'."
            )

    async def synthesize(self, text: str) -> bytes:
        """Synthesize *text* using the active backend and return WAV bytes.

        Args:
            text: The string to speak. Should be a complete sentence or
                  paragraph — very short strings may sound clipped.

        Returns:
            WAV audio bytes to send to the frontend.

        Raises:
            RuntimeError: If the backend is not initialised or synthesis fails.
        """
        if self._backend is None:
            raise RuntimeError("TextToSpeech.init() has not been called.")
        return await self._backend.synthesize(text)

    async def close(self) -> None:
        """Shut down the active backend and release resources."""
        if self._backend is not None:
            await self._backend.close()
            self._backend = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Convert MP3 bytes to WAV bytes using soundfile + pydub fallback.

    edge-tts and ElevenLabs both return MP3. The frontend AudioContext
    can decode MP3 directly in the browser, so this conversion is mainly
    for server-side audio processing consistency.

    If the environment has neither soundfile (which doesn't decode MP3) nor
    pydub, the raw MP3 bytes are returned as-is — the frontend handles it.

    Args:
        mp3_bytes: Raw MP3-encoded audio bytes.

    Returns:
        WAV-encoded bytes, or the original MP3 bytes if conversion is unavailable.
    """
    # Try pydub (requires ffmpeg) — most flexible
    try:
        from pydub import AudioSegment  # type: ignore[import]
        segment = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        wav_buf = io.BytesIO()
        segment.export(wav_buf, format="wav")
        return wav_buf.getvalue()
    except Exception:
        pass

    # Fall through: return MP3 as-is — browser can decode MP3 natively
    logger.debug("MP3→WAV conversion unavailable — returning raw MP3 bytes.")
    return mp3_bytes
