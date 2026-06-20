"""
Text-to-speech abstraction layer.

Supports:
  - **Coqui TTS** — local inference, no API key required
  - **ElevenLabs** — cloud API, premium quality
  - **edge-tts** — free Microsoft neural voices, no API key required

The active engine is selected by ``settings.tts_engine`` at startup.

All third-party imports are **lazy** (inside methods, not at module level)
so the module never crashes at import time if one backend is missing.

Imports allowed: config.settings, logging, standard library.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Facade that wraps the configured TTS backend.

    Usage::

        tts = TextToSpeech()
        tts.load()

        wav_bytes = tts.synthesize("Hello from IRIS.")
        if wav_bytes:
            # send to frontend via WebSocket
            ...
    """

    def __init__(self) -> None:
        """Initialise engine reference to ``None``.

        The actual backend is loaded lazily via :meth:`load` based on
        ``settings.tts_engine``.
        """
        self._engine: Any = None
        self._engine_name: str = ""
        logger.debug("TextToSpeech instance created (engine not yet loaded).")

    # ─────────────────────────── lifecycle ────────────────────────────────────

    def load(self) -> None:
        """Load the TTS backend specified by ``settings.tts_engine``.

        Supported values:

        - ``"coqui"`` — local Coqui TTS (tacotron2-DDC, CPU)
        - ``"elevenlabs"`` — cloud ElevenLabs API
        - ``"edge-tts"`` / ``"edge"`` — free Microsoft neural voices

        Logs the engine name and load time at INFO level.

        Raises:
            ValueError:    If ``tts_engine`` is not one of the supported values.
            RuntimeError:  If the backend fails to initialise (missing package,
                           missing API key, etc.).
        """
        engine = settings.tts_engine.lower().strip()
        t0 = time.monotonic()

        if engine == "coqui":
            self._load_coqui()
        elif engine == "elevenlabs":
            self._load_elevenlabs()
        elif engine in ("edge-tts", "edge"):
            self._load_edge_tts()
        else:
            raise ValueError(
                f"Unknown TTS_ENGINE '{settings.tts_engine}'. "
                "Valid options: 'coqui' | 'elevenlabs' | 'edge-tts'."
            )

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info("TTS loaded: engine='%s' in %.0f ms.", self._engine_name, elapsed_ms)

    # ─────────────────────────── synthesis ────────────────────────────────────

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize *text* to audio bytes using the active backend.

        Args:
            text: The string to speak. Should be a complete sentence or
                  paragraph — very short strings may sound clipped.

        Returns:
            Raw audio bytes (WAV for Coqui/edge-tts, MP3 for ElevenLabs),
            or ``None`` if the engine is not loaded or synthesis fails.
        """
        if self._engine is None:
            logger.warning("synthesize() called before load() — returning None.")
            return None

        try:
            if self._engine_name == "coqui":
                return self._synthesize_coqui(text)
            elif self._engine_name == "elevenlabs":
                return self._synthesize_elevenlabs(text)
            elif self._engine_name == "edge-tts":
                return self._synthesize_edge_tts(text)
        except Exception as exc:
            logger.error("TTS synthesis error (%s): %s", self._engine_name, exc)
            return None

        return None

    # ─────────────────────────── status ────────────────────────────────────────

    def is_loaded(self) -> bool:
        """Return ``True`` if the TTS engine is ready to synthesize.

        Returns:
            ``True`` when the engine has been successfully loaded.
        """
        return self._engine is not None

    # ─────────────────────────── loaders (private) ────────────────────────────

    def _load_coqui(self) -> None:
        """Load Coqui TTS with the tacotron2-DDC English model."""
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore[import]
        except ImportError:
            raise RuntimeError(
                "Coqui TTS is not installed. Run: pip install TTS"
            )

        logger.info("Loading Coqui TTS model: tts_models/en/ljspeech/tacotron2-DDC …")
        self._engine = CoquiTTS(
            model_name="tts_models/en/ljspeech/tacotron2-DDC",
            gpu=False,
        )
        self._engine_name = "coqui"

    def _load_elevenlabs(self) -> None:
        """Prepare ElevenLabs API client."""
        if not settings.elevenlabs_api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not set. "
                "Add it to .env or switch TTS_ENGINE=edge-tts."
            )
        try:
            import elevenlabs  # type: ignore[import]
            elevenlabs.set_api_key(settings.elevenlabs_api_key)
        except ImportError:
            raise RuntimeError(
                "elevenlabs package is not installed. Run: pip install elevenlabs"
            )

        self._engine = elevenlabs
        self._engine_name = "elevenlabs"
        logger.info("ElevenLabs API key configured.")

    def _load_edge_tts(self) -> None:
        """Prepare edge-tts (Microsoft neural voices, free)."""
        try:
            import edge_tts  # type: ignore[import]
        except ImportError:
            raise RuntimeError(
                "edge-tts is not installed. Run: pip install edge-tts"
            )

        voice = settings.elevenlabs_voice_id or "en-US-AriaNeural"
        self._engine = {"module": edge_tts, "voice": voice}
        self._engine_name = "edge-tts"
        logger.info("edge-tts ready (voice=%s).", voice)

    # ─────────────────────────── synthesizers (private) ───────────────────────

    def _synthesize_coqui(self, text: str) -> bytes:
        """Synthesize via Coqui TTS → temp WAV file → bytes."""
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self._engine.tts_to_file(text=text, file_path=tmp_path)
            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()
            logger.debug("Coqui synthesized %d chars → %d bytes WAV.", len(text), len(wav_bytes))
            return wav_bytes
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _synthesize_elevenlabs(self, text: str) -> bytes:
        """Synthesize via the ElevenLabs API → MP3 bytes."""
        voice_id = settings.elevenlabs_voice_id or "Rachel"
        audio = self._engine.generate(
            text=text,
            voice=voice_id,
        )
        mp3_bytes = b"".join(audio) if hasattr(audio, "__iter__") else audio
        logger.debug("ElevenLabs synthesized %d chars → %d bytes MP3.", len(text), len(mp3_bytes))
        return mp3_bytes

    def _synthesize_edge_tts(self, text: str) -> bytes:
        """Synthesize via edge-tts → in-memory MP3 bytes.

        edge-tts is async-native, so we run a small event loop if we
        are not already inside one.
        """
        import asyncio

        edge_tts = self._engine["module"]
        voice = self._engine["voice"]

        async def _run() -> bytes:
            communicate = edge_tts.Communicate(text, voice)
            chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        # Run the coroutine — handle both "no loop" and "loop already running"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            mp3_bytes = asyncio.run(_run())
        else:
            # Already in an async context — create a new thread to avoid
            # blocking the running event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                mp3_bytes = pool.submit(asyncio.run, _run()).result()

        if not mp3_bytes:
            raise RuntimeError("edge-tts returned empty audio.")

        logger.debug("edge-tts synthesized %d chars → %d bytes MP3.", len(text), len(mp3_bytes))
        return mp3_bytes
