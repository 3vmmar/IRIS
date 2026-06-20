"""
Speech-to-text using faster-whisper for transcription and silero-VAD
for voice activity detection. Both run entirely locally — no external API.

Imports allowed: config.settings, faster_whisper, silero_vad, numpy,
                 sounddevice, soundfile, logging, threading, standard library.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
from faster_whisper import WhisperModel

from config import settings

logger = logging.getLogger(__name__)


class SpeechToText:
    """Local speech-to-text engine backed by faster-whisper + silero-VAD.

    Provides a simple load-then-transcribe interface. The microphone capture
    and VAD segmentation are handled externally (by the WebSocket handler
    or a dedicated audio pipeline) — this class only handles transcription
    of already-captured audio buffers.

    Usage::

        stt = SpeechToText()
        stt.load()

        text = stt.transcribe_bytes(raw_pcm_bytes)
        if text:
            print("User said:", text)
    """

    def __init__(self) -> None:
        """Initialise model references to ``None``.

        Models are loaded lazily via :meth:`load` so the import of this
        module is lightweight and never triggers a multi-second download.
        """
        self._whisper: Optional[WhisperModel] = None
        self._vad: Any = None   # silero returns OnnxWrapper — use Any to avoid coupling
        logger.debug("SpeechToText instance created (models not yet loaded).")

    # ─────────────────────────── lifecycle ────────────────────────────────────

    def load(self) -> None:
        """Load faster-whisper and silero-VAD models.

        Reads ``settings.whisper_model`` for the Whisper model size
        (``tiny``, ``base``, ``small``, ``medium``). Uses CPU with int8
        quantisation for ~2× speedup with minimal accuracy loss.

        Logs model identifiers and total load time at INFO level.

        Raises:
            RuntimeError: If either model fails to load (missing files,
                          incompatible Python version, etc.).
        """
        t0 = time.monotonic()

        # ── Whisper ───────────────────────────────────────────────────────────
        logger.info("Loading faster-whisper model: '%s' …", settings.whisper_model)
        try:
            self._whisper = WhisperModel(
                settings.whisper_model,
                device="cpu",
                compute_type="int8",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Whisper model '{settings.whisper_model}': {exc}"
            ) from exc

        # ── Silero VAD ────────────────────────────────────────────────────────
        logger.info("Loading silero-VAD model …")
        try:
            from silero_vad import load_silero_vad
            self._vad = load_silero_vad()
        except Exception as exc:
            raise RuntimeError(f"Failed to load silero-VAD: {exc}") from exc

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "STT models loaded in %.0f ms (whisper='%s', vad=silero).",
            elapsed_ms, settings.whisper_model,
        )

    # ─────────────────────────── transcription ────────────────────────────────

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16_000,
    ) -> Optional[str]:
        """Transcribe raw PCM audio bytes to text.

        Accepts 16-bit signed integer PCM, mono channel, at the given
        sample rate. Internally converts to float32 normalised to
        [-1.0, 1.0] before passing to faster-whisper.

        Args:
            audio_bytes: Raw PCM audio data (16-bit signed integers, mono).
            sample_rate: Sample rate of the input audio in Hz.
                         Defaults to 16 000 (Whisper's native rate).

        Returns:
            The transcribed text string, or ``None`` if the result is empty
            or only whitespace.
        """
        if self._whisper is None:
            logger.warning("transcribe_bytes called before load() — returning None.")
            return None

        t0 = time.monotonic()

        # ── Convert PCM bytes → float32 array ─────────────────────────────────
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_f32 = audio_int16.astype(np.float32) / 32768.0

        # ── Run faster-whisper inference ──────────────────────────────────────
        try:
            segments, _info = self._whisper.transcribe(
                audio_f32,
                language="en",
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            logger.warning("Whisper transcription error: %s", exc)
            return None

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if not text:
            logger.debug("Transcription returned empty string (%.0f ms).", elapsed_ms)
            return None

        logger.debug("Transcribed in %.0f ms: %r", elapsed_ms, text)
        return text

    # ─────────────────────────── status ────────────────────────────────────────

    def is_loaded(self) -> bool:
        """Return ``True`` if both Whisper and VAD models are loaded.

        Returns:
            ``True`` when the engine is ready to transcribe.
        """
        return self._whisper is not None and self._vad is not None
