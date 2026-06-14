"""
Speech-to-text using faster-whisper with silero-VAD
for automatic voice activity detection and audio segmentation.
"""

from __future__ import annotations


import io
import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

from config import settings

logger = logging.getLogger(__name__)

# ── Audio constants ────────────────────────────────────────────────────────────
_SAMPLE_RATE   = 16_000   # Hz — Whisper's native rate
_CHANNELS      = 1
_CHUNK_MS      = 30       # ms per audio chunk fed to silero-VAD
_CHUNK_SAMPLES = int(_SAMPLE_RATE * _CHUNK_MS / 1000)
_SILENCE_PAD_S = 0.3      # seconds of silence appended after VAD end to avoid cut-off


class SpeechToText:
    """Listens to the microphone, detects speech with silero-VAD, and transcribes
    each utterance with faster-whisper.

    The transcription callback is called on a dedicated thread with the
    recognised text so it never blocks the main async event loop.

    Usage::

        def on_transcript(text: str) -> None:
            print("IRIS heard:", text)

        stt = SpeechToText(callback=on_transcript)
        stt.start()
        # … user speaks …
        stt.stop()
    """

    def __init__(self, callback: Callable[[str], None]) -> None:
        """
        Args:
            callback: Called with the transcribed string after each utterance.
                      Invoked from a background thread — use thread-safe mechanisms
                      (e.g. ``asyncio.run_coroutine_threadsafe``) to hand off to async code.
        """
        self._callback = callback
        self._model: Optional[WhisperModel] = None
        self._vad_model: Any = None
        self._stop_event = threading.Event()
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._listen_thread: Optional[threading.Thread] = None
        self._transcribe_thread: Optional[threading.Thread] = None

    # ─────────────────────────── lifecycle ────────────────────────────────────

    def start(self) -> None:
        """Load models and start the microphone listener + transcription threads.

        Raises:
            RuntimeError: If the Whisper or VAD model cannot be loaded.
        """
        logger.info("Loading faster-whisper model: %s …", settings.whisper_model)
        try:
            self._model = WhisperModel(
                settings.whisper_model,
                device="cpu",
                compute_type="int8",   # int8 gives ~2× speedup on CPU with minimal WER cost
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load Whisper model '{settings.whisper_model}': {exc}") from exc

        logger.info("Loading silero-VAD model …")
        try:
            self._vad_model = load_silero_vad()
        except Exception as exc:
            raise RuntimeError(f"Failed to load silero-VAD: {exc}") from exc

        self._stop_event.clear()

        self._listen_thread = threading.Thread(
            target=self._listen_loop,
            name="iris-mic-listener",
            daemon=True,
        )
        self._transcribe_thread = threading.Thread(
            target=self._transcribe_loop,
            name="iris-transcriber",
            daemon=True,
        )

        self._listen_thread.start()
        self._transcribe_thread.start()
        logger.info("STT started (model=%s, VAD=silero, sample_rate=%d Hz).",
                    settings.whisper_model, _SAMPLE_RATE)

    def stop(self) -> None:
        """Signal both threads to stop and join with a 3-second timeout each."""
        logger.info("Stopping STT …")
        self._stop_event.set()
        for t in (self._listen_thread, self._transcribe_thread):
            if t is not None and t.is_alive():
                t.join(timeout=3.0)
        logger.info("STT stopped.")

    def is_running(self) -> bool:
        """Return True if the listener thread is alive."""
        return self._listen_thread is not None and self._listen_thread.is_alive()

    # ─────────────────────────── transcribe on demand ─────────────────────────

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe a raw PCM or WAV byte blob sent over WebSocket.

        Intended for the ``audio_chunk`` WebSocket message type where the
        client captures audio and sends it as base64-decoded bytes.

        Args:
            audio_bytes: Raw audio data. Accepts WAV (with header) or raw
                         16-bit PCM at 16 kHz mono.

        Returns:
            Transcribed text string, stripped of leading/trailing whitespace.
            Empty string on failure.
        """
        if self._model is None:
            logger.warning("transcribe_audio_bytes called before STT is started.")
            return ""
        try:
            audio_buf = io.BytesIO(audio_bytes)
            audio_np, sr = sf.read(audio_buf, dtype="float32")
            if sr != _SAMPLE_RATE:
                # Basic resampling: soundfile doesn't resample so we note the mismatch
                logger.warning("Audio sample rate %d Hz != expected %d Hz.", sr, _SAMPLE_RATE)
            segments, _ = self._model.transcribe(audio_np, language="en", beam_size=5)
            text = " ".join(seg.text for seg in segments).strip()
            logger.debug("Transcribed (bytes input): %r", text)
            return text
        except Exception as exc:
            logger.warning("transcribe_audio_bytes failed: %s", exc)
            return ""

    # ─────────────────────────── internal threads ──────────────────────────────

    def _listen_loop(self) -> None:
        """Continuously read microphone chunks and push speech segments to the queue.

        Uses silero-VAD to detect start/end of speech. Accumulates audio while
        VAD reports speech activity, then enqueues the complete utterance for
        transcription.
        """
        accumulated: list[np.ndarray] = []
        in_speech = False
        silence_counter = 0
        # Number of consecutive silent chunks before we consider the utterance done
        silence_chunks_threshold = int(0.6 * 1000 / _CHUNK_MS)   # ~600 ms

        def audio_callback(indata: np.ndarray, frames: int,
                           time_info: object, status: object) -> None:
            nonlocal accumulated, in_speech, silence_counter

            if status:
                logger.debug("Sounddevice status: %s", status)

            chunk = indata[:, 0].copy().astype(np.float32)
            chunk_tensor = torch.from_numpy(chunk)

            try:
                vad = self._vad_model
                if vad is not None:
                    speech_prob = vad(chunk_tensor, _SAMPLE_RATE).item()
                else:
                    speech_prob = 0.0
            except Exception:
                speech_prob = 0.0

            if speech_prob > 0.5:
                if not in_speech:
                    logger.debug("VAD: speech start detected (prob=%.2f).", speech_prob)
                    in_speech = True
                silence_counter = 0
                accumulated.append(chunk)
            elif in_speech:
                accumulated.append(chunk)   # include trailing silence for natural cut
                silence_counter += 1
                if silence_counter >= silence_chunks_threshold:
                    # Speech ended — enqueue the utterance
                    utterance = np.concatenate(accumulated)
                    self._audio_queue.put(utterance)
                    logger.debug("VAD: speech end — queued %.2f s of audio.",
                                 len(utterance) / _SAMPLE_RATE)
                    accumulated = []
                    in_speech = False
                    silence_counter = 0

        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype="float32",
                blocksize=_CHUNK_SAMPLES,
                callback=audio_callback,
            ):
                while not self._stop_event.is_set():
                    time.sleep(0.1)
        except Exception as exc:
            logger.error("Microphone listener error: %s", exc)

    def _transcribe_loop(self) -> None:
        """Pull audio utterances off the queue and transcribe them via Whisper.

        Calls ``self._callback(text)`` for every non-empty transcript.
        """
        while not self._stop_event.is_set():
            try:
                audio_np: np.ndarray = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._model is None:
                continue

            try:
                t0 = time.monotonic()
                segments, info = self._model.transcribe(
                    audio_np,
                    language="en",
                    beam_size=5,
                    vad_filter=False,   # VAD already applied upstream
                )
                text = " ".join(seg.text for seg in segments).strip()
                elapsed_ms = (time.monotonic() - t0) * 1000

                if text:
                    logger.info("Transcribed in %.0f ms: %r", elapsed_ms, text)
                    self._callback(text)
                else:
                    logger.debug("Transcription returned empty string — skipping.")
            except Exception as exc:
                logger.warning("Transcription error: %s", exc)
