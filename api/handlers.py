"""
Per-client WebSocket protocol handler.

Incoming message types  : text_query | audio_chunk | request_snapshot
Outgoing message types  : detections | text_token | text_done
                          audio_chunk | memory_update | error

Runs a background asyncio task that pushes detection frames to the client
every 150ms independently of user queries.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from agent.brain import IRISBrain

from core.camera import CameraCapture
from core.detector import DetectionResult, ObjectDetector
from core.memory import VisualMemory
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)

# How often to push detection results to the client (ms)
_DETECTION_PUSH_INTERVAL_S = 0.15


class WebSocketHandler:
    """Manages a single client's WebSocket session.

    One instance is created per connected client. It owns:
    - A :class:`~agent.context.ConversationContext` scoped to this client.
    - A background asyncio task that pushes detection frames at 150ms intervals.
    - Message dispatch for all incoming protocol message types.

    The handler does **not** own the camera, detector, memory, brain, or TTS —
    those are singletons shared across all clients and injected at construction.

    Usage (called from the FastAPI WebSocket route)::

        handler = WebSocketHandler(ws, detector, memory, brain, tts, stt)
        await handler.run()
    """

    def __init__(
        self,
        websocket: WebSocket,
        detector: ObjectDetector,
        memory: VisualMemory,
        brain: IRISBrain,
        tts: TextToSpeech,
        stt: Optional[SpeechToText],
        camera: CameraCapture,
    ) -> None:
        self._ws = websocket
        self._detector = detector
        self._memory = memory
        self._brain = brain
        self._tts = tts
        self._stt = stt
        self._camera = camera

        self._detection_task: Optional[asyncio.Task] = None
        self._last_sent_frame_id: int = -1

    # ─────────────────────────── session lifecycle ─────────────────────────────

    async def run(self) -> None:
        """Accept the connection, start the detection push task, and enter the
        message receive loop. Cleans up when the client disconnects.
        """
        await self._ws.accept()
        logger.info("WebSocket client connected: %s", self._ws.client)

        self._detection_task = asyncio.create_task(self._detection_push_loop())

        try:
            await self._receive_loop()
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected: %s", self._ws.client)
        except Exception as exc:
            logger.error("WebSocket session error: %s", exc)
        finally:
            if self._detection_task and not self._detection_task.done():
                self._detection_task.cancel()
            logger.info("WebSocket session cleaned up: %s", self._ws.client)

    # ─────────────────────────── receive loop ──────────────────────────────────

    async def _receive_loop(self) -> None:
        """Continuously receive and dispatch incoming WebSocket messages."""
        while True:
            raw = await self._ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error("Invalid JSON payload.")
                continue

            msg_type = msg.get("type", "")

            if msg_type == "text_query":
                asyncio.create_task(self._handle_text_query(msg))
            elif msg_type == "audio_chunk":
                asyncio.create_task(self._handle_audio_chunk(msg))
            elif msg_type == "request_snapshot":
                asyncio.create_task(self._handle_snapshot())
            else:
                await self._send_error(f"Unknown message type: '{msg_type}'.")

    # ─────────────────────────── message handlers ─────────────────────────────

    async def _handle_text_query(self, msg: dict) -> None:
        """Process a ``text_query`` message: stream Claude's response back token-by-token.

        Args:
            msg: Parsed message dict containing ``"text"`` key.
        """
        query = msg.get("text", "").strip()
        if not query:
            await self._send_error("text_query message missing 'text' field.")
            return

        logger.info("text_query: %r", query[:80])

        result = self._detector.get_latest()
        frame = self._camera.get_frame()

        full_tokens: list[str] = []

        try:
            async for token in self._brain.query(
                user_text=query,
                frame=frame,
                detection_result=result,
            ):
                await self._ws.send_json({"type": "text_token", "token": token})
                full_tokens.append(token)
        except Exception as exc:
            logger.error("Brain query error: %s", exc)
            await self._send_error(str(exc))
            return

        full_text = "".join(full_tokens)
        await self._ws.send_json({"type": "text_done", "full": full_text})

        # Synthesize TTS in background — don't block the response being shown
        asyncio.create_task(self._synthesize_and_send(full_text))

    async def _handle_audio_chunk(self, msg: dict) -> None:
        """Process an ``audio_chunk`` message from the client's microphone.

        Decodes base64 PCM/WAV bytes, transcribes via Whisper, then routes
        the transcript through the same text_query pipeline.

        Args:
            msg: Parsed message dict containing ``"data"`` key (base64 string).
        """
        data_b64 = msg.get("data", "")
        if not data_b64:
            await self._send_error("audio_chunk message missing 'data' field.")
            return

        if self._stt is None:
            await self._send_error("STT is not available (no microphone configured).")
            return

        try:
            audio_bytes = base64.b64decode(data_b64)
        except Exception:
            await self._send_error("audio_chunk 'data' is not valid base64.")
            return

        # Transcription is CPU-bound — run in executor
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(
                None, self._stt.transcribe_audio_bytes, audio_bytes
            )
        except Exception as exc:
            logger.warning("Audio transcription error: %s", exc)
            await self._send_error("Transcription failed.")
            return

        if not text:
            logger.debug("Audio chunk produced empty transcript — ignoring.")
            return

        # Route transcript through the standard query pipeline
        await self._handle_text_query({"type": "text_query", "text": text})

    async def _handle_snapshot(self) -> None:
        """Process a ``request_snapshot`` message.

        Sends the latest detection result immediately (bypasses the 150ms push interval).
        """
        result = self._detector.get_latest()
        if result is None:
            await self._send_error("No detections available yet.")
            return
        await self._send_detections(result)

    # ─────────────────────────── detection push loop ──────────────────────────

    async def _detection_push_loop(self) -> None:
        """Push detection bounding boxes to the client every 150ms.

        Runs as an independent asyncio task so queries don't stall the overlay.
        Only sends if a new inference result is available (frame_id changed).
        """
        while True:
            try:
                result = self._detector.get_latest()
                if result is not None and result.frame_id != self._last_sent_frame_id:
                    await self._send_detections(result)
                    self._last_sent_frame_id = result.frame_id
            except Exception as exc:
                logger.debug("Detection push error: %s", exc)

            await asyncio.sleep(_DETECTION_PUSH_INTERVAL_S)

    # ─────────────────────────── TTS helper ───────────────────────────────────

    async def _synthesize_and_send(self, text: str) -> None:
        """Synthesize TTS for *text* and send the audio over the WebSocket.

        Args:
            text: The full response text to synthesize.
        """
        try:
            wav_bytes = await self._tts.synthesize(text)
            b64_audio = base64.b64encode(wav_bytes).decode("utf-8")
            await self._ws.send_json({"type": "audio_chunk", "data": b64_audio})
        except Exception as exc:
            logger.warning("TTS synthesis error: %s", exc)

    # ─────────────────────────── send helpers ─────────────────────────────────

    async def _send_detections(self, result: DetectionResult) -> None:
        """Serialize and send a ``detections`` message.

        Args:
            result: The :class:`~core.detector.DetectionResult` to serialize.
        """
        boxes = [d.to_dict() for d in result.detections]
        await self._ws.send_json({
            "type": "detections",
            "boxes": boxes,
            "frame_id": result.frame_id,
        })

    async def _send_error(self, message: str) -> None:
        """Send an ``error`` message to the client.

        Args:
            message: Human-readable error description.
        """
        logger.warning("Sending error to client: %s", message)
        try:
            await self._ws.send_json({"type": "error", "message": message})
        except Exception:
            pass   # client may have already disconnected
